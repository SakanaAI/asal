import time
import torch
import taichi as ti
from .substrate import Substrate


@ti.data_oriented
class Visualization:
    """
    Taichi GGUI visualizer for a Substrate.

    Renders up to 3 channels as RGB, with selectable normalization and view modes.
    Optionally reserves a side panel area so controls don't overlap the simulation.

    Parameters
    ----------
    substrate   : Substrate
    chids       : list of up to 3 channel name strings (e.g. ["energy", "infra"])
    panel_width : int  — if > 0, window is widened by this many pixels; the right
                  strip is rendered dark and sub_windows are positioned there.
                  Access self.panel_x / self.panel_wfrac for sub_window placement.

    Normalization modes (self.norm_mode):
        0  Max-norm    — divide by per-channel max (default; relative scale)
        1  Reinhard    — raw / (raw + midpoint); never clips, handles outliers
        2  Percentile  — divide by per-channel Nth percentile then clip to 1
        3  Log         — log(raw+1) / log(max+2)

    View modes (self.view_mode):
        0  RGB         — ch0→R, ch1→G, ch2→B (standard blend)
        1  ch0 only    — show first channel in red, others black
        2  ch1 only    — show second channel in green, others black
        3  ch2 only    — show third channel in blue, others black
        4  ch0 + ch1   — first two channels, no blue

    Sub_window insertion
    --------------------
    Override render_opt_window() and call super().render_opt_window() first.
    All sub_windows must be created before window.show() (called by show()).
    Or use render() + show() directly in your loop instead of update().
    """

    def __init__(self,
                 substrate: Substrate,
                 chids: list = None,
                 chinds: list = None,
                 name: str = None,
                 scale: int = None,
                 panel_width: int = 0):

        self.substrate = substrate
        self.w = substrate.w
        self.h = substrate.h
        self.chids = chids or []
        self.panel_width = panel_width

        chinds_vec = substrate.get_inds_tivec(chids)
        self.chinds = torch.tensor(list(chinds_vec), device=substrate.torch_device)

        # Ensure exactly 3 entries (pad with last channel if fewer given)
        while len(self.chinds) < 3:
            self.chinds = torch.cat([self.chinds, self.chinds[-1:]])

        # Scale: target 800 px on the longer axis for the simulation area
        if scale is None:
            max_dim = max(self.w, self.h)
            scale = max(1, 800 // max_dim)
        self.scale = scale
        self.img_w = self.w * scale
        self.img_h = self.h * scale

        # Window dimensions (simulation area + optional side panel)
        self.sim_w   = self.img_w                          # simulation pixel width
        total_w      = self.img_w + panel_width
        self.name    = name or "Vis"

        # Panel fractional coords for sub_window positioning
        if panel_width > 0:
            self.panel_x     = self.img_w / total_w + 0.005
            self.panel_wfrac = (panel_width / total_w) - 0.01
        else:
            # Default: top-left overlay
            self.panel_x     = 0.05
            self.panel_wfrac = min(480 / total_w, 1.0)

        # Taichi image field (full window width including panel)
        self.image = ti.Vector.field(n=3, dtype=ti.f32, shape=(total_w, self.img_h))

        self.window = ti.ui.Window(
            self.name, (total_w, self.img_h), fps_limit=200, vsync=True)
        self.canvas = self.window.get_canvas()
        self.gui    = self.window.get_gui()

        # Rendering / normalization state
        self.norm_mode    = 0       # 0=max 1=Reinhard 2=percentile 3=log
        self.view_mode    = 0       # 0=RGB 1=ch0 2=ch1 3=ch2 4=ch0+ch1
        self.reinhard_mid = 0.5     # Reinhard knee
        self.percentile_p = 99.0   # percentile for mode 2

        # Interaction state
        self.paused               = False
        self.brush_radius         = 4
        self.mutating             = False
        self.perturbation_strength = 0.1
        self.drawing              = False
        self.prev_time            = time.time()
        self.prev_pos             = self.window.get_cursor_pos()
        self.channel_to_paint     = 0
        self.val_to_paint         = 0.1

    # ------------------------------------------------------------------
    # Taichi kernels
    # ------------------------------------------------------------------

    @ti.kernel
    def add_val_to_loc(self,
                       val: ti.f32,
                       pos_x: ti.f32,
                       pos_y: ti.f32,
                       radius: ti.i32,
                       channel_to_paint: ti.i32,
                       mem: ti.types.ndarray()):
        # pos_x/pos_y are fractions of the SIMULATION area (not full window)
        ind_x = int(pos_x * self.w)
        ind_y = int(pos_y * self.h)
        for i, j in ti.ndrange((-radius, radius), (-radius, radius)):
            if (i ** 2) + j ** 2 < radius ** 2:
                mem[0, channel_to_paint,
                    (i + ind_x) % self.w,
                    (j + ind_y) % self.h] += val

    @ti.kernel
    def write_to_renderer(self,
                          mem:        ti.types.ndarray(),
                          scale_vals: ti.types.ndarray(),
                          chinds:     ti.types.ndarray(),
                          mask:       ti.types.ndarray(),
                          norm_mode:  ti.i32,
                          sim_w:      ti.i32):
        for i, j in self.image:
            if i < sim_w:
                xind = (i // self.scale) % self.w
                yind = (j // self.scale) % self.h
                for k in ti.static(range(3)):
                    if mask[k] == 1:
                        raw = mem[0, chinds[k], xind, yind]
                        sc  = scale_vals[k]
                        if norm_mode == 0:    # max-norm
                            self.image[i, j][k] = raw / (sc + 1e-8)
                        elif norm_mode == 1:  # Reinhard
                            self.image[i, j][k] = raw / (raw + sc + 1e-8)
                        elif norm_mode == 2:  # percentile-clip
                            self.image[i, j][k] = ti.min(raw / (sc + 1e-8), 1.0)
                        else:                 # log
                            self.image[i, j][k] = (
                                ti.log(raw + 1.0) / ti.log(sc + 2.0))
                    else:
                        self.image[i, j][k] = 0.0
            else:
                # Side panel — dark background
                self.image[i, j][0] = 0.06
                self.image[i, j][1] = 0.06
                self.image[i, j][2] = 0.06

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _compute_render_params(self):
        """Return (scale_vals, mask) tensors for the current norm/view modes."""
        dev = self.substrate.torch_device

        # --- scale_vals ---
        ch_maxes = torch.tensor(
            [float(self.substrate.mem[0, int(ch)].max()) for ch in self.chinds],
            device=dev)

        if self.norm_mode == 0:      # max-norm
            scale_vals = ch_maxes
        elif self.norm_mode == 1:    # Reinhard — scale_vals = midpoint
            scale_vals = torch.full((3,), self.reinhard_mid, device=dev)
        elif self.norm_mode == 2:    # percentile
            scale_vals = torch.tensor([
                float(torch.quantile(
                    self.substrate.mem[0, int(ch)].flatten().float(),
                    self.percentile_p / 100.0))
                for ch in self.chinds
            ], device=dev)
        else:                        # log — still uses max for denominator
            scale_vals = ch_maxes

        # --- mask (which RGB slots are active) ---
        mask_map = {
            0: [1, 1, 1],   # RGB
            1: [1, 0, 0],   # ch0 only
            2: [0, 1, 0],   # ch1 only
            3: [0, 0, 1],   # ch2 only
            4: [1, 1, 0],   # ch0 + ch1
        }
        mask = torch.tensor(mask_map[self.view_mode], dtype=torch.int32, device=dev)
        return scale_vals, mask

    # ------------------------------------------------------------------
    # GUI controls
    # ------------------------------------------------------------------

    def opt_window(self, sub_w):
        """Populate the Options sub_window. Override or extend in subclasses."""
        # --- Paint controls ---
        ch_name = (self.chids[self.channel_to_paint]
                   if self.channel_to_paint < len(self.chids)
                   else str(self.channel_to_paint))
        self.channel_to_paint = sub_w.slider_int(
            f"Paint ch: {ch_name}", self.channel_to_paint,
            0, self.substrate.mem.shape[1] - 1)
        self.val_to_paint = sub_w.slider_float(
            "Paint value", self.val_to_paint, -1.0, 1.0)
        self.val_to_paint = round(self.val_to_paint * 10) / 10
        self.brush_radius = sub_w.slider_int(
            "Brush radius", self.brush_radius, 1, 200)
        self.paused = sub_w.checkbox("Pause", self.paused)
        self.mutating = sub_w.checkbox("Perturb weights", self.mutating)
        self.perturbation_strength = sub_w.slider_float(
            "Perturbation strength", self.perturbation_strength, 0.0, 5.0)

    def _draw_norm_controls(self, sub_w):
        """Norm mode buttons + conditional parameter slider."""
        norm_labels = ["* Max-norm", "* Reinhard", "* Percentile", "* Log"]
        plain_labels = ["  Max-norm", "  Reinhard", "  Percentile", "  Log"]
        for i, (active, plain) in enumerate(zip(norm_labels, plain_labels)):
            label = active if self.norm_mode == i else plain
            if sub_w.button(label):
                self.norm_mode = i
        if self.norm_mode == 1:
            self.reinhard_mid = sub_w.slider_float(
                "Reinhard mid", self.reinhard_mid, 0.01, 5.0)
        elif self.norm_mode == 2:
            self.percentile_p = sub_w.slider_float(
                "Percentile", self.percentile_p, 80.0, 100.0)

    def _draw_view_controls(self, sub_w):
        """View mode buttons (labels derived from initial chids).

        NOTE: these only change self.view_mode (the channel mask). They do NOT
        update self.chinds. Subclasses that allow chinds to change (e.g. via
        sliders) should use preset buttons that update both chinds and view_mode
        together rather than calling this method.
        """
        def _ch_name(i):
            return self.chids[i] if i < len(self.chids) else f"ch{i}"

        view_labels = [
            "RGB (all)",
            _ch_name(0),
            _ch_name(1) if len(self.chids) > 1 else "ch1",
            _ch_name(2) if len(self.chids) > 2 else "ch2",
            f"{_ch_name(0)} + {_ch_name(1) if len(self.chids) > 1 else 'ch1'}",
        ]
        for i, label in enumerate(view_labels):
            marker = "* " if self.view_mode == i else "  "
            if sub_w.button(marker + label):
                self.view_mode = i

    def _norm_view_window(self, sub_w):
        """Norm controls + view controls combined (for simple experiments with
        fixed chinds). For experiments with mutable chinds (sliders), call
        _draw_norm_controls() and manage view/channel selection yourself."""
        self._draw_norm_controls(sub_w)
        self._draw_view_controls(sub_w)

    def render_opt_window(self):
        """Draw the base GUI panel. Subclasses can override this entirely and
        call self._norm_view_window(sw) / self.opt_window(sw) directly inside
        their own sub_window blocks for custom layouts."""
        with self.gui.sub_window("Controls", self.panel_x, 0.01,
                                 self.panel_wfrac, 0.98) as sub_w:
            self._norm_view_window(sub_w)
            self.opt_window(sub_w)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def check_events(self):
        for e in self.window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                exit()
            if e.key == ti.ui.LMB and self.window.is_pressed(ti.ui.SHIFT):
                self.drawing = True
            elif e.key == ti.ui.SPACE:
                self.substrate.mem *= 0.0
        for e in self.window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                self.drawing = False

    # ------------------------------------------------------------------
    # Main render / update loop
    # ------------------------------------------------------------------

    def render(self):
        """Render image + draw base GUI panels. Call show() when done adding
        any extra sub_windows."""
        current_time = time.time()
        current_pos  = self.window.get_cursor_pos()

        if not self.paused:
            self.check_events()
            if self.drawing and (current_time - self.prev_time) > 0.05:
                # Map cursor from full-window fraction → simulation fraction
                sim_frac = self.sim_w / self.image.shape[0]
                cx_sim = current_pos[0] / sim_frac  # re-scale to [0,1] over sim
                self.add_val_to_loc(
                    self.val_to_paint, cx_sim, current_pos[1],
                    self.brush_radius, self.channel_to_paint,
                    self.substrate.mem)
                self.prev_time = current_time
                self.prev_pos  = current_pos

        scale_vals, mask = self._compute_render_params()
        self.write_to_renderer(
            self.substrate.mem, scale_vals, self.chinds, mask,
            self.norm_mode, self.sim_w)

        self.render_opt_window()
        self.canvas.set_image(self.image)

    def show(self):
        """Display the current frame. Must be called after render() and any
        extra sub_windows."""
        self.window.show()

    def update(self):
        """Convenience wrapper: render() + show(). For simple experiments that
        don't need extra sub_windows after the base panels."""
        self.render()
        self.show()
