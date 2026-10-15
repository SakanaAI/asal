import time
import taichi as ti
from ..utils.ti_struct_factory import TaichiStructFactory
from ..substrate.substrate import Substrate
# from .vis_params import VisParams, VIS_CHIDS

@ti.data_oriented
class Vis:
    def __init__(
        self,
        world: Substrate,
        chids: list,
        name: str = None,
        scale: int = None,
    ):
        self.world = world
        self.chids = chids
        if name is None:
            name = "Vis_" + str(time.time())
        self.name = name

        chindices = self.world.get_inds_tivec(self.chids)
        if chindices.n != 3:
            raise ValueError("Vis: ch_cmaps must have 3 channels")
        self.chindices = chindices

        if scale is None:
            max_dim = max(self.world.w, self.world.h)
            desired_max_dim = 800
            scale = desired_max_dim // max_dim

        self.img_w = self.world.w * scale
        self.img_h = self.world.h * scale
        self.image = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.img_w, self.img_h))

        self.window = ti.ui.Window(
            f"{self.name}", (self.img_w, self.img_h), fps_limit=60, vsync=True
        )
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.prev_time = time.time()

        builder = TaichiStructFactory()
        builder.add_i("scale", scale)
        self.add_dynamic_params(builder)
        self.params = builder.build()


    def add_dynamic_params(self, builder: TaichiStructFactory):
        """
        This is to clean up parameters that change during the simulation and thus
        have to be passed to taichi kernels
        """
        builder.add_timat_f("chlims", self.world.get_lims_timat(self.chids))
        builder.add_tivec_i("chindices", self.chindices)
        builder.add_i("brush_radius", 4)
        builder.add_i("chnum_to_paint", 0)
        builder.add_i("chindex_to_paint", self.chindices[0])
        builder.add_f("val_to_paint", 1.5)
        builder.add_f("val_to_paint_dt", -1)
        builder.add_i("drawing", False)
        builder.add_f("mouse_posx", 0.0)
        builder.add_f("mouse_posy", 0.0)
        builder.add_f("perturb_strength", 0.1)
        builder.add_i("is_perturbing_weights", False)
        builder.add_i("is_perturbing_biases", False)
        builder.add_f("test", -1)


    def update(self):
        ps = self.params[None]
        self.check_events(self.params)
        self.render_opt_window(self.params)

        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        pos = self.window.get_cursor_pos()
        ps.mouse_posx = pos[0]
        ps.mouse_posy = pos[1]
        ps.val_to_paint_dt = ps.val_to_paint * dt * 5

        self.update_vis(self.world.mem, self.params)
        self.canvas.set_background_color((1, 1, 1))
        self.canvas.set_image(self.image)
        self.window.show()
        self.params[None] = ps


    @ti.kernel
    def update_vis(self, mem: ti.types.ndarray(), params: ti.template()):
        ps = params[None]
        if ps.drawing:
            self.paint_cursor_value(mem, params)
        self.write_to_image(mem, params)
        params[None] = ps


    @ti.func
    def write_to_image(self, mem: ti.types.ndarray(), params: ti.template()):
        ps = params[None]
        for i, j in self.image:
            xind = i // ps.scale
            yind = j // ps.scale
            for chnum in ti.static(range(3)):
                self.image[i, j][chnum] = mem[0, ps.chindices[chnum], xind, yind]
        params[None] = ps


    @ti.func
    def paint_cursor_value(self, mem: ti.types.ndarray(), params: ti.template()):
        """
        Use cursor position to paint selected value of selected channel - take into account pixel sizes
        """
        ps = params[None]
        ind_x = int(ps.mouse_posx * mem.shape[2])
        ind_y = int(ps.mouse_posy * mem.shape[3])
        for i, j in ti.ndrange(
            (-ps.brush_radius, ps.brush_radius), (-ps.brush_radius, ps.brush_radius)
        ):
            if (i**2) + j**2 < ps.brush_radius**2:
                ix = (i + ind_x) % mem.shape[2]
                jy = (j + ind_y) % mem.shape[3]
                mem[0, ps.chindex_to_paint, ix, jy] += ti.math.clamp(
                    mem[0, ps.chindex_to_paint, ix, jy] + ps.val_to_paint_dt,
                    ps.chlims[ps.chnum_to_paint, 0],
                    ps.chlims[ps.chnum_to_paint, 1],
                )
        params[None] = ps


    def render_opt_window(self, params):
        """
        Slider for brush size
        Paint channel and value
        Any sliders for world metadata desired
        """
        ps = params[None]
        opt_w = min(480 / self.img_w, self.img_w)
        opt_h = min(180 / self.img_h, self.img_h)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as w:
            ps.brush_radius = w.slider_int("Brush Radius", ps.brush_radius, 1, 20)
            ps.chnum_to_paint = w.slider_int("Channel to Paint", ps.chnum_to_paint, 0, 2)
            ps.val_to_paint = w.slider_float("Paint delta", ps.val_to_paint, -2, 2)
            ps.perturb_strength = w.slider_float(
                "Perturb Strength", ps.perturb_strength, 0.0, 5.0
            )
            ps.is_perturbing_weights = w.checkbox(
                "Perturb Weights", ps.is_perturbing_weights
            )
            ps.is_perturbing_biases = w.checkbox(
                "Perturb Biases", ps.is_perturbing_biases
            )

            ps.chindex_to_paint = ps.chindices[ps.chnum_to_paint]
            w.text(f"Painting: {self.chids[ps.chnum_to_paint]}")
        params[None] = ps


    def check_events(self, params):
        """
        Keys for pause and reset
        Cursor motion for painting
        """
        ps = params[None]
        for e in self.window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                self.window.running = False
            # if e.key == 'p':
            #     ps.is_perturbing_weights = True

        if self.window.is_pressed(ti.ui.LMB) and self.window.is_pressed(ti.ui.SPACE):
            ps.drawing = True
        else:
            ps.drawing = False

        for e in self.window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                ps.drawing = False
            # if e.key == 'p':
            #     ps.is_perturbing_weights = False
        params[None] = ps
