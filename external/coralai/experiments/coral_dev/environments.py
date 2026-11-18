"""
Non-homogeneous start environments for coral_dev.

Each class wraps spatial mask logic (where 0 = uninhabitable, 1 = habitable)
and implements the StartEnvironment interface. Replaces the old mask-based
functions in env.py.

Usage:
    from environments import make_env
    env = make_env("hole", param=0.35)
    env.seed(substrate)          # called once after evolver is ready
    if env.persist:
        env.step(substrate)      # called each step to keep barren zones dead
"""

import random

import torch

from coralai.environment import StartEnvironment

import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circle_mask(w, h, cx, cy, radius, device):
    """Binary mask: 1 outside circle, 0 inside (hole). radius in pixels."""
    ys = torch.arange(h, device=device, dtype=torch.float32).view(h, 1).expand(h, w)
    xs = torch.arange(w, device=device, dtype=torch.float32).view(1, w).expand(h, w)
    dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
    return (dist_sq > radius ** 2).float()


def _apply_mask(substrate, mask):
    """Zero energy, infra, and genome where mask==0 (barren cells)."""
    inds = substrate.ti_indices[None]
    inv = (1.0 - mask).bool()
    substrate.mem[0, inds.energy].masked_fill_(inv, 0.0)
    substrate.mem[0, inds.infra].masked_fill_(inv, 0.0)
    substrate.mem[0, inds.genome].masked_fill_(inv, -1.0)


def _seed_habitable(substrate, mask, energy=0.4, infra=0.4):
    """Add energy and infra only where mask==1."""
    inds = substrate.ti_indices[None]
    substrate.mem[0, inds.energy].add_(mask * energy).clamp_(0.01, 100)
    substrate.mem[0, inds.infra].add_(mask * infra).clamp_(0.01, 100)


# ---------------------------------------------------------------------------
# Flat (default)
# ---------------------------------------------------------------------------

class FlatEnvironment(StartEnvironment):
    """Homogeneous environment — no spatial structure. seed and step are no-ops."""
    name = "flat"


# ---------------------------------------------------------------------------
# Hole / Ring (circular barren centre)
# ---------------------------------------------------------------------------

class HoleEnvironment(StartEnvironment):
    """Circular barren zone in the grid centre.

    param radius_frac: fraction of min(W,H)/2 that is barren (default 0.35).
    """
    name = "hole"

    def __init__(self, radius_frac: float = 0.35):
        self.radius_frac = radius_frac
        self._mask_cache = None

    def _mask(self, substrate):
        mem = substrate.mem
        w, h = mem.shape[2], mem.shape[3]
        device = substrate.torch_device
        if self._mask_cache is None or self._mask_cache.shape != (w, h):
            radius = self.radius_frac * min(w, h) / 2.0
            self._mask_cache = _circle_mask(w, h, w / 2.0, h / 2.0,
                                            radius, device)
        return self._mask_cache

    def seed(self, substrate):
        mask = self._mask(substrate)
        _apply_mask(substrate, mask)
        _seed_habitable(substrate, mask)

    def step(self, substrate):
        _apply_mask(substrate, self._mask(substrate))

    @property
    def persist(self) -> bool:
        return True

    def to_dict(self) -> dict:
        return {"name": self.name, "radius_frac": self.radius_frac}


class RingEnvironment(HoleEnvironment):
    """Same as HoleEnvironment (barren centre, ring of life around it)."""
    name = "ring"


# ---------------------------------------------------------------------------
# Stripes (alternating vertical bands)
# ---------------------------------------------------------------------------

class StripesEnvironment(StartEnvironment):
    """Alternating habitable / barren vertical bands.

    param n_stripes: total number of stripes (default 5); even-indexed = habitable.
    """
    name = "stripes"

    def __init__(self, n_stripes: int = 5):
        self.n_stripes = max(1, int(n_stripes))
        self._mask_cache = None

    def _mask(self, substrate):
        mem = substrate.mem
        w, h = mem.shape[2], mem.shape[3]
        device = substrate.torch_device
        if self._mask_cache is None or self._mask_cache.shape != (w, h):
            n = self.n_stripes
            xs = torch.arange(w, device=device, dtype=torch.float32)
            stripe = (xs * n / w).long().clamp(0, n - 1)
            mask_1d = (stripe % 2 == 0).float()
            # Expand to (w, h) — mask is indexed [x, y]
            self._mask_cache = mask_1d.view(w, 1).expand(w, h)
        return self._mask_cache

    def seed(self, substrate):
        mask = self._mask(substrate)
        _apply_mask(substrate, mask)
        _seed_habitable(substrate, mask)

    def step(self, substrate):
        _apply_mask(substrate, self._mask(substrate))

    @property
    def persist(self) -> bool:
        return True

    def to_dict(self) -> dict:
        return {"name": self.name, "n_stripes": self.n_stripes}


# ---------------------------------------------------------------------------
# Corners (four diagonal barren corners)
# ---------------------------------------------------------------------------

class CornersEnvironment(StartEnvironment):
    """Four diagonal barren corners; life fills a central diamond.

    param corner_frac: fraction of side removed per corner (default 0.25).
    """
    name = "corners"

    def __init__(self, corner_frac: float = 0.25):
        self.corner_frac = max(0.01, min(0.49, float(corner_frac)))
        self._mask_cache = None

    def _mask(self, substrate):
        mem = substrate.mem
        w, h = mem.shape[2], mem.shape[3]
        device = substrate.torch_device
        if self._mask_cache is None or self._mask_cache.shape != (w, h):
            frac = self.corner_frac
            ys = torch.arange(h, device=device, dtype=torch.float32).view(1, h).expand(w, h)
            xs = torch.arange(w, device=device, dtype=torch.float32).view(w, 1).expand(w, h)
            tl = (xs / w + ys / h) < (frac * 2)
            tr = ((w - 1 - xs) / w + ys / h) < (frac * 2)
            bl = (xs / w + (h - 1 - ys) / h) < (frac * 2)
            br = ((w - 1 - xs) / w + (h - 1 - ys) / h) < (frac * 2)
            self._mask_cache = (~(tl | tr | bl | br)).float()
        return self._mask_cache

    def seed(self, substrate):
        mask = self._mask(substrate)
        _apply_mask(substrate, mask)
        _seed_habitable(substrate, mask)

    def step(self, substrate):
        _apply_mask(substrate, self._mask(substrate))

    @property
    def persist(self) -> bool:
        return True

    def to_dict(self) -> dict:
        return {"name": self.name, "corner_frac": self.corner_frac}


# ---------------------------------------------------------------------------
# Patches (spatial nutrient patches with injection + capacity regen)
# ---------------------------------------------------------------------------

class PatchesEnvironment(StartEnvironment):
    """Nutrient patches with extinction and birth dynamics.

    Patches are the *sole* energy source — global day/night injection has been
    removed.  Each patch cell holds a finite energy reserve; once that reserve
    is exhausted the cell goes dark (extinction).  Every `spawn_interval` steps
    a wave of new patches spawns at random locations (birth), keeping the
    ecosystem in a non-equilibrium state that rewards foraging.

    param n_patches: patches to seed at start and per spawn wave (passed via
                     --env-param; default 8).

    Keyword parameters (edit class or subclass to change without CLI):
      patch_radius_frac : float  — patch radius as fraction of min(W,H) (0.07)
      max_capacity      : float  — energy reserve per cell at birth (30.0)
      inject_rate       : float  — energy released per cell per step (0.06)
      spawn_interval    : int    — steps between spawn waves (250)
      n_spawn           : int    — patches per spawn wave; None → same as n_patches
      seed_infra        : float  — infra seeded into fresh patches at birth (0.4)
    """
    name = "patches"

    def __init__(self, n_patches: int = 8, *,
                 patch_radius_frac: float = 0.07,
                 max_capacity: float = 30.0,
                 inject_rate: float = 0.06,
                 spawn_interval: int = 250,
                 n_spawn: int = None,
                 seed_infra: float = 0.4):
        self.n_patches         = max(1, int(n_patches))
        self.patch_radius_frac = float(patch_radius_frac)
        self.max_capacity      = float(max_capacity)
        self.inject_rate       = float(inject_rate)
        self.spawn_interval    = max(1, int(spawn_interval))
        self.n_spawn           = int(n_spawn) if n_spawn is not None else self.n_patches
        self.seed_infra        = float(seed_infra)
        self._capacity  = None   # (W, H) tensor; 0 = dead, >0 = energy reserve
        self._step_count = 0
        self._xs = None          # cached arange tensors for patch drawing
        self._ys = None

    def _ensure_coords(self, w, h, device):
        if self._xs is None or self._xs.shape[0] != w:
            self._xs = torch.arange(w, device=device, dtype=torch.float32).view(w, 1)
            self._ys = torch.arange(h, device=device, dtype=torch.float32).view(1, h)

    def _add_patches(self, n, w, h, substrate):
        """Draw n circles at random locations; set their capacity to max_capacity
        and seed infra into the substrate at those locations."""
        device = substrate.torch_device
        self._ensure_coords(w, h, device)
        radius = self.patch_radius_frac * min(w, h)
        inds = substrate.ti_indices[None]
        for _ in range(n):
            cx = random.uniform(0, w)
            cy = random.uniform(0, h)
            dist_sq = (self._xs - cx) ** 2 + (self._ys - cy) ** 2
            circle = (dist_sq <= radius ** 2).float()
            # Only activate cells that are currently dead (capacity == 0)
            new_cells = circle * (self._capacity == 0).float()
            self._capacity += new_cells * self.max_capacity
            substrate.mem[0, inds.infra].add_(new_cells * self.seed_infra).clamp_(0, 100)

    def seed(self, substrate):
        mem = substrate.mem
        w, h = mem.shape[2], mem.shape[3]
        device = substrate.torch_device
        self._capacity = torch.zeros(w, h, device=device)
        self._step_count = 0
        self._add_patches(self.n_patches, w, h, substrate)

    def step(self, substrate):
        inds = substrate.ti_indices[None]
        # Release energy proportional to remaining reserve; deplete reserve
        injection = self._capacity.clamp(max=self.inject_rate)
        substrate.mem[0, inds.energy].add_(injection).clamp_(0, 100)
        self._capacity.sub_(injection).clamp_(min=0)

        self._step_count += 1
        if self._step_count % self.spawn_interval == 0:
            mem = substrate.mem
            w, h = mem.shape[2], mem.shape[3]
            self._add_patches(self.n_spawn, w, h, substrate)

    @property
    def persist(self) -> bool:
        return True

    def to_dict(self) -> dict:
        return {
            "name":              self.name,
            "n_patches":         self.n_patches,
            "patch_radius_frac": self.patch_radius_frac,
            "max_capacity":      self.max_capacity,
            "inject_rate":       self.inject_rate,
            "spawn_interval":    self.spawn_interval,
            "n_spawn":           self.n_spawn,
        }


# ---------------------------------------------------------------------------
# Oases (grid of isolated regions with narrow bridges + patch energy)
# ---------------------------------------------------------------------------

class OasesEnvironment(StartEnvironment):
    """n_cols × n_rows oases separated by blacked-out walls with narrow bridges.

    Walls are enforced every step — nothing survives there.  Each oasis
    contains a small number of nutrient patches which are the sole energy
    source.  Patches deplete over time and respawn within their oasis every
    `spawn_interval` steps, creating foraging pressure and boom/bust dynamics.

    param n_cols: number of oasis columns (default 3; passed via --env-param).

    Keyword parameters:
      n_rows            : int   — oasis rows; None → same as n_cols (default None)
      wall_width        : int   — wall thickness in cells (default 10)
      bridge_width      : int   — bridge gap in cells perpendicular to wall (default 4)
      patches_per_region: int   — nutrient patches per oasis (default 2)
      patch_radius_frac : float — patch radius as fraction of min(region_w, region_h) (0.12)
      max_capacity      : float — energy reserve per patch cell at birth (25.0)
      inject_rate       : float — energy released per patch cell per step (0.05)
      spawn_interval    : int   — steps between patch respawn waves (300)
      seed_infra        : float — infra seeded at patch birth (0.4)
    """
    name = "oases"

    def __init__(self, n_cols: int = 3, *,
                 n_rows: int = None,
                 wall_width: int = 10,
                 bridge_width: int = 4,
                 patches_per_region: int = 2,
                 patch_radius_frac: float = 0.12,
                 max_capacity: float = 25.0,
                 inject_rate: float = 0.05,
                 spawn_interval: int = 300,
                 seed_infra: float = 0.4):
        self.n_cols             = max(1, int(n_cols))
        self.n_rows             = max(1, int(n_rows)) if n_rows is not None else self.n_cols
        self.wall_width         = max(2, int(wall_width))
        self.bridge_width       = max(1, int(bridge_width))
        self.patches_per_region = max(0, int(patches_per_region))
        self.patch_radius_frac  = float(patch_radius_frac)
        self.max_capacity       = float(max_capacity)
        self.inject_rate        = float(inject_rate)
        self.spawn_interval     = max(1, int(spawn_interval))
        self.seed_infra         = float(seed_infra)
        self._mask         = None   # (w, h) float32: 0=wall, 1=oasis/bridge
        self._region_boxes = None   # list of (x0, x1, y0, y1) pixel bounds
        self._capacity     = None   # (w, h) float32 patch energy reserve
        self._step_count   = 0

    # ------------------------------------------------------------------
    def _build_topology(self, w, h, device):
        """Return (mask, region_boxes).

        Walls are punched first, then bridges are restored within each wall
        at the midpoint of every adjacent oasis.  Bridges are never placed at
        wall–wall intersections (corners stay blocked).
        """
        ww = self.wall_width
        bw = min(self.bridge_width, ww)

        # Compute wall ranges and oasis spans along x (vertical walls between cols)
        x_wall_ranges = []
        x_region_edges = [0]
        for c in range(1, self.n_cols):
            wc = int(round(c * w / self.n_cols))
            ws = max(0, wc - ww // 2)
            we = min(w, ws + ww)
            x_wall_ranges.append((ws, we))
            x_region_edges.append(ws)   # end of col c-1
            x_region_edges.append(we)   # start of col c
        x_region_edges.append(w)
        # oasis col spans: edges at indices [2c, 2c+1]
        x_spans = [(x_region_edges[2 * c], x_region_edges[2 * c + 1])
                   for c in range(self.n_cols)]

        # Same along y (horizontal walls between rows)
        y_wall_ranges = []
        y_region_edges = [0]
        for r in range(1, self.n_rows):
            wc = int(round(r * h / self.n_rows))
            ws = max(0, wc - ww // 2)
            we = min(h, ws + ww)
            y_wall_ranges.append((ws, we))
            y_region_edges.append(ws)
            y_region_edges.append(we)
        y_region_edges.append(h)
        y_spans = [(y_region_edges[2 * r], y_region_edges[2 * r + 1])
                   for r in range(self.n_rows)]

        # Base mask: habitable everywhere, then punch walls
        mask = torch.ones(w, h, device=device)
        for (ws, we) in x_wall_ranges:
            mask[ws:we, :] = 0.0
        for (ws, we) in y_wall_ranges:
            mask[:, ws:we] = 0.0

        # Bridges through vertical walls — gap in y, placed at each y-oasis centre
        for (ws, we) in x_wall_ranges:
            for (y0, y1) in y_spans:
                yc = (y0 + y1) // 2
                bys = max(0, yc - bw // 2)
                bye = min(h, bys + bw)
                mask[ws:we, bys:bye] = 1.0

        # Bridges through horizontal walls — gap in x, placed at each x-oasis centre
        for (ws, we) in y_wall_ranges:
            for (x0, x1) in x_spans:
                xc = (x0 + x1) // 2
                bxs = max(0, xc - bw // 2)
                bxe = min(w, bxs + bw)
                mask[bxs:bxe, ws:we] = 1.0

        region_boxes = [(x0, x1, y0, y1)
                        for (x0, x1) in x_spans
                        for (y0, y1) in y_spans]
        return mask, region_boxes

    # ------------------------------------------------------------------
    def _spawn_patches(self, substrate):
        """Place patches randomly inside each oasis (only into dead cells)."""
        mem = substrate.mem
        w, h = mem.shape[2], mem.shape[3]
        device = substrate.torch_device
        inds = substrate.ti_indices[None]
        xs = torch.arange(w, device=device, dtype=torch.float32).view(w, 1)
        ys = torch.arange(h, device=device, dtype=torch.float32).view(1, h)

        for (x0, x1, y0, y1) in self._region_boxes:
            rw, rh = x1 - x0, y1 - y0
            if rw <= 0 or rh <= 0:
                continue
            patch_r = max(1.0, self.patch_radius_frac * min(rw, rh))
            margin = int(patch_r) + 1
            cx_lo = x0 + margin;  cx_hi = max(cx_lo + 1, x1 - margin)
            cy_lo = y0 + margin;  cy_hi = max(cy_lo + 1, y1 - margin)
            for _ in range(self.patches_per_region):
                cx = random.uniform(cx_lo, cx_hi)
                cy = random.uniform(cy_lo, cy_hi)
                dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
                circle = (dist_sq <= patch_r ** 2).float()
                # Only activate dead cells; never bleed into walls
                new_cells = circle * (self._capacity == 0).float() * self._mask
                self._capacity.add_(new_cells * self.max_capacity)
                mem[0, inds.infra].add_(new_cells * self.seed_infra).clamp_(0, 100)

    # ------------------------------------------------------------------
    def seed(self, substrate):
        mem = substrate.mem
        w, h = mem.shape[2], mem.shape[3]
        device = substrate.torch_device
        self._mask, self._region_boxes = self._build_topology(w, h, device)
        self._capacity   = torch.zeros(w, h, device=device)
        self._step_count = 0
        self._spawn_patches(substrate)
        _apply_mask(substrate, self._mask)

    def step(self, substrate):
        inds = substrate.ti_indices[None]
        # Keep walls dead
        _apply_mask(substrate, self._mask)
        # Release energy from patch reserves
        injection = self._capacity.clamp(max=self.inject_rate)
        substrate.mem[0, inds.energy].add_(injection).clamp_(0, 100)
        self._capacity.sub_(injection).clamp_(min=0)
        # Periodic patch respawn inside oases
        self._step_count += 1
        if self._step_count % self.spawn_interval == 0:
            self._spawn_patches(substrate)

    @property
    def persist(self) -> bool:
        return True

    def to_dict(self) -> dict:
        return {
            "name":              self.name,
            "n_cols":            self.n_cols,
            "n_rows":            self.n_rows,
            "wall_width":        self.wall_width,
            "bridge_width":      self.bridge_width,
            "patches_per_region": self.patches_per_region,
            "patch_radius_frac": self.patch_radius_frac,
            "max_capacity":      self.max_capacity,
            "inject_rate":       self.inject_rate,
            "spawn_interval":    self.spawn_interval,
        }


### TRAIL ENVIRONMENT

# ---------------------------------------------------------------------------
# Trail (food along a winding path; a few founders spawn at the start)
# ---------------------------------------------------------------------------

class TrailEnvironment(StartEnvironment):
    """A serpentine food trail starting in the top-left corner.

    Life is seeded only as a few small founder blobs at the trail's start.
    The trail holds finite energy reserves (released into the substrate at
    inject_rate per step) that grow richer toward the far end.

    param n_lanes: number of back-and-forth lanes (default 4; --env-param).

    Keyword parameters:
      trail_width    : float — trail width in cells (4.0)
      wiggle_frac    : float — lane wobble as fraction of min(W,H) (0.03)
      margin_frac    : float — edge margin; also trail start offset (0.08)
      max_capacity   : float — reserve per trail cell at the start (20.0)
      goal_boost     : float — far end holds (1 + goal_boost)x reserve (2.0)
      inject_rate    : float — energy released per trail cell per step (0.02)
      init_energy    : float — standing energy on trail + founders at t=0 (0.4)
      seed_infra     : float — infra added on trail + founders (0.4)
      n_founders     : int   — founder blobs at the start (3)
      founder_radius : float — founder blob radius in cells (3.0)
      spawn_frac     : float — side of the founder spawn box, fraction of min(W,H) (0.12)
      shift_interval : int   — completed updates between reserve translations; 0 disables
      shift_cells    : int   — cells to translate remaining reserve along y, with wrapping
    """
    name = "trail"

    def __init__(self, n_lanes: int = 4, *,
                 spawn: str = "spread",
                 trail_width: float = 4.0,
                 wiggle_frac: float = 0.03,
                 margin_frac: float = 0.08,
                 max_capacity: float = 20.0,
                 goal_boost: float = 2.0,
                 inject_rate: float = 0.02,
                 init_energy: float = 0.4,
                 seed_infra: float = 0.4,
                 n_founders: int = 1,
                 founder_radius: float = 3.0,
                 spawn_frac: float = 0.01,
                 shift_interval: int = 0,
                 shift_cells: int = 1):
        if spawn not in {"spread", "corner"}:
            raise ValueError("trail spawn must be spread or corner")
        for key, value in (("trail_width", trail_width), ("max_capacity", max_capacity),
                           ("inject_rate", inject_rate)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"trail {key} must be finite and positive")
        for key, value in (("shift_interval", shift_interval), ("shift_cells", shift_cells)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"trail {key} must be a nonnegative integer")
        self.n_lanes        = max(1, int(n_lanes))
        self.spawn          = spawn
        self.trail_width    = float(trail_width)
        self.wiggle_frac    = float(wiggle_frac)
        self.margin_frac    = float(margin_frac)
        self.max_capacity   = float(max_capacity)
        self.goal_boost     = float(goal_boost)
        self.inject_rate    = float(inject_rate)
        self.init_energy    = float(init_energy)
        self.seed_infra     = float(seed_infra)
        self.n_founders     = max(1, int(n_founders))
        self.founder_radius = float(founder_radius)
        self.spawn_frac     = float(spawn_frac)
        # Periodically translate only the remaining food reserve along y,
        # wrapping at the boundary. Never move cells or replenish the reserve.
        self.shift_interval = shift_interval
        self.shift_cells = shift_cells
        self._capacity = None   # (w, h) energy reserve left on the trail

    # ------------------------------------------------------------------
    def _waypoints(self, w, h):
        """Serpentine polyline in (x, y) cell coordinates, starting at (m, m)."""
        m = self.margin_frac * min(w, h)
        x_lo, x_hi = m, w - m
        n = self.n_lanes
        lane_ys = [m] if n == 1 else [m + k * (h - 2 * m) / (n - 1) for k in range(n)]
        spacing = (h - 2 * m) / (n - 1) if n > 1 else h
        amp = min(self.wiggle_frac * min(w, h), 0.3 * spacing)
        pts, n_samples = [], 32
        for k, y in enumerate(lane_ys):
            for i in range(n_samples + 1):
                u = i / n_samples
                x = x_lo + (x_hi - x_lo) * (u if k % 2 == 0 else 1.0 - u)
                # sin has period 1/2 lane -> zero at both ends, so lanes join cleanly
                yy = y + ((-1) ** k) * amp * math.sin(2 * math.pi * 2 * u)
                pts.append((x, yy))
        return pts

    def _build_trail(self, w, h, device):
        """Return (trail_mask, progress): mask in {0,1}, progress in [0,1] along path."""
        pts = self._waypoints(w, h)
        seg_len = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        total = max(sum(seg_len), 1e-6)

        xs = torch.arange(w, device=device, dtype=torch.float32).view(w, 1)
        ys = torch.arange(h, device=device, dtype=torch.float32).view(1, h)
        best = torch.full((w, h), float("inf"), device=device)
        prog = torch.zeros(w, h, device=device)

        cum = 0.0
        for (ax, ay), (bx, by), L in zip(pts[:-1], pts[1:], seg_len):
            abx, aby = bx - ax, by - ay
            L2 = max(abx * abx + aby * aby, 1e-9)
            t = (((xs - ax) * abx + (ys - ay) * aby) / L2).clamp(0.0, 1.0)
            d2 = (xs - (ax + t * abx)) ** 2 + (ys - (ay + t * aby)) ** 2
            closer = d2 < best
            best = torch.where(closer, d2, best)
            prog = torch.where(closer, (cum + t * L) / total, prog)
            cum += L

        trail = (best <= (self.trail_width / 2.0) ** 2).float()
        return trail, prog

    def _founder_map(self, w, h, device):
        """(w, h) float tensor: -1 everywhere; founder blob i holds genome index i."""
        m = self.margin_frac * min(w, h)
        half = self.spawn_frac * min(w, h) / 2.0
        xs = torch.arange(w, device=device, dtype=torch.float32).view(w, 1)
        ys = torch.arange(h, device=device, dtype=torch.float32).view(1, h)
        gmap = torch.full((w, h), -1.0, device=device)
        for i in range(self.n_founders):
            cx = random.uniform(max(0.0, m - half), m + half)
            cy = random.uniform(max(0.0, m - half), m + half)
            blob = (xs - cx) ** 2 + (ys - cy) ** 2 <= self.founder_radius ** 2
            gmap = torch.where(blob, torch.full_like(gmap, float(i)), gmap)
        return gmap

    def seed(self, substrate):
        mem = substrate.mem
        w, h = mem.shape[2], mem.shape[3]
        device = substrate.torch_device
        inds = substrate.ti_indices[None]

        trail, prog = self._build_trail(w, h, device)
        self._capacity = trail * self.max_capacity * (1.0 + self.goal_boost * prog)
        self._step_count = 0
        self._initial_capacity = float(self._capacity.double().sum())
        self._requested = self._delivered = self._clipped = 0.0
        self._last_injection = torch.zeros_like(self._capacity)

        if self.spawn == "corner":
            gmap = self._founder_map(w, h, device)
            founders = (gmap >= 0).float()
            mem[0, inds.genome].copy_(gmap)
            mem[0, inds.infra].zero_()
        else:
            founders = torch.zeros(w, h, device=device)   # genomes and infra untouched

        # Trail is the only food; founders (corner mode) get a starting meal too
        lit = torch.maximum(trail, founders)
        mem[0, inds.energy].zero_()
        mem[0, inds.energy].add_(lit * self.init_energy).clamp_(0, 100)
        mem[0, inds.infra].add_(lit * self.seed_infra).clamp_(0, 100)

    def step(self, substrate):
        inds = substrate.ti_indices[None]
        self._step_count += 1
        if self.shift_interval and self._step_count % self.shift_interval == 0:
            self._capacity = torch.roll(self._capacity, self.shift_cells, dims=1)
        injection = self._capacity.clamp(max=self.inject_rate)
        before = substrate.mem[0, inds.energy].double()
        substrate.mem[0, inds.energy].add_(injection).clamp_(0, 100)
        self._requested += float(injection.double().sum())
        self._delivered += float((substrate.mem[0, inds.energy].double() - before).sum())
        self._clipped += float((before + injection.double() - 100).clamp(min=0).sum())
        self._last_injection = injection
        self._capacity.sub_(injection).clamp_(min=0)

    def resource_state(self):
        """Food state and measured external supply after completed updates."""
        return {
            "completed_update": self._step_count,
            "capacity": self._capacity.detach().cpu(),
            "last_injection": self._last_injection.detach().cpu(),
            "initial_capacity": self._initial_capacity,
            "remaining_capacity": float(self._capacity.double().sum()),
            "cumulative_requested": self._requested,
            "cumulative_delivered": self._delivered,
            "cumulative_clipped": self._clipped,
        }

    @property
    def persist(self) -> bool:
        return True

    def to_dict(self) -> dict:
        return {
            "name": self.name, "n_lanes": self.n_lanes,
            "spawn": self.spawn,
            "trail_width": self.trail_width, "wiggle_frac": self.wiggle_frac,
            "margin_frac": self.margin_frac, "max_capacity": self.max_capacity,
            "goal_boost": self.goal_boost, "inject_rate": self.inject_rate,
            "init_energy": self.init_energy, "seed_infra": self.seed_infra,
            "n_founders": self.n_founders, "founder_radius": self.founder_radius,
            "spawn_frac": self.spawn_frac,
            "shift_interval": self.shift_interval, "shift_cells": self.shift_cells,
        }

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ENVIRONMENTS = {
    "flat":    FlatEnvironment,
    "hole":    HoleEnvironment,
    "ring":    RingEnvironment,
    "stripes": StripesEnvironment,
    "corners": CornersEnvironment,
    "patches": PatchesEnvironment,
    "oases":   OasesEnvironment,
    "trail": TrailEnvironment,
}


def make_env(env_name: str, param=None, options=None) -> StartEnvironment:
    """Instantiate by name/numeric param; optional supported trail settings."""
    if env_name not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown env {env_name!r}. Choose from: {list(ENVIRONMENTS)}")
    cls = ENVIRONMENTS[env_name]
    options = {} if options is None else options
    if not isinstance(options, dict):
        raise ValueError("env_options must be a JSON object")
    supported = {"spawn", "trail_width", "max_capacity", "inject_rate", "shift_interval", "shift_cells"}
    if options and (env_name != "trail" or set(options) - supported):
        raise ValueError("env_options supports only trail spawn, trail_width, max_capacity, inject_rate, shift_interval, shift_cells")
    return cls(param, **options) if param is not None else cls(**options)
