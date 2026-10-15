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
}


def make_env(env_name: str, param=None) -> StartEnvironment:
    """Instantiate a StartEnvironment by name with an optional numeric param."""
    if env_name not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown env {env_name!r}. Choose from: {list(ENVIRONMENTS)}")
    cls = ENVIRONMENTS[env_name]
    return cls(param) if param is not None else cls()
