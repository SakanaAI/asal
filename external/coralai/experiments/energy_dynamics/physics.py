"""
Energy-only physics — no organisms, no NEAT.

Channels:
    energy  — free energy at each cell
    infra   — infrastructure landscape (static or decaying)
    source  — per-cell energy injection rate per step

Five physics modes:
    infra_attract     — energy flows toward high-infra neighbors (proportional)
    infra_attract_cap — infra attract + overflow spill when energy > max_energy
    diffuse           — energy averages equally with all neighbors (Laplacian)
    infra_repel       — energy flows toward low-infra neighbors (inverse proportion)
    infra_decay       — infra attract + cap, but infra slowly decays over time

Six environments (init_env):
    patch    — source patches at corners + center, flat low infra
    gradient — uniform source, diagonal infra gradient
    ring     — concentric infra peak at center, source strongest at edges
    rivers   — diagonal sinusoidal infra channels, peak-aligned sources
    volcano  — center source, ring of infra surrounding it
    random   — smooth sinusoidal noise for both infra and source
"""

import math
import torch
import taichi as ti


# ---------------------------------------------------------------------------
# Taichi kernels
# ---------------------------------------------------------------------------

@ti.kernel
def _flow_energy_up(mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                    kernel: ti.types.ndarray(), ti_inds: ti.template()):
    """Energy at each cell flows to neighbors in proportion to their infra."""
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        energy = mem[0, inds.energy, i, j]
        infra_sum = 0.0
        for n in ti.ndrange(kernel.shape[0]):
            nx = (i + kernel[n, 0]) % mem.shape[2]
            ny = (j + kernel[n, 1]) % mem.shape[3]
            infra_sum += mem[0, inds.infra, nx, ny]
        if infra_sum > 1e-8:
            for n in ti.ndrange(kernel.shape[0]):
                nx = (i + kernel[n, 0]) % mem.shape[2]
                ny = (j + kernel[n, 1]) % mem.shape[3]
                out_mem[nx, ny] += energy * (mem[0, inds.infra, nx, ny] / infra_sum)
        else:
            out_mem[i, j] += energy


@ti.kernel
def _distribute_energy(mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                       max_energy: ti.f32, kernel: ti.types.ndarray(),
                       ti_inds: ti.template()):
    """Cells above max_energy spill energy evenly to all neighbors."""
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        if mem[0, inds.energy, i, j] > max_energy:
            for n in ti.ndrange(kernel.shape[0]):
                nx = (i + kernel[n, 0]) % mem.shape[2]
                ny = (j + kernel[n, 1]) % mem.shape[3]
                out_mem[nx, ny] += mem[0, inds.energy, i, j] / kernel.shape[0]
        else:
            out_mem[i, j] += mem[0, inds.energy, i, j]


@ti.kernel
def _diffuse_energy(mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                    kernel: ti.types.ndarray(), ti_inds: ti.template()):
    """Energy averages equally with all neighbors (Laplacian diffusion)."""
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        total = 0.0
        for n in ti.ndrange(kernel.shape[0]):
            nx = (i + kernel[n, 0]) % mem.shape[2]
            ny = (j + kernel[n, 1]) % mem.shape[3]
            total += mem[0, inds.energy, nx, ny]
        out_mem[i, j] = total / kernel.shape[0]


@ti.kernel
def _flow_energy_down(mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                      kernel: ti.types.ndarray(), ti_inds: ti.template()):
    """Energy flows to neighbors inversely proportional to their infra.
    Low-infra cells attract energy. Opposite of flow_energy_up."""
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        energy = mem[0, inds.energy, i, j]
        inv_sum = 0.0
        for n in ti.ndrange(kernel.shape[0]):
            nx = (i + kernel[n, 0]) % mem.shape[2]
            ny = (j + kernel[n, 1]) % mem.shape[3]
            inv_sum += 1.0 / (mem[0, inds.infra, nx, ny] + 1e-3)
        for n in ti.ndrange(kernel.shape[0]):
            nx = (i + kernel[n, 0]) % mem.shape[2]
            ny = (j + kernel[n, 1]) % mem.shape[3]
            inv_infra = 1.0 / (mem[0, inds.infra, nx, ny] + 1e-3)
            out_mem[nx, ny] += energy * (inv_infra / inv_sum)


# ---------------------------------------------------------------------------
# Scratch buffer
# ---------------------------------------------------------------------------

_buf = {}


def _get_buf(substrate):
    inds = substrate.ti_indices[None]
    grid_shape = substrate.mem[0, inds.energy].shape
    dev = substrate.torch_device
    if "buf" not in _buf or _buf["buf"].shape != grid_shape:
        _buf["buf"] = torch.zeros(grid_shape, dtype=substrate.mem.dtype, device=dev)
    return _buf["buf"]


# ---------------------------------------------------------------------------
# Shared: energy injection
# ---------------------------------------------------------------------------

def inject_energy(substrate, rate):
    """Add source * rate to energy each step."""
    inds = substrate.ti_indices[None]
    substrate.mem[0, inds.energy] += substrate.mem[0, inds.source] * rate


# ---------------------------------------------------------------------------
# Physics modes
# ---------------------------------------------------------------------------

def physics_infra_attract(substrate, kernel, max_energy):
    """Energy flows toward high-infra neighbors. No overflow cap — energy
    accumulates at infra peaks until they become very bright."""
    inds = substrate.ti_indices[None]
    buf = _get_buf(substrate)
    buf.zero_()
    _flow_energy_up(substrate.mem, buf, kernel, substrate.ti_indices)
    substrate.mem[0, inds.energy] = buf


def physics_infra_attract_cap(substrate, kernel, max_energy):
    """Infra attract + overflow spill when energy > max_energy.
    Energy channels through infra landscape, overflows peaks and floods outward."""
    inds = substrate.ti_indices[None]
    buf = _get_buf(substrate)
    buf.zero_()
    _flow_energy_up(substrate.mem, buf, kernel, substrate.ti_indices)
    substrate.mem[0, inds.energy] = buf
    buf.zero_()
    _distribute_energy(substrate.mem, buf, max_energy, kernel, substrate.ti_indices)
    substrate.mem[0, inds.energy] = buf


def physics_diffuse(substrate, kernel, max_energy):
    """Pure Laplacian diffusion — energy equalizes among neighbors.
    Infra landscape has no effect. Energy spreads smoothly from sources."""
    inds = substrate.ti_indices[None]
    buf = _get_buf(substrate)
    buf.zero_()
    _diffuse_energy(substrate.mem, buf, kernel, substrate.ti_indices)
    substrate.mem[0, inds.energy] = buf


def physics_infra_repel(substrate, kernel, max_energy):
    """Energy flows toward low-infra neighbors — the opposite of infra attract.
    Energy pools in infra deserts. Infra ridges become energy barriers."""
    inds = substrate.ti_indices[None]
    buf = _get_buf(substrate)
    buf.zero_()
    _flow_energy_down(substrate.mem, buf, kernel, substrate.ti_indices)
    substrate.mem[0, inds.energy] = buf


def physics_infra_decay(substrate, kernel, max_energy, decay_rate=0.002):
    """Infra attract + cap, but infra decays each step.
    Energy follows moving peaks as the landscape slowly collapses."""
    inds = substrate.ti_indices[None]
    buf = _get_buf(substrate)
    buf.zero_()
    _flow_energy_up(substrate.mem, buf, kernel, substrate.ti_indices)
    substrate.mem[0, inds.energy] = buf
    buf.zero_()
    _distribute_energy(substrate.mem, buf, max_energy, kernel, substrate.ti_indices)
    substrate.mem[0, inds.energy] = buf
    substrate.mem[0, inds.infra] *= (1.0 - decay_rate)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _coords(W, H, device):
    xs = torch.arange(W, device=device).float().unsqueeze(1).expand(W, H)
    ys = torch.arange(H, device=device).float().unsqueeze(0).expand(W, H)
    return xs, ys


def _circle_mask(W, H, cx, cy, r, device):
    xs, ys = _coords(W, H, device)
    return ((xs - cx) ** 2 + (ys - cy) ** 2) < r ** 2


def _sine_noise(W, H, device, n_freqs=4):
    """Smooth noise as sum of sinusoids at different frequencies and random phases."""
    noise = torch.zeros(W, H, device=device)
    xs = torch.arange(W, device=device).float()
    ys = torch.arange(H, device=device).float()
    for freq in range(1, n_freqs + 1):
        px = torch.rand(1).item() * 2 * math.pi
        py = torch.rand(1).item() * 2 * math.pi
        amp = 1.0 / freq
        noise += amp * (
            torch.sin(xs * freq * 2 * math.pi / W + px).unsqueeze(1)
            * torch.sin(ys * freq * 2 * math.pi / H + py).unsqueeze(0)
        )
    mn, mx = noise.min(), noise.max()
    return (noise - mn) / (mx - mn + 1e-8)


# ---------------------------------------------------------------------------
# Environment initializers
# ---------------------------------------------------------------------------

def init_env(substrate, env_name):
    """Zero the substrate and initialize channels for the named environment."""
    inds = substrate.ti_indices[None]
    W, H = substrate.w, substrate.h
    dev = substrate.torch_device
    mem = substrate.mem
    mem.zero_()
    xs, ys = _coords(W, H, dev)

    if env_name == "patch":
        # Source patches at corners + center; flat low infra.
        # With infra_attract_cap: energy pools at sources, then overflows outward as waves.
        # With diffuse: energy spreads evenly from all patches simultaneously.
        # With infra_repel: energy pools between patches (the infra valleys).
        mem[0, inds.infra] = 0.1
        patch_centers = [
            (W // 4, H // 4), (3 * W // 4, H // 4),
            (W // 4, 3 * H // 4), (3 * W // 4, 3 * H // 4),
            (W // 2, H // 2),
        ]
        r = max(5, W // 12)
        for cx, cy in patch_centers:
            mask = _circle_mask(W, H, cx, cy, r, dev)
            mem[0, inds.source] += mask.float()
        mem[0, inds.source].clamp_(0, 1)
        mem[0, inds.energy] = torch.rand(W, H, device=dev) * 0.05

    elif env_name == "gradient":
        # Uniform source; infra increases diagonally bottom-left → top-right.
        # With infra_attract: energy is pulled steadily toward the high corner.
        # With infra_repel: energy pools at the bottom-left instead — reversed.
        mem[0, inds.infra] = (xs / W + ys / H) / 2.0
        mem[0, inds.source] = 0.5
        mem[0, inds.energy] = torch.rand(W, H, device=dev) * 0.1

    elif env_name == "ring":
        # Concentric infra peak at center; source strongest at edges.
        # With infra_attract: energy flows inward and funnels to center.
        # With infra_repel: energy is trapped at the edges.
        cx, cy = W / 2.0, H / 2.0
        dist = torch.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        max_dist = math.sqrt((W / 2) ** 2 + (H / 2) ** 2)
        mem[0, inds.infra] = torch.clamp(1.0 - dist / (max_dist * 0.7), 0, 1)
        mem[0, inds.source] = torch.clamp(dist / max_dist, 0, 1) ** 2
        mem[0, inds.energy] = torch.rand(W, H, device=dev) * 0.05

    elif env_name == "rivers":
        # Diagonal sinusoidal infra channels; source concentrated on the ridges.
        # With infra_attract: energy travels along the channel ridges.
        # With infra_repel: energy pools in the valleys between channels.
        stripe = torch.sin((xs + ys) * math.pi * 6 / W)
        mem[0, inds.infra] = torch.clamp(stripe, 0, 1)
        mem[0, inds.source] = 0.15
        peak_mask = stripe > 0.85
        mem[0, inds.source] += peak_mask.float() * 0.85
        mem[0, inds.source].clamp_(0, 1)
        mem[0, inds.energy] = torch.rand(W, H, device=dev) * 0.05

    elif env_name == "volcano":
        # Single center source; ring of infra surrounding the center void.
        # Energy builds at center, gets pulled outward into the ring, then
        # erupts along the ring rim when capped.
        cx, cy = W / 2.0, H / 2.0
        dist = torch.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        ring_r = W * 0.22
        ring_w = W * 0.07
        ring = torch.exp(-((dist - ring_r) ** 2) / (2 * ring_w ** 2))
        mem[0, inds.infra] = ring
        center_mask = dist < (W * 0.1)
        mem[0, inds.source] = center_mask.float()
        mem[0, inds.energy] = torch.rand(W, H, device=dev) * 0.05

    elif env_name == "random":
        # Smooth random infra + random source clusters.
        # Different physics modes produce visibly different steady-state patterns.
        mem[0, inds.infra] = _sine_noise(W, H, dev, n_freqs=5)
        raw_src = _sine_noise(W, H, dev, n_freqs=3)
        mem[0, inds.source] = (raw_src > 0.65).float()
        mem[0, inds.energy] = torch.rand(W, H, device=dev) * 0.1

    else:
        raise ValueError(f"Unknown environment: {env_name!r}")
