"""
Headless REPL for Coralai — replaces the Taichi GGUI for interaction on
headless Linux VMs.  Outputs a video of the run and saves a JSON log of
every command and per-step statistics.

Usage:
    source .venv/bin/activate
    python headless_repl.py [--experiment minimal|nca|coral] [--steps-per-frame 1]
                            [--shape 64] [--max-frames 2000] [--auto N]

Commands inside the REPL:
    step [N]         — advance N steps (default 10)
    mutate [strength]— mutate / perturb organism weights
    status           — print channel statistics
    channels R G B   — set which channel indices map to RGB
    paint X Y VAL CH — set substrate.mem[0, CH, X, Y] += VAL
    clear            — zero the substrate memory
    save_frame       — save current frame as PNG
    snapshot         — save substrate memory tensor to disk
    quit / exit      — compile video, save log, and exit
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import taichi as ti
from PIL import Image
import imageio

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Headless Coralai REPL")
parser.add_argument("--experiment", type=str, default="minimal",
                    choices=["minimal", "nca", "coral"],
                    help="Which experiment to run")
parser.add_argument("--steps-per-frame", type=int, default=1,
                    help="Simulation steps between rendered frames")
parser.add_argument("--shape", type=int, default=64,
                    help="Grid side length (shape x shape)")
parser.add_argument("--max-frames", type=int, default=2000,
                    help="Maximum number of frames to save")
parser.add_argument("--auto", type=int, default=0,
                    help="If >0, run this many steps non-interactively then quit")
parser.add_argument("--fps", type=int, default=30,
                    help="Video frames per second")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Taichi + Torch setup (CPU / headless)
# ---------------------------------------------------------------------------
ti.init(ti.cpu)
TORCH_DEVICE = torch.device("cpu")

from coralai.substrate import Substrate
# Legacy experiment classes (minimal, nca, coral-old) are broken stubs — imported
# lazily inside their setup functions so they don't break the coral experiment path.

# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------
RUN_ID = datetime.now().strftime("%y%m%d-%H%M%S")
RUN_DIR = os.path.join("runs", f"{args.experiment}_{RUN_ID}")
FRAMES_DIR = os.path.join(RUN_DIR, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
LOG = {"experiment": args.experiment, "shape": args.shape,
       "run_id": RUN_ID, "events": [], "step_stats": []}


def log_event(kind, **kwargs):
    entry = {"time": time.time(), "kind": kind, **kwargs}
    LOG["events"].append(entry)


def log_step(step, substrate, channel_names):
    stats = {"step": step}
    for name in channel_names:
        try:
            inds = substrate.windex[name]
            vals = substrate.mem[0, inds]
            stats[name] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "sum": float(vals.sum()),
            }
        except Exception:
            pass
    LOG["step_stats"].append(stats)


def save_log():
    path = os.path.join(RUN_DIR, "log.json")
    with open(path, "w") as f:
        json.dump(LOG, f, indent=2, default=str)
    print(f"  Log saved → {path}")

# ---------------------------------------------------------------------------
# Headless renderer
# ---------------------------------------------------------------------------
class HeadlessRenderer:
    def __init__(self, substrate, vis_channel_names, run_dir, frames_dir,
                 max_frames=2000):
        self.substrate = substrate
        self.run_dir = run_dir
        self.frames_dir = frames_dir
        self.max_frames = max_frames
        self.frame_idx = 0
        self.frames = []
        self.set_channels(vis_channel_names)

    def set_channels(self, channel_names):
        """channel_names: list of up to 3 channel name strings or ints."""
        self.vis_names = list(channel_names)
        chinds = []
        for c in channel_names:
            if isinstance(c, int):
                chinds.append(c)
            elif isinstance(c, tuple):
                chinds.append(self.substrate.windex[c])
            else:
                idx = self.substrate.windex[c]
                if isinstance(idx, list):
                    chinds.append(idx[0])
                else:
                    chinds.append(idx)
        while len(chinds) < 3:
            chinds.append(chinds[-1] if chinds else 0)
        self.chinds = chinds[:3]

    def render_frame(self):
        """Return an (H, W, 3) uint8 numpy array."""
        w, h = self.substrate.w, self.substrate.h
        img = np.zeros((h, w, 3), dtype=np.float32)
        for k in range(3):
            ch_data = self.substrate.mem[0, self.chinds[k]].detach().cpu()
            if ch_data.dim() > 2:
                ch_data = ch_data.mean(dim=0)
            ch = ch_data.numpy()
            mx = ch.max()
            mn = ch.min()
            rng = mx - mn
            if rng > 1e-8:
                ch = (ch - mn) / rng
            else:
                ch = np.zeros_like(ch)
            img[:, :, k] = ch.T  # (w,h) → (h,w) for image convention
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def capture(self):
        if self.frame_idx >= self.max_frames:
            return
        img = self.render_frame()
        self.frames.append(img)
        self.frame_idx += 1

    def save_frame_png(self, tag=""):
        img = self.render_frame()
        fname = f"frame_{self.frame_idx:06d}{tag}.png"
        path = os.path.join(self.frames_dir, fname)
        Image.fromarray(img).save(path)
        print(f"  Frame saved → {path}")
        return path

    def compile_video(self, fps=30):
        if not self.frames:
            print("  No frames to compile.")
            return None
        video_path = os.path.join(self.run_dir, "simulation.mp4")
        writer = imageio.get_writer(video_path, fps=fps, codec="libx264",
                                    quality=8)
        for frame in self.frames:
            writer.append_data(frame)
        writer.close()
        print(f"  Video saved → {video_path}  ({len(self.frames)} frames)")
        return video_path

# ---------------------------------------------------------------------------
# Experiment setups
# ---------------------------------------------------------------------------
def setup_minimal(shape):
    substrate = Substrate(
        shape=shape,
        torch_dtype=torch.float32,
        torch_device=TORCH_DEVICE,
        channels={"bw": ti.f32},
    )
    substrate.malloc()
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir,
                               "experiments/minimal/minimal_neat.config")
    kernel = [[-1, -1], [0, -1], [1, -1],
              [-1,  0], [0,  0], [1,  0],
              [-1,  1], [0,  1], [1,  1]]
    sense_chs = ["bw"]
    act_chs = ["bw"]

    genome_key = 0
    genome_map = torch.zeros(shape[0], shape[1], dtype=torch.int32,
                             device=TORCH_DEVICE)

    organism = MinimalOrganismCNN(substrate, kernel, sense_chs, act_chs,
                                 TORCH_DEVICE)

    # Seed initial pattern
    cx, cy = shape[0] // 2, shape[1] // 2
    r = max(1, shape[0] // 8)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                x = (cx + dx) % shape[0]
                y = (cy + dy) % shape[1]
                substrate.mem[0, 0, x, y] = torch.rand(1).item()

    vis_channels = ["bw"]
    channel_names = ["bw"]

    def step_fn():
        organism.forward(substrate.mem)

    def mutate_fn(strength=0.1):
        organism.mutate(strength)
        log_event("mutate", strength=strength)

    return substrate, organism, vis_channels, channel_names, step_fn, \
        mutate_fn, genome_map


def setup_nca(shape):
    substrate = Substrate(
        shape=shape,
        torch_dtype=torch.float32,
        torch_device=TORCH_DEVICE,
        channels={
            "rgb": ti.types.struct(r=ti.f32, g=ti.f32, b=ti.f32),
            "hidden": ti.types.vector(n=10, dtype=ti.f32),
            "genome": ti.f32,
        },
    )
    substrate.malloc()
    inds = substrate.ti_indices[None]

    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir,
                               "coralai/instances/nca/nca_neat.config")

    import neat
    from coralai.evolution.neat_evolver import NEATEvolver
    from coralai.nn_lib import ch_norm

    kernel = torch.tensor([[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]],
                          device=TORCH_DEVICE)
    sense_chs = ["rgb", "hidden"]
    act_chs = ["rgb", "hidden"]

    neat_evolver = NEATEvolver(config_path, substrate, kernel, 2,
                               sense_chs, act_chs)
    genome = neat.DefaultGenome(str(0))
    genome.configure_new(neat_evolver.neat_config.genome_config)
    net = neat_evolver.create_torch_net(genome)
    weights = net.weights.unsqueeze(0)
    biases = net.biases.unsqueeze(0)

    # All cells belong to genome 0 (single genome NCA).
    # On CPU, genome=-1 causes out-of-bounds access in the Taichi kernel.
    substrate.mem[0, inds.genome, ...] = 0

    def nca_activation(mem_slice):
        mean = mem_slice.mean(dim=(0, 2, 3), keepdim=True)
        var = mem_slice.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        mem_slice.sub_(mean).div_(torch.sqrt(var + 1e-5))
        return torch.sigmoid(mem_slice)

    vis_channels = [("rgb", "r"), ("rgb", "g"), ("rgb", "b")]
    channel_names = ["rgb", "hidden", "genome"]

    def step_fn():
        substrate.mem[:, neat_evolver.act_chinds] += (
            torch.rand_like(substrate.mem[:, neat_evolver.act_chinds]) * 0.1)
        neat_evolver.forward(weights, biases)
        substrate.mem[:, neat_evolver.act_chinds] = nca_activation(
            substrate.mem[:, neat_evolver.act_chinds])

    def mutate_fn(strength=None):
        nonlocal weights, biases
        genome.mutate(neat_evolver.neat_config.genome_config)
        net = neat_evolver.create_torch_net(genome)
        weights = net.weights.unsqueeze(0)
        biases = net.biases.unsqueeze(0)
        log_event("mutate")

    return substrate, neat_evolver, vis_channels, channel_names, step_fn, \
        mutate_fn, None


def setup_coral(shape):
    from coralai.evolution.ecosystem import Ecosystem
    from coralai.evolution.hyper_organism import HyperOrganism
    from coralai.instances.coral.coral_physics_old import apply_physics

    substrate = Substrate(
        shape=shape,
        torch_dtype=torch.float32,
        torch_device=TORCH_DEVICE,
        channels={
            "genome": ti.f32,
            "energy": ti.f32,
            "infra": ti.f32,
            "acts": ti.types.struct(
                invest=ti.f32,
                liquidate=ti.f32,
                explore=ti.types.vector(n=5, dtype=ti.f32),
            ),
            "com": ti.types.struct(a=ti.f32, b=ti.f32, c=ti.f32, d=ti.f32),
        },
    )
    substrate.malloc()
    inds = substrate.ti_indices[None]
    substrate.mem[0, inds.genome, ...] = -1

    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir,
                               "experiments/coral/neat.config")
    kernel = torch.tensor([[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]],
                          device=TORCH_DEVICE)
    sense_chs = ["energy", "infra", "com"]
    act_chs = ["acts", "com"]

    ecosystem = [None]

    def _create_organism(genome_key, genome=None):
        org = HyperOrganism(config_path, substrate, kernel, sense_chs,
                            act_chs, TORCH_DEVICE)
        if genome is None:
            genome = org.gen_random_genome(genome_key)
        org.set_genome(genome_key, genome=genome)
        org.genome.fitness = 0.1
        org.create_torch_net()
        return org

    def _apply_physics():
        apply_physics(substrate, ecosystem[0], kernel)

    ecosystem[0] = Ecosystem(substrate, _create_organism, _apply_physics,
                             min_size=1, max_size=1)

    vis_channels = ["energy", "infra", "genome"]
    channel_names = ["genome", "energy", "infra"]

    def step_fn():
        ecosystem[0].update()

    def mutate_fn(strength=None):
        for gk in list(ecosystem[0].population.keys()):
            ecosystem[0].mutate(gk, report=True)
        log_event("mutate")

    return substrate, ecosystem[0], vis_channels, channel_names, step_fn, \
        mutate_fn, None

# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------
def print_status(substrate, channel_names, step):
    print(f"\n{'='*60}")
    print(f"  Step {step}  |  Substrate shape: {list(substrate.mem.shape)}")
    print(f"{'='*60}")
    for name in channel_names:
        try:
            inds = substrate.windex[name]
            vals = substrate.mem[0, inds]
            if vals.dim() == 2:
                print(f"  {name:>12s}:  mean={vals.mean():.4f}  "
                      f"std={vals.std():.4f}  min={vals.min():.4f}  "
                      f"max={vals.max():.4f}  sum={vals.sum():.4f}")
            else:
                for i in range(vals.shape[0]):
                    v = vals[i]
                    print(f"  {name}[{i}]:  mean={v.mean():.4f}  "
                          f"std={v.std():.4f}  min={v.min():.4f}  "
                          f"max={v.max():.4f}")
        except Exception as e:
            print(f"  {name}: (error reading: {e})")
    print()


def run_repl():
    shape = (args.shape, args.shape)
    print(f"\n{'#'*60}")
    print(f"  Coralai Headless REPL — {args.experiment}")
    print(f"  Grid: {shape[0]}x{shape[1]}  |  Run: {RUN_ID}")
    print(f"  Output dir: {RUN_DIR}")
    print(f"{'#'*60}\n")

    if args.experiment == "minimal":
        substrate, organism, vis_chs, ch_names, step_fn, mutate_fn, gmap = \
            setup_minimal(shape)
    elif args.experiment == "nca":
        substrate, organism, vis_chs, ch_names, step_fn, mutate_fn, gmap = \
            setup_nca(shape)
    elif args.experiment == "coral":
        substrate, organism, vis_chs, ch_names, step_fn, mutate_fn, gmap = \
            setup_coral(shape)
    else:
        print(f"Unknown experiment: {args.experiment}")
        sys.exit(1)

    renderer = HeadlessRenderer(substrate, vis_chs, RUN_DIR, FRAMES_DIR,
                                max_frames=args.max_frames)

    current_step = 0
    log_event("start", experiment=args.experiment, shape=list(shape))

    # Auto mode: run N steps, capture frames, then save and exit
    if args.auto > 0:
        print(f"  Auto-running {args.auto} steps ...")
        t0 = time.time()
        for i in range(args.auto):
            step_fn()
            current_step += 1
            if i % args.steps_per_frame == 0:
                renderer.capture()
            if i % max(1, args.auto // 20) == 0:
                log_step(current_step, substrate, ch_names)
        elapsed = time.time() - t0
        print(f"  {args.auto} steps in {elapsed:.2f}s "
              f"({args.auto / elapsed:.0f} steps/s)")
        log_step(current_step, substrate, ch_names)
        log_event("auto_complete", steps=args.auto, elapsed=elapsed)
        print_status(substrate, ch_names, current_step)
        renderer.save_frame_png("_final")
        renderer.compile_video(fps=args.fps)
        save_log()
        return

    # Interactive REPL
    print("Commands: step [N], mutate [strength], status, channels R G B,")
    print("          paint X Y VAL CH, clear, save_frame, snapshot, quit\n")

    renderer.capture()

    while True:
        try:
            line = input(f"coralai [{current_step}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            line = "quit"

        if not line:
            continue

        log_event("command", input=line)
        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            break

        elif cmd == "step":
            n = int(parts[1]) if len(parts) > 1 else 10
            t0 = time.time()
            for i in range(n):
                step_fn()
                current_step += 1
                if i % args.steps_per_frame == 0:
                    renderer.capture()
            elapsed = time.time() - t0
            log_step(current_step, substrate, ch_names)
            print(f"  {n} steps in {elapsed:.2f}s "
                  f"({n / max(elapsed, 1e-9):.0f} steps/s)")
            print_status(substrate, ch_names, current_step)

        elif cmd == "mutate":
            strength = float(parts[1]) if len(parts) > 1 else 0.1
            mutate_fn(strength)
            print(f"  Mutated (strength={strength})")

        elif cmd == "status":
            print_status(substrate, ch_names, current_step)

        elif cmd == "channels":
            if len(parts) >= 4:
                try:
                    new_chs = [int(x) for x in parts[1:4]]
                    renderer.chinds = new_chs
                    print(f"  Vis channels → {new_chs}")
                    log_event("channels_changed", channels=new_chs)
                except ValueError:
                    ch_names_in = parts[1:4]
                    renderer.set_channels(ch_names_in)
                    print(f"  Vis channels → {ch_names_in}")
                    log_event("channels_changed", channels=ch_names_in)
            else:
                print(f"  Current vis channel indices: {renderer.chinds}")
                print(f"  Available channels ({substrate.mem.shape[1]} total):")
                for name in ch_names:
                    try:
                        idx = substrate.windex[name]
                        print(f"    {name} → index {idx}")
                    except Exception:
                        pass

        elif cmd == "paint":
            if len(parts) >= 5:
                x, y = int(parts[1]), int(parts[2])
                val = float(parts[3])
                ch = int(parts[4])
                substrate.mem[0, ch, x % substrate.w, y % substrate.h] += val
                print(f"  Painted {val} at ({x},{y}) channel {ch}")
                log_event("paint", x=x, y=y, val=val, channel=ch)
            else:
                print("  Usage: paint X Y VAL CH")

        elif cmd == "clear":
            substrate.mem *= 0.0
            print("  Substrate memory cleared")
            log_event("clear")

        elif cmd == "save_frame":
            renderer.save_frame_png(f"_manual_{current_step}")

        elif cmd == "snapshot":
            path = os.path.join(RUN_DIR, f"substrate_step{current_step}.pt")
            torch.save(substrate.mem, path)
            print(f"  Snapshot saved → {path}")
            log_event("snapshot", path=path)

        elif cmd == "help":
            print("Commands: step [N], mutate [strength], status, "
                  "channels R G B,")
            print("          paint X Y VAL CH, clear, save_frame, "
                  "snapshot, quit")

        else:
            print(f"  Unknown command: {cmd}. Type 'help' for commands.")

    # Finalize
    print("\nFinalizing...")
    renderer.save_frame_png("_final")
    video_path = renderer.compile_video(fps=args.fps)
    log_event("finish", total_steps=current_step)
    save_log()
    print(f"\nRun complete. Output in {RUN_DIR}/")
    return video_path


if __name__ == "__main__":
    run_repl()
