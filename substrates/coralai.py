from pathlib import Path
import argparse
import numpy as np
import torch
from PIL import Image


def render_checkpoint(checkpoint):
    state = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )

    if not torch.is_tensor(state) or state.ndim != 4:
        raise ValueError("Expected a saved Coralai cell tensor.")
    if state.shape[0] != 1 or state.shape[1] != 14:
        raise ValueError("Expected the 14-channel coral_dev format.")

    cells = state.detach().cpu().numpy()[0]
    if not np.isfinite(cells).all():
        raise ValueError("Checkpoint contains invalid cell values.")

    # Fixed scales across runs: red=energy, green=infrastructure.
    red = np.clip(cells[0] / 10.0, 0, 1)
    green = np.clip(cells[1] / 10.0, 0, 1)
    blue = (cells[13] >= 0).astype(np.float32) * 0.25

    # Coralai uses x,y; images use row=y,column=x.
    rgb = np.stack([red, green, blue], axis=-1).transpose(1, 0, 2)
    image = Image.fromarray(np.rint(rgb * 255).astype(np.uint8))
    image = image.resize((224, 224), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.float32) / 255.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/week7/coralai_frame.png")
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        pattern = (
            "**/coral_dev_20261001_174547/"
            "checkpoint_0001000/substrate.pt"
        )
        matches = []
        for folder in [Path("data/raw"), Path("external/coralai")]:
            matches.extend(folder.glob(pattern))
        if not matches:
            parser.error(
                "Could not find the saved patch run. "
                "Supply the path to a substrate.pt checkpoint."
            )
        args.checkpoint = sorted(matches)[0]

    rgb = render_checkpoint(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(args.output)

    print("Checkpoint:", args.checkpoint)
    print("Saved:", args.output)
    print("RGB shape:", rgb.shape)
