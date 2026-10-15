from pathlib import Path
import argparse
import json
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import jax.numpy as jnp
from PIL import Image
from foundation_models import create_foundation_model
from asal_metrics import calc_supervised_target_score

parser = argparse.ArgumentParser()
parser.add_argument("--image", default="results/week7/coralai_frame.png")
parser.add_argument("--prompt", default="branching colonies")
parser.add_argument(
    "--output", default="results/week7/coralai_clip_score.json"
)
args = parser.parse_args()

with Image.open(args.image) as image:
    image = image.convert("RGB")
    if image.size != (224, 224):
        raise ValueError("Expected a 224 x 224 image.")
    rgb = jnp.asarray(np.asarray(image, dtype=np.float32) / 255.0)

print("Loading CLIP. First run may download model weights.", flush=True)
model = create_foundation_model("clip")

print("Scoring the image...", flush=True)
image_embedding = jnp.asarray(model.embed_img(rgb)).reshape(1, -1)
text_embedding = jnp.asarray(model.embed_txt([args.prompt])).reshape(1, -1)

loss = float(
    calc_supervised_target_score(image_embedding, text_embedding)
)
if not np.isfinite(loss):
    raise ValueError("ASAL returned an invalid score.")

result = {
    "image": args.image,
    "prompt": args.prompt,
    "model": "ASAL CLIP",
    "asal_target_loss": loss,
    "alignment": -loss,
    "interpretation": "Lower target loss means stronger prompt alignment."
}

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n")

print(json.dumps(result, indent=2))
print("Saved:", output)
