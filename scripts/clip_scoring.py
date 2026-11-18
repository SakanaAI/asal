"""Shared ASAL CLIP scoring with explicit JAX device placement."""

import time

import numpy as np
from PIL import Image


def load_clip_image(path, *, resize=False):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if resize:
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
        elif image.size != (224, 224):
            raise ValueError("Expected a 224 x 224 image.")
        return np.asarray(image, dtype=np.float32) / 255.0


def select_device(name):
    import jax

    if name not in ("cpu", "gpu"):
        raise ValueError(f"Unsupported scoring device: {name}")
    try:
        devices = jax.devices(name)
    except RuntimeError as exc:
        raise ValueError(f"Requested {name.upper()} is unavailable: {exc}") from exc
    if not devices:
        raise ValueError(f"Requested {name.upper()} is unavailable: no JAX devices found")
    return devices[0]


class ClipScorer:
    def __init__(self, device="cpu", *, model_path=None):
        import jax
        from foundation_models.clip import CLIP

        self.device = select_device(device)
        started = time.perf_counter()
        with jax.default_device(self.device):
            self.model = CLIP(model_path=str(model_path) if model_path is not None else None)
            self.model.clip_model.params = jax.device_put(self.model.clip_model.params, self.device)
            self.model.img_mean = jax.device_put(self.model.img_mean, self.device)
            self.model.img_std = jax.device_put(self.model.img_std, self.device)
            jax.block_until_ready((self.model.clip_model.params, self.model.img_mean, self.model.img_std))
        self.model_load_seconds = time.perf_counter() - started
        leaves = jax.tree_util.tree_leaves(self.model.clip_model.params)
        if any(leaf.device != self.device for leaf in leaves):
            raise ValueError("CLIP parameters were not placed on the requested device")
        self.parameter_dtypes = sorted({str(leaf.dtype) for leaf in leaves})

    def text_embeddings(self, prompts):
        import jax

        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("Prompts must not be empty.")
        with jax.default_device(self.device), jax.default_matmul_precision("highest"):
            return jax.block_until_ready(self.model.embed_txt(prompts))

    def image_embeddings(self, images):
        """Normalized CLIP embeddings for ASAL temporal scoring; no text prompt."""
        import jax
        import jax.numpy as jnp

        if not hasattr(self, "_embed_image"):
            self._embed_image = jax.jit(self.model.embed_img)
        with jax.default_device(self.device), jax.default_matmul_precision("highest"):
            return jax.block_until_ready(jnp.stack([
                self._embed_image(jax.device_put(image, self.device)) for image in images
            ]))

    def losses(self, images, text_embeddings):
        """Return an asynchronous image × prompt loss matrix using ASAL's metric."""
        import jax
        import jax.numpy as jnp
        from asal_metrics import calc_supervised_target_score

        with jax.default_device(self.device), jax.default_matmul_precision("highest"):
            rows = []
            for image in images:
                rgb = jax.device_put(image, self.device)
                embedding = self.model.embed_img(rgb).reshape(1, -1)
                rows.append(jnp.stack([
                    calc_supervised_target_score(embedding, text_embeddings[index:index + 1])
                    for index in range(text_embeddings.shape[0])
                ]))
            return jnp.stack(rows)

    def score(self, image, prompt):
        import jax

        text = self.text_embeddings([prompt])
        loss = float(jax.device_get(self.losses([image], text))[0, 0])
        if not np.isfinite(loss):
            raise ValueError("ASAL returned an invalid score.")
        return loss
