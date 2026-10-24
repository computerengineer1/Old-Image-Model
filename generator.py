"""
OLD MODEL IMAGES — generation engine.

Design constraints (RTX 2050 4 GB VRAM / Ryzen 5 7535HS / 8 GB system RAM):
  * SD 1.5 only (DreamShaper 8), fp16, no safety checker.
  * Heavy imports (torch/diffusers) are LAZY: importing this module costs
    nothing, so app.py can boot and serve /api/health before any model loads.
  * Default offload mode is "safe" (sequential CPU offload): the UNet/text
    encoder/VAE live in system RAM and stream through the GPU layer-by-layer.
    VRAM peak stays around ~1.5 GB, leaving headroom for VAE tiling at 2x res.
  * VAE slicing + tiling are always on (required for the 1824x1024 hires pass).
  * LoRAs are loaded per-generation and unloaded afterwards to keep RAM flat.
  * Every CUDA call is wrapped: torch.cuda.OutOfMemoryError (and any
    "out of memory" RuntimeError) becomes a FriendlyGenerationError with an
    actionable hint. Tracebacks only ever go to logs/app.log.
  * Real-ESRGAN upscaling is vendored here (BSD-3 architecture, weights from
    ai-forever/Real-ESRGAN) and runs TILED with feathered blending, so the
    optional 4x pass cannot blow up VRAM. On any failure it degrades to the
    hires-fix result with a user-visible warning.

The hires-fix pipeline (per spec):
    base pass @ 512-class resolution  ->  second pass @ 2x with denoise 0.35-0.45
    (img2img by default; ESRGAN x2 selectable)  ->  optional ESRGAN x4 "Ultra".
"""

from __future__ import annotations

import gc
import json
import logging
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image, ImageChops, ImageEnhance, PngImagePlugin

from safety import FORCED_NEGATIVE, blocked_message, scan_prompt

APP_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("omi.generator")

try:  # keep module import cheap; real requirement is enforced in ensure_loaded()
    import torch  # type: ignore
except ImportError:  # pragma: no cover - only when deps are missing
    torch = None  # type: ignore


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class FriendlyGenerationError(Exception):
    """An error with a user-facing message plus an actionable hint."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint


# --------------------------------------------------------------------------- #
# Aspect ratios (base pass -> final 2x output), per spec
# --------------------------------------------------------------------------- #

RATIOS: dict[str, dict[str, Any]] = {
    "1:1":  {"label": "Square 1:1",    "base_w": 512, "base_h": 512, "final_w": 1024, "final_h": 1024},
    "4:3":  {"label": "Landscape 4:3", "base_w": 680, "base_h": 512, "final_w": 1360, "final_h": 1024},
    "3:4":  {"label": "Portrait 3:4",  "base_w": 512, "base_h": 680, "final_w": 1024, "final_h": 1360},
    "16:9": {"label": "Wide 16:9",     "base_w": 912, "base_h": 512, "final_w": 1824, "final_h": 1024},
    "9:16": {"label": "Tall 9:16",     "base_w": 512, "base_h": 912, "final_w": 1024, "final_h": 1824},
    "3:2":  {"label": "Photo 3:2",     "base_w": 768, "base_h": 512, "final_w": 1536, "final_h": 1024},
    "2:3":  {"label": "Photo 2:3",     "base_w": 512, "base_h": 768, "final_w": 1024, "final_h": 1536},
}

# UI sampler name -> (diffusers scheduler class name, extra from_config kwargs)
SAMPLERS: dict[str, tuple[str, dict[str, Any]]] = {
    "Euler a":          ("EulerAncestralDiscreteScheduler", {}),
    "Euler":            ("EulerDiscreteScheduler", {}),
    "DPM++ 2M Karras":  ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "DPM++ SDE Karras": ("DPMSolverSinglestepScheduler", {"use_karras_sigmas": True}),
    "DDIM":             ("DDIMScheduler", {}),
    "LMS":              ("LMSDiscreteScheduler", {}),
    "PNDM":             ("PNDMScheduler", {}),
}

PERFORMANCE_MODES = ("safe", "balanced", "fast")
HIRES_METHODS = ("img2img", "esrgan_x2")


def ratios_for_api() -> list[dict[str, Any]]:
    return [
        {"key": k, "label": v["label"], "base_w": v["base_w"], "base_h": v["base_h"],
         "final_w": v["final_w"], "final_h": v["final_h"]}
        for k, v in RATIOS.items()
    ]


# --------------------------------------------------------------------------- #
# Style library (loaded from styles/styles.json — nothing hardcoded)
# --------------------------------------------------------------------------- #

class StyleLibrary:
    def __init__(self, styles_path: Path | str, lora_dir: Path | str):
        self.styles_path = Path(styles_path)
        self.lora_dir = Path(lora_dir)
        self._styles: dict[str, dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        with open(self.styles_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        entries = data.get("styles", [])
        self._styles = {}
        for entry in entries:
            key = entry.get("key")
            if not key:
                continue
            self._styles[key] = entry

    def keys(self) -> list[str]:
        return list(self._styles.keys())

    def get(self, key: str) -> Optional[dict[str, Any]]:
        return self._styles.get(key)

    def lora_file(self, style: dict[str, Any]) -> Optional[Path]:
        rel = style.get("lora_path")
        if not rel:
            return None
        path = APP_ROOT / rel
        return path if path.exists() and path.stat().st_size > 500_000 else None

    def list_for_api(self) -> list[dict[str, Any]]:
        out = []
        for key, s in self._styles.items():
            out.append({
                "key": key,
                "name": s.get("name", key),
                "tagline": s.get("tagline", ""),
                "description": s.get("description", ""),
                "thumbnail": s.get("thumbnail", ""),
                "lora_available": self.lora_file(s) is not None,
                "recommended_cfg": s.get("recommended_cfg", 7.0),
                "recommended_steps": s.get("recommended_steps", 25),
                "negative_prompt": s.get("negative_prompt", ""),
                "has_post_fx": bool(s.get("post_fx")),
            })
        return out


def _dedupe_csv(*parts: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        for token in part.split(","):
            t = token.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
    return ", ".join(out)


def build_final_prompt(style: dict[str, Any], user_prompt: str, lora_available: bool) -> str:
    if lora_available:
        prefix = style.get("prompt_prefix", "")
        suffix = style.get("prompt_suffix", "")
    else:
        prefix = style.get("fallback_prompt_prefix", style.get("prompt_prefix", ""))
        suffix = style.get("fallback_prompt_suffix", style.get("prompt_suffix", ""))
    parts = (
        prefix.strip().rstrip(",").strip(),
        user_prompt.strip(),
        suffix.strip().lstrip(",").strip(),
    )
    return ", ".join(p for p in parts if p)


def build_final_negative(style: dict[str, Any], user_negative: str) -> str:
    return _dedupe_csv(style.get("negative_prompt", ""), user_negative.strip())


# --------------------------------------------------------------------------- #
# Gallery index (outputs/gallery.json), newest-first
# --------------------------------------------------------------------------- #

class GalleryStore:
    def __init__(self, outputs_dir: Path | str):
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.outputs_dir / "gallery.json"
        self._lock = threading.Lock()

    def load(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._load_unlocked()

    def _load_unlocked(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        try:
            with open(self.index_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            logger.warning("gallery.json unreadable; starting fresh index")
            return []

    def _write_unlocked(self, items: list[dict[str, Any]]) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(items, fh, indent=1)
        tmp.replace(self.index_path)

    def add(self, entry: dict[str, Any]) -> None:
        with self._lock:
            items = self._load_unlocked()
            items = [i for i in items if i.get("filename") != entry.get("filename")]
            items.insert(0, entry)
            items = items[:500]  # index cap; files themselves are untouched
            self._write_unlocked(items)

    def remove(self, filename: str) -> bool:
        with self._lock:
            items = self._load_unlocked()
            kept = [i for i in items if i.get("filename") != filename]
            removed = len(kept) != len(items)
            if removed:
                self._write_unlocked(kept)
            target = self.outputs_dir / Path(filename).name
            if target.exists():
                try:
                    target.unlink()
                except OSError:
                    logger.exception("could not delete %s", target)
                    return False
            return removed

    def list_existing(self, limit: int = 200) -> list[dict[str, Any]]:
        items = self.load()
        return [i for i in items if (self.outputs_dir / i.get("filename", "")).exists()][:limit]


# --------------------------------------------------------------------------- #
# Vendored Real-ESRGAN (RRDBNet), architecture per ai-forever/Real-ESRGAN
# (BSD-3). Weights: RealESRGAN_x2.pth / RealESRGAN_x4.pth from the same repo.
# --------------------------------------------------------------------------- #

def _extract_state_dict(raw: Any) -> Any:
    """Handle the common checkpoint envelopes: raw sd, {'params':...},
    {'params_ema':...}, {'model':...}, {'state_dict':...}."""
    if isinstance(raw, dict):
        for key in ("params_ema", "params", "model", "state_dict"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
        cleaned = {}
        for k, v in raw.items():
            if k.startswith("module."):
                k = k[len("module."):]
            cleaned[k] = v
        return cleaned
    raise FriendlyGenerationError(
        "Upscaler weights file is in an unexpected format.",
        "Re-run download_models.py to fetch clean Real-ESRGAN weights.")


def _build_esrgan_model(scale: int, weights_path: Path, device: Any) -> Any:
    import torch
    from torch import nn
    from torch.nn import functional as F

    def pixel_unshuffle(x, s: int):
        b, c, hh, hw = x.shape
        out_c = c * (s ** 2)
        h, w = hh // s, hw // s
        return x.view(b, c, h, s, w, s).permute(0, 1, 3, 5, 2, 4).reshape(b, out_c, h, w)

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat: int, num_grow_ch: int = 32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb3(self.rdb2(self.rdb1(x)))
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4,
                     num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32):
            super().__init__()
            self.scale = scale
            in_ch = num_in_ch * 4 if scale == 2 else num_in_ch
            self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat = pixel_unshuffle(x, 2) if self.scale == 2 else x
            feat = self.conv_first(feat)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    model = RRDBNet(scale=scale)
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(_extract_state_dict(state), strict=True)
    model.eval()
    return model.to(device)


class RealESRGANUpscaler:
    """Tiled Real-ESRGAN inference. Tiles keep VRAM usage ~0.5 GB even for
    1824x1024 -> 4x. Falls back to full-image CPU inference on GPU OOM."""

    def __init__(self, weights_path: Path, scale: int, tile_size: int = 384, tile_overlap: int = 32):
        self.weights_path = Path(weights_path)
        self.scale = scale
        self.tile_size = max(96, int(tile_size))
        self.tile_overlap = max(8, int(tile_overlap))
        self._model = None
        self._device = None

    def available(self) -> bool:
        return self.weights_path.exists() and self.weights_path.stat().st_size > 10_000_000

    def _ensure_model(self, device: Any) -> Any:
        if self._model is None or self._device != device:
            self._model = _build_esrgan_model(self.scale, self.weights_path, device)
            self._device = device
        return self._model

    def _forward(self, np_img, device) -> "Any":
        import numpy as np
        model = self._ensure_model(device)
        x = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        if self.scale == 2 and (x.shape[2] % 2 or x.shape[3] % 2):  # pixel_unshuffle needs even dims
            pad_h, pad_w = x.shape[2] % 2, x.shape[3] % 2
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        with torch.no_grad():
            y = model(x)
        out = y.squeeze(0).permute(1, 2, 0).clamp_(0, 1).cpu().numpy()
        out_h, out_w = np_img.shape[0] * self.scale, np_img.shape[1] * self.scale
        return out[:out_h, :out_w, :]

    def upscale(self, img: Image.Image) -> Image.Image:
        import numpy as np

        if not self.available():
            raise FriendlyGenerationError(
                "Real-ESRGAN weights are missing.",
                "Run download_models.py, or switch the hires method back to 'Hires-fix (img2img)'.")

        img = img.convert("RGB")
        src = np.asarray(img).astype(np.float32) / 255.0
        H, W = src.shape[:2]
        s = self.scale
        use_gpu = torch is not None and torch.cuda.is_available()
        device = torch.device("cuda") if use_gpu else torch.device("cpu")
        if use_gpu:
            torch.cuda.empty_cache()

        tile = self.tile_size
        try:
            out = self._tiled_forward(src, tile, device)
        except Exception as exc:
            if use_gpu and _is_oom_error(exc):
                logger.warning("ESRGAN GPU tile failed (%s); retrying with smaller tile, then CPU", exc)
                torch.cuda.empty_cache()
                try:
                    out = self._tiled_forward(src, max(128, tile // 2), device)
                except Exception:
                    torch.cuda.empty_cache()
                    gc.collect()
                    out = self._tiled_forward(src, tile, torch.device("cpu"))
            else:
                raise FriendlyGenerationError(
                    "The upscaler failed to run.",
                    "Close other GPU applications and try again, or disable the Ultra 4x pass.") from exc

        out_u8 = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype("uint8")
        result = Image.fromarray(out_u8)
        assert result.size == (W * s, H * s)
        return result

    def _tiled_forward(self, src, tile: int, device):
        import numpy as np

        H, W = src.shape[:2]
        s = self.scale
        OH, OW = H * s, W * s

        if W <= tile and H <= tile:
            return self._forward(src, device)

        overlap = min(self.tile_overlap, tile // 4)
        stride = max(1, tile - overlap)

        def starts(total: int) -> list[int]:
            if total <= tile:
                return [0]
            pos = list(range(0, total - tile + 1, stride))
            if pos[-1] != total - tile:
                pos.append(total - tile)
            return pos

        acc = np.zeros((OH, OW, 3), dtype=np.float32)
        wsum = np.zeros((OH, OW, 1), dtype=np.float32)

        for y0 in starts(H):
            for x0 in starts(W):
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                crop = src[y0:y1, x0:x1, :]
                out = self._forward(crop, device)

                # feathered blend mask: linear ramps across the overlap zones
                oh, ow = out.shape[0], out.shape[1]
                ramp_o = overlap * s
                wx = np.ones((ow,), dtype=np.float32)
                wy = np.ones((oh,), dtype=np.float32)
                if x0 > 0 and ramp_o > 0:
                    wx[:ramp_o] = np.linspace(0.0, 1.0, ramp_o, dtype=np.float32)
                if x1 < W and ramp_o > 0:
                    wx[-ramp_o:] = np.linspace(1.0, 0.0, ramp_o, dtype=np.float32)
                if y0 > 0 and ramp_o > 0:
                    wy[:ramp_o] = np.linspace(0.0, 1.0, ramp_o, dtype=np.float32)
                if y1 < H and ramp_o > 0:
                    wy[-ramp_o:] = np.linspace(1.0, 0.0, ramp_o, dtype=np.float32)
                weight = (wy[:, None] * wx[None, :])[:, :, None]

                oy0, ox0 = y0 * s, x0 * s
                acc[oy0:oy0 + oh, ox0:ox0 + ow, :] += out * weight
                wsum[oy0:oy0 + oh, ox0:ox0 + ow, :] += weight

        np.maximum(wsum, 1e-6, out=wsum)
        return acc / wsum


def build_shared_img2img(pipe):
    """Create an img2img pipeline sharing ALL modules with `pipe` (one copy of
    the weights in RAM). Component keys are filtered against the img2img
    __init__ signature because txt2img pipelines expose extra components
    (e.g. image_processor in diffusers 0.27) that img2img constructs itself."""
    import inspect
    from diffusers import StableDiffusionImg2ImgPipeline

    init_params = set(inspect.signature(StableDiffusionImg2ImgPipeline.__init__).parameters)
    shared_components = {k: v for k, v in pipe.components.items() if k in init_params}
    shared_components["requires_safety_checker"] = False
    return StableDiffusionImg2ImgPipeline(**shared_components)


def _is_oom_error(exc: BaseException) -> bool:
    if torch is not None:
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# --------------------------------------------------------------------------- #
# Post-processing effects (pure PIL, cheap, run on CPU)
# --------------------------------------------------------------------------- #

def fx_crt_vhs(img: Image.Image) -> Image.Image:
    """Chroma bleed + scanlines + tape grain."""
    img = img.convert("RGB")
    w, h = img.size
    shift = max(2, w // 640)
    r, g, b = img.split()
    img = Image.merge("RGB", (ImageChops.offset(r, shift, 0), g, ImageChops.offset(b, -shift, 0)))

    # scanline mask: dark line every 3rd pixel row
    pattern = Image.new("L", (1, 3))
    pattern.putdata([0, 0, 130])
    mask = pattern.resize((w, h), Image.NEAREST)
    dark = ImageEnhance.Brightness(img).enhance(0.55)
    img = Image.composite(dark, img, mask)

    grain = Image.effect_noise((w, h), 9)
    grain = Image.merge("RGB", (grain, grain, grain))
    return Image.blend(img, grain, 0.055)


def fx_polaroid(img: Image.Image) -> Image.Image:
    """Warm fade + corner light leak + classic Polaroid frame."""
    img = img.convert("RGB")
    w, h = img.size

    warm = Image.new("RGB", (w, h), (255, 246, 222))
    img = Image.blend(img, warm, 0.12)
    img = ImageEnhance.Color(img).enhance(0.82)
    img = ImageEnhance.Brightness(img).enhance(1.06)
    img = ImageEnhance.Contrast(img).enhance(0.92)

    big_side = int(max(w, h) * 1.5)
    grad = Image.radial_gradient("L").resize((big_side, big_side))
    half = big_side // 2
    leak_mask = grad.crop((half, half, half + w, half + h))
    leak_mask = leak_mask.point(lambda v: int(v * 0.32))
    orange = Image.new("RGB", (w, h), (255, 148, 66))
    img = Image.composite(orange, img, leak_mask)

    b = max(18, int(w * 0.062))
    bb = int(b * 3.2)
    canvas = Image.new("RGB", (w + 2 * b, h + b + bb), (244, 242, 234))
    canvas.paste(img, (b, b))
    return canvas


POST_FX: dict[str, Callable[[Image.Image], Image.Image]] = {
    "crt_vhs": fx_crt_vhs,
    "polaroid_frame": fx_polaroid,
}


# --------------------------------------------------------------------------- #
# The generator
# --------------------------------------------------------------------------- #

class OldModelGenerator:
    """Lazy-loading Stable Diffusion 1.5 engine tuned for 4 GB VRAM / 8 GB RAM."""

    def __init__(self, config: dict, style_lib: StyleLibrary, outputs_dir: Path | str):
        self.config = config
        self.styles = style_lib
        self.gallery = GalleryStore(outputs_dir)
        self.outputs_dir = Path(outputs_dir)

        self._pipe = None
        self._img2img = None
        self._mode: Optional[str] = None
        self._lora_loaded: Optional[Path] = None
        self._esrgan: dict[int, RealESRGANUpscaler] = {}
        self._checker = None
        self._checker_proc = None
        self._checker_failed = False
        self._load_lock = threading.Lock()
        self.last_load_seconds: float = 0.0
        self.model_label = config.get("model", {}).get("name_label", "Stable Diffusion 1.5")

    # ---------------- paths & availability ----------------

    def _model_dir(self) -> Path:
        return APP_ROOT / self.config["model"]["local_dir"]

    def checkpoint_present(self) -> bool:
        return (self._model_dir() / "model_index.json").exists()

    def _detect_variant(self) -> Optional[str]:
        d = self._model_dir()
        needed = [
            d / "unet" / "diffusion_pytorch_model.fp16.safetensors",
            d / "text_encoder" / "model.fp16.safetensors",
            d / "vae" / "diffusion_pytorch_model.fp16.safetensors",
        ]
        return "fp16" if all(p.exists() for p in needed) else None

    def _build_load_kwargs(self) -> dict[str, Any]:
        """from_pretrained kwargs for DreamShaper 8.

        safety_checker AND feature_extractor are both forced to None: the
        checkpoint's model_index.json declares both components, and diffusers
        would otherwise try to load feature_extractor/preprocessor_config.json
        (which download_models.py intentionally does not fetch, because the
        safety checker stack is disabled to save RAM/VRAM)."""
        kwargs: dict[str, Any] = dict(
            torch_dtype=torch.float16,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            use_safetensors=True,
            local_files_only=True,
        )
        if self._detect_variant():
            kwargs["variant"] = "fp16"
        return kwargs

    def _incomplete_model_error(self) -> FriendlyGenerationError:
        return FriendlyGenerationError(
            "DreamShaper 8 could not be loaded \u2014 the model folder looks incomplete or corrupted.",
            "Re-run:  .venv\\Scripts\\python.exe download_models.py   (it repairs and resumes the "
            f"folder at models/checkpoints), then restart the app. Details: logs/app.log")

    def esrgan(self, scale: int) -> Optional[RealESRGANUpscaler]:
        if scale in self._esrgan:
            return self._esrgan[scale]
        up = self.config.get("upscaler", {})
        path = APP_ROOT / up.get("local_dir", "models/upscalers") / up.get(f"x{scale}_file", f"RealESRGAN_x{scale}.pth")
        esc = RealESRGANUpscaler(path, scale,
                                 tile_size=up.get("tile_size", 384),
                                 tile_overlap=up.get("tile_overlap", 32))
        if not esc.available():
            return None
        self._esrgan[scale] = esc
        return esc

    # ---------------- NSFW output checker (CLIP-based, CPU) ----------------

    def _safety_cfg(self) -> dict[str, Any]:
        return self.config.get("safety", {})

    def _safety_dir(self) -> Path:
        return APP_ROOT / self._safety_cfg().get("checker_dir", "models/safety")

    def checker_available(self) -> bool:
        d = self._safety_dir()
        has_cfg = (d / "safety_checker" / "config.json").exists()
        has_weights = any((d / "safety_checker" / f).exists()
                          for f in ("model.safetensors", "model.fp16.safetensors"))
        has_fx = (d / "feature_extractor" / "preprocessor_config.json").exists()
        return has_cfg and has_weights and has_fx

    def _ensure_output_checker(self) -> bool:
        """Lazy-load the Stable-Diffusion CLIP safety checker on CPU.
        Returns False (once) if weights are missing or loading fails, so a
        broken scanner degrades to a warning instead of killing generation."""
        if self._checker is not None:
            return True
        if self._checker_failed or not self.checker_available():
            return False
        try:
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )
            from transformers import CLIPImageProcessor

            d = self._safety_dir()
            self._checker_proc = CLIPImageProcessor.from_pretrained(str(d / "feature_extractor"))
            self._checker = StableDiffusionSafetyChecker.from_pretrained(
                str(d / "safety_checker"), torch_dtype=torch.float32).eval()
            logger.info("NSFW output checker loaded on CPU (~1.2 GB RAM)")
            return True
        except Exception:
            logger.exception("failed to load the NSFW output checker")
            self._checker_failed = True
            self._checker = None
            self._checker_proc = None
            return False

    def check_image_nsfw(self, img: Image.Image) -> bool:
        """True when the CLIP safety checker classifies the image as NSFW."""
        import numpy as np

        img = img.convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        clip_input = self._checker_proc(images=[img], return_tensors="pt").pixel_values
        with torch.no_grad():
            _images, has_nsfw = self._checker(images=[arr], clip_input=clip_input)
        return bool(has_nsfw[0])

    def health(self) -> dict[str, Any]:
        cuda_ok = torch is not None and torch.cuda.is_available()
        info: dict[str, Any] = {
            "torch_installed": torch is not None,
            "cuda_available": bool(cuda_ok),
            "checkpoint_present": self.checkpoint_present(),
            "esrgan_x2": self.esrgan(2) is not None,
            "esrgan_x4": self.esrgan(4) is not None,
            "model_loaded": self._pipe is not None,
            "performance_mode": self._mode,
            "lora_loaded": self._lora_loaded.name if self._lora_loaded else None,
            "safety": {
                "prompt_filter": bool(self._safety_cfg().get("prompt_filter", True)),
                "force_negative": bool(self._safety_cfg().get("force_negative", True)),
                "check_outputs": bool(self._safety_cfg().get("check_outputs", True)),
                "checker_weights_present": self.checker_available(),
                "checker_loaded": self._checker is not None,
            },
        }
        if cuda_ok:
            try:
                props = torch.cuda.get_device_properties(0)
                info["gpu_name"] = props.name
                info["vram_total_mb"] = round(props.total_memory / 1048576)
                info["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 1048576)
            except Exception:
                logger.exception("could not read CUDA device properties")
        return info

    # ---------------- pipeline lifecycle ----------------

    def unload(self) -> None:
        self._pipe = None
        self._img2img = None
        self._mode = None
        self._lora_loaded = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_loaded(self, mode: str, progress: Optional[Any] = None) -> None:
        if self._pipe is not None and self._mode == mode:
            return
        with self._load_lock:
            if self._pipe is not None and self._mode == mode:
                return
            if self._pipe is not None:
                logger.info("performance mode %s -> %s: rebuilding pipeline", self._mode, mode)
                self.unload()

            if torch is None:
                raise FriendlyGenerationError(
                    "PyTorch is not installed in this environment.",
                    "Run setup.bat once to install all dependencies, then run.bat again.")
            if not torch.cuda.is_available():
                raise FriendlyGenerationError(
                    "No CUDA GPU was detected. OLD MODEL IMAGES needs your NVIDIA GPU.",
                    "Update your NVIDIA driver (CUDA 12.x), make sure no other app hogs the GPU, then restart the app.")
            if not self.checkpoint_present():
                raise FriendlyGenerationError(
                    "The DreamShaper 8 model files are not downloaded yet.",
                    "Run download_models.py once (setup.bat does this for you). Check logs/download_report.txt for details.")

            if progress:
                progress.update(phase="loading_model", message="Loading DreamShaper 8 (first run takes a while)...")
            t0 = time.time()

            from diffusers import StableDiffusionPipeline

            load_kwargs = self._build_load_kwargs()
            try:
                pipe = StableDiffusionPipeline.from_pretrained(str(self._model_dir()), **load_kwargs)
            except Exception:
                if load_kwargs.pop("variant", None):
                    logger.warning("fp16 variant load failed; retrying without variant",
                                   exc_info=True)
                    try:
                        pipe = StableDiffusionPipeline.from_pretrained(str(self._model_dir()),
                                                                       **load_kwargs)
                    except Exception:
                        logger.exception("checkpoint load failed (both fp16 and full precision)")
                        raise self._incomplete_model_error() from None
                else:
                    logger.exception("checkpoint load failed")
                    raise self._incomplete_model_error() from None

            img2img = build_shared_img2img(pipe)

            gen_cfg = self.config.get("generation", {})
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()

            if mode == "safe":
                pipe.enable_sequential_cpu_offload()
                if gen_cfg.get("attention_slicing_in_safe_mode", True):
                    pipe.enable_attention_slicing("max")
            elif mode == "balanced":
                pipe.enable_model_cpu_offload()
            else:  # "fast": everything on GPU — 4 GB is tight, user's choice
                pipe.to("cuda")

            if gen_cfg.get("use_xformers_if_available", True):
                try:
                    from diffusers.utils import is_xformers_available
                    if is_xformers_available():
                        pipe.enable_xformers_memory_efficient_attention()
                        logger.info("xformers memory-efficient attention enabled")
                except Exception:
                    logger.info("xformers unavailable — using PyTorch SDPA attention instead")

            self._pipe = pipe
            self._img2img = img2img
            self._mode = mode
            self.last_load_seconds = time.time() - t0
            logger.info("pipeline loaded in %.1fs (mode=%s, variant=%s)", self.last_load_seconds, mode, variant)
            if progress:
                progress.update(phase="loading_model", message=f"Model ready ({mode} mode).")

    def _set_scheduler(self, sampler_name: str) -> None:
        import diffusers

        cls_name, kwargs = SAMPLERS[sampler_name]
        cls = getattr(diffusers, cls_name)
        sched = cls.from_config(self._pipe.scheduler.config, **kwargs)
        self._pipe.scheduler = sched
        self._img2img.scheduler = sched

    # ---------------- prompt helpers ----------------

    def _clip_prompt(self, prompt: str, warnings: list[str]) -> str:
        """SD 1.5 CLIP accepts 77 tokens; clip cleanly instead of letting the
        tokenizer truncate mid-word."""
        try:
            tok = self._pipe.tokenizer
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) > 75:
                prompt = tok.decode(ids[:75]).strip()
                warnings.append("Prompt was longer than the CLIP token limit and has been clipped to 75 tokens.")
        except Exception:
            logger.exception("prompt clipping failed; passing prompt through")
        return prompt

    # ---------------- LoRA handling ----------------

    def _apply_style_lora(self, style: dict[str, Any], warnings: list[str]) -> Optional[dict[str, Any]]:
        """Load the style LoRA if the file exists. Returns cross-attention
        kwargs (or None). Always call _clear_lora() afterwards."""
        lora_file = self.styles.lora_file(style)
        declared = style.get("lora_path")
        if declared and not lora_file:
            warnings.append(f"LoRA file '{Path(declared).name}' not found — using this style's prompt-engineering fallback.")
            return None
        if not lora_file:
            return None
        try:
            if self._lora_loaded is not None:
                self._pipe.unload_lora_weights()
                self._lora_loaded = None
            weight = float(style.get("lora_weight", 0.8) or 0.8)
            self._pipe.load_lora_weights(str(lora_file.parent), weight_name=lora_file.name)
            self._lora_loaded = lora_file
            logger.info("LoRA loaded: %s @ %.2f", lora_file.name, weight)
            return {"scale": weight}
        except Exception:
            logger.exception("LoRA load failed for %s", lora_file)
            self._lora_loaded = None
            warnings.append(f"LoRA '{lora_file.name}' failed to load — using prompt fallback for this style.")
            return None

    def _clear_lora(self) -> None:
        if self._lora_loaded is not None:
            try:
                self._pipe.unload_lora_weights()
            except Exception:
                logger.exception("LoRA unload failed")
            self._lora_loaded = None
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------- main entry point ----------------

    def generate(self, request: dict[str, Any], progress: Any) -> dict[str, Any]:
        """Run one full generation. `progress` is any object with
        .update(**fields). Returns the result dict consumed by app.py / UI."""
        warnings: list[str] = []
        t_start = time.time()

        style_key = request["style_key"]
        style = self.styles.get(style_key)
        if style is None:
            raise FriendlyGenerationError(
                f"Unknown style '{style_key}'.", "Pick one of the styles shown in the UI.")
        ratio_key = request["ratio"]
        ratio = RATIOS[ratio_key]

        seed = int(request["seed"])
        if seed < 0:
            seed = random.SystemRandom().randint(0, 2**32 - 1)

        steps = int(request["steps"])
        cfg = float(request["cfg"])
        sampler = request["sampler"]
        hires_method = request["hires_method"]
        ultra = bool(request["ultra"])
        mode = request.get("performance_mode") or self.config["generation"]["performance_mode"]

        self.ensure_loaded(mode, progress)
        self._set_scheduler(sampler)

        user_prompt = request["prompt"].strip()
        lora_file = self.styles.lora_file(style)
        final_prompt = build_final_prompt(style, user_prompt, lora_file is not None)
        final_prompt = self._clip_prompt(final_prompt, warnings)
        negative = build_final_negative(style, request.get("negative_prompt", ""))

        # ---- content safety layer 1: deterministic prompt filter ----
        safety_cfg = self._safety_cfg()
        if safety_cfg.get("prompt_filter", True):
            hit = scan_prompt(user_prompt)
            if hit is not None:
                if safety_cfg.get("block_logging", True):
                    logger.warning("content policy: blocked a prompt "
                                   "(category=%s, severity=%s, term=%s)",
                                   hit["category"], hit["severity"], hit["term"])
                message, hint = blocked_message(hit)
                raise FriendlyGenerationError(message, hint)

        # ---- content safety layer 2: forced NSFW negative prompt ----
        if safety_cfg.get("force_negative", True):
            negative = _dedupe_csv(FORCED_NEGATIVE, negative)
        negative = self._clip_prompt(negative, warnings) if negative else negative

        if hires_method == "esrgan_x2" and self.esrgan(2) is None:
            warnings.append("ESRGAN x2 weights missing — used img2img hires-fix instead.")
            hires_method = "img2img"
        if ultra and self.esrgan(4) is None and self.esrgan(2) is None:
            warnings.append("Real-ESRGAN weights missing — skipped the Ultra pass. Run download_models.py to enable it.")
            ultra = False

        device = self._pipe._execution_device
        gen = torch.Generator(device=device).manual_seed(seed)
        gen_hires = torch.Generator(device=device).manual_seed(seed)

        xak: Optional[dict[str, Any]] = None
        try:
            xak = self._apply_style_lora(style, warnings)
            base_w, base_h = ratio["base_w"], ratio["base_h"]
            final_w, final_h = ratio["final_w"], ratio["final_h"]

            # ---- pass 1: base generation ---------------------------------
            progress.update(phase="generating", step=0, total=steps,
                            message=f"Base pass at {base_w}x{base_h}...")

            def cb_base(_p, step_index, _timestep, cb_kwargs):
                progress.update(phase="generating", step=step_index + 1, total=steps)
                return cb_kwargs

            try:
                base_img = self._pipe(
                    prompt=final_prompt,
                    negative_prompt=negative or None,
                    width=base_w,
                    height=base_h,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=gen,
                    cross_attention_kwargs=xak,
                    callback_on_step_end=cb_base,
                ).images[0]
            except Exception as exc:
                raise self._translate_error(exc, "base pass") from exc

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ---- pass 2: hires fix ----------------------------------------
            strength = float(self.config["generation"].get("hires_strength", 0.4))
            strength = min(0.45, max(0.35, strength))

            if hires_method == "esrgan_x2":
                progress.update(phase="hires", message="Hires pass: Real-ESRGAN x2...")
                try:
                    hires_img = self.esrgan(2).upscale(base_img)
                except FriendlyGenerationError:
                    raise
                except Exception as exc:
                    raise self._translate_error(exc, "ESRGAN x2 pass") from exc
            else:
                progress.update(phase="hires", step=0, total=steps,
                                message=f"Hires-fix 2x at {final_w}x{final_h} (denoise {strength:.2f})...")

                def cb_hires(_p, step_index, _timestep, cb_kwargs):
                    effective = max(1, int(steps * strength))
                    progress.update(phase="hires", step=min(step_index + 1, effective), total=effective)
                    return cb_kwargs

                # img2img derives the output size from the input image (its
                # __call__ has no width/height args in diffusers 0.27), so the
                # base render is resized to the final dimensions up front.
                # All RATIOS final sizes are multiples of 8 — VAE-safe.
                hires_input = base_img.resize((final_w, final_h), Image.LANCZOS)

                try:
                    hires_img = self._img2img(
                        prompt=final_prompt,
                        negative_prompt=negative or None,
                        image=hires_input,
                        strength=strength,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=gen_hires,
                        cross_attention_kwargs=xak,
                        callback_on_step_end=cb_hires,
                    ).images[0]
                except Exception as exc:
                    raise self._translate_error(exc, "hires-fix pass") from exc

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ---- content safety layer 3: AI scan of the rendered image ----
            if safety_cfg.get("check_outputs", True):
                if self._ensure_output_checker():
                    progress.update(phase="safety_scan",
                                    message="Scanning the image with the NSFW filter...")
                    try:
                        if self.check_image_nsfw(hires_img):
                            logger.warning("content policy: output image blocked by the NSFW scanner "
                                           "(style=%s, seed=%s)", style_key, seed)
                            raise FriendlyGenerationError(
                                "This generation was blocked by the NSFW output filter and was NOT saved.",
                                "The finished image was classified as sexual/NSFW by the on-device AI "
                                "scanner. Rephrase your prompt, or try another seed or style.")
                    except FriendlyGenerationError:
                        raise
                    except Exception:
                        logger.exception("NSFW output scan errored for this job")
                        warnings.append("The NSFW output scanner hit an internal error and was skipped "
                                        "for this image (the prompt filter stayed active).")
                elif not self._checker_failed:
                    warnings.append("NSFW output-scanner weights are missing \u2014 the prompt filter stayed "
                                    "active. Run download_models.py to enable the full 3-layer protection.")

            # ---- optional pass 3: ultra 4x (with a RAM guard) --------------
            # A 4x pass on the widest hires output (1824x1024) would produce a
            # ~30 MP image; the float32 accumulators alone would eat ~0.7 GB of
            # the 8 GB system RAM. Above the guard we downgrade to x2 and tell
            # the user, instead of risking swap death.
            if ultra:
                uw, uh = hires_img.size
                target_px = (uw * 4) * (uh * 4)
                ultra_scale = 4
                if target_px > 30_000_000 and self.esrgan(2) is not None:
                    ultra_scale = 2
                    warnings.append(
                        "Ultra 4x would exceed the 30-megapixel RAM guard on an 8 GB system - "
                        "the final pass ran at Real-ESRGAN x2 instead.")
                esc = self.esrgan(ultra_scale) or self.esrgan(4) or self.esrgan(2)
                progress.update(phase="upscaling",
                                message=f"Ultra pass: Real-ESRGAN x{ultra_scale} (tiled)...")
                try:
                    if esc is None:
                        raise FriendlyGenerationError(
                            "No Real-ESRGAN weights available.",
                            "Run download_models.py to fetch them.")
                    hires_img = esc.upscale(hires_img)
                except FriendlyGenerationError as exc:
                    warnings.append(f"Ultra pass skipped: {exc.message}")
                except Exception:
                    logger.exception("ultra upscale failed")
                    warnings.append("Ultra pass failed (likely VRAM) - kept the hires-fix result.")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ---- post fx ---------------------------------------------------
            fx_name = style.get("post_fx")
            if fx_name and fx_name in POST_FX:
                progress.update(phase="post_fx", message="Applying style post-effects...")
                try:
                    hires_img = POST_FX[fx_name](hires_img)
                except Exception:
                    logger.exception("post fx %s failed", fx_name)
                    warnings.append(f"Post-effect '{fx_name}' failed — saved the clean render instead.")

            # ---- save -------------------------------------------------------
            progress.update(phase="saving", message="Saving to gallery...")
            elapsed = round(time.time() - t_start, 1)
            out_w, out_h = hires_img.size
            filename = self._make_filename(style_key, seed)
            meta = {
                "filename": filename,
                "prompt": user_prompt,
                "final_prompt": final_prompt,
                "negative_prompt": negative,
                "style_key": style_key,
                "style_name": style.get("name", style_key),
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler": sampler,
                "ratio": ratio_key,
                "base_w": base_w, "base_h": base_h,
                "width": out_w, "height": out_h,
                "hires_method": hires_method,
                "hires_strength": strength if hires_method == "img2img" else None,
                "ultra": ultra,
                "performance_mode": mode,
                "lora_used": lora_file.name if (lora_file and xak) else None,
                "model": self.model_label,
                "elapsed": elapsed,
                "warnings": warnings,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._save_image(hires_img, filename, meta)
            self.gallery.add(meta)

            progress.update(phase="done", message="Done.")
            result = dict(meta)
            result["image_url"] = f"/outputs/{filename}"
            result["download_url"] = f"/outputs/{filename}?download=1"
            return result
        finally:
            self._clear_lora()

    # ---------------- internals ----------------

    def _translate_error(self, exc: Exception, phase: str) -> FriendlyGenerationError:
        if isinstance(exc, FriendlyGenerationError):
            return exc
        if _is_oom_error(exc):
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.error("CUDA OOM during %s", phase)
            logger.debug("OOM traceback", exc_info=exc)
            return FriendlyGenerationError(
                f"The GPU ran out of memory during the {phase}.",
                "Try: switch Performance Mode to 'Safe', pick a smaller ratio, lower the step count, "
                "turn off the Ultra 4x pass, or close other GPU-hungry apps (browser tabs, games, video players).")
        logger.exception("generation failed during %s", phase)
        return FriendlyGenerationError(
            f"Generation failed during the {phase}.",
            "The details were written to logs/app.log. Try again — if it repeats, lower steps/resolution or restart the app.")

    def _make_filename(self, style_key: str, seed: int) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}_{style_key}_seed{seed}.png"
        candidate = self.outputs_dir / base
        n = 1
        while candidate.exists():
            candidate = self.outputs_dir / f"{stamp}_{style_key}_seed{seed}_{n}.png"
            n += 1
        return candidate.name

    def _save_image(self, img: Image.Image, filename: str, meta: dict[str, Any]) -> Path:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        path = self.outputs_dir / filename
        pnginfo = PngImagePlugin.PngInfo()
        for key in ("prompt", "final_prompt", "negative_prompt", "style_key", "seed", "steps",
                    "cfg", "sampler", "ratio", "width", "height", "hires_method", "ultra",
                    "lora_used", "model", "elapsed"):
            pnginfo.add_text(f"omi:{key}", str(meta.get(key, "")))
        img.save(path, format="PNG", pnginfo=pnginfo)
        return path
