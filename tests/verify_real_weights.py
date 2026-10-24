"""Verification against the REAL downloaded weights (run from workspace root)."""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # .../old-model-images
VERIFY = ROOT / "models" / "upscalers"
LORA_DIR = ROOT / "styles" / "lora"
if not (VERIFY / "RealESRGAN_x2.pth").exists():
    VERIFY = ROOT.parent / "verify"
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from PIL import Image
import generator as G

fails = 0
def check(name, cond, detail=""):
    global fails
    print(("  [PASS] " if cond else "  [FAIL] ") + name + ((" - " + detail) if (detail and not cond) else ""))
    if not cond:
        fails += 1

print("== REAL official weights: RealESRGAN_x2.pth strict load ==")
dev = torch.device("cpu")
x2_path = VERIFY / "RealESRGAN_x2.pth"
model = G._build_esrgan_model(2, x2_path, dev)
params = sum(p.numel() for p in model.parameters())
check("strict=True load succeeded (arch matches official file exactly)", model is not None)
check("param count consistent with 67,061,725-byte file", params == 16703171, str(params))

up = G.RealESRGANUpscaler(x2_path, 2, tile_size=192, tile_overlap=32)
yy, xx = np.mgrid[0:220, 0:300]
img_arr = np.stack([xx / 300 * 255, yy / 220 * 255, (xx + yy) / 520 * 255], axis=-1).astype("uint8")
img = Image.fromarray(img_arr)
out = up.upscale(img)
check("real x2 upscale 300x220 -> 600x440 (tiled, real weights)", out.size == (600, 440), str(out.size))
o = np.asarray(out).astype(np.float32)
check("output finite & in range", np.isfinite(o).all() and o.min() >= 0 and o.max() <= 255)
ref = np.asarray(img.resize((600, 440), Image.BILINEAR)).astype(np.float32)
mad = np.abs(o - ref).mean()
check("upscaled gradient close to smooth reference (weights loaded, not garbage)", mad < 40, f"MAD={mad:.1f}")

print("== REAL LoRA file: PixelArtRedmond key format ==")
from safetensors.torch import load_file
lora_file = LORA_DIR / "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors"
if not lora_file.exists():
    lora_file = ROOT.parent / "verify" / "pixel.safetensors"
sd = load_file(str(lora_file))
keys = list(sd.keys())
check("loads as safetensors", len(keys) > 0)
unet_keys = [k for k in keys if "unet" in k]
te_keys = [k for k in keys if ("te_" in k or "text_encoder" in k)]
check("contains UNet LoRA tensors (kohya or diffusers naming)", len(unet_keys) > 0, str(keys[:3]))
print(f"         info: {len(keys)} tensors, {len(unet_keys)} unet, {len(te_keys)} text-encoder; e.g. {keys[0]}")
check("2-D LoRA weight matrices present (kohya/diffusers convertible)",
      any(v.dim() == 2 for v in sd.values()))

print("== prompt join fix regression ==")
style = json.loads((ROOT / "styles" / "styles.json").read_text(encoding="utf-8"))["styles"][0]
p = G.build_final_prompt(style, "a castle on a hill", lora_available=True)
check("no double spaces / stray commas", "  " not in p and ",," not in p and not p.startswith(","), p)
check("trigger word leads when LoRA present", p.startswith("PIXARFK"), p)
pf = G.build_final_prompt(style, "a castle on a hill", lora_available=False)
check("fallback omits trigger word", "PIXARFK" not in pf, pf)

print(f"\nREAL-WEIGHTS RESULT: {fails} failed")
sys.exit(1 if fails else 0)
