"""
OLD MODEL IMAGES — end-to-end generation test on CPU (no GPU needed).

Downloads the ~1 MB hf-internal-testing tiny Stable Diffusion pipeline, wires
it into the REAL OldModelGenerator (bypassing only the CUDA device gate), and
runs the full generate() flow with 64x64 "resolutions":

  * img2img hires-fix path (incl. the shared-components construction fix)
  * esrgan_x2 hires path using the REAL RealESRGAN_x2.pth weights
  * ultra 4x/2x fallback chain
  * LoRA-missing prompt-fallback warning
  * post-FX styles (crt_vhs, polaroid_frame)
  * seed determinism, PNG metadata, gallery.json persistence
  * scheduler (sampler) switching for all 7 samplers incl. LMS+scipy

Run:  python tests/test_generation_cpu.py   (needs the deps from requirements.txt
      and ../verify/RealESRGAN_x2.pth from tests/verify_real_weights.py)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT / "models" / "upscalers"   # canonical in-project location
if not (VERIFY / "RealESRGAN_x2.pth").exists() and (ROOT.parent / "verify" / "RealESRGAN_x2.pth").exists():
    VERIFY = ROOT.parent / "verify"
sys.path.insert(0, str(ROOT))

import generator as G  # noqa: E402

PASSED = 0
FAILED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {('- ' + detail) if detail else ''}")


class StubProgress:
    def __init__(self):
        self.phases = []

    def update(self, **fields):
        if "phase" in fields and (not self.phases or self.phases[-1] != fields["phase"]):
            self.phases.append(fields["phase"])


def main() -> int:
    import torch
    from diffusers import StableDiffusionPipeline
    from huggingface_hub import snapshot_download
    from PIL import Image, PngImagePlugin

    print("== fetching tiny SD pipeline (~1 MB) ==")
    tiny_dir = Path(tempfile.gettempdir()) / "omi-tiny-sd"
    if not (tiny_dir / "model_index.json").exists():
        repo = snapshot_download("hf-internal-testing/tiny-stable-diffusion-torch",
                                 local_dir=str(tiny_dir), local_dir_use_symlinks=False)
        tiny_dir = Path(repo)
    check("tiny pipeline downloaded", (tiny_dir / "model_index.json").exists())

    # Re-save as safetensors so the generator's use_safetensors=True path is exercised.
    sd_dir = Path(tempfile.gettempdir()) / "omi-tiny-sd-safetensors"
    if not (sd_dir / "model_index.json").exists():
        tmp_pipe = StableDiffusionPipeline.from_pretrained(str(tiny_dir))
        tmp_pipe.save_pretrained(str(sd_dir), safe_serialization=True)
        del tmp_pipe
    check("tiny pipeline re-saved as safetensors",
          (sd_dir / "unet" / "diffusion_pytorch_model.safetensors").exists())

    print("== wiring the real OldModelGenerator (CPU) ==")
    out_dir = Path(tempfile.mkdtemp(prefix="omi-outputs-"))
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    config["upscaler"]["local_dir"] = str(VERIFY)
    config["upscaler"]["x2_file"] = "RealESRGAN_x2.pth"
    style_lib = G.StyleLibrary(ROOT / "styles" / "styles.json", ROOT / "styles" / "lora")
    gen = G.OldModelGenerator(config, style_lib, out_dir)

    pipe = StableDiffusionPipeline.from_pretrained(str(sd_dir), safety_checker=None,
                                                   requires_safety_checker=False, use_safetensors=True)
    gen._pipe = pipe
    gen._img2img = G.build_shared_img2img(pipe)
    gen._mode = "cpu-test"
    check("build_shared_img2img constructs from real components", gen._img2img is not None)
    check("esrgan x2 real weights found", gen.esrgan(2) is not None)

    # Shrink the resolution table so CPU runs take seconds, not minutes.
    G.RATIOS["1:1"] = {"label": "t", "base_w": 64, "base_h": 64, "final_w": 128, "final_h": 128}
    G.RATIOS["16:9"] = {"label": "t", "base_w": 96, "base_h": 64, "final_w": 192, "final_h": 128}

    def base_request(**over):
        req = {"prompt": "a lighthouse on a stormy coast", "style_key": "synthwave",
               "ratio": "1:1", "seed": 42, "steps": 4, "cfg": 7.0, "sampler": "Euler a",
               "performance_mode": "cpu-test", "hires_method": "img2img", "ultra": False,
               "negative_prompt": "blurry"}
        req.update(over)
        return req

    print("== run 1: img2img hires path ==")
    prog = StubProgress()
    r1 = gen.generate(base_request(), prog)
    check("result file exists", (out_dir / r1["filename"]).exists())
    check("final size == ratio final (128x128)", (r1["width"], r1["height"]) == (128, 128),
          f'{r1["width"]}x{r1["height"]}')
    check("seed honored", r1["seed"] == 42)
    check("phases ran in order",
          prog.phases == ["generating", "hires", "saving", "done"], str(prog.phases))
    check("no spurious LoRA warnings for prompt-only style",
          not any("LoRA" in w for w in r1["warnings"]), str(r1["warnings"]))
    # NOTE: the tiny test tokenizer has a <77-token limit, so a benign clip
    # warning can legitimately appear here; real CLIP handles the full prompt.
    check("synthwave prompt template applied",
          r1["final_prompt"].startswith("synthwave retrofuturism") and "PIXARFK" not in r1["final_prompt"])
    check("forced NSFW negative injected (safety layer 2)",
          "nsfw" in r1["negative_prompt"] and "nude" in r1["negative_prompt"],
          r1["negative_prompt"][:80])
    # the tiny test tokenizer has a small vocab -> the merged negative can be
    # legitimately clipped there; on real CLIP (77 tokens) it all fits.
    clipped = any("clipped" in w.lower() for w in r1["warnings"])
    check("style negative merged with user negative (or clipped by tiny tokenizer)",
          ("blurry" in r1["negative_prompt"] and "daylight" in r1["negative_prompt"]) or clipped,
          r1["negative_prompt"][-80:])
    check("gallery.json entry written",
          any(i["filename"] == r1["filename"] for i in gen.gallery.list_existing()))
    check("result schema complete for UI",
          {"image_url", "download_url", "filename", "seed", "elapsed", "style_key", "style_name",
           "final_prompt", "steps", "cfg", "sampler", "ratio", "hires_method", "ultra",
           "lora_used", "warnings", "model", "base_w", "base_h"} <= set(r1))

    with Image.open(out_dir / r1["filename"]) as saved:
        check("PNG metadata omi:seed embedded", saved.info.get("omi:seed") == "42",
              str(list(saved.info.keys())[:6]))
        check("PNG metadata omi:prompt embedded", "lighthouse" in saved.info.get("omi:prompt", ""))

    print("== run 1b: pixel_8bit LoRA resilience (real LoRA vs tiny test UNet) ==")
    # The bundled PixelArtRedmond LoRA targets a real SD1.5 UNet; against the tiny
    # test UNet the injection must either load cleanly or fail GRACEFULLY into the
    # prompt fallback — never crash the job.
    try:
        r1b = gen.generate(base_request(style_key="pixel_8bit", seed=43), StubProgress())
        lora_ok = r1b["lora_used"] is not None
        graceful = lora_ok or any("fallback" in w.lower() for w in r1b["warnings"])
        check("pixel_8bit produced an image despite LoRA/tiny-UNet mismatch", True)
        check("LoRA either applied or degraded to prompt fallback with warning",
              graceful, str(r1b["warnings"]))
        check("LoRA state cleaned up after generation", gen._lora_loaded is None)
    except G.FriendlyGenerationError as exc:
        check("pixel_8bit degrades gracefully (friendly error, no traceback leak)",
              bool(exc.message and exc.hint is not None), exc.message)

    print("== run 2: seed determinism ==")
    r2 = gen.generate(base_request(), StubProgress())
    with Image.open(out_dir / r1["filename"]) as a, Image.open(out_dir / r2["filename"]) as b:
        check("same seed -> identical pixels", list(a.getdata()) == list(b.getdata()))

    if gen.esrgan(2) is None:
        print("== runs 3-4 skipped: real RealESRGAN_x2.pth not present (dev-only check) ==")
        shutil.rmtree(out_dir, ignore_errors=True)
        print(f"\nE2E-CPU RESULT: {PASSED} passed, {FAILED} failed")
        return 0 if FAILED == 0 else 1

    print("== run 3: esrgan_x2 hires with REAL weights ==")
    r3 = gen.generate(base_request(hires_method="esrgan_x2", seed=7), StubProgress())
    check("esrgan_x2 output 128x128", (r3["width"], r3["height"]) == (128, 128),
          f'{r3["width"]}x{r3["height"]}')
    check("hires_method recorded", r3["hires_method"] == "esrgan_x2")

    print("== run 4: ultra fallback chain (x4 missing -> x2 real weights) ==")
    r4 = gen.generate(base_request(hires_method="esrgan_x2", ultra=True, seed=8), StubProgress())
    check("ultra ran at x2 fallback -> 256x256", (r4["width"], r4["height"]) == (256, 256),
          f'{r4["width"]}x{r4["height"]}')

    print("== run 5: 16:9 geometry ==")
    r5 = gen.generate(base_request(ratio="16:9", seed=9), StubProgress())
    check("16:9 base 96x64 -> final 192x128", (r5["width"], r5["height"]) == (192, 128),
          f'{r5["width"]}x{r5["height"]}')

    print("== run 6: post-FX styles ==")
    r6 = gen.generate(base_request(style_key="crt_vhs", seed=10), StubProgress())
    check("crt_vhs keeps final size", (r6["width"], r6["height"]) == (128, 128))
    r7 = gen.generate(base_request(style_key="polaroid", seed=11), StubProgress())
    check("polaroid frame enlarges image", r7["width"] > 128 and r7["height"] > 128,
          f'{r7["width"]}x{r7["height"]}')

    print("== run 7: all samplers switch cleanly ==")
    for name in G.SAMPLERS:
        try:
            gen._set_scheduler(name)
            check(f"sampler '{name}' applies", type(gen._pipe.scheduler).__name__ == G.SAMPLERS[name][0])
            check(f"sampler '{name}' propagates to img2img",
                  gen._img2img.scheduler is gen._pipe.scheduler)
        except Exception as exc:
            check(f"sampler '{name}' applies", False, repr(exc))

    print("== run 8: random seed + clip ==")
    r8 = gen.generate(base_request(seed=-1, prompt="word " * 200, style_key="no_style"), StubProgress())
    check("random seed assigned", r8["seed"] >= 0)
    check("long prompt clipped with warning",
          any("clipped" in w.lower() for w in r8["warnings"]), str(r8["warnings"]))
    check("no_style adds no prefix", r8["final_prompt"].startswith("word"))

    shutil.rmtree(out_dir, ignore_errors=True)
    print(f"\nE2E-CPU RESULT: {PASSED} passed, {FAILED} failed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
