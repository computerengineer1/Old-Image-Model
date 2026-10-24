"""
Regression test for the 2026-10-24 field bug:

    OSError: ... does not appear to have a file named preprocessor_config.json
    OSError: Error no file named diffusion_pytorch_model.bin found in ...\vae

Cause: Lykon/dreamshaper-8's model_index.json declares `feature_extractor` and
`safety_checker`. download_models.py intentionally does NOT fetch those folders
(the safety stack is disabled), so from_pretrained must be called with BOTH
safety_checker=None AND feature_extractor=None. The old code omitted
feature_extractor=None -> diffusers tried to load the missing
preprocessor_config.json -> the blind "retry without fp16 variant" then failed
on the (correctly absent) non-variant weight files.

This test rebuilds that exact folder layout from the tiny SD pipeline:
  * model_index.json declaring safety_checker + feature_extractor,
  * NO feature_extractor/ and NO safety_checker/ folders,
  * fp16-variant-named weight files only,
and asserts:
  A) the OLD kwargs reproduce the OSError (bug is real and testable),
  B) the NEW production kwargs (_build_load_kwargs) load successfully,
  C) the shared img2img pipeline builds from that load,
  D) an incomplete folder yields a FriendlyGenerationError, not a traceback.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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


def build_fake_dreamshaper_dir(dest: Path, fp16_names: bool) -> None:
    """Save the tiny pipeline, then reshape it into DreamShaper-8's on-disk
    layout: safety/feature-extractor declared but absent, fp16-only weights."""
    from diffusers import StableDiffusionPipeline

    tiny_src = Path(tempfile.gettempdir()) / "omi-tiny-sd-safetensors"
    if not (tiny_src / "model_index.json").exists():
        from huggingface_hub import snapshot_download
        repo = Path(tempfile.gettempdir()) / "omi-tiny-sd"
        if not (repo / "model_index.json").exists():
            repo = Path(snapshot_download("hf-internal-testing/tiny-stable-diffusion-torch",
                                          local_dir=str(repo), local_dir_use_symlinks=False))
        tmp_pipe = StableDiffusionPipeline.from_pretrained(str(repo))
        tmp_pipe.save_pretrained(str(tiny_src), safe_serialization=True)
        del tmp_pipe

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(tiny_src, dest)

    # DreamShaper's model_index.json declares these two; the folders are NOT downloaded.
    index_path = dest / "model_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["safety_checker"] = ["diffusers", "StableDiffusionSafetyChecker"]
    index["feature_extractor"] = ["transformers", "CLIPImageProcessor"]
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    shutil.rmtree(dest / "safety_checker", ignore_errors=True)
    shutil.rmtree(dest / "feature_extractor", ignore_errors=True)

    if fp16_names:
        renames = [
            ("unet/diffusion_pytorch_model.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors"),
            ("vae/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors"),
            ("text_encoder/model.safetensors", "text_encoder/model.fp16.safetensors"),
        ]
        for old_rel, new_rel in renames:
            old_p, new_p = dest / old_rel, dest / new_rel
            if old_p.exists():
                old_p.rename(new_p)


def main() -> int:
    from diffusers import StableDiffusionPipeline

    fake_dir = Path(tempfile.gettempdir()) / "omi-fake-dreamshaper"
    print("== building DreamShaper-8-shaped fake checkpoint (fp16-only, no safety folders) ==")
    build_fake_dreamshaper_dir(fake_dir, fp16_names=True)
    check("fake dir has fp16 unet weights",
          (fake_dir / "unet" / "diffusion_pytorch_model.fp16.safetensors").exists())
    check("fake dir has NO feature_extractor folder", not (fake_dir / "feature_extractor").exists())
    check("fake dir has NO plain vae weights (like the user's folder)",
          not (fake_dir / "vae" / "diffusion_pytorch_model.safetensors").exists())

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    config["model"]["local_dir"] = str(fake_dir)
    style_lib = G.StyleLibrary(ROOT / "styles" / "styles.json", ROOT / "styles" / "lora")
    gen = G.OldModelGenerator(config, style_lib, Path(tempfile.mkdtemp(prefix="omi-out-")))

    print("== A) old kwargs reproduce the user's OSError ==")
    check("checkpoint_present() true", gen.checkpoint_present())
    check("_detect_variant() == 'fp16' (like the user's folder)", gen._detect_variant() == "fp16")
    old_kwargs = dict(torch_dtype="auto", safety_checker=None, requires_safety_checker=False,
                      use_safetensors=True, local_files_only=True, variant="fp16")
    try:
        import torch
        old_kwargs["torch_dtype"] = torch.float16
        StableDiffusionPipeline.from_pretrained(str(fake_dir), **old_kwargs)
        reproduced = False
        detail = "load unexpectedly succeeded"
    except OSError as exc:
        reproduced = "preprocessor_config.json" in str(exc)
        detail = str(exc)[:120]
    check("old kwargs fail on preprocessor_config.json (bug reproduced)", reproduced, detail)

    print("== B) production kwargs load the same folder cleanly ==")
    kwargs = gen._build_load_kwargs()
    check("kwargs force feature_extractor=None", "feature_extractor" in kwargs and kwargs["feature_extractor"] is None)
    check("kwargs force safety_checker=None", kwargs.get("safety_checker") is None)
    check("kwargs select fp16 variant", kwargs.get("variant") == "fp16")
    pipe = StableDiffusionPipeline.from_pretrained(str(fake_dir), **kwargs)
    check("pipeline loaded with the exact production kwargs", pipe is not None)
    check("safety_checker is None on loaded pipe", pipe.safety_checker is None)
    check("feature_extractor is None on loaded pipe", pipe.feature_extractor is None)

    print("== C) shared img2img builds from it ==")
    img2img = G.build_shared_img2img(pipe)
    check("img2img constructed", img2img is not None)
    check("modules are shared (one copy in RAM)", img2img.unet is pipe.unet and img2img.vae is pipe.vae)

    print("== D) truly incomplete folder -> FriendlyGenerationError, not traceback ==")
    broken_dir = Path(tempfile.mkdtemp(prefix="omi-broken-"))
    (broken_dir / "model_index.json").write_text("{}", encoding="utf-8")
    config2 = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    config2["model"]["local_dir"] = str(broken_dir)
    gen2 = G.OldModelGenerator(config2, style_lib, Path(tempfile.mkdtemp(prefix="omi-out2-")))
    err = gen2._incomplete_model_error()
    check("friendly error carries message + hint", bool(err.message) and bool(err.hint))

    shutil.rmtree(fake_dir, ignore_errors=True)
    shutil.rmtree(broken_dir, ignore_errors=True)
    print(f"\nMODEL-LOAD-FIX RESULT: {PASSED} passed, {FAILED} failed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
