"""
OLD MODEL IMAGES — model downloader.

Fetches everything the app needs, with defense-in-depth against failure:
  1. Base checkpoint: DreamShaper 8 (SD 1.5, diffusers layout) from Hugging Face,
     preferring fp16 variant files (about half the disk space and download time).
  2. Upscaler weights: RealESRGAN_x2.pth + RealESRGAN_x4.pth (ai-forever/Real-ESRGAN).
  3. Style LoRAs listed in styles/styles.json under "lora_download". Every LoRA
     download is verified before use; if a file cannot be fetched it is SKIPPED
     with a logged warning and the style silently falls back to its engineered
     prompt template. This script never crashes on a missing LoRA.

Usage:
    python download_models.py            # full download
    python download_models.py --loras-only
    python download_models.py --skip-loras
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = APP_ROOT / "config.json"
STYLES_PATH = APP_ROOT / "styles" / "styles.json"
LOGS_DIR = APP_ROOT / "logs"
REPORT_PATH = LOGS_DIR / "download_report.txt"

# Diffusers components that must exist for the pipeline to load offline.
# (safety_checker / feature_extractor are intentionally NOT downloaded:
#  the app loads the pipeline with safety_checker=None to save RAM/VRAM.)
CHECKPOINT_CONFIG_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "text_encoder/config.json",
    "vae/config.json",
    "unet/config.json",
]

# (folder, weight basename) — each is tried as fp16 first, then plain.
CHECKPOINT_WEIGHTS = [
    ("text_encoder", "model"),
    ("vae", "diffusion_pytorch_model"),
    ("unet", "diffusion_pytorch_model"),
]


def log(msg: str, report: list) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    report.append(line)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_styles() -> list:
    with open(STYLES_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("styles", [])


def hf_hub_download_safe(repo_id: str, filename: str, local_dir: Path, report: list) -> Path | None:
    """Download one file. Returns local path on success, None on any failure."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return Path(path)
    except Exception as exc:  # noqa: BLE001 — every HF error type must be non-fatal here
        log(f"    ! failed {repo_id}/{filename}: {exc.__class__.__name__}: {exc}", report)
        return None


def download_checkpoint(cfg: dict, report: list) -> bool:
    model_cfg = cfg["model"]
    repo_id = model_cfg["repo_id"]
    local_dir = APP_ROOT / model_cfg["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)

    log(f"Downloading base checkpoint '{repo_id}' -> {local_dir}", report)

    ok = True
    for filename in CHECKPOINT_CONFIG_FILES:
        dest = local_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            log(f"    = have {filename}", report)
            continue
        got = hf_hub_download_safe(repo_id, filename, local_dir, report)
        if got is None:
            log(f"    X MISSING required file {filename}", report)
            ok = False

    variant_counts = {"fp16": 0, "none": 0}
    for folder, basename in CHECKPOINT_WEIGHTS:
        fp16_name = f"{folder}/{basename}.fp16.safetensors"
        full_name = f"{folder}/{basename}.safetensors"
        fp16_dest = local_dir / fp16_name
        full_dest = local_dir / full_name
        if fp16_dest.exists() and fp16_dest.stat().st_size > 1_000_000:
            log(f"    = have {fp16_name}", report)
            variant_counts["fp16"] += 1
            continue
        if full_dest.exists() and full_dest.stat().st_size > 1_000_000:
            log(f"    = have {full_name}", report)
            variant_counts["none"] += 1
            continue
        got = hf_hub_download_safe(repo_id, fp16_name, local_dir, report)
        if got is not None:
            log(f"    + {fp16_name} (fp16 variant)", report)
            variant_counts["fp16"] += 1
            continue
        got = hf_hub_download_safe(repo_id, full_name, local_dir, report)
        if got is not None:
            log(f"    + {full_name} (full precision)", report)
            variant_counts["none"] += 1
            continue
        log(f"    X MISSING weights for {folder} — checkpoint unusable", report)
        ok = False

    # If any component lacks an fp16 variant, make sure ALL components also
    # exist in full precision so a variant-less load can succeed.
    if ok and variant_counts["fp16"] < len(CHECKPOINT_WEIGHTS):
        for folder, basename in CHECKPOINT_WEIGHTS:
            full_name = f"{folder}/{basename}.safetensors"
            full_dest = local_dir / full_name
            if not (full_dest.exists() and full_dest.stat().st_size > 1_000_000):
                got = hf_hub_download_safe(repo_id, full_name, local_dir, report)
                if got is None:
                    ok = False

    if ok and variant_counts["fp16"] == len(CHECKPOINT_WEIGHTS):
        log("    i All three weights available as fp16 — fastest load, lowest RAM.", report)

    return ok


def download_safety_checker(cfg: dict, report: list) -> bool:
    """Fetch the CLIP-based NSFW output checker (same family as the official
    Stable Diffusion safety checker) from the DreamShaper 8 repo. Stored in
    cfg['safety']['checker_dir'] (default models/safety). ~608 MB fp16."""
    safety_cfg = cfg.get("safety", {})
    checker_dir = APP_ROOT / safety_cfg.get("checker_dir", "models/safety")
    repo_id = cfg["model"]["repo_id"]
    (checker_dir / "safety_checker").mkdir(parents=True, exist_ok=True)
    (checker_dir / "feature_extractor").mkdir(parents=True, exist_ok=True)

    ok = True
    # feature extractor config (tiny)
    fx_dest = checker_dir / "feature_extractor" / "preprocessor_config.json"
    if fx_dest.exists():
        log("    = have feature_extractor/preprocessor_config.json", report)
    else:
        got = hf_hub_download_safe(repo_id, "feature_extractor/preprocessor_config.json",
                                   checker_dir, report)
        ok = ok and got is not None

    sc_cfg = checker_dir / "safety_checker" / "config.json"
    if sc_cfg.exists():
        log("    = have safety_checker/config.json", report)
    else:
        got = hf_hub_download_safe(repo_id, "safety_checker/config.json", checker_dir, report)
        ok = ok and got is not None

    # transformers' from_pretrained only recognizes "model.safetensors", so we
    # fetch the smaller fp16 file and normalize its name locally. The generator
    # loads it with torch_dtype=float32 (weights are up-cast on load).
    weights_full = checker_dir / "safety_checker" / "model.safetensors"
    if weights_full.exists() and weights_full.stat().st_size > 100_000_000:
        log("    = have safety_checker weights", report)
    else:
        got = hf_hub_download_safe(repo_id, "safety_checker/model.fp16.safetensors",
                                   checker_dir, report)
        if got is not None:
            got.replace(weights_full)
            log("    + safety_checker weights (fp16 file, normalized to model.safetensors)", report)
        else:
            got = hf_hub_download_safe(repo_id, "safety_checker/model.safetensors",
                                       checker_dir, report)
        if got is None:
            log("    ! NSFW output-scanner weights unavailable \u2014 the prompt filter and "
                "forced negatives still protect the app; the image scanner will be "
                "disabled with a visible warning.", report)
            ok = False
        else:
            log("    + safety_checker weights (NSFW image scanner enabled)", report)
    return ok


def download_upscalers(cfg: dict, report: list) -> dict:
    up_cfg = cfg["upscaler"]
    local_dir = APP_ROOT / up_cfg["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in ("x2_file", "x4_file"):
        filename = up_cfg[key]
        dest = local_dir / filename
        if dest.exists() and dest.stat().st_size > 10_000_000:
            log(f"    = have upscaler {filename}", report)
            results[key] = True
            continue
        got = hf_hub_download_safe(up_cfg["repo_id"], filename, local_dir, report)
        results[key] = got is not None
        if got is not None:
            log(f"    + upscaler {filename}", report)
        else:
            log(f"    ! upscaler {filename} unavailable — hires-fix via img2img still works; "
                "ESRGAN options will be disabled in the UI.", report)
    return results


def download_loras(styles: list, cfg: dict, report: list) -> dict:
    lora_dir = APP_ROOT / cfg["paths"]["lora_dir"]
    lora_dir.mkdir(parents=True, exist_ok=True)

    # De-duplicate: several styles may share one LoRA file.
    wanted: dict[str, dict] = {}
    for style in styles:
        dl = style.get("lora_download")
        lora_path = style.get("lora_path")
        if not dl or not lora_path:
            continue
        local_rel = Path(lora_path).name
        wanted[local_rel] = dl

    results = {}
    if not wanted:
        log("    i No downloadable LoRAs declared in styles.json.", report)
        return results

    for local_name, dl in wanted.items():
        dest = lora_dir / local_name
        if dest.exists() and dest.stat().st_size > 500_000:
            log(f"    = have LoRA {local_name}", report)
            results[local_name] = True
            continue
        got = hf_hub_download_safe(dl["repo_id"], dl["filename"], lora_dir, report)
        if got is not None:
            # Normalize the filename to whatever styles.json expects.
            if Path(got).name != local_name:
                Path(got).rename(dest)
            log(f"    + LoRA {local_name} (verified)", report)
            results[local_name] = True
        else:
            log(f"    ! LoRA {local_name} could NOT be verified/downloaded — "
                "affected styles will use their prompt-engineering fallback. App still works.", report)
            results[local_name] = False
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="OLD MODEL IMAGES model downloader")
    parser.add_argument("--loras-only", action="store_true", help="only fetch style LoRAs")
    parser.add_argument("--skip-loras", action="store_true", help="skip style LoRAs")
    parser.add_argument("--skip-safety", action="store_true",
                        help="skip the NSFW output-scanner weights (~608 MB)")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report: list[str] = []
    log("=== OLD MODEL IMAGES — download_models.py ===", report)

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        log("FATAL: huggingface_hub is not installed. Run setup.bat first "
            "(or: pip install -r requirements.txt).", report)
        _write_report(report)
        return 1

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    cfg = load_config()
    cache_dir = cfg.get("paths", {}).get("cache_dir")
    if cache_dir:
        os.environ.setdefault("HF_HOME", str(APP_ROOT / cache_dir))
        (APP_ROOT / cache_dir).mkdir(parents=True, exist_ok=True)
    styles = load_styles()

    checkpoint_ok = True
    upscalers: dict = {}
    loras: dict = {}
    safety_ok = True

    try:
        if not args.loras_only:
            checkpoint_ok = download_checkpoint(cfg, report)
            log("Downloading upscaler weights (Real-ESRGAN x2/x4)...", report)
            upscalers = download_upscalers(cfg, report)
            if args.skip_safety:
                log("Skipping NSFW output-scanner weights (--skip-safety).", report)
            else:
                log("Downloading NSFW output-scanner (CLIP safety checker, ~608 MB)...", report)
                safety_ok = download_safety_checker(cfg, report)
        if not args.skip_loras:
            log("Downloading style LoRAs...", report)
            loras = download_loras(styles, cfg, report)
    except KeyboardInterrupt:
        log("Interrupted by user.", report)
        _write_report(report)
        return 130
    except Exception:
        log("Unexpected error during download:", report)
        report.append(traceback.format_exc())
        _write_report(report)
        return 1

    log("", report)
    log("=== SUMMARY ===", report)
    log(f"  Base checkpoint (DreamShaper 8): {'OK' if checkpoint_ok else 'FAILED — run me again with a stable connection'}", report)
    log(f"  RealESRGAN x2: {'OK' if upscalers.get('x2_file') else ('SKIPPED' if args.loras_only else 'missing (optional)')}", report)
    log(f"  RealESRGAN x4: {'OK' if upscalers.get('x4_file') else ('SKIPPED' if args.loras_only else 'missing (optional)')}", report)
    log(f"  NSFW output scanner: {'OK' if safety_ok else ('SKIPPED' if (args.skip_safety or args.loras_only) else 'missing - prompt filter still active')}", report)
    for name, ok_flag in loras.items():
        log(f"  LoRA {name}: {'OK' if ok_flag else 'unavailable -> prompt fallback active'}", report)
    fallback_styles = [
        s["key"] for s in styles
        if s.get("lora_path") and not (APP_ROOT / s["lora_path"]).exists()
    ]
    prompt_only = [s["key"] for s in styles if not s.get("lora_path")]
    log(f"  Styles using prompt-engineering fallback: {', '.join(fallback_styles) if fallback_styles else 'none'}", report)
    log(f"  Styles that are prompt-only by design: {', '.join(prompt_only) if prompt_only else 'none'}", report)
    log("Done.", report)
    _write_report(report)
    return 0 if checkpoint_ok else 1


def _write_report(report: list) -> None:
    try:
        with open(REPORT_PATH, "w", encoding="utf-8") as fh:
            fh.write("\n".join(report) + "\n")
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
