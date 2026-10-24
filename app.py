"""
OLD MODEL IMAGES — Flask application.

Routes (all payloads are JSON except "/" and "/outputs/<file>"):
  GET  /                       -> the app UI
  GET  /api/styles             -> style catalog + ratios + samplers + defaults
  POST /api/generate           -> start a generation job (single-job queue)
  GET  /api/progress/<job_id>  -> poll job state (phase/step/total/result/error)
  GET  /api/gallery            -> persisted gallery index (newest first)
  POST /api/gallery/delete     -> {"filename": ...} delete one output
  POST /api/settings           -> runtime settings (performance mode, hires, ultra)
  GET  /api/health             -> model/GPU/RAM status for the header badge
  GET  /outputs/<filename>     -> serve generated PNGs (?download=1 to save)

Safety rules implemented here:
  * Exactly ONE generation at a time (threading.Lock + busy check on submit).
    A second submit returns HTTP 409 with a friendly message — never a parallel
    job that would OOM the GPU.
  * Every exception path returns {"error": {"message", "hint"}} — raw
    tracebacks only ever go to logs/app.log.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import secrets
import threading
import time
import uuid
import webbrowser
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, render_template, request, send_from_directory

from generator import (
    APP_ROOT,
    HIRES_METHODS,
    PERFORMANCE_MODES,
    SAMPLERS,
    FriendlyGenerationError,
    GalleryStore,
    OldModelGenerator,
    StyleLibrary,
    ratios_for_api,
)
from safety import blocked_message, scan_prompt

# --------------------------------------------------------------------------- #
# Configuration & logging
# --------------------------------------------------------------------------- #

def load_config() -> dict:
    config_path = APP_ROOT / "config.json"
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


CONFIG = load_config()

# Honor the configurable HF cache dir (config.json -> paths.cache_dir).
# run.bat also sets HF_HOME; the environment always wins over this default.
import os as _os
_os.environ.setdefault("HF_HOME", str(APP_ROOT / CONFIG["paths"].get("cache_dir", "models/hf-cache")))

LOGS_DIR = APP_ROOT / CONFIG["paths"]["logs_dir"]
OUTPUTS_DIR = APP_ROOT / CONFIG["paths"]["outputs_dir"]
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("omi")
logger.setLevel(logging.INFO)
_file_handler = logging.handlers.RotatingFileHandler(
    LOGS_DIR / "app.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logger.addHandler(_file_handler)
_gen_logger = logging.getLogger("omi.generator")
_gen_logger.addHandler(_file_handler)
_gen_logger.setLevel(logging.INFO)
_gen_logger.propagate = False  # prevent duplicate lines via the parent "omi" logger

# --------------------------------------------------------------------------- #
# App state
# --------------------------------------------------------------------------- #

STYLE_LIB = StyleLibrary(APP_ROOT / CONFIG["paths"]["styles"], APP_ROOT / CONFIG["paths"]["lora_dir"])
GALLERY = GalleryStore(OUTPUTS_DIR)
ENGINE = OldModelGenerator(CONFIG, STYLE_LIB, OUTPUTS_DIR)

RUNTIME_SETTINGS = {
    "performance_mode": CONFIG["generation"].get("performance_mode", "safe"),
    "hires_method": CONFIG["generation"].get("hires_method", "img2img"),
    "ultra": bool(CONFIG["generation"].get("ultra_upscale_default", False)),
}

# --------------------------------------------------------------------------- #
# Security: per-boot session token, host allow-list, rate limiting
# --------------------------------------------------------------------------- #

SESSION_TOKEN = secrets.token_urlsafe(32)   # regenerated every boot; injected
                                            # into the served page and required
                                            # on every /api/* call (CSRF + blind
                                            # localhost-request protection)
_SEC = CONFIG.get("security", {})
ALLOWED_HOSTS = {h.lower() for h in _SEC.get("allowed_hosts", ["127.0.0.1", "localhost"])}
ALLOWED_HOSTS.add(str(CONFIG.get("server", {}).get("host", "127.0.0.1")).split(":")[0].strip().lower())
ALLOW_LAN = bool(_SEC.get("allow_lan", False))
API_RATE_LIMIT = int(_SEC.get("api_rate_limit_per_minute", 240))
GEN_RATE_LIMIT = int(_SEC.get("generate_rate_limit_per_minute", 30))
RATE_WINDOW: dict[str, list[float]] = {"api": [], "generate": []}
RATE_LOCK = threading.Lock()

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


def _rate_ok(bucket: str, limit_per_minute: int) -> bool:
    now = time.time()
    with RATE_LOCK:
        hits = [t for t in RATE_WINDOW[bucket] if now - t < 60.0]
        hits.append(now)
        RATE_WINDOW[bucket] = hits
        return len(hits) <= limit_per_minute


def _signed_url(filename: str, download: bool = False) -> str:
    url = f"/outputs/{filename}?tk={SESSION_TOKEN}"
    return url + ("&download=1" if download else "")


GENERATION_LOCK = threading.Lock()          # held for the duration of a GPU job
_active_job_id: Optional[str] = None        # guarded by _active_lock
_active_lock = threading.Lock()


class JobProgress:
    """Thread-safe progress record shared between the worker and the poller."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self._lock = threading.Lock()
        self.status = "queued"          # queued | running | done | error
        self.phase = "queued"           # queued/loading_model/generating/hires/upscaling/post_fx/saving/done
        self.step = 0
        self.total = 0
        self.message = "Waiting in queue..."
        self.result: Optional[dict] = None
        self.error: Optional[dict] = None
        self.started_at = time.time()

    def update(self, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def finish(self, result: dict) -> None:
        with self._lock:
            self.result = result
            self.status = "done"
            self.phase = "done"
            self.message = "Done."

    def fail(self, message: str, hint: str) -> None:
        with self._lock:
            self.error = {"message": message, "hint": hint}
            self.status = "error"
            self.phase = "error"
            self.message = message

    def snapshot(self) -> dict:
        with self._lock:
            total = self.total or 0
            percent = round(100.0 * self.step / total) if total and self.phase in ("generating", "hires") else None
            return {
                "job_id": self.job_id,
                "status": self.status,
                "phase": self.phase,
                "step": self.step,
                "total": total,
                "percent": percent,
                "message": self.message,
                "result": self.result,
                "error": self.error,
            }


JOBS: "OrderedDict[str, JobProgress]" = OrderedDict()
JOBS_LOCK = threading.Lock()
MAX_JOBS_KEPT = 20


def _remember_job(job: JobProgress) -> None:
    with JOBS_LOCK:
        JOBS[job.job_id] = job
        while len(JOBS) > MAX_JOBS_KEPT:
            JOBS.popitem(last=False)


def _job_is_finished(job_id: str) -> bool:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    return job is None or job.status in ("done", "error")


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _err(message: str, hint: str = "", status: int = 400):
    return jsonify({"error": {"message": message, "hint": hint}}), status


def _clamp_int(value: Any, lo: int, hi: int, default: int, name: str) -> tuple[int, Optional[str]]:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default, f"'{name}' must be a whole number — using {default}."
    if v < lo or v > hi:
        return max(lo, min(hi, v)), f"'{name}' was clamped to the {lo}–{hi} range."
    return v, None


def validate_generate_payload(data: dict) -> tuple[dict, list[str]]:
    """Return a normalized request dict or raise ValueError with a friendly msg."""
    notes: list[str] = []
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Please write a prompt first — the model needs something to dream about.")
    if len(prompt) > 1200:
        raise ValueError("Prompt is too long (max 1200 characters). Trim it and try again.")

    # content-safety layer 1 at the API boundary (generator re-checks too)
    if CONFIG.get("safety", {}).get("prompt_filter", True):
        hit = scan_prompt(prompt)
        if hit is not None:
            if CONFIG.get("safety", {}).get("block_logging", True):
                logger.warning("content policy: API rejected a prompt (category=%s, severity=%s)",
                               hit["category"], hit["severity"])
            message, hint = blocked_message(hit)
            raise ValueError(message + " " + hint)

    style_key = str(data.get("style_key", "no_style"))
    if STYLE_LIB.get(style_key) is None:
        raise ValueError(f"Unknown style '{style_key}'. Refresh the page to reload the style list.")

    ratio = str(data.get("ratio", "1:1"))
    from generator import RATIOS
    if ratio not in RATIOS:
        raise ValueError(f"Unknown aspect ratio '{ratio}'.")

    seed, note = _clamp_int(data.get("seed", -1), -1, 2**31 - 1, -1, "Seed")
    if note:
        notes.append(note)
    steps, note = _clamp_int(data.get("steps", 25), 4, 60, 25, "Steps")
    if note:
        notes.append(note)

    try:
        cfg = float(data.get("cfg", 7.0))
    except (TypeError, ValueError):
        cfg, notes = 7.0, notes + ["CFG was not a number — using 7.0."]
    cfg = max(1.0, min(20.0, cfg))

    sampler = str(data.get("sampler", "Euler a"))
    if sampler not in SAMPLERS:
        sampler = "Euler a"
        notes.append(f"Unknown sampler — fell back to 'Euler a'.")

    performance_mode = str(data.get("performance_mode") or RUNTIME_SETTINGS["performance_mode"])
    if performance_mode not in PERFORMANCE_MODES:
        performance_mode = "safe"
        notes.append("Unknown performance mode — using 'Safe'.")

    hires_method = str(data.get("hires_method") or RUNTIME_SETTINGS["hires_method"])
    if hires_method not in HIRES_METHODS:
        hires_method = "img2img"
        notes.append("Unknown hires method — using img2img hires-fix.")

    negative = str(data.get("negative_prompt", "")).strip()
    if len(negative) > 1200:
        raise ValueError("Negative prompt is too long (max 1200 characters).")

    return {
        "prompt": prompt,
        "style_key": style_key,
        "ratio": ratio,
        "seed": seed,
        "steps": steps,
        "cfg": round(cfg, 2),
        "sampler": sampler,
        "performance_mode": performance_mode,
        "hires_method": hires_method,
        "ultra": bool(data.get("ultra", RUNTIME_SETTINGS["ultra"])),
        "negative_prompt": negative,
    }, notes


# --------------------------------------------------------------------------- #
# Generation worker
# --------------------------------------------------------------------------- #

def _run_job(job: JobProgress, req: dict) -> None:
    global _active_job_id
    try:
        with GENERATION_LOCK:
            job.update(status="running", phase="loading_model",
                       message="Starting engine...")
            result = ENGINE.generate(req, job)
            job.finish(result)
            logger.info("job %s done: %s seed=%s (%.1fs)", job.job_id,
                        result.get("filename"), result.get("seed"), result.get("elapsed", 0))
    except FriendlyGenerationError as exc:
        logger.warning("job %s failed (friendly): %s | hint: %s", job.job_id, exc.message, exc.hint)
        job.fail(exc.message, exc.hint)
    except Exception:
        logger.exception("job %s crashed", job.job_id)
        job.fail(
            "Something went wrong inside the generator.",
            "The full traceback is in logs/app.log. Try restarting the app; if it keeps happening, "
            "switch Performance Mode to 'Safe' and lower the resolution.")
    finally:
        with _active_lock:
            if _active_job_id == job.job_id:
                _active_job_id = None


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # tiny JSON payloads only


@app.before_request
def _security_gate():
    """Hard gate on EVERY request: local-only, host allow-list, rate limit,
    and the per-boot session token for all API traffic."""
    # 1) localhost-only (even if someone edits server.host to 0.0.0.0)
    if not ALLOW_LAN and request.remote_addr not in ("127.0.0.1", "::1"):
        logger.warning("rejected non-local request from %s", request.remote_addr)
        return jsonify({"error": {"message": "This server only accepts connections from this computer.",
                                  "hint": "Set security.allow_lan in config.json only if you fully trust your network."}}), 403

    # 2) Host header allow-list (defeats DNS-rebinding against localhost)
    host = (request.host or "").split(":")[0].strip().lower()
    if host not in ALLOWED_HOSTS:
        logger.warning("rejected request with Host header %r", request.host)
        return jsonify({"error": {"message": "Blocked: invalid Host header.",
                                  "hint": "Open the app via http://127.0.0.1:8756 directly."}}), 403

    # 3) rate limits
    if request.path.startswith("/api/"):
        if not _rate_ok("api", API_RATE_LIMIT):
            return jsonify({"error": {"message": "Too many requests.",
                                      "hint": "Slow down \u2014 the limit resets within a minute."}}), 429
        if request.path == "/api/generate" and request.method == "POST":
            if not _rate_ok("generate", GEN_RATE_LIMIT):
                return jsonify({"error": {"message": "Too many generation requests.",
                                          "hint": "Wait a moment before submitting again."}}), 429

        # 4) session token on every API call
        if request.headers.get("X-OMI-Token", "") != SESSION_TOKEN:
            logger.warning("API call without a valid token from %s", request.remote_addr)
            return jsonify({"error": {"message": "Unauthorized: missing or invalid session token.",
                                      "hint": "Open the app page in your browser (http://127.0.0.1:8756) "
                                              "\u2014 the token is issued with the page."}}), 401


@app.after_request
def _security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=(), payment=()"
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    elif request.path == "/":
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def index():
    return render_template("index.html",
                           brand="OLD MODEL IMAGES",
                           model_label=CONFIG["model"].get("name_label", "DreamShaper 8 (SD 1.5)"),
                           session_token=SESSION_TOKEN)


@app.route("/api/styles")
def api_styles():
    return jsonify({
        "styles": STYLE_LIB.list_for_api(),
        "ratios": ratios_for_api(),
        "samplers": list(SAMPLERS.keys()),
        "performance_modes": list(PERFORMANCE_MODES),
        "hires_methods": list(HIRES_METHODS),
        "defaults": {
            "steps": 25,
            "cfg": 7.0,
            "seed": -1,
            "sampler": "Euler a",
            "negative_prompt": "",
            "performance_mode": RUNTIME_SETTINGS["performance_mode"],
            "hires_method": RUNTIME_SETTINGS["hires_method"],
            "ultra": RUNTIME_SETTINGS["ultra"],
            "model_label": CONFIG["model"].get("name_label", "DreamShaper 8 (SD 1.5)"),
            "safety_output_filter": bool(CONFIG.get("safety", {}).get("check_outputs", True)),
        },
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    global _active_job_id
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _err("The request body must be JSON.", "This looks like a bug — refresh the page and retry.")

    try:
        req, notes = validate_generate_payload(data)
    except ValueError as exc:
        return _err(str(exc))

    with _active_lock:
        if _active_job_id is not None and not _job_is_finished(_active_job_id):
            return _err(
                "A generation is already running.",
                "Wait for the current image to finish — this app runs one job at a time to protect your 4 GB GPU.",
                status=409)
        job = JobProgress(uuid.uuid4().hex[:12])
        _active_job_id = job.job_id

    _remember_job(job)
    if notes:
        logger.info("generate notes for job %s: %s", job.job_id, "; ".join(notes))

    worker = threading.Thread(target=_run_job, args=(job, req), name=f"omi-job-{job.job_id}", daemon=True)
    worker.start()
    return jsonify({"job_id": job.job_id, "notes": notes}), 202


@app.route("/api/progress/<job_id>")
def api_progress(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return _err("Unknown or expired job id.", "Start a new generation.", status=404)
    snap = job.snapshot()
    if snap.get("result"):
        result = dict(snap["result"])
        result["image_url"] = _signed_url(result["filename"])
        result["download_url"] = _signed_url(result["filename"], download=True)
        snap["result"] = result
    return jsonify(snap)


@app.route("/api/gallery")
def api_gallery():
    items = []
    for item in GALLERY.list_existing(limit=200):
        decorated = dict(item)
        decorated["url"] = _signed_url(item["filename"])
        decorated["download_url"] = _signed_url(item["filename"], download=True)
        items.append(decorated)
    return jsonify({"items": items})


@app.route("/api/gallery/delete", methods=["POST"])
def api_gallery_delete():
    data = request.get_json(silent=True) or {}
    filename = Path(str(data.get("filename", ""))).name  # basename only: no traversal
    if not filename or not filename.lower().endswith(".png"):
        return _err("A valid PNG filename is required to delete an image.")
    removed = GALLERY.remove(filename)
    if not removed:
        return _err("That image is not in the gallery (it may already be deleted).", "", status=404)
    logger.info("deleted output %s", filename)
    return jsonify({"ok": True, "filename": filename})


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _err("The request body must be JSON.")
    changed = []
    if "performance_mode" in data:
        mode = str(data["performance_mode"])
        if mode not in PERFORMANCE_MODES:
            return _err(f"Unknown performance mode '{mode}'.", "Use one of: safe, balanced, fast.")
        if mode != RUNTIME_SETTINGS["performance_mode"]:
            RUNTIME_SETTINGS["performance_mode"] = mode
            changed.append(f"performance_mode={mode}")
    if "hires_method" in data:
        method = str(data["hires_method"])
        if method not in HIRES_METHODS:
            return _err(f"Unknown hires method '{method}'.", "Use 'img2img' or 'esrgan_x2'.")
        RUNTIME_SETTINGS["hires_method"] = method
        changed.append(f"hires_method={method}")
    if "ultra" in data:
        RUNTIME_SETTINGS["ultra"] = bool(data["ultra"])
        changed.append(f"ultra={RUNTIME_SETTINGS['ultra']}")
    if "safety_output_filter" in data:
        CONFIG.setdefault("safety", {})["check_outputs"] = bool(data["safety_output_filter"])
        changed.append(f"safety_output_filter={CONFIG['safety']['check_outputs']}")
    # NOTE: the prompt filter and forced negatives are intentionally NOT
    # toggleable through the API \u2014 they can only be changed in config.json
    # by the machine owner.
    if changed:
        logger.info("runtime settings changed: %s", ", ".join(changed))
    return jsonify({"ok": True, "effective": dict(RUNTIME_SETTINGS)})


@app.route("/api/health")
def api_health():
    import psutil

    vm = psutil.virtual_memory()
    health = ENGINE.health()
    health.update({
        "status": "ok",
        "busy": GENERATION_LOCK.locked(),
        "active_job": _active_job_id,
        "ram_total_mb": round(vm.total / 1048576),
        "ram_used_mb": round(vm.used / 1048576),
        "ram_percent": vm.percent,
        "settings": dict(RUNTIME_SETTINGS),
    })
    return jsonify(health)


@app.route("/outputs/<path:filename>")
def serve_output(filename: str):
    if request.args.get("tk", "") != SESSION_TOKEN:
        logger.warning("outputs access without token from %s", request.remote_addr)
        return _err("Unauthorized.", "Open images through the app UI.", status=401)
    safe_name = Path(filename).name
    full = OUTPUTS_DIR / safe_name
    if not full.exists():
        return _err("Image not found.", "It may have been deleted.", status=404)
    return send_from_directory(OUTPUTS_DIR, safe_name,
                               as_attachment=request.args.get("download") == "1",
                               download_name=safe_name,
                               mimetype="image/png")


# Friendly JSON for any unhandled error / 404 on API paths.
@app.errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api/"):
        return _err("Endpoint not found.", "", status=404)
    return render_template("index.html", brand="OLD MODEL IMAGES",
                           model_label=CONFIG["model"].get("name_label", "DreamShaper 8 (SD 1.5)"),
                           session_token=SESSION_TOKEN), 404


@app.errorhandler(500)
def server_error(e):
    logger.exception("unhandled server error: %s", e)
    return _err("Internal server error.", "Check logs/app.log for details.", status=500)


@app.errorhandler(413)
def too_large(_e):
    return _err("Request body too large.", "This endpoint only accepts small JSON payloads.", status=413)


def main() -> None:
    server = CONFIG.get("server", {})
    host = server.get("host", "127.0.0.1")
    port = int(server.get("port", 8756))
    url = f"http://{host}:{port}/"

    logger.info("OLD MODEL IMAGES starting on %s (model dir: %s)", url, ENGINE._model_dir())
    print("=" * 62)
    print("   OLD MODEL IMAGES  —  local retro AI image studio")
    print(f"   Open your browser at:  {url}")
    if not ENGINE.checkpoint_present():
        print("   [!] Model files not found. Run:  python download_models.py")
    print("   Press Ctrl+C in this window to stop the server.")
    print("=" * 62, flush=True)

    if server.get("open_browser", True):
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=bool(server.get("debug", False)), threaded=True)


if __name__ == "__main__":
    main()
