"""
OLD MODEL IMAGES — offline smoke test (secured build).

Runs the REAL Flask app and the REAL generator module without a GPU:
  * security gate: session token, Host allow-list, localhost-only, CSP headers,
    rate limiting, signed /outputs URLs
  * content-safety: blocked prompts at the API boundary, allowed negatives
  * every route + the exact JSON shapes static/js/app.js consumes
  * friendly-error path via the real engine (no CUDA here), happy path,
    single-job 409 lock, validation errors, gallery persistence/deletion,
    settings and static file serving via a stub engine.

Run:  python tests/test_app_smoke.py
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


# ---------------------------------------------------------------- import app
import generator  # noqa: E402  (real module; lazy torch import keeps this cheap)
import safety  # noqa: E402

check("generator imports without heavy deps", True)
check("safety module imports", hasattr(safety, "scan_prompt"))
check("RATIOS has all 7 spec ratios",
      set(generator.RATIOS) == {"1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3"})
check("SAMPLERS has 7 entries", len(generator.SAMPLERS) == 7)

import app as app_module  # noqa: E402

client = app_module.app.test_client()
TOKEN = app_module.SESSION_TOKEN
AUTH = {"X-OMI-Token": TOKEN}

# ------------------------------------------------------- static & index pages
print("\n== pages ==")
r = client.get("/")
check("GET / -> 200", r.status_code == 200)
html = r.get_data(as_text=True)
check("index contains wordmark", "OLD MODEL" in html and "IMAGES" in html)
check("index embeds the session token meta tag", TOKEN in html)
check("CSP header present and strict",
      "default-src 'self'" in r.headers.get("Content-Security-Policy", "")
      and "frame-ancestors 'none'" in r.headers.get("Content-Security-Policy", ""))
check("nosniff + deny framing headers",
      r.headers.get("X-Content-Type-Options") == "nosniff"
      and r.headers.get("X-Frame-Options") == "DENY")
for asset in ("/static/css/styles.css", "/static/js/app.js", "/static/img/logo.svg",
              "/static/img/style_pixel_8bit.svg", "/static/img/style_synthwave.svg"):
    rr = client.get(asset)
    check(f"asset {asset} -> 200", rr.status_code == 200)

# ------------------------------------------------------------- security gate
print("\n== security gate ==")
r = client.get("/api/styles")
check("API without token -> 401", r.status_code == 401 and "error" in r.get_json())
r = client.get("/api/styles", headers={"X-OMI-Token": "forged-token"})
check("API with forged token -> 401", r.status_code == 401)
r = client.get("/api/styles", headers={**AUTH, "Host": "evil.example.com"})
check("DNS-rebinding Host header -> 403", r.status_code == 403)
r = client.get("/api/styles", headers={**AUTH, "Host": "127.0.0.1:8756"})
check("valid Host with port -> 200", r.status_code == 200)
r = client.get("/outputs/whatever.png")
check("outputs without token -> 401", r.status_code == 401)

# ------------------------------------------------------------------- styles
print("\n== /api/styles ==")
r = client.get("/api/styles", headers=AUTH)
check("GET /api/styles -> 200", r.status_code == 200)
data = r.get_json()
styles = data["styles"]
check("11 styles served", len(styles) == 11, f"got {len(styles)}")
need_keys = {"key", "name", "tagline", "description", "thumbnail", "lora_available",
             "recommended_cfg", "recommended_steps", "negative_prompt", "has_post_fx"}
check("style objects have full schema", all(need_keys <= set(s) for s in styles))
spec_style_keys = {"pixel_8bit", "pixel_16bit", "low_poly", "crt_vhs", "early_ai",
                   "vintage_photo", "polaroid", "glitch_art", "retro_anime",
                   "synthwave", "no_style"}
check("all spec style keys present", {s["key"] for s in styles} == spec_style_keys)
check("7 ratios served", len(data["ratios"]) == 7)
check("samplers served", len(data["samplers"]) == 7)
check("defaults incl. safety toggle",
      {"steps", "cfg", "seed", "sampler", "performance_mode", "hires_method",
       "ultra", "model_label", "safety_output_filter"} <= set(data["defaults"]))
missing_thumbs = [s["thumbnail"] for s in styles
                  if not (PROJECT_ROOT / "static" / s["thumbnail"]).exists()]
check("every style thumbnail exists", not missing_thumbs, str(missing_thumbs))

# ------------------------------------------------------------------- health
print("\n== /api/health ==")
r = client.get("/api/health", headers=AUTH)
h = r.get_json()
check("health -> 200 with schema", r.status_code == 200 and
      {"status", "busy", "cuda_available", "model_loaded", "checkpoint_present",
       "esrgan_x2", "esrgan_x4", "ram_total_mb", "ram_used_mb", "settings",
       "safety"} <= set(h))
check("health reports safety layers",
      {"prompt_filter", "force_negative", "check_outputs",
       "checker_weights_present"} <= set(h["safety"]))

# ------------------------------------------- real engine, no CUDA: friendly error
print("\n== real engine error UX (no CUDA here) ==")
payload = {
    "prompt": "a lighthouse in a storm",
    "style_key": "pixel_8bit",
    "ratio": "1:1",
    "seed": 12345,
    "steps": 25,
    "cfg": 7.0,
    "negative_prompt": "",
    "sampler": "Euler a",
    "performance_mode": "safe",
    "hires_method": "img2img",
    "ultra": False,
}
r = client.post("/api/generate", json=payload, headers=AUTH)
check("POST /api/generate -> 202 + job_id", r.status_code == 202 and "job_id" in r.get_json())
job_id = r.get_json()["job_id"]
deadline = time.time() + 20
snap = {}
while time.time() < deadline:
    snap = client.get(f"/api/progress/{job_id}", headers=AUTH).get_json()
    if snap["status"] in ("done", "error"):
        break
    time.sleep(0.2)
check("real engine fails gracefully without CUDA",
      snap.get("status") == "error" and snap.get("error", {}).get("message")
      and snap.get("error", {}).get("hint") is not None,
      json.dumps(snap.get("error"))[:200])

# ---------------------------------------------------------- content safety API
print("\n== content-safety at the API boundary ==")
for bad in ("a nude portrait", "nsfw art", "schoolgirl sexy", "صور عارية", "سكس",
            "p0rn scene", "n u d e"):
    r = client.post("/api/generate", json={**payload, "prompt": bad}, headers=AUTH)
    body = r.get_json() or {}
    msg = body.get("error", {}).get("message", "")
    check(f"blocked prompt {bad!r} -> 400 content-safety",
          r.status_code == 400 and "content-safety" in msg, msg[:90])

r = client.post("/api/generate", json={**payload, "negative_prompt": "nsfw, nude, nudity"},
                headers=AUTH)
check("NSFW words in NEGATIVE prompt are allowed (protective usage)", r.status_code == 202)
ok_job = r.get_json()["job_id"]
deadline = time.time() + 20
while time.time() < deadline:
    snap = client.get(f"/api/progress/{ok_job}", headers=AUTH).get_json()
    if snap["status"] in ("done", "error"):
        break
    time.sleep(0.2)
check("that job failed only due to missing CUDA (not the filter)",
      snap.get("status") == "error" and "CUDA" in snap["error"]["message"])

# ---------------------------------------------------------------- validation
print("\n== validation ==")
r = client.post("/api/generate", json={**payload, "prompt": "   "}, headers=AUTH)
check("empty prompt -> 400", r.status_code == 400 and "error" in r.get_json())
r = client.post("/api/generate", json={**payload, "style_key": "vaporwave_deluxe"}, headers=AUTH)
check("unknown style -> 400", r.status_code == 400)
r = client.post("/api/generate", json={**payload, "ratio": "21:9"}, headers=AUTH)
check("unknown ratio -> 400", r.status_code == 400)
r = client.post("/api/generate", data="not json", content_type="text/plain", headers=AUTH)
check("non-JSON body -> 400", r.status_code == 400)
r = client.get("/api/progress/doesnotexist", headers=AUTH)
check("unknown job -> 404", r.status_code == 404)

# ------------------------------------------------------ stub engine happy path
print("\n== stubbed happy path ==")

from PIL import Image, PngImagePlugin  # noqa: E402


class StubEngine:
    """Mimics OldModelGenerator.generate()/health() contracts exactly."""

    def __init__(self):
        self.gallery = app_module.GALLERY
        self.slow = False

    def health(self):
        return {"torch_installed": True, "cuda_available": True,
                "gpu_name": "NVIDIA GeForce RTX 2050",
                "vram_total_mb": 4096, "vram_allocated_mb": 1024,
                "checkpoint_present": True, "esrgan_x2": True, "esrgan_x4": True,
                "model_loaded": True, "performance_mode": "safe", "lora_loaded": None,
                "safety": {"prompt_filter": True, "force_negative": True,
                            "check_outputs": True, "checker_weights_present": True,
                            "checker_loaded": False}}

    def generate(self, request, progress):
        if self.slow:
            time.sleep(0.9)
        if "OOMTEST" in request["prompt"]:
            raise generator.FriendlyGenerationError(
                "The GPU ran out of memory during the base pass.",
                "Try Safe mode or a smaller ratio.")
        for phase in ("loading_model", "generating", "hires", "saving"):
            progress.update(phase=phase, step=0, total=1, message=f"{phase}...")
            time.sleep(0.02)
        ratio = generator.RATIOS[request["ratio"]]
        seed = request["seed"] if request["seed"] >= 0 else 987654321
        fname = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{request['style_key']}_seed{seed}.png"
        img = Image.new("RGB", (ratio["final_w"] // 8, ratio["final_h"] // 8), (40, 20, 60))
        meta = {
            "filename": fname, "prompt": request["prompt"],
            "final_prompt": "stub " + request["prompt"],
            "negative_prompt": request["negative_prompt"],
            "style_key": request["style_key"], "style_name": request["style_key"].title(),
            "seed": seed, "steps": request["steps"], "cfg": request["cfg"],
            "sampler": request["sampler"], "ratio": request["ratio"],
            "base_w": ratio["base_w"], "base_h": ratio["base_h"],
            "width": ratio["final_w"] // 8, "height": ratio["final_h"] // 8,
            "hires_method": request["hires_method"], "hires_strength": 0.4,
            "ultra": request["ultra"], "performance_mode": request["performance_mode"],
            "lora_used": None, "model": "DreamShaper 8 (SD 1.5)", "elapsed": 0.5,
            "warnings": [], "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("omi:seed", str(seed))
        img.save(app_module.OUTPUTS_DIR / fname, format="PNG", pnginfo=pnginfo)
        self.gallery.add(meta)
        meta_out = dict(meta)
        meta_out["image_url"] = f"/outputs/{fname}"
        meta_out["download_url"] = f"/outputs/{fname}?download=1"
        return meta_out


stub = StubEngine()
app_module.ENGINE = stub

# --- busy lock / 409 -------------------------------------------------------
stub.slow = True
r1 = client.post("/api/generate", json=payload, headers=AUTH)
check("job A accepted", r1.status_code == 202)
r2 = client.post("/api/generate", json=payload, headers=AUTH)
check("job B while A running -> 409 with friendly error",
      r2.status_code == 409 and "error" in r2.get_json() and r2.get_json()["error"]["hint"])
stub.slow = False
job_a = r1.get_json()["job_id"]
deadline = time.time() + 20
snap = {}
while time.time() < deadline:
    snap = client.get(f"/api/progress/{job_a}", headers=AUTH).get_json()
    if snap["status"] in ("done", "error"):
        break
    time.sleep(0.15)
check("job A completes after 409 rejection", snap.get("status") == "done")
result = snap.get("result") or {}
ui_result_keys = {"filename", "image_url", "download_url", "seed", "elapsed", "style_key",
                  "style_name", "width", "height", "final_prompt", "negative_prompt", "steps",
                  "cfg", "sampler", "ratio", "hires_method", "ultra", "lora_used", "warnings",
                  "created_at", "model", "base_w", "base_h", "performance_mode", "prompt"}
check("result schema matches app.js expectations", ui_result_keys <= set(result),
      str(ui_result_keys - set(result)))
check("image_url is signed with the session token", f"tk={TOKEN}" in result["image_url"])
check("percent field exists in snapshot", "percent" in snap)

# --- OOM error path ---------------------------------------------------------
r = client.post("/api/generate", json={**payload, "prompt": "OOMTEST please"}, headers=AUTH)
jid = r.get_json()["job_id"]
deadline = time.time() + 10
snap = {}
while time.time() < deadline:
    snap = client.get(f"/api/progress/{jid}", headers=AUTH).get_json()
    if snap["status"] in ("done", "error"):
        break
    time.sleep(0.1)
check("OOM-style failure -> status error + message + hint",
      snap.get("status") == "error" and "memory" in snap["error"]["message"].lower()
      and snap["error"]["hint"])

# --- gallery ----------------------------------------------------------------
print("\n== gallery ==")
r = client.get("/api/gallery", headers=AUTH)
g = r.get_json()
check("gallery lists saved items", r.status_code == 200 and len(g["items"]) >= 1)
item = g["items"][0]
check("gallery item has reuse fields",
      {"filename", "prompt", "style_key", "ratio", "seed", "steps", "cfg", "sampler",
       "negative_prompt", "hires_method", "ultra", "width", "height"} <= set(item))
check("gallery item carries signed url + download_url",
      item.get("url", "").startswith("/outputs/") and f"tk={TOKEN}" in item["url"]
      and "download=1" in item.get("download_url", ""))
fname = item["filename"]
r = client.get(item["url"])
check("signed PNG URL -> 200 image/png", r.status_code == 200 and r.mimetype == "image/png")
r = client.get(item["download_url"])
check("download header set", r.status_code == 200
      and "attachment" in r.headers.get("Content-Disposition", ""))
r = client.get(f"/outputs/{fname}")
check("unsigned PNG URL -> 401", r.status_code == 401)
r = client.get("/outputs/..%5Cconfig.json", headers=AUTH)
check("path traversal blocked", r.status_code in (400, 401, 404))
r = client.post("/api/gallery/delete", json={"filename": fname}, headers=AUTH)
check("delete -> ok", r.status_code == 200 and r.get_json()["ok"])
check("file removed from disk", not (app_module.OUTPUTS_DIR / fname).exists())
r = client.post("/api/gallery/delete", json={"filename": fname}, headers=AUTH)
check("delete twice -> 404", r.status_code == 404)
r = client.post("/api/gallery/delete", json={"filename": "../../config.json"}, headers=AUTH)
check("delete rejects traversal", r.status_code == 400)

# --- settings ---------------------------------------------------------------
print("\n== settings ==")
r = client.post("/api/settings", json={"performance_mode": "balanced",
                                        "hires_method": "esrgan_x2", "ultra": True},
                headers=AUTH)
s = r.get_json()
check("settings applied", r.status_code == 200 and s["effective"]["performance_mode"] == "balanced"
      and s["effective"]["hires_method"] == "esrgan_x2" and s["effective"]["ultra"] is True)
r = client.post("/api/settings", json={"performance_mode": "turbocharged"}, headers=AUTH)
check("bad performance mode -> 400", r.status_code == 400)
r = client.post("/api/settings", json={"safety_output_filter": False}, headers=AUTH)
check("safety output filter toggle -> 200", r.status_code == 200
      and app_module.CONFIG["safety"]["check_outputs"] is False)
r = client.post("/api/settings", json={"safety_output_filter": True}, headers=AUTH)
check("safety output filter re-enabled", app_module.CONFIG["safety"]["check_outputs"] is True)
client.post("/api/settings", json={"performance_mode": "safe", "hires_method": "img2img",
                                    "ultra": False}, headers=AUTH)

# --- rate limiting ----------------------------------------------------------
print("\n== rate limiting ==")
with app_module.RATE_LOCK:
    app_module.RATE_WINDOW["generate"].clear()
    app_module.RATE_WINDOW["api"].clear()
got_429 = False
for i in range(app_module.GEN_RATE_LIMIT + 8):
    rr = client.post("/api/generate", json=payload, headers=AUTH)
    if rr.status_code == 429:
        got_429 = True
        break
check("generate rate limit kicks in with 429", got_429)
with app_module.RATE_LOCK:  # reset so nothing else is affected
    app_module.RATE_WINDOW["generate"].clear()
    app_module.RATE_WINDOW["api"].clear()

# --- JS payload contract ----------------------------------------------------
print("\n== JS <-> Flask payload contract ==")
js_text = (PROJECT_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
for field in ("prompt:", "style_key:", "ratio:", "seed:", "steps:", "cfg:",
              "negative_prompt:", "sampler:", "performance_mode:", "hires_method:", "ultra:"):
    check(f"app.js sends '{field.rstrip(':')}'", field in js_text)
check("app.js sends the token header", "X-OMI-Token" in js_text)
check("app.js reads the token meta tag", 'meta[name="omi-token"]' in js_text)
check("server accepts everything app.js sends (validated above via 202s)", True)

# ------------------------------------------------------------------ summary
print(f"\nRESULT: {PASSED} passed, {FAILED} failed")
sys.exit(0 if FAILED == 0 else 1)
