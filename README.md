# OLD MODEL IMAGES

**A branded, fully-offline text-to-image studio for vintage, retro and "old AI" aesthetics.**
Built and tuned for one specific machine class: Windows 11 laptops with a 4 GB NVIDIA GPU
(developed against an RTX 2050 / Ryzen 5 7535HS / **8 GB system RAM**). No cloud, no accounts,
no telemetry — your prompts and images never leave your laptop.

---

## Architecture in one breath

A Flask server (`app.py`) exposes a small JSON API and serves a hand-built dark/amber/magenta
CRT-branded UI (vanilla JS + CSS, no build step, minimal RAM). All image work happens in
`generator.py`: **Stable Diffusion 1.5 (DreamShaper 8, fp16)** via `diffusers`, run in
sequential-CPU-offload mode by default so the 4 GB GPU never overflows, with a mandatory 2×
hires-fix second pass, optional tiled Real-ESRGAN ultra upscale, per-style LoRA load/unload,
style post-FX, a single-job queue and OOM defense-in-depth. Styles are pure data
(`styles/styles.json`) — every style ships a verified LoRA **or** a fully-engineered prompt
fallback, so the app never breaks when a LoRA is missing.

---

## Quick start (5 steps)

1. **Install Python 3.10, 3.11 or 3.12 (64-bit)** from <https://www.python.org/downloads/windows/>
   and tick **"Add python.exe to PATH"** during install. (Skip if you already have it — `setup.bat`
   can also offer to install it via `winget`.)
2. **Unzip / copy the `old-model-images` folder** anywhere you like (e.g. `C:\old-model-images`).
   Paths with spaces are fine.
3. **Double-click `setup.bat`.** It creates a private `.venv`, installs the pinned dependency set
   (torch 2.2.1 + CUDA 12.1 wheels ≈ 2.5 GB) and then downloads the models
   (DreamShaper 8 fp16 ≈ 2 GB, Real-ESRGAN x2/x4 ≈ 130 MB, Pixel-Art LoRA ≈ 27 MB).
   Needs ~8 GB free disk and a stable connection.
4. **Double-click `run.bat`.** Your browser opens `http://127.0.0.1:8756` automatically.
5. **Type a prompt, pick a style, press GENERATE.** The *first* generation loads the model
   (~1–2 min on this hardware); every generation after that starts immediately.

That's it. To relaunch later: `run.bat`. To stop: close the console window (or Ctrl+C in it).

---

## Hardware envelope & expert decisions

Every choice below was made for **4 GB VRAM + 8 GB system RAM**. They are deliberate — documented
here so you can change them knowingly.

| Decision | Why |
|---|---|
| **SD 1.5 only (DreamShaper 8)** | SDXL/Flux/SD3 cannot run in 4 GB VRAM + 8 GB RAM. DreamShaper 8 is the strongest all-round SD 1.5 checkpoint and plays well with LoRAs. |
| **fp16 variant weights** | ~2 GB instead of ~4 GB on disk; halves load-time RAM pressure. `download_models.py` falls back to full precision automatically if a fp16 file is ever missing. |
| **`safety_checker=None`** | Skips loading CLIP safety weights → less RAM/VRAM, faster startup. |
| **Sequential CPU offload by default ("Safe" mode)** | Keeps VRAM peak ≈ 1.0–1.5 GB. It is the *only* mode that reliably survives 16:9 hires passes on a 4 GB card. "Balanced" (model-level offload) and "Fast" (all-GPU) are selectable in Advanced → Performance mode; Fast **will** OOM at large ratios — that is expected and handled gracefully. |
| **VAE slicing + tiling always on** | Required to decode 1824×1024 latents inside 4 GB. |
| **Attention slicing on in Safe mode only** | Belt-and-braces for the 512-class + hires passes; skipped in faster modes where SDPA/xformers already fits. |
| **xformers optional** | `setup.bat` tries `xformers==0.0.25` (the build matching torch 2.2.1). If it fails, PyTorch 2.x **SDPA** is used automatically — nearly identical speed/VRAM on this GPU. The app checks at runtime; nothing to configure. |
| **`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`** (set in `run.bat`) | Fights VRAM *fragmentation*, the silent killer on 4 GB cards. |
| **`HF_HOME=%~dp0models`** (set in `run.bat`) | All Hugging Face caches stay inside the project folder — no surprises on `C:`. |
| **One job at a time (hard lock)** | Two parallel generations = guaranteed OOM. The API answers `409` and the UI tells you politely. |
| **LoRA loaded per-generation, unloaded after** | Keeps RAM flat across a long session; switching styles never accumulates weights. |
| **Real-ESRGAN vendored, tiled, with fallbacks** | The `realesrgan`/`basicsr` PyPI packages drag in heavy, Windows-flaky deps (`gfpgan`, `torchvision` pinning, the `functional_tensor` breakage). Instead `generator.py` contains the exact BSD-licensed RRDBNet architecture (matching `ai-forever/Real-ESRGAN` weights byte-for-byte) and runs it **tiled with feathered blending** (384 px tiles → ~0.5 GB VRAM peak). On GPU OOM it retries with smaller tiles, then on CPU; if all that fails you still get the hires-fix image plus a visible warning. |
| **30-megapixel ultra guard** | A 4× pass on the widest output (1824×1024 → 7296×4096) would need ~0.7 GB of float32 accumulators — dangerous on 8 GB RAM. Above 30 MP the ultra pass automatically drops to 2× and tells you in the image notes. |
| **Lazy torch import** | `app.py` boots, serves the UI and `/api/health` before any model loads; a broken GPU driver shows up as a friendly badge, not a dead server. |

### Aspect ratios & resolutions

Base pass (SD 1.5 native territory) → mandatory 2× hires-fix (img2img, denoise 0.40, clamped to the
0.35–0.45 band) → optional Ultra pass:

| Ratio | Base pass | Final (standard) | Final (Ultra 4×)* |
|---|---|---|---|
| 1:1  | 512×512  | 1024×1024 | 4096×4096 |
| 4:3  | 680×512  | 1360×1024 | 4096×3072 (auto-capped → 2720×2048) |
| 3:4  | 512×680  | 1024×1360 | 3072×4096 (auto-capped → 2048×2720) |
| 16:9 | 912×512  | 1824×1024 | auto-capped → 3648×2048 |
| 9:16 | 512×912  | 1024×1824 | auto-capped → 2048×3648 |
| 3:2  | 768×512  | 1536×1024 | auto-capped → 3072×2048 |
| 2:3  | 512×768  | 1024×1536 | auto-capped → 2048×3072 |

\* "auto-capped" = the 30 MP RAM guard silently switches the ultra pass to Real-ESRGAN **x2**;
the UI notes it under the image. Hires method is switchable: `img2img` (default, adds detail via
re-denoising) or `esrgan_x2` (faster, purely reconstructive).

### Realistic timings on the target hardware (Safe mode)

- 1:1, 25 steps, standard hires: **~70–110 s**
- 16:9, 25 steps, standard hires: **~2–3 min**
- Balanced mode is ~30–50 % faster *if* it fits; Fast mode is fastest but OOMs on big ratios.

---

## Style catalog

Styles live in `styles/styles.json` — data, not code. Each entry carries a LoRA slot, prompt
prefix/suffix, a **fallback prompt** used automatically when the LoRA file is absent, a style
negative prompt, recommended CFG/steps and an optional post-FX hook.

| Key | Look | LoRA status at build time |
|---|---|---|
| `pixel_8bit` | NES/GameBoy chunky pixels | ✅ **PixelArtRedmond 1.5V** (`PIXARFK` trigger) — verified on HF, auto-downloaded |
| `pixel_16bit` | SNES/Genesis sprite art | ✅ same LoRA at weight 0.7 |
| `low_poly` | PS1-era faceted 3D | 🟡 prompt-engineered (no SD1.5 LoRA verifiable at build time) |
| `crt_vhs` | Scanlines, chroma bleed, tape grain | 🟡 prompt-engineered **+ real post-FX pass** (RGB split, scanlines, grain) |
| `early_ai` | 2015–2018 DeepDream/GAN hallucinations | 🟡 prompt-engineered |
| `vintage_photo` | 1800s–1950s sepia/silver-gelatin | 🟡 prompt-engineered |
| `polaroid` | Faded 70s instant film | 🟡 prompt-engineered **+ real Polaroid frame post-FX** (fade, light leak, frame) |
| `glitch_art` | Datamosh, RGB split, pixel sorting | 🟡 prompt-engineered |
| `retro_anime` | 80s–90s cel-shaded stills | 🟡 prompt-engineered (the main HF candidate turned out to be a **Flux** LoRA — incompatible with SD 1.5, rejected) |
| `synthwave` | Neon grid, chrome, magenta sunsets | 🟡 prompt-only by design (per spec) |
| `no_style` | Clean DreamShaper output | — none |

🟡 is **not** a degraded mode by accident: the fallback prompts are full engineered templates, and
the UI badge on each card tells you which path is active (`LoRA` / `PROMPT` / `FX`).

**Adding a LoRA later:** drop any SD 1.5 `.safetensors` LoRA into `styles/lora/`, then in
`styles/styles.json` set that style's `lora_path`, `lora_weight` and put its trigger word at the
start of `prompt_prefix`. Restart the app — the card badge flips to `LoRA` automatically.
**Adding a brand-new style:** copy any JSON entry, change `key`/prompts, add an SVG thumbnail to
`static/img/`. No code changes.

---

## Security architecture (v1.1 "secured build")

This app is a **local server**, so the threat model is: malicious websites probing your
localhost, DNS-rebinding, request smuggling into the API, path traversal, and abuse by
anyone else who sits at your computer. Every layer below is active by default.

| # | Layer | Implementation |
|---|---|---|
| 1 | **Localhost-only** | Requests from any IP other than `127.0.0.1`/`::1` are rejected with 403 *before routing*, even if `server.host` is edited to `0.0.0.0`. (`security.allow_lan` exists for advanced users and defaults to `false`.) |
| 2 | **Host-header allow-list** | Defeats DNS-rebinding: only `127.0.0.1` / `localhost` (+ configured host) are accepted; anything else → 403. |
| 3 | **Per-boot session token** | A 256-bit token (`secrets.token_urlsafe(32)`) is generated at startup, embedded in the served page, and required as `X-OMI-Token` on **every** `/api/*` call → 401 otherwise. Malicious pages cannot read it (same-origin policy) nor forge it. |
| 4 | **Signed image URLs** | `/outputs/<file>` requires `?tk=<token>`; gallery/result URLs are signed server-side. Filenames are timestamp+seed based (unguessable), served via `send_from_directory` with basename-only sanitization → no path traversal. |
| 5 | **Strict CSP + headers** | `default-src 'self'; script-src 'self'; frame-ancestors 'none'; object-src 'none'…`, plus `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Permissions-Policy` denying camera/mic/geo, COOP/CORP same-origin. No inline scripts/styles anywhere in the UI. |
| 6 | **Rate limiting** | Sliding-window: 240 req/min on `/api/*`, 30 req/min on `/api/generate` → 429 with a friendly body. |
| 7 | **Single-job GPU lock** | One generation at a time (409 on a second submit) — also a denial-of-service guard for the 4 GB GPU. |
| 8 | **Input validation everywhere** | Strict typing/clamping on every field, 1 MB body cap, prompt length caps, JSON-only endpoints, `debug=False` (no Werkzeug console). |
| 9 | **Error hygiene** | API returns `{message, hint}` only; full tracebacks go to `logs/app.log` exclusively. Blocked-content events are logged **by category only** (your prompts are not written to logs). |
| 10 | **Supply chain** | Fully pinned dependency set, weights fetched over HTTPS from official HF repos, `weights_only=True` on all `torch.load` calls (no pickle RCE), vendored upscaler code (no third-party runtime dep). |

## Content safety — zero tolerance for NSFW (3 independent layers)

| Layer | Where | What it does |
|---|---|---|
| **1. Prompt filter** (`safety.py`) | API boundary **and** generator (double-checked) | ~450 blocked terms + phrase-combination regexes across **English, Arabic, Russian, Korean, Japanese, Chinese, Spanish, French, Dutch/Turkish**. Bypass-resistant: case folding, leet-speak (`p0rn`, `$exy`, `nud3`), separators (`n u d e`, `n-u-d-e`, `N.S.F.W`), letter-run collapse (`sexxx`), Arabic diacritic removal and letter-variant unification (`إباحي` = `اباحي`, `ة` = `ه`). Word-boundary matching keeps legitimate words safe (assassin, classic, peninsula, Scunthorpe, chicken breast, XXXL…). Minor-safety and sexual-violence hits get **critical severity** with zero-tolerance messaging. 228-case test suite: `tests/test_safety.py`. |
| **2. Forced negative prompt** | every generation | A core NSFW negative (`nsfw, nude, nudity, naked, explicit, sexual, porn, hentai…`) is **always** merged into the negative prompt — it cannot be removed from the UI, only via `config.json` by the machine owner. |
| **3. AI output scanner** | after the hires pass, **before** save | The official CLIP-based Stable-Diffusion safety checker (608 MB, auto-downloaded) scans the finished image **on CPU** (zero VRAM cost). If flagged: the image is **never written to disk**, never enters the gallery, and you get a clear "blocked by the NSFW output filter" message. The `🛡 PROTECTED` header badge shows live status (3/3 layers, or 2/3 if weights are missing / toggle off). |

**Honest limitations** (documented on purpose): no filter is mathematically perfect — the
CLIP checker has known false-positive/negative rates, and anyone with full control of the
machine can edit local files. The design goal is: *accidental, guest, or web-driven abuse is
effectively impossible, and every generation passes three independent gates.* The prompt
filter and forced negatives cannot be disabled from the UI; the output scanner can be
toggled in Advanced (e.g. to save ~1.2 GB RAM on a tight session) and the UI then shows
`🛡 FILTER 2/3` so the state is always visible.

**Privacy:** nothing ever leaves your machine — no telemetry, `HF_HUB_DISABLE_TELEMETRY=1`,
blocked events logged by category only, images and prompts stay in `outputs/`.

---

## Using the app

- **Prompt** → **Style card** → **Ratio card** → **Resolution** (Standard 2× / Ultra 4×) → **GENERATE**
  (or Ctrl+Enter).
- **Advanced** panel: steps, CFG, sampler (7 schedulers), seed (+ 🎲 random, ♻ reuse, "Variation
  seed+1" buttons), negative prompt (merged & de-duplicated with the style's own negatives),
  hires method, performance mode.
- Selecting a style auto-fills its **recommended steps/CFG** (you can always override).
- The header badges show live GPU/VRAM/RAM/model state; the progress overlay shows the exact phase
  (model load → base pass step x/y → hires step x/y → upscale → save).
- **Gallery tab**: every output ever produced (persisted in `outputs/gallery.json` + PNG metadata),
  with lightbox view, **↺ Reuse** (restores *all* settings incl. seed), **⬇ Download** and **✕ Delete**.
- Files are saved as `outputs/20261024-153001_pixel_8bit_seed12345.png` and carry the full recipe
  in PNG text chunks (`omi:prompt`, `omi:seed`, …) — readable by any metadata tool.

---

## Project layout

```
old-model-images/
├── README.md               ← you are here
├── requirements.txt        ← pinned deps (torch 2.2.1+cu121, diffusers 0.27.2, …)
├── setup.bat               ← one-time: venv + deps + model downloads + report
├── run.bat                 ← launches the server (sets CUDA allocator + HF_HOME)
├── config.json             ← model paths, defaults, port, memory knobs
├── app.py                  ← Flask API + security gate (token/Host/CSP/rate-limit) + job queue
├── generator.py            ← engine: SD1.5 pipeline, LoRA, hires, tiled ESRGAN, post-FX, NSFW scan
├── safety.py               ← multi-language content filter + forced negatives (pure Python)
├── download_models.py      ← checkpoint/upscaler/LoRA/NSFW-scanner fetcher (never crashes on a missing LoRA)
├── styles/
│   ├── styles.json         ← the entire style catalog (data-driven)
│   └── lora/               ← downloaded LoRAs land here
├── models/
│   ├── checkpoints/        ← DreamShaper 8 (diffusers layout, fp16)
│   ├── upscalers/          ← RealESRGAN_x2.pth / x4.pth
│   └── safety/             ← CLIP NSFW output-scanner weights (~608 MB)
├── static/
│   ├── css/styles.css      ← brand theme (dark, amber/magenta, scanlines)
│   ├── js/app.js           ← UI logic (vanilla, polls /api/progress)
│   └── img/                ← logo + 11 hand-built SVG style thumbnails
├── templates/index.html
├── outputs/                ← your images + gallery.json index
├── logs/                   ← app.log (tracebacks live ONLY here) + download_report.txt
├── .gitignore              ← keeps weights/outputs/venv out of git
├── .github/workflows/ci.yml← GitHub Actions: runs all 4 test suites on every push
└── tests/
    ├── test_app_smoke.py   ← API/security/UI-contract suite (stubs the GPU engine)
    ├── test_safety.py      ← 228-case content-filter suite (EN/AR/RU/KO/JA/ZH + evasions)
    ├── test_generation_cpu.py ← end-to-end generation on CPU (tiny SD + real ESRGAN)
    ├── test_model_load_fix.py ← DreamShaper load-layout regression suite
    └── verify_real_weights.py ← strict-load proof against official Real-ESRGAN weights
```

### HTTP API (consumed by `static/js/app.js`; shapes match exactly)

All `/api/*` routes require the `X-OMI-Token` header (issued with the page);
`/outputs/*` requires the `?tk=` signature. Both come from the per-boot session token.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/styles` | GET | styles + ratios + samplers + defaults |
| `/api/generate` | POST | submit job → `202 {job_id}` / `409 busy` / `400 validation` |
| `/api/progress/<job_id>` | GET | `{status, phase, step, total, percent, message, result, error}` |
| `/api/gallery` | GET | persisted gallery items (newest first) |
| `/api/gallery/delete` | POST | `{filename}` → deletes PNG + index entry |
| `/api/settings` | POST | live-update performance mode / hires method / ultra |
| `/api/health` | GET | GPU, VRAM, RAM, model presence, busy flag |
| `/outputs/<file>.png` | GET | image serving (`?download=1` → attachment) |

Errors are always `{"error": {"message", "hint"}}` — never a raw traceback.

---

## Troubleshooting

**"The GPU ran out of memory…" toast**
The OOM watchdog caught it, cleared the CUDA cache and kept the server alive. In order of
effectiveness: ① Performance mode → **Safe**; ② disable the **Ultra 4×** pass; ③ pick a smaller
ratio (1:1 / 4:3 are cheapest, 16:9 / 9:16 the heaviest); ④ lower steps to ~20; ⑤ close browsers
with many tabs, games, video players — on 8 GB RAM every process counts; ⑥ reboot (Windows
compresses GPU memory poorly after long sessions).

**First run / first generation is very slow**
Normal. `setup.bat` compiles nothing, but the *first* generation loads ~2 GB of fp16 weights from
disk into RAM and CUDA initializes its context — budget 1–2 minutes. Subsequent generations in the
same session skip all of that. Also make sure the laptop is **plugged in** and on a performance
power plan; on battery the 50 W GPU is throttled hard.

**`xformers` missing / setup.bat printed a xformers WARN**
Ignorable by design. The app checks `is_xformers_available()` at load time and silently uses
PyTorch 2.x SDPA attention otherwise — within a few percent of xformers on Ada hardware. If you
want it later: `.venv\Scripts\pip install xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu121`
(must match torch 2.2.1 exactly — do **not** let it upgrade torch).

**Windows Defender / SmartScreen warnings**
`setup.bat` and `run.bat` are unsigned local scripts; SmartScreen may ask "Keep anyway?" — that is
expected for any self-built tool. If Defender quarantines a downloaded `.safetensors`/`.pth`
(it occasionally flags large ML blobs heuristically): restore it in *Virus & threat protection →
Protection history*, then add the `old-model-images` folder as an **exclusion**, and re-run
`download_models.py`. Never exclude folders you don't control.

**Model download failures**
`download_models.py` is resumable and idempotent — just run it again:
`.venv\Scripts\python.exe download_models.py`. It writes `logs/download_report.txt` explaining every
file (downloaded / already present / skipped). LoRA failures are **warnings, never fatal** — the
affected style switches to its prompt fallback. If Hugging Face is blocked on your network, set a
proxy before running: `set HTTPS_PROXY=http://your-proxy:port`.

**`torch.cuda.is_available()` says False in setup**
Update the NVIDIA driver (any Game Ready with CUDA 12.x, e.g. your 621.01, is fine), reboot, and
re-run `setup.bat` step 5. In Device Manager confirm the RTX 2050 has no warning icon.

**Port 8756 already in use**
Edit `config.json → server.port`, then `run.bat` again. The UI follows automatically.

**`does not appear to have a file named preprocessor_config.json` / `no file named diffusion_pytorch_model.bin found in ...\vae`**
Fixed in the current build (v1.0.1): DreamShaper 8's `model_index.json` declares a
safety-checker stack (`feature_extractor`) that this app deliberately never downloads, so the
loader now passes `feature_extractor=None` alongside `safety_checker=None`. If you see this error
you are running an old `generator.py` — replace it with the current one and restart; your
downloaded model files are fine and do **not** need re-downloading.

**App feels starved / Windows starts swapping**
Close other big apps before generating — with 8 GB RAM, the SD 1.5 fp16 weights (~2.5 GB resident
in Safe mode) plus browser plus Windows leaves little slack. Never run two copies of this app; the
single-job lock is per-process, not per-machine.

**Image looks "wrong" for the style**
Check the badge on the style card: `LoRA` (trained weights active) vs `PROMPT` (engineered
fallback). Prompt-fallback styles depend more on your prompt — be concrete ("a pixel-art treasure
chest" beats "treasure chest"). Also try the style's recommended steps/CFG (auto-filled on select).

---

## Testing

Four suites, all runnable **without a GPU** (CI runs them on every push):

```
.venv\Scripts\python.exe tests\test_safety.py          # 228 content-filter checks
.venv\Scripts\python.exe tests\test_app_smoke.py       # 82 API + security checks
.venv\Scripts\python.exe tests\test_generation_cpu.py  # 43 end-to-end generation checks
.venv\Scripts\python.exe tests\test_model_load_fix.py  # 15 checkpoint-layout regressions
```

`test_generation_cpu.py` and `verify_real_weights.py` additionally exercise the real tiled
Real-ESRGAN path when `models/upscalers/RealESRGAN_x2.pth` is present (they skip gracefully
when it is not).

## Credits & licenses

- **DreamShaper 8** by Lykon — CreativeML Open RAIL-M.
- **Real-ESRGAN** weights (ai-forever mirror) & RRDBNet architecture — BSD-3 (xinntao / AI Forewer);
  the vendored copy in `generator.py` preserves the architecture exactly.
- **PixelArtRedmond 1.5V** LoRA by artificialguybr — see the model card on Hugging Face.
- diffusers/transformers/accelerate/peft — Apache-2.0. Flask — BSD-3. Pillow — MIT-CMU.

*OLD MODEL IMAGES is a local tool; generated images are yours.*
