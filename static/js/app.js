/* ============================================================
   OLD MODEL IMAGES — frontend logic (vanilla JS, no deps)
   Talks to Flask: /api/styles /api/generate /api/progress/<id>
   /api/gallery /api/gallery/delete /api/settings /api/health
   ============================================================ */
"use strict";

const $ = (id) => document.getElementById(id);
const TOKEN = (document.querySelector('meta[name="omi-token"]') || {}).content || "";

const state = {
  styles: [],
  ratios: [],
  samplers: [],
  defaults: {},
  selectedStyle: null,
  selectedRatio: "1:1",
  currentJobId: null,
  pollTimer: null,
  busy: false,
  lastResult: null,
  galleryItems: [],
};

/* ---------------------------------------------------------- toasts */

function toast(kind, title, message, hint) {
  const stack = $("toast-stack");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML =
    `<span class="toast-title">${escapeHtml(title)}</span>` +
    (message ? `${escapeHtml(message)}` : "") +
    (hint ? `<span class="toast-hint">💡 ${escapeHtml(hint)}</span>` : "");
  const close = () => el.remove();
  el.addEventListener("click", close);
  stack.appendChild(el);
  setTimeout(close, kind === "error" ? 12000 : 5000);
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------------------------------------------------------- API helpers */

async function api(path, options) {
  const opts = Object.assign({}, options || {});
  opts.headers = Object.assign({ "X-OMI-Token": TOKEN }, opts.headers || {});
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { body = null; }
  if (!res.ok) {
    const err = (body && body.error) || { message: `Request failed (HTTP ${res.status})`, hint: "" };
    const e = new Error(err.message);
    e.hint = err.hint || "";
    e.status = res.status;
    throw e;
  }
  return body;
}

/* ---------------------------------------------------------- rendering */

const RATIO_SHAPE_PX = {
  "1:1":  [26, 26], "4:3":  [32, 24], "3:4":  [24, 32],
  "16:9": [36, 20], "9:16": [20, 36], "3:2":  [33, 22], "2:3": [22, 33],
};

function renderStyles() {
  const grid = $("style-grid");
  grid.innerHTML = "";
  for (const s of state.styles) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "style-card" + (state.selectedStyle === s.key ? " selected" : "");
    card.dataset.key = s.key;
    const badge = s.lora_available
      ? '<span class="lora-dot" title="LoRA weights installed">LoRA</span>'
      : (s.has_post_fx
        ? '<span class="lora-dot fx" title="Includes post-processing effects">FX</span>'
        : '<span class="lora-dot fx" title="Prompt-engineered style">PROMPT</span>');
    card.innerHTML =
      `<img class="style-thumb" src="/static/${s.thumbnail}" alt="" loading="lazy">` +
      `<span class="style-body">` +
        `<span class="style-name">${escapeHtml(s.name)} ${badge}</span>` +
        `<span class="style-tagline">${escapeHtml(s.tagline)}</span>` +
      `</span>`;
    card.title = s.description;
    card.addEventListener("click", () => selectStyle(s.key));
    grid.appendChild(card);
  }
  updateStyleNote();
}

function selectStyle(key) {
  state.selectedStyle = key;
  const s = state.styles.find((x) => x.key === key);
  document.querySelectorAll(".style-card").forEach((c) =>
    c.classList.toggle("selected", c.dataset.key === key));
  if (s) {
    $("steps").value = s.recommended_steps;
    $("cfg").value = s.recommended_cfg;
    $("steps-rec").textContent = `(recommended: ${s.recommended_steps})`;
    $("cfg-rec").textContent = `(recommended: ${s.recommended_cfg})`;
  }
  updateStyleNote();
}

function updateStyleNote() {
  const s = state.styles.find((x) => x.key === state.selectedStyle);
  const note = $("style-lora-note");
  if (!s) { note.textContent = ""; return; }
  note.textContent = s.lora_available
    ? "— LoRA active"
    : (s.key === "no_style" ? "— base model" : "— prompt-engineered fallback active");
}

function renderRatios() {
  const row = $("ratio-row");
  row.innerHTML = "";
  for (const r of state.ratios) {
    const [w, h] = RATIO_SHAPE_PX[r.key] || [26, 26];
    const card = document.createElement("button");
    card.type = "button";
    card.className = "ratio-card" + (state.selectedRatio === r.key ? " selected" : "");
    card.dataset.key = r.key;
    card.title = `${r.label} — base ${r.base_w}×${r.base_h}, final ${r.final_w}×${r.final_h}`;
    const shape = document.createElement("span");
    shape.className = "ratio-shape";
    shape.style.width = `${w}px`;
    shape.style.height = `${h}px`;
    const label = document.createElement("span");
    label.className = "ratio-label";
    label.textContent = r.key;
    card.appendChild(shape);
    card.appendChild(label);
    card.addEventListener("click", () => selectRatio(r.key));
    row.appendChild(card);
  }
  updateResolutionHint();
}

function selectRatio(key) {
  state.selectedRatio = key;
  document.querySelectorAll(".ratio-card").forEach((c) =>
    c.classList.toggle("selected", c.dataset.key === key));
  updateResolutionHint();
}

function updateResolutionHint() {
  const r = state.ratios.find((x) => x.key === state.selectedRatio);
  const ultra = $("resolution").value === "ultra";
  if (!r) return;
  $("resolution-hint").textContent = ultra
    ? `Standard pass: ${r.final_w}×${r.final_h}. Ultra may finish at ${r.final_w * 2}×${r.final_h * 2} (auto-reduced on huge ratios to protect your 8 GB RAM).`
    : `Final output: ${r.final_w}×${r.final_h} (base ${r.base_w}×${r.base_h} → 2× hires-fix).`;
}

function renderSamplers() {
  const sel = $("sampler");
  sel.innerHTML = "";
  for (const name of state.samplers) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name === state.defaults.sampler) opt.selected = true;
    sel.appendChild(opt);
  }
}

/* ---------------------------------------------------------- gallery */

function renderGallery() {
  const grid = $("gallery-grid");
  const empty = $("gallery-empty");
  grid.innerHTML = "";
  $("gallery-count").textContent = String(state.galleryItems.length);
  empty.classList.toggle("hidden", state.galleryItems.length > 0);

  for (const item of state.galleryItems) {
    const card = document.createElement("div");
    card.className = "gcard";
    card.innerHTML =
      `<div class="gcard-img-wrap"><img src="${item.url}" alt="" loading="lazy"></div>` +
      `<div class="gcard-body">` +
        `<div class="gcard-prompt" title="${escapeHtml(item.prompt)}">${escapeHtml(item.prompt || "(no prompt)")}</div>` +
        `<div class="gcard-meta"><em>${escapeHtml(item.style_key)}</em><span>${item.ratio}</span><span>${item.width}×${item.height}</span><span>seed ${item.seed}</span></div>` +
        `<div class="gcard-actions">` +
          `<button type="button" class="btn act-reuse" title="Load these settings into the form">↺ Reuse</button>` +
          `<a class="btn" href="${item.download_url}" download="${item.filename}" title="Download PNG">⬇</a>` +
          `<button type="button" class="btn btn-danger act-del" title="Delete image">✕</button>` +
        `</div>` +
      `</div>`;
    card.querySelector(".gcard-img-wrap").addEventListener("click", () => openLightbox(item.url));
    card.querySelector(".act-reuse").addEventListener("click", () => reuseSettings(item));
    card.querySelector(".act-del").addEventListener("click", () => deleteGalleryItem(item));
    grid.appendChild(card);
  }
}

async function loadGallery() {
  try {
    const data = await api("/api/gallery");
    state.galleryItems = data.items || [];
    renderGallery();
  } catch (e) {
    toast("error", "Gallery unavailable", e.message, e.hint);
  }
}

async function deleteGalleryItem(item) {
  if (!confirm(`Delete ${item.filename}? This removes the PNG from outputs/ permanently.`)) return;
  try {
    await api("/api/gallery/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: item.filename }),
    });
    state.galleryItems = state.galleryItems.filter((x) => x.filename !== item.filename);
    renderGallery();
    toast("info", "Deleted", item.filename);
  } catch (e) {
    toast("error", "Delete failed", e.message, e.hint);
  }
}

function reuseSettings(item) {
  $("prompt").value = item.prompt || "";
  updatePromptCount();
  if (state.styles.some((s) => s.key === item.style_key)) selectStyle(item.style_key);
  if (state.ratios.some((r) => r.key === item.ratio)) selectRatio(item.ratio);
  $("steps").value = item.steps;
  $("cfg").value = item.cfg;
  $("sampler").value = item.sampler;
  $("seed").value = item.seed;
  $("negative").value = item.negative_prompt || "";
  $("hires-method").value = item.hires_method || "img2img";
  $("resolution").value = item.ultra ? "ultra" : "standard";
  switchTab("create");
  toast("success", "Settings loaded", `Seed ${item.seed} and all parameters from this image are back in the form.`);
}

/* ---------------------------------------------------------- lightbox */

function openLightbox(src) {
  const box = document.createElement("div");
  box.className = "lightbox";
  box.innerHTML = `<img src="${src}" alt="">`;
  box.addEventListener("click", () => box.remove());
  document.body.appendChild(box);
  const onKey = (e) => { if (e.key === "Escape") { box.remove(); document.removeEventListener("keydown", onKey); } };
  document.addEventListener("keydown", onKey);
}

/* ---------------------------------------------------------- generation */

function collectPayload() {
  return {
    prompt: $("prompt").value.trim(),
    style_key: state.selectedStyle || "no_style",
    ratio: state.selectedRatio,
    seed: parseInt($("seed").value, 10),
    steps: parseInt($("steps").value, 10),
    cfg: parseFloat($("cfg").value),
    negative_prompt: $("negative").value.trim(),
    sampler: $("sampler").value,
    performance_mode: $("perf-mode").value,
    hires_method: $("hires-method").value,
    ultra: $("resolution").value === "ultra",
  };
}

function setBusy(busy) {
  state.busy = busy;
  const btn = $("btn-generate");
  btn.disabled = busy;
  btn.classList.toggle("busy", busy);
  btn.querySelector(".btn-generate-label").textContent = busy ? "◼ GENERATING" : "▶ GENERATE";
  $("queue-note").textContent = busy ? "One job at a time — your 4 GB GPU says thanks." : "";
}

async function startGeneration() {
  if (state.busy) return;
  const payload = collectPayload();
  if (!payload.prompt) {
    toast("warn", "Empty prompt", "Describe the image you want first.");
    $("prompt").focus();
    return;
  }
  if (Number.isNaN(payload.seed)) payload.seed = -1;
  if (Number.isNaN(payload.steps)) payload.steps = state.defaults.steps || 25;
  if (Number.isNaN(payload.cfg)) payload.cfg = state.defaults.cfg || 7;

  setBusy(true);
  $("canvas-idle").classList.add("hidden");
  $("output-image").classList.add("hidden");
  showProgress("queued", "Submitting job…", null, null);

  try {
    const data = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.currentJobId = data.job_id;
    if (data.notes && data.notes.length) toast("warn", "Heads up", data.notes.join(" "));
    startPolling(data.job_id);
  } catch (e) {
    setBusy(false);
    hideProgress();
    if (e.status === 409) {
      toast("warn", "GPU busy", e.message, e.hint);
    } else if (e.status === 400 && /content-safety|blocked/i.test(e.message)) {
      toast("error", "Blocked by safety filter", e.message, e.hint);
    } else if (e.status === 401 || e.status === 403) {
      toast("error", "Security check failed", e.message,
            "Open the app directly at http://127.0.0.1:8756 in this browser, then retry.");
    } else {
      toast("error", "Could not start", e.message, e.hint);
    }
  }
}

function startPolling(jobId) {
  stopPolling();
  state.pollTimer = setInterval(() => pollOnce(jobId), 600);
  pollOnce(jobId);
}

function stopPolling() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

const PHASE_FALLBACK_PCT = {
  queued: 3, loading_model: 8, generating: null, hires: null,
  safety_scan: 91, upscaling: 93, post_fx: 96, saving: 98, done: 100,
};

async function pollOnce(jobId) {
  let snap;
  try {
    snap = await api(`/api/progress/${jobId}`);
  } catch (e) {
    stopPolling();
    setBusy(false);
    hideProgress();
    toast("error", "Lost contact with the server", e.message, "Is the app window still open? Restart run.bat if needed.");
    return;
  }
  if (jobId !== state.currentJobId) { stopPolling(); return; }  // stale poller from an older job

  const pct = snap.percent != null ? snap.percent : (PHASE_FALLBACK_PCT[snap.phase] ?? 50);
  showProgress(snap.phase, snap.message || "", snap.step, snap.total, pct);

  if (snap.status === "done") {
    stopPolling();
    setBusy(false);
    hideProgress();
    onGenerationDone(snap.result);
  } else if (snap.status === "error") {
    stopPolling();
    setBusy(false);
    hideProgress();
    $("canvas-idle").classList.toggle("hidden", !!state.lastResult);
    $("output-image").classList.toggle("hidden", !state.lastResult);
    const err = snap.error || { message: "Unknown error", hint: "" };
    toast("error", "Generation failed", err.message, err.hint);
  }
}

function showProgress(phase, message, step, total, pct) {
  const overlay = $("progress-overlay");
  overlay.classList.remove("hidden");
  const nice = {
    queued: "Waiting in queue…",
    loading_model: "Warming up the model…",
    generating: "Base pass",
    hires: "Hires-fix 2× pass",
    safety_scan: "NSFW safety scan",
    upscaling: "Real-ESRGAN ultra upscale",
    post_fx: "Applying retro post-effects",
    saving: "Saving to gallery",
  }[phase] || phase;
  $("progress-phase").textContent = message && phase !== "generating" && phase !== "hires" ? message : nice;
  $("progress-fill").style.width = `${Math.max(2, Math.min(100, pct ?? 5))}%`;
  $("progress-steps").textContent = (step != null && total) ? `step ${step} / ${total}` : "";
}

function hideProgress() {
  $("progress-overlay").classList.add("hidden");
}

function onGenerationDone(result) {
  if (!result) return;
  state.lastResult = result;

  const img = $("output-image");
  img.src = `${result.image_url}?t=${Date.now()}`;
  img.classList.remove("hidden");
  $("canvas-idle").classList.add("hidden");
  img.onerror = () => toast("error", "Could not display the image", result.filename);
  img.onclick = () => openLightbox(result.image_url);
  img.style.cursor = "zoom-in";

  $("info-seed").textContent = result.seed;
  $("info-style").textContent = `${result.style_name} (${result.style_key})`;
  $("info-model").textContent = result.model;
  $("info-size").textContent = `${result.width}×${result.height} (base ${result.base_w}×${result.base_h})`;
  $("info-params").textContent = `${result.steps} steps · CFG ${result.cfg} · ${result.sampler}`;
  $("info-lora").textContent = result.lora_used || "none (prompt-engineered)";
  $("info-time").textContent = `${result.elapsed}s · ${result.performance_mode} mode · ${result.hires_method}${result.ultra ? " + ultra" : ""}`;

  const dl = $("btn-download");
  dl.href = result.download_url;
  dl.setAttribute("download", result.filename);
  dl.classList.remove("hidden");
  $("btn-reuse-seed").classList.remove("hidden");
  $("btn-variation").classList.remove("hidden");

  const warnBox = $("info-warnings");
  if (result.warnings && result.warnings.length) {
    warnBox.innerHTML = "<strong>Notes:</strong><ul>" +
      result.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("") + "</ul>";
    warnBox.classList.remove("hidden");
  } else {
    warnBox.classList.add("hidden");
  }

  toast("success", "Image ready", `${result.filename} · ${result.elapsed}s · seed ${result.seed}`);
  loadGallery();
}

/* ---------------------------------------------------------- tabs, settings, health */

function switchTab(which) {
  const isCreate = which === "create";
  $("tab-create").classList.toggle("active", isCreate);
  $("tab-gallery").classList.toggle("active", !isCreate);
  $("tab-create").setAttribute("aria-selected", String(isCreate));
  $("tab-gallery").setAttribute("aria-selected", String(!isCreate));
  $("page-create").classList.toggle("hidden", !isCreate);
  $("page-gallery").classList.toggle("hidden", isCreate);
}

async function pushSettings(patch) {
  try {
    await api("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
  } catch (e) {
    toast("error", "Setting not applied", e.message, e.hint);
  }
}

function fmtMb(mb) { return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`; }

async function pollHealth() {
  try {
    const h = await api("/api/health");
    const gpu = $("badge-gpu"), ram = $("badge-ram"), model = $("badge-model"), busy = $("badge-busy");

    if (h.cuda_available) {
      const name = (h.gpu_name || "GPU").replace(/NVIDIA GeForce /, "");
      gpu.textContent = h.vram_allocated_mb != null
        ? `GPU: ${name} · ${fmtMb(h.vram_allocated_mb)}/${fmtMb(h.vram_total_mb)}`
        : `GPU: ${name}`;
      gpu.className = "badge ok";
    } else {
      gpu.textContent = "GPU: none";
      gpu.className = "badge bad";
    }

    ram.textContent = `RAM: ${fmtMb(h.ram_used_mb)}/${fmtMb(h.ram_total_mb)}`;
    ram.className = "badge " + (h.ram_percent > 90 ? "bad" : h.ram_percent > 75 ? "warn" : "ok");

    model.textContent = h.model_loaded
      ? `MODEL: loaded (${h.performance_mode})`
      : (h.checkpoint_present ? "MODEL: ready" : "MODEL: not downloaded");
    model.className = "badge " + (h.checkpoint_present ? "ok" : "bad");

    busy.textContent = h.busy ? "GENERATING…" : "IDLE";
    busy.className = "badge " + (h.busy ? "busy" : "");

    const res = $("resolution");
    if (!h.esrgan_x4 && res.value === "ultra") {
      res.value = "standard";
      updateResolutionHint();
    }
    res.options[1].disabled = !h.esrgan_x4;
    res.options[1].textContent = h.esrgan_x4
      ? "Ultra 4× Real-ESRGAN (slow, VRAM-hungry)"
      : "Ultra 4× — weights missing (run download_models.py)";

    const hm = $("hires-method");
    if (!h.esrgan_x2) hm.options[1].disabled = true;

    const sb = $("badge-safety");
    const sf = h.safety || {};
    if (sf.check_outputs && sf.checker_weights_present) {
      sb.textContent = sf.checker_loaded ? "\u{1F6E1} PROTECTED \u00B7 3/3" : "\u{1F6E1} PROTECTED";
      sb.className = "badge safety-on";
    } else if (sf.check_outputs && !sf.checker_weights_present) {
      sb.textContent = "\u{1F6E1} SCANNER WEIGHTS MISSING";
      sb.className = "badge safety-partial";
      sb.title = "Run download_models.py to fetch the NSFW output scanner (~608 MB). Prompt filter is still active.";
    } else {
      sb.textContent = "\u{1F6E1} FILTER 2/3";
      sb.className = "badge safety-partial";
      sb.title = "AI output scan disabled in Advanced settings. Prompt filter + forced negatives remain active.";
    }
  } catch {
    $("badge-gpu").textContent = "GPU: ?";
    $("badge-gpu").className = "badge bad";
  }
}

/* ---------------------------------------------------------- init */

function updatePromptCount() {
  $("prompt-count").textContent = String($("prompt").value.length);
}

async function init() {
  try {
    const data = await api("/api/styles");
    state.styles = data.styles;
    state.ratios = data.ratios;
    state.samplers = data.samplers;
    state.defaults = data.defaults;

    const firstLora = data.styles.find((s) => s.lora_available);
    state.selectedStyle = (firstLora || data.styles.find((s) => s.key === "synthwave") || data.styles[0])?.key || "no_style";

    renderStyles();
    renderRatios();
    renderSamplers();
    selectStyle(state.selectedStyle);

    $("safety-scan").checked = data.defaults.safety_output_filter !== false;
    $("perf-mode").value = data.defaults.performance_mode;
    $("hires-method").value = data.defaults.hires_method;
    $("resolution").value = data.defaults.ultra ? "ultra" : "standard";
    $("negative").value = data.defaults.negative_prompt || "";
  } catch (e) {
    toast("error", "Could not load styles", e.message, "Check that the server console shows no errors, then refresh.");
  }

  await loadGallery();
  pollHealth();
  setInterval(pollHealth, 5000);

  /* events */
  $("btn-generate").addEventListener("click", startGeneration);
  $("prompt").addEventListener("input", updatePromptCount);
  $("prompt").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); startGeneration(); }
  });
  $("btn-random-seed").addEventListener("click", () => {
    $("seed").value = -1;
    toast("info", "Seed randomized", "The next generation will pick a fresh seed (shown in the info panel).");
  });
  $("btn-reuse-seed").addEventListener("click", () => {
    if (state.lastResult) {
      $("seed").value = state.lastResult.seed;
      $("advanced").open = true;
      toast("info", "Seed reused", `Seed ${state.lastResult.seed} locked in.`);
    }
  });
  $("btn-variation").addEventListener("click", () => {
    if (state.lastResult) {
      $("seed").value = state.lastResult.seed + 1;
      $("advanced").open = true;
      toast("info", "Variation armed", `Seed set to ${state.lastResult.seed + 1}. Press GENERATE.`);
    }
  });
  $("tab-create").addEventListener("click", () => switchTab("create"));
  $("tab-gallery").addEventListener("click", () => { switchTab("gallery"); loadGallery(); });
  $("resolution").addEventListener("change", (e) => {
    pushSettings({ ultra: e.target.value === "ultra" });
    updateResolutionHint();
  });
  $("perf-mode").addEventListener("change", (e) => {
    pushSettings({ performance_mode: e.target.value });
    if (e.target.value === "fast") {
      toast("warn", "Fast mode selected", "The whole model lives in VRAM now. On a 4 GB RTX 2050 this can OOM at 16:9 — Safe or Balanced is smarter.");
    }
  });
  $("hires-method").addEventListener("change", (e) => pushSettings({ hires_method: e.target.value }));
  $("safety-scan").addEventListener("change", (e) => {
    pushSettings({ safety_output_filter: e.target.checked });
    toast(e.target.checked ? "success" : "warn",
          e.target.checked ? "Output scanner ON" : "Output scanner OFF",
          e.target.checked
            ? "Finished images will be AI-scanned and NSFW results destroyed unsaved."
            : "The prompt filter and forced negatives still protect the app, but finished images will no longer be AI-scanned.");
  });

  window.addEventListener("beforeunload", (e) => {
    if (state.busy) { e.preventDefault(); e.returnValue = ""; }
  });
}

document.addEventListener("DOMContentLoaded", init);
