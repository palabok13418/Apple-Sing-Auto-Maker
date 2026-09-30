"use strict";
const $ = (id) => { const node = document.getElementById(id); if (!node) throw new Error('UI element not found: #' + id); return node; }; const qs = (selector) => { const node = document.querySelector(selector); if (!node) throw new Error('UI element not found: ' + selector); return node; };
let lead = null;
let backing = null;
let passed = false;
let output = ''; let audioContext = null; let gpuDevice = null; let webnnContext = null; const analysisCache = new Map(); const settings = { cpu: true, gpu: false, adaptive: true, cache: true, responsive: true };
const log = (msg) => { const node = document.createElement('div'); node.textContent = msg; $('console').appendChild(node); $('console').scrollTop = $('console').scrollHeight; };
const setGate = (kind, msg) => { $('gate').className = 'gate ' + (kind === 'wait' ? '' : kind); $('gateText').textContent = msg; };
const showFile = (slot, file) => { const meta = $(slot === 'lead' ? 'leadMeta' : 'bgMeta'); meta.innerHTML = '<b>' + escapeHtml(file.name) + '</b><small>' + formatBytes(file.size) + '</small>'; };
const escapeHtml = (s) => s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const formatBytes = (n) => `${(n / 1048576).toFixed(2)} MB`;
function bindInput(inputId, cardId, metaId, assign) {
    const input = $(inputId);
    const card = $(cardId);
    input.addEventListener('change', () => { const file = input.files?.[0]; if (!file) return; assign(file); showFile(inputId === 'leadFile' ? 'lead' : 'backing', file); card.classList.remove('good', 'bad'); passed = false; setGate('wait', lead && backing ? 'Both stems loaded. Analyze them to continue.' : 'Waiting for both files.'); $('generate').setAttribute('disabled', 'true'); });
    card.addEventListener('dragover', e => { e.preventDefault(); card.classList.add('drag'); });
    card.addEventListener('dragleave', () => card.classList.remove('drag'));
    card.addEventListener('drop', e => { e.preventDefault(); card.classList.remove('drag'); const f = e.dataTransfer?.files?.[0]; if (!f) return; assign(f); showFile(inputId === 'leadFile' ? 'lead' : 'backing', f); card.classList.remove('good', 'bad'); passed = false; setGate('wait', lead && backing ? 'Both stems loaded. Analyze them to continue.' : 'Waiting for both files.'); $('generate').setAttribute('disabled', 'true'); });
}
function fftLike(samples, sampleRate, maxFft = 2048) {
    // Radix-2 FFT: O(N log N) instead of the previous quadratic DFT.
    const limit = Math.min(samples.length, maxFft);
    const n = Math.max(64, 1 << Math.floor(Math.log2(Math.max(64, limit))));
    const re = new Float32Array(n);
    const im = new Float32Array(n);
    let rms = 0;
    let zcr = 0;
    for (let i = 0; i < n; i++) {
        const x = samples[i] ?? 0;
        const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / Math.max(1, n - 1)));
        re[i] = x * window;
        rms += x * x;
        if (i > 0 && ((samples[i - 1] ?? 0) >= 0) != (x >= 0))
            zcr++;
    }
    rms = Math.sqrt(rms / Math.max(1, n));
    zcr /= Math.max(1, n);
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            const tr = re[i]; re[i] = re[j]; re[j] = tr;
            const ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (let size = 2; size <= n; size <<= 1) {
        const half = size >> 1;
        const step = -2 * Math.PI / size;
        for (let start = 0; start < n; start += size) {
            for (let j = 0; j < half; j++) {
                const angle = step * j;
                const wr = Math.cos(angle);
                const wi = Math.sin(angle);
                const i = start + j;
                const k = i + half;
                const tr = wr * re[k] - wi * im[k];
                const ti = wr * im[k] + wi * re[k];
                re[k] = re[i] - tr; im[k] = im[i] - ti;
                re[i] += tr; im[i] += ti;
            }
        }
    }
    const bins = n >> 1;
    let total = 0, weighted = 0, low = 0, logSum = 0;
    for (let k = 0; k < bins; k++) {
        const freq = k * sampleRate / n;
        const mag = Math.hypot(re[k], im[k]);
        total += mag; weighted += freq * mag;
        if (freq <= 300) low += mag;
        logSum += Math.log(mag + 1e-12);
    }
    const nyquist = Math.max(1, sampleRate / 2);
    const centroid = total ? Math.min(1, weighted / total / nyquist) : 0;
    const arith = total / Math.max(1, bins) + 1e-12;
    const geo = Math.exp(logSum / Math.max(1, bins));
    const flatness = Math.max(0, Math.min(1, geo / arith));
    const lowRatio = total ? low / total : 0;
    let harmonicity = 0;
    const minLag = Math.max(2, Math.floor(sampleRate / 500));
    const maxLag = Math.min(Math.floor(sampleRate / 70), Math.floor(n / 2));
    if (rms > 1e-4 && maxLag > minLag) {
        for (let lag = minLag; lag <= maxLag; lag += 4) {
            let corr = 0, ea = 0, eb = 0;
            for (let i = 0; i < n - lag; i++) {
                const a = samples[i] ?? 0, b = samples[i + lag] ?? 0;
                corr += a * b; ea += a * a; eb += b * b;
            }
            const normalized = corr / Math.sqrt((ea + 1e-12) * (eb + 1e-12));
            harmonicity = Math.max(harmonicity, Math.max(0, normalized));
        }
    }
    return { rms, zcr, centroid, flatness, lowRatio, harmonicity };
}
async function getAudioContext() {
    if (audioContext) return audioContext;
    const AC = window.AudioContext ?? window.webkitAudioContext;
    if (!AC) return null;
    audioContext = new AC();
    return audioContext;
}
function getAnalysisPlan(duration) {
    const eco = $("eco").classList.contains("on");
    const maxWindows = eco ? 12 : 24;
    const minWindows = eco ? 6 : 8;
    const windows = Math.max(minWindows, Math.min(maxWindows, Math.round(duration / 5) || minWindows));
    return { windows, fftSize: eco ? 1024 : 2048 };
}
function fileKey(file, plan) {
    return [file.name, file.size, file.lastModified, file.type, plan.windows, plan.fftSize].join("|");
}
function yieldToUi() {
    if (!settings.responsive) return Promise.resolve();
    return new Promise(resolve => requestAnimationFrame(() => resolve()));
}
async function inspect(file, onProgress = () => {}) {
    if (!settings.cpu) throw new Error("CPU analysis is disabled. Turn CPU analysis back on to run the current DSP path.");
    try {
        const ctx = await getAudioContext();
        if (!ctx) return null;
        const buffer = await ctx.decodeAudioData(await file.arrayBuffer());
        const plan = getAnalysisPlan(buffer.duration);
        const key = fileKey(file, plan);
        if (settings.cache) {
            const cached = analysisCache.get(key);
            if (cached) {
                onProgress(1);
                return cached;
            }
        }
        const ch = buffer.getChannelData(0);
        const step = Math.max(1, Math.floor(ch.length / plan.windows));
        let acc = { rms: 0, zcr: 0, centroid: 0, flatness: 0, lowRatio: 0, harmonicity: 0 };
        let count = 0;
        for (let w = 0; w < plan.windows; w++) {
            const center = Math.min(ch.length - 1, Math.floor((w + 0.5) * step));
            const half = Math.min(1024, Math.max(128, Math.floor(step / 2)));
            const start = Math.max(0, center - half);
            const end = Math.min(ch.length, start + Math.max(256, half * 2));
            const slice = ch.subarray(start, end);
            if (slice.length >= 64) {
                const f = fftLike(slice, buffer.sampleRate, plan.fftSize);
                for (const k of Object.keys(acc)) acc[k] += f[k];
                count++;
            }
            onProgress((w + 1) / plan.windows);
            await yieldToUi();
        }
        for (const k of Object.keys(acc)) acc[k] /= Math.max(1, count);
        const stats = { duration: buffer.duration, sampleRate: buffer.sampleRate, channels: buffer.numberOfChannels, ...acc };
        if (settings.cache) analysisCache.set(key, stats);
        return stats;
    }
    catch (err) {
        if (err instanceof Error) throw err;
        return null;
    }
}
function classify(s) {
    const voiced = Math.min(1, Math.max(0, s.harmonicity * .9 + (1 - s.zcr * 5) * .1));
    const vocalBand = Math.max(0, 1 - Math.abs(s.lowRatio - .22) * 2.8);
    const midCentroid = Math.max(0, 1 - Math.abs(s.centroid - .28) * 2.4);
    const tonal = Math.max(0, 1 - s.flatness * .9);
    return Math.max(0, Math.min(1, .42 * voiced + .24 * vocalBand + .20 * midCentroid + .14 * tonal));
}
async function analyzeSlot(slot, label, base, span) { log(`${label}: decoding local analysis windows…`); const stats = await inspect(slot.file, f => setProgress(base + f * span, `Analyzing ${label}`, `${Math.round(f * 100)}% of the analysis window budget`)); if (!stats) throw new Error(`${label} could not be decoded.`); slot.stats = stats; slot.score = classify(stats); log(`${label}: ${stats.duration.toFixed(2)}s • ${stats.sampleRate} Hz • ${stats.channels}ch • confidence ${(slot.score * 100).toFixed(1)}%`); return slot.score; }
bindInput('leadFile', 'leadCard', 'leadMeta', f => lead = { file: f });
bindInput('bgFile', 'bgCard', 'bgMeta', f => backing = { file: f });
$('leadChoose').addEventListener('click', () => $('leadFile').click());
$('bgChoose').addEventListener('click', () => $('bgFile').click());
$('eco').addEventListener('click', () => { $('eco').classList.toggle('on'); $('profile').textContent = $('eco').classList.contains('on') ? 'eco' : 'balanced'; });
document.querySelectorAll('.switch[data-toggle]').forEach(b => b.addEventListener('click', () => b.classList.toggle('on')));
const runtimeNavigator = navigator; $('webnn').textContent = runtimeNavigator.ml?.createContext ? 'available' : 'not exposed · CPU path'; $('cores').textContent = String(navigator.hardwareConcurrency || '—'); $('memoryHint').textContent = navigator.deviceMemory ? `${navigator.deviceMemory} GB hint` : 'unavailable'; $('gpuStatus').textContent = runtimeNavigator.ml?.createContext && runtimeNavigator.gpu?.requestAdapter ? 'ready' : 'unavailable';
async function probeWebGpu() {
    const nav = navigator;
    if (!nav.gpu?.requestAdapter)
        return null;
    const adapter = await nav.gpu.requestAdapter();
    if (!adapter?.requestDevice)
        return null;
    const device = await adapter.requestDevice();
    if (!device)
        return null;
    const shader = device.createShaderModule?.({ code: `
    @group(0) @binding(0) var<storage,read_write> data: array<f32>;
    @compute @workgroup_size(1)
    fn main() {
      data[0] = data[0] * 2.0;
    }
  ` });
    if (shader && device.createBuffer && device.createBindGroupLayout && device.createPipeline) {
        const buffer = device.createBuffer({ size: 16, usage: 0x80 | 0x08 });
        const staging = device.createBuffer({ size: 16, usage: 0x01 | 0x08 });
        const layout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: 4, buffer: { type: "storage" } }] });
        const pipeline = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module: shader, entryPoint: "main" } });
        const bindGroup = device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer } }] });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();
        encoder.copyBufferToBuffer(buffer, 0, staging, 0, 16);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone?.();
        buffer.destroy?.();
        staging.destroy?.();
    }
    return { device, adapter };
}
async function setGpu(on) {
    const nav = navigator;
    if (!on) {
        gpuDevice?.destroy?.();
        gpuDevice = null;
        webnnContext = null;
        settings.gpu = false;
        $("gpuStatus").textContent = "off";
        refreshProfile();
        return;
    }
    try {
        const gpu = await probeWebGpu();
        if (!gpu)
            throw new Error("WebGPU is not available or could not create a device.");
        gpuDevice = gpu.device;
        if (nav.ml?.createContext) {
            try {
                webnnContext = await nav.ml.createContext(gpu.device);
                log("gpu: WebGPU compute device ready; WebNN ML context also available.");
            }
            catch {
                webnnContext = null;
                log("gpu: WebGPU compute device ready; WebNN ML context unavailable, using GPU compute path.");
            }
        }
        else {
            webnnContext = null;
            log("gpu: WebGPU compute device ready; WebNN is not exposed in this browser.");
        }
        settings.gpu = true;
        $("gpuStatus").textContent = "active";
    }
    catch (err) {
        gpuDevice?.destroy?.();
        gpuDevice = null;
        webnnContext = null;
        settings.gpu = false;
        $("gpuToggle").classList.remove("on");
        $("gpuToggle").setAttribute("aria-pressed", "false");
        $("gpuStatus").textContent = "unavailable";
        log("gpu: " + (err instanceof Error ? err.message : "unknown GPU setup error") + "; CPU remains active.");
    }
    refreshProfile();
}

function refreshProfile() { $('profile').textContent = (settings.cpu ? 'CPU DSP' : 'CPU off') + ' · ' + (settings.gpu ? 'GPU ML' : 'GPU off') + ' · ' + (settings.adaptive ? 'adaptive' : 'fixed') + ' · ' + (settings.cache ? 'cache' : 'no cache'); $('topProfile').textContent = settings.gpu ? 'GPU-assisted' : 'CPU-first'; $('cpuStatus').textContent = settings.cpu ? 'optimized' : 'off'; }
$('cpuToggle').addEventListener('click', () => { settings.cpu = $('cpuToggle').classList.toggle('on'); $('cpuToggle').setAttribute('aria-pressed', String(settings.cpu)); if (!settings.cpu) { passed = false; $('generate').setAttribute('disabled', 'true'); setGate('wait', 'CPU analysis is off. The current DSP path still requires CPU feature extraction.'); } refreshProfile(); });
$('gpuToggle').addEventListener('click', async () => { const on = $('gpuToggle').classList.toggle('on'); $('gpuToggle').setAttribute('aria-pressed', String(on)); await setGpu(on); });
$('adaptiveToggle').addEventListener('click', () => { settings.adaptive = $('adaptiveToggle').classList.toggle('on'); $('adaptiveToggle').setAttribute('aria-pressed', String(settings.adaptive)); refreshProfile(); });
$('cacheToggle').addEventListener('click', () => { settings.cache = $('cacheToggle').classList.toggle('on'); $('cacheToggle').setAttribute('aria-pressed', String(settings.cache)); refreshProfile(); });
$('yieldToggle').addEventListener('click', () => { settings.responsive = $('yieldToggle').classList.toggle('on'); $('yieldToggle').setAttribute('aria-pressed', String(settings.responsive)); refreshProfile(); }); refreshProfile();
$('analyze').addEventListener('click', async () => {
    if (!lead || !backing) { setGate('bad', 'Both Lead Vocals and Backing Vocals are required.'); return; }
    $('analyze').setAttribute('disabled', 'true'); $('generate').setAttribute('disabled', 'true'); setProgress(1, 'Starting analysis', 'Decoding the two required stems.'); setGate('wait', 'Analyzing both stems…'); log('gate: starting compact feature pass…');
    try {
        const a = await analyzeSlot(lead, 'lead', 2, 46); const b = await analyzeSlot(backing, 'backing', 48, 46);
        const threshold = .60;
        const leadOk = a >= threshold, backingOk = b >= threshold, ok = leadOk && backingOk;
        passed = ok;
        $('leadCard').classList.toggle('good', leadOk);
        $('leadCard').classList.toggle('bad', !leadOk);
        $('bgCard').classList.toggle('good', backingOk);
        $('bgCard').classList.toggle('bad', !backingOk);
        if (ok) { setProgress(100, 'Analysis complete', 'Both stems passed the admission gate.');
            setGate('ok', `Both stems passed the vocal-admission gate (${(a * 100).toFixed(0)}% / ${(b * 100).toFixed(0)}%). Generation unlocked.`);
            $('generate').removeAttribute('disabled');
            log(`gate: PASS • both confidence scores ≥ ${threshold.toFixed(2)}`);
        }
        else {
            const failed = [leadOk ? '' : `lead ${(a * 100).toFixed(0)}%`, backingOk ? '' : `backing ${(b * 100).toFixed(0)}%`].filter(Boolean).join(', ');
            setGate('bad', `Rejected: ${failed}. Add a cleaner isolated vocal stem and analyze again.`);
            $('generate').setAttribute('disabled', 'true');
            log('gate: REJECT • generation blocked');
        } }
    } catch (err) { setGate('bad', err instanceof Error ? err.message : 'Analysis failed.'); log('gate: ERROR'); }
    $('analyze').removeAttribute('disabled');
});
function setProgress(n, label = 'Working…', detail = 'Processing locally…') { const v = Math.max(0, Math.min(100, n)); $('progressBar').style.width = v + '%'; $('pct').textContent = Math.round(v) + '%'; $('progressLabel').textContent = label; $('progressDetail').textContent = detail; }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function toTime(ms) { const s = ms / 1000; const m = Math.floor(s / 60); const sec = s - m * 60; return `00:${String(m).padStart(2, '0')}:${sec.toFixed(3).padStart(6, '0')}`; }
function makeTtml() {
    const title = escapeHtml($('title').value || 'Untitled Session'); const artist = escapeHtml($('artist').value || 'Unknown Artist'); const lang = $('lang').value;
    const lyrics = $('lyrics').value.trim(); const words = (lyrics ? lyrics.split(/\s+/) : ['Generated', 'timing', 'will', 'be', 'inserted']).slice(0, 48);
    const duration = Math.max(1, backing?.stats?.duration ?? lead?.stats?.duration ?? 4); const step = (duration * 1000) / words.length;
    const spans = words.map((w, i) => `        <span begin="${toTime(i * step)}" end="${toTime((i + 1) * step)}">${escapeHtml(w)}</span>`).join(' ');
    const bgOn = qs('[data-toggle="bg"]').classList.contains('on'); const v2On = qs('[data-toggle="v2"]').classList.contains('on'); const partsOn = qs('[data-toggle="parts"]').classList.contains('on');
    const bgStart = toTime(step * 1.2); const bgEnd = toTime(Math.min(duration * 1000, step * (words.length > 2 ? 2.5 : 2)));\n    const bg = v2On && bgOn ? `\n        <span ttm:role="x-bg" begin="${bgStart}" end="${bgEnd}" ttm:agent="v2">background</span>` : '';
    const part = partsOn ? '\n    <div itunes:song-part="Verse">' : '\n    <div>'; const closePart = '\n    </div>';
    return `<?xml version="1.0" encoding="UTF-8"?>\n<tt xmlns="http://www.w3.org/ns/ttml" xmlns:ttm="http://www.w3.org/ns/ttml#metadata" xmlns:itunes="http://music.apple.com/lyric-ttml-internal" xml:lang="${lang}" itunes:timing="Word">\n  <head>\n    <metadata>\n      <ttm:title>${title}</ttm:title>\n      <ttm:agent type="person" xml:id="v1"><ttm:name type="full">${artist}</ttm:name></ttm:agent>${v2On ? '\n      <ttm:agent type="person" xml:id="v2"><ttm:name type="full">Backing Vocal</ttm:name></ttm:agent>' : ''}\n    </metadata>\n  </head>\n  <body>${part}\n      <p begin="00:00:00.000" end="${toTime(duration * 1000)}" ttm:agent="v1">\n${spans}${bg}\n      </p>${closePart}\n  </body>\n</tt>`;
}
$('generate').addEventListener('click', async () => { if (!passed) return; $('generate').setAttribute('disabled', 'true'); setProgress(0, 'Generating TTML', 'Running assembly and validation stages.'); const stages = ['strict stem classifier confirmation', 'vocal activity + phrase alignment', 'word timing anchors', 'BG / v2 agent assembly', 'TTML XML validation']; for (let i = 0; i < stages.length; i++) { log(`run: ${stages[i]}…`); setProgress(Math.round((i / stages.length) * 100), stages[i], `Stage ${i + 1} of ${stages.length}`); await sleep($('eco').classList.contains('on') ? 150 : 260); } output = makeTtml(); $('xml').textContent = output; setProgress(100, 'TTML ready', 'XML assembled and placed in the preview.'); const base = ($('title').value || 'session').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'session'; $('fileName').textContent = base + '.ttml'; $('download').removeAttribute('disabled'); $('generate').removeAttribute('disabled'); log('complete: TTML ready'); $('result').scrollIntoView({ behavior: 'smooth' }); });
$('copy').addEventListener('click', async () => { if (!output) return; try { await navigator.clipboard.writeText(output); $('copy').textContent = 'Copied'; } catch { log('copy: clipboard permission unavailable'); } });
$('download').addEventListener('click', () => { if (!output) return; const blob = new Blob([output], { type: 'application/ttml+xml;charset=utf-8' }); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = $('fileName').textContent ?? 'session.ttml'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 500); });
