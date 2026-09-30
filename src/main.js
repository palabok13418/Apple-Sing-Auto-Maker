const $ = id => {
  const node = document.getElementById(id);
  if (!node) throw new Error('UI element not found: #' + id);
  return node;
};
const qs = selector => {
  const node = document.querySelector(selector);
  if (!node) throw new Error('UI element not found: ' + selector);
  return node;
};

let lead = null;
let backing = null;
let passed = false;
let output = '';
let audioContext = null;
let gpuAdapter = null;
let gpuDevice = null;
let webnnContext = null;
let alignmentPipelinePromise = null;
const analysisCache = new Map();
const settings = { cpu: true, gpu: false, adaptive: true, cache: true, responsive: true };

const log = msg => {
  const node = document.createElement('div');
  node.textContent = msg;
  $('console').appendChild(node);
  $('console').scrollTop = $('console').scrollHeight;
};
const setGate = (kind, msg) => {
  $('gate').className = 'gate ' + (kind === 'wait' ? '' : kind);
  $('gateText').textContent = msg;
};
const escapeHtml = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const formatBytes = n => (n / 1048576).toFixed(2) + ' MB';
const clamp01 = n => Math.max(0, Math.min(1, Number.isFinite(n) ? n : 0));

function showFile(slot, file) {
  const meta = $(slot === 'lead' ? 'leadMeta' : 'bgMeta');
  meta.innerHTML = '<b>' + escapeHtml(file.name) + '</b><small>' + formatBytes(file.size) + '</small>';
}
function bindInput(inputId, cardId, assign) {
  const input = $(inputId);
  const card = $(cardId);
  const reset = () => {
    passed = false;
    $('generate').setAttribute('disabled', 'true');
    $('v2Status').textContent = 'waiting…';
    $('bgStatus').textContent = 'waiting…';
    setGate('wait', lead && backing ? 'Both stems loaded. Analyze them to continue.' : 'Waiting for both files.');
  };
  input.addEventListener('change', () => {
    const file = input.files && input.files[0];
    if (!file) return;
    assign(file);
    showFile(inputId === 'leadFile' ? 'lead' : 'backing', file);
    card.classList.remove('good', 'bad');
    reset();
  });
  card.addEventListener('dragover', e => { e.preventDefault(); card.classList.add('drag'); });
  card.addEventListener('dragleave', () => card.classList.remove('drag'));
  card.addEventListener('drop', e => {
    e.preventDefault();
    card.classList.remove('drag');
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    assign(file);
    showFile(inputId === 'leadFile' ? 'lead' : 'backing', file);
    card.classList.remove('good', 'bad');
    reset();
  });
}

function fftFeatures(samples, sampleRate, maxFft) {
  const limit = Math.min(samples.length, maxFft || 2048);
  const n = Math.max(64, 1 << Math.floor(Math.log2(Math.max(64, limit))));
  const re = new Float32Array(n), im = new Float32Array(n);
  let rms = 0, zcr = 0;
  for (let i = 0; i < n; i++) {
    const x = samples[i] || 0;
    const win = 0.5 * (1 - Math.cos(2 * Math.PI * i / Math.max(1, n - 1)));
    re[i] = x * win;
    rms += x * x;
    if (i && (((samples[i - 1] || 0) >= 0) !== (x >= 0))) zcr++;
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
        const wr = Math.cos(angle), wi = Math.sin(angle);
        const i = start + j, k = i + half;
        const tr = wr * re[k] - wi * im[k];
        const ti = wr * im[k] + wi * re[k];
        re[k] = re[i] - tr; im[k] = im[i] - ti;
        re[i] += tr; im[i] += ti;
      }
    }
  }

  let total = 0, weighted = 0, low = 0, logSum = 0;
  const bins = n >> 1;
  for (let k = 0; k < bins; k++) {
    const f = k * sampleRate / n;
    const mag = Math.hypot(re[k], im[k]);
    total += mag; weighted += f * mag;
    if (f <= 300) low += mag;
    logSum += Math.log(mag + 1e-12);
  }
  const centroid = total ? Math.min(1, weighted / total / Math.max(1, sampleRate / 2)) : 0;
  const arith = total / Math.max(1, bins) + 1e-12;
  const geo = Math.exp(logSum / Math.max(1, bins));
  const flatness = clamp01(geo / arith);
  const lowRatio = total ? low / total : 0;

  const candidates = [];
  let harmonicity = 0;
  const minLag = Math.max(2, Math.floor(sampleRate / 500));
  const maxLag = Math.min(Math.floor(sampleRate / 70), Math.floor(n / 2));
  for (let lag = minLag; lag <= maxLag; lag += 2) {
    let corr = 0, ea = 0, eb = 0;
    for (let i = 0; i < n - lag; i++) {
      const a = samples[i] || 0, b = samples[i + lag] || 0;
      corr += a * b; ea += a * a; eb += b * b;
    }
    const normalized = corr / Math.sqrt((ea + 1e-12) * (eb + 1e-12));
    harmonicity = Math.max(harmonicity, Math.max(0, normalized));
    if (normalized > 0.12) candidates.push({ freq: sampleRate / lag, score: normalized });
  }
  candidates.sort((a, b) => b.score - a.score);
  const primary = candidates[0];
  let secondary = null;
  if (primary) {
    for (const candidate of candidates.slice(1)) {
      const ratio = candidate.freq / primary.freq;
      const semitones = Math.abs(12 * Math.log2(Math.max(1e-6, ratio)));
      const harmonicRatio = Math.abs(ratio - 2) < 0.08 || Math.abs(ratio - 3) < 0.10 || Math.abs(ratio - 4) < 0.12 ||
        Math.abs(ratio - 0.5) < 0.03 || Math.abs(ratio - 1 / 3) < 0.03 || Math.abs(ratio - 0.25) < 0.025;
      if (semitones >= 2.5 && !harmonicRatio && candidate.score >= primary.score * 0.68) {
        secondary = candidate;
        break;
      }
    }
  }
  const secondaryVoice = primary && secondary
    ? clamp01(((secondary.score / Math.max(0.01, primary.score)) - 0.58) / 0.42) * clamp01((primary.score - 0.18) / 0.52)
    : 0;
  const vocalActivity = clamp01(harmonicity * 0.74 + Math.min(1, rms * 24) * 0.16 + (1 - flatness) * 0.10);
  return { rms, zcr, centroid, flatness, lowRatio, harmonicity, secondaryVoice, vocalActivity };
}

function mergeIntervals(intervals, gap) {
  if (!intervals.length) return [];
  const sorted = intervals.slice().sort((a, b) => a.start - b.start);
  const out = [{ start: Math.max(0, sorted[0].start), end: Math.max(0, sorted[0].end) }];
  for (const item of sorted.slice(1)) {
    const last = out[out.length - 1];
    if (item.start <= last.end + gap) last.end = Math.max(last.end, item.end);
    else out.push({ start: item.start, end: item.end });
  }
  return out;
}
function normalizeIntervals(intervals, duration, gap) {
  return mergeIntervals(intervals.map(x => ({
    start: Math.max(0, Math.min(duration, Number.isFinite(x.start) ? x.start : 0)),
    end: Math.max(0, Math.min(duration, Number.isFinite(x.end) ? x.end : 0))
  })).filter(x => x.end > x.start), gap || 0.10).filter(x => x.end - x.start >= 0.04);
}
function fastPitchConfidence(samples, sampleRate) {
  if (samples.length < 64) return 0;
  const factor = Math.max(1, Math.floor(sampleRate / 4000));
  const ds = [];
  for (let i = 0; i < samples.length; i += factor) ds.push(samples[i] || 0);
  const sr = sampleRate / factor;
  const minLag = Math.max(2, Math.floor(sr / 500));
  const maxLag = Math.min(Math.floor(sr / 70), Math.floor(ds.length / 2));
  let best = 0;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let corr = 0, ea = 0, eb = 0;
    for (let i = 0; i < ds.length - lag; i++) {
      const a = ds[i] || 0, b = ds[i + lag] || 0;
      corr += a * b; ea += a * a; eb += b * b;
    }
    if (ea > 1e-8 && eb > 1e-8) best = Math.max(best, corr / Math.sqrt(ea * eb));
  }
  return clamp01(best);
}
function extractVocalEvidence(samples, sampleRate) {
  const frame = Math.max(1024, Math.round(sampleRate * 0.04));
  const frames = Math.max(1, Math.ceil(samples.length / frame));
  const scores = new Float32Array(frames);
  const rms = new Float32Array(frames);
  const zcr = new Float32Array(frames);
  let maxRms = 0;
  for (let f = 0; f < frames; f++) {
    const start = f * frame, end = Math.min(samples.length, start + frame);
    let energy = 0, crosses = 0;
    for (let i = start; i < end; i++) {
      const x = samples[i] || 0;
      energy += x * x;
      if (i > start && (((samples[i - 1] || 0) >= 0) !== (x >= 0))) crosses++;
    }
    const count = Math.max(1, end - start);
    const r = Math.sqrt(energy / count);
    rms[f] = r;
    zcr[f] = crosses / count;
    maxRms = Math.max(maxRms, r);
  }
  const sorted = Array.from(rms).sort((a, b) => a - b);
  const q20 = sorted[Math.floor(sorted.length * 0.20)] || 0;
  const threshold = Math.max(0.0025, q20 * 1.6, maxRms * 0.08);
  for (let f = 0; f < frames; f++) {
    const start = f * frame, end = Math.min(samples.length, start + frame);
    const normalized = rms[f] / Math.max(maxRms, 1e-6);
    const energyScore = clamp01((rms[f] - threshold) / Math.max(0.001, maxRms - threshold));
    const pitch = fastPitchConfidence(samples.subarray(start, end), sampleRate);
    const zcrScore = clamp01(1 - zcr[f] / 0.45);
    scores[f] = clamp01(energyScore * 0.38 + pitch * 0.42 + zcrScore * 0.15 + normalized * 0.05);
  }
  const active = Array.from(scores, (score, f) => score >= 0.46 && rms[f] >= threshold && zcr[f] < 0.48);
  for (let f = 1; f < frames - 1; f++) {
    if (!active[f] && active[f - 1] && active[f + 1] && scores[f] >= 0.30) active[f] = true;
    if (active[f] && !active[f - 1] && !active[f + 1] && scores[f] < 0.62) active[f] = false;
  }
  const raw = [];
  let begin = -1;
  for (let f = 0; f < frames; f++) {
    if (active[f] && begin < 0) begin = f;
    const close = (!active[f] && begin >= 0) || f === frames - 1;
    if (close) {
      const ef = active[f] ? f + 1 : f;
      const a = begin * frame / sampleRate;
      const b = Math.min(samples.length / sampleRate, ef * frame / sampleRate);
      if (b - a >= 0.08) raw.push({ start: a, end: b });
      begin = -1;
    }
  }
  const intervals = normalizeIntervals(raw, samples.length / sampleRate, 0.12);
  const peaks = [];
  for (let f = 1; f < frames - 1; f++) {
    if (scores[f] >= scores[f - 1] && scores[f] >= scores[f + 1] && scores[f] >= 0.48) {
      const prominence = scores[f] - Math.min(scores[f - 1], scores[f + 1]);
      const strength = clamp01(scores[f] * 0.72 + prominence * 0.95);
      if (strength >= 0.50) peaks.push({ time: (f + 0.5) * frame / sampleRate, strength });
    }
  }
  const reduced = [];
  for (const peak of peaks) {
    const last = reduced[reduced.length - 1];
    if (!last || peak.time - last.time >= 0.08) reduced.push(peak);
    else if (peak.strength > last.strength) reduced[reduced.length - 1] = peak;
  }
  const covered = intervals.reduce((sum, x) => sum + x.end - x.start, 0);
  return {
    intervals,
    peaks: reduced,
    coverage: clamp01(covered / Math.max(1, samples.length / sampleRate))
  };
}

async function getAudioContext() {
  if (audioContext) return audioContext;
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return null;
  audioContext = new AC();
  return audioContext;
}
function getAnalysisPlan(duration) {
  const eco = $('eco').classList.contains('on');
  if (!settings.adaptive) return { windows: 16, fftSize: eco ? 1024 : 2048 };
  const maxWindows = eco ? 12 : 24, minWindows = eco ? 6 : 8;
  return { windows: Math.max(minWindows, Math.min(maxWindows, Math.round(duration / 5) || minWindows)), fftSize: eco ? 1024 : 2048 };
}
function fileKey(file, plan) {
  return [file.name, file.size, file.lastModified, file.type, plan.windows, plan.fftSize].join('|');
}
function yieldToUi() {
  if (!settings.responsive) return Promise.resolve();
  return new Promise(resolve => requestAnimationFrame(resolve));
}
async function inspect(file, onProgress) {
  if (!settings.cpu) throw new Error('CPU analysis is disabled. Turn CPU analysis back on to analyze the audio.');
  const ctx = await getAudioContext();
  if (!ctx) throw new Error('Web Audio is unavailable.');
  const buffer = await ctx.decodeAudioData(await file.arrayBuffer());
  const plan = getAnalysisPlan(buffer.duration);
  const key = fileKey(file, plan);
  if (settings.cache && analysisCache.has(key)) {
    onProgress(1);
    return analysisCache.get(key);
  }
  const channel = buffer.getChannelData(0);
  const step = Math.max(1, Math.floor(channel.length / plan.windows));
  let rms = 0, zcr = 0, centroid = 0, flatness = 0, lowRatio = 0, harmonicity = 0, secondaryVoice = 0, secondaryVoicePeak = 0, vocalActivity = 0;
  const secondaryIntervals = [];
  for (let w = 0; w < plan.windows; w++) {
    const center = Math.min(channel.length - 1, Math.floor((w + 0.5) * step));
    const half = Math.min(1024, Math.max(128, Math.floor(step / 2)));
    const slice = channel.subarray(Math.max(0, center - half), Math.min(channel.length, center + half));
    if (slice.length >= 64) {
      const f = fftFeatures(slice, buffer.sampleRate, plan.fftSize);
      rms += f.rms; zcr += f.zcr; centroid += f.centroid; flatness += f.flatness; lowRatio += f.lowRatio;
      harmonicity += f.harmonicity; secondaryVoice += f.secondaryVoice; secondaryVoicePeak = Math.max(secondaryVoicePeak, f.secondaryVoice); vocalActivity += f.vocalActivity;
      if (f.secondaryVoice >= 0.62) secondaryIntervals.push({ start: Math.max(0, center - half) / buffer.sampleRate, end: Math.min(channel.length, center + half) / buffer.sampleRate });
    }
    onProgress((w + 1) / plan.windows);
    await yieldToUi();
  }
  const n = Math.max(1, plan.windows);
  const evidence = extractVocalEvidence(channel, buffer.sampleRate);
  const stats = {
    duration: buffer.duration,
    sampleRate: buffer.sampleRate,
    channels: buffer.numberOfChannels,
    rms: rms / n, zcr: zcr / n, centroid: centroid / n, flatness: flatness / n, lowRatio: lowRatio / n,
    harmonicity: harmonicity / n, secondaryVoice: secondaryVoice / n, secondaryVoicePeak,
    vocalActivity: vocalActivity / n, vocalCoverage: evidence.coverage,
    secondaryIntervals: normalizeIntervals(secondaryIntervals, buffer.duration, 0.10),
    vocalIntervals: evidence.intervals,
    syllablePeaks: evidence.peaks,
    backingPeaks: evidence.peaks
  };

  if (settings.cache) analysisCache.set(key, stats);
  return stats;
}
function classify(s) {
  const periodicity = clamp01(s.harmonicity * 0.82 + s.vocalActivity * 0.18);
  const voiceTexture = clamp01((1 - s.flatness) * 0.58 + Math.max(0, 1 - s.zcr * 4) * 0.22 + (1 - s.lowRatio) * 0.20);
  const voiceBand = clamp01(1 - Math.abs(s.lowRatio - 0.20) * 3.6);
  const formantBand = clamp01(1 - Math.abs(s.centroid - 0.30) * 2.8);
  const coverage = clamp01(s.vocalCoverage / 0.42);
  const broadbandRisk = clamp01(s.flatness * 0.90 + s.zcr * 0.46 + Math.max(0, s.centroid - 0.56) * 0.60);
  const ambiguity = clamp01(Math.abs(periodicity - voiceTexture) * 1.25 + broadbandRisk * 0.55);
  const raw = 0.31 * periodicity + 0.19 * voiceTexture + 0.14 * voiceBand + 0.12 * formantBand + 0.15 * coverage + 0.09 * s.vocalActivity;
  return clamp01(raw - 0.18 * broadbandRisk - 0.10 * ambiguity);
}
async function analyzeSlot(slot, label, base, span) {
  log(label + ': decoding and scanning actual audio…');
  const stats = await inspect(slot.file, p => setProgress(base + p * span, 'Analyzing ' + label, Math.round(p * 100) + '% complete'));
  slot.stats = stats;
  slot.score = classify(stats);
  log(label + ': ' + stats.duration.toFixed(2) + 's • vocal coverage ' + (stats.vocalCoverage * 100).toFixed(1) + '% • confidence ' + (slot.score * 100).toFixed(1) + '%');
  log(label + ': detected ' + stats.syllablePeaks.length + ' vocal timing peaks');
  return slot.score;
}

function normalizedToken(text) {
  return String(text).normalize('NFKD').toLowerCase().replace(/\p{M}/gu, '').replace(/[^\p{L}\p{N}]+/gu, '');
}
function similarity(a, b) {
  const aa = normalizedToken(a), bb = normalizedToken(b);
  if (!aa || !bb) return 0;
  if (aa === bb) return 1;
  const prev = new Array(bb.length + 1);
  const next = new Array(bb.length + 1);
  for (let j = 0; j <= bb.length; j++) prev[j] = j;
  for (let i = 1; i <= aa.length; i++) {
    next[0] = i;
    for (let j = 1; j <= bb.length; j++) {
      const cost = aa[i - 1] === bb[j - 1] ? 0 : 1;
      next[j] = Math.min(prev[j] + 1, next[j - 1] + 1, prev[j - 1] + cost);
    }
    for (let j = 0; j <= bb.length; j++) prev[j] = next[j];
  }
  return 1 - prev[bb.length] / Math.max(aa.length, bb.length);
}
function transcriptWords(chunks) {
  const out = [];
  for (const chunk of Array.isArray(chunks) ? chunks : []) {
    const text = String((chunk && chunk.text) || '').trim();
    const ts = chunk && (chunk.timestamp || chunk.timestamps);
    if (!text || !Array.isArray(ts) || !Number.isFinite(ts[0]) || !Number.isFinite(ts[1])) continue;
    const start = Math.max(0, Number(ts[0])), end = Math.max(start, Number(ts[1]));
    const pieces = text.match(/\S+/gu) || [];
    if (pieces.length === 1) {
      if (end > start) out.push({ text: pieces[0], start, end });
      continue;
    }
    const weights = pieces.map(p => Math.max(1, normalizedToken(p).length));
    const total = weights.reduce((a, b) => a + b, 0) || 1;
    let cursor = start;
    pieces.forEach((piece, i) => {
      const next = i === pieces.length - 1 ? end : cursor + (end - start) * weights[i] / total;
      if (next > cursor) out.push({ text: piece, start: cursor, end: next });
      cursor = next;
    });
  }
  return out.sort((a, b) => a.start - b.start);
}
function getLyricLines() {
  const raw = String(($('lyrics').value || '')).replace(/\r\n?/g, '\n');
  const lines = raw.split('\n').map(x => x.trim()).filter(Boolean);
  if (!lines.length) throw new Error('Paste the existing lyrics into the lyric box before generating.');
  return lines;
}
function alignLyrics(lines, observed, minCoverage, requireEveryLine) {
  const known = [];
  lines.forEach((line, lineIndex) => splitWords(line).forEach(word => known.push({ text: word, line: lineIndex })));
  if (!observed.length) throw new Error('The acoustic model returned no timed words.');
  const N = known.length, M = observed.length, width = M + 1, NEG = -1e9;
  const dp = new Float32Array((N + 1) * width); dp.fill(NEG);
  const back = new Int8Array((N + 1) * width);
  const at = (i, j) => i * width + j;
  dp[0] = 0;
  for (let i = 0; i <= N; i++) {
    for (let j = 0; j <= M; j++) {
      const cur = dp[at(i, j)];
      if (cur <= NEG / 2) continue;
      if (i < N && j < M) {
        const sim = similarity(known[i].text, observed[j].text);
        const score = cur + (sim >= 0.55 ? 1.60 * sim + 0.25 : -0.85);
        const k = at(i + 1, j + 1);
        if (score > dp[k]) { dp[k] = score; back[k] = 1; }
      }
      if (i < N) {
        const k = at(i + 1, j);
        if (cur - 1.20 > dp[k]) { dp[k] = cur - 1.20; back[k] = 2; }
      }
      if (j < M) {
        const k = at(i, j + 1);
        if (cur - 0.42 > dp[k]) { dp[k] = cur - 0.42; back[k] = 3; }
      }
    }
  }
  const mapped = new Array(N).fill(null);
  let i = N, j = M;
  while (i || j) {
    const action = back[at(i, j)];
    if (action === 1) {
      const sim = similarity(known[i - 1].text, observed[j - 1].text);
      if (sim >= 0.55) mapped[i - 1] = { text: known[i - 1].text, start: observed[j - 1].start, end: observed[j - 1].end, score: sim };
      i--; j--;
    } else if (action === 2) i--;
    else if (action === 3) j--;
    else break;
  }
  const matched = mapped.filter(Boolean).length;
  const coverage = matched / Math.max(1, N);
  if (coverage < minCoverage) throw new Error('Acoustic alignment matched only ' + (coverage * 100).toFixed(1) + '% of the supplied lyrics.');
  const out = lines.map((text, lineIndex) => {
    const words = [];
    known.forEach((entry, index) => {
      if (entry.line === lineIndex && mapped[index]) words.push(mapped[index]);
    });
    return words.length ? { text, begin: Math.min(...words.map(w => w.start)), end: Math.max(...words.map(w => w.end)), words } : { text, begin: null, end: null, words: [] };
  });
  if (requireEveryLine) {
    const missing = out.findIndex(x => !x.words.length);
    if (missing >= 0) throw new Error('Lyric line ' + (missing + 1) + ' could not be acoustically aligned.');
  }
  return { lines: out, matchedWords: matched, totalWords: N, coverage, model: 'Whisper tiny word timestamps + monotonic forced lyric alignment' };
}
function splitWords(line) {
  return line.match(/\S+/gu) || [];
}
function splitSyllables(word, lang) {
  if (!word) return [];
  const chars = Array.from(word);
  if (/^ja(?:-|$)/u.test(lang)) {
    const small = /^[ぁぃぅぇぉゃゅょゎっァィゥェォャュョヮッ]$/u;
    const out = [];
    for (const c of chars) {
      if (small.test(c) && out.length) out[out.length - 1] += c;
      else out.push(c);
    }
    return out;
  }
  if (/^ko(?:-|$)/u.test(lang) || /^zh(?:-|$)/u.test(lang)) return chars;
  const lead = (word.match(/^[^\p{L}\p{N}]*/u) || [''])[0];
  const trail = (word.match(/[^\p{L}\p{N}]*$/u) || [''])[0];
  const core = word.slice(lead.length, Math.max(lead.length, word.length - trail.length));
  const nuclei = [];
  const re = /[aeiouy]+/giu;
  let m;
  while ((m = re.exec(core)) !== null) nuclei.push({ start: m.index, end: m.index + m[0].length });
  if (!nuclei.length) return [word];
  const out = [];
  let start = 0;
  for (let i = 0; i < nuclei.length - 1; i++) {
    const cluster = core.slice(nuclei[i].end, nuclei[i + 1].start);
    const boundary = nuclei[i].end + Math.floor(Math.max(0, cluster.length - 1) / 2);
    out.push(core.slice(start, boundary));
    start = boundary;
  }
  out.push(core.slice(start));
  out[0] = lead + out[0];
  out[out.length - 1] += trail;
  return out;
}
function nearestPeak(peaks, targetMs, fromIndex, minMs) {
  let best = null;
  for (let i = Math.max(0, fromIndex || 0); i < peaks.length; i++) {
    const ms = peaks[i].time * 1000;
    if (ms < minMs) continue;
    const candidate = { index: i, timeMs: ms, strength: peaks[i].strength };
    if (!best || Math.abs(candidate.timeMs - targetMs) < Math.abs(best.timeMs - targetMs)) best = candidate;
    if (ms > targetMs + 700) break;
  }
  return best;
}
function refineWordSyllables(word, startMs, endMs, lang, peaks) {
  const parts = splitSyllables(word, lang);
  if (!parts.length) return [];
  const span = Math.max(16, endMs - startMs);
  const local = peaks.filter(p => p.time * 1000 >= startMs - 60 && p.time * 1000 <= endMs + 60).sort((a, b) => a.time - b.time);
  const out = [];
  let cursor = startMs, peakIndex = 0;
  const ideal = span / parts.length;
  for (let i = 0; i < parts.length; i++) {
    const nominal = startMs + i * ideal;
    const startPeak = nearestPeak(local, nominal, peakIndex, cursor);
    const begin = i === 0 ? startMs : Math.max(cursor, startPeak ? startPeak.timeMs - 12 : nominal);
    const nextNominal = i === parts.length - 1 ? endMs : startMs + (i + 1) * ideal;
    const endPeak = nearestPeak(local, nextNominal, startPeak ? startPeak.index + 1 : peakIndex + 1, begin);
    const end = i === parts.length - 1 ? endMs : Math.min(endMs, Math.max(begin + 8, endPeak ? endPeak.timeMs - 6 : nextNominal));
    out.push({ text: parts[i], begin: Math.max(startMs, Math.min(endMs, begin)), end: Math.max(begin, Math.min(endMs, end)) });
    cursor = out[out.length - 1].end;
    peakIndex = endPeak ? endPeak.index : (startPeak ? startPeak.index : peakIndex);
  }
  if (out.length) {
    out[0].begin = startMs;
    for (let i = 1; i < out.length; i++) out[i].begin = Math.max(out[i].begin, out[i - 1].end);
    for (let i = 0; i < out.length - 1; i++) out[i].end = Math.max(out[i].begin + 4, Math.min(endMs, out[i + 1].begin));
    out[out.length - 1].end = Math.max(out[out.length - 1].begin, endMs);
  }
  return out;
}
function alignmentToUnits(alignment, peaks, lang) {
  return alignment.lines.map(line => {
    const syllables = [];
    line.words.forEach((word, wi) => {
      refineWordSyllables(word.text, word.start * 1000, word.end * 1000, lang, peaks).forEach(u => syllables.push({ text: u.text, begin: u.begin, end: u.end, wordIndex: wi }));
    });
    return { text: line.text, begin: line.begin * 1000, end: line.end * 1000, syllables };
  });
}
function escapeXmlText(text) {
  return escapeHtml(text);
}
function toTime(ms) {
  const safe = Math.max(0, Math.floor(Number.isFinite(ms) ? ms : 0));
  const total = Math.floor(safe / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total - h * 3600) / 60);
  const s = total - h * 3600 - m * 60;
  return String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0') + '.' + String(safe % 1000).padStart(3, '0');
}
function renderSyllables(units) {
  return units.map((u, i) => {
    const next = units[i + 1];
    const spacer = next && next.wordIndex !== u.wordIndex ? ' ' : '';
    return '<span begin="' + toTime(u.begin) + '" end="' + toTime(u.end) + '">' + escapeXmlText(u.text) + '</span>' + spacer;
  }).join('');
}
function documentHead(title, artist, lang, hasV2) {
  return '<?xml version="1.0" encoding="UTF-8"?>\n<tt xmlns="http://www.w3.org/ns/ttml" xmlns:ttm="http://www.w3.org/ns/ttml#metadata" xmlns:itunes="http://music.apple.com/lyric-ttml-internal" xml:lang="' + lang + '" itunes:timing="Word">\n  <head>\n    <metadata>\n      <ttm:title>' + title + '</ttm:title>\n      <ttm:agent type="person" xml:id="v1"><ttm:name type="full">' + artist + '</ttm:name></ttm:agent>' + (hasV2 ? '\n      <ttm:agent type="person" xml:id="v2"><ttm:name type="full">Secondary Voice</ttm:name></ttm:agent>' : '') + '\n    </metadata>\n  </head>\n  <body>\n    <div itunes:song-part="Verse">\n';
}
function documentTail() {
  return '    </div>\n  </body>\n</tt>';
}
function overlapSeconds(a, b) {
  return Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));
}
function lineHasSecondary(line, intervals) {
  if (line.begin == null || line.end == null) return false;
  const begin = line.begin / 1000, end = line.end / 1000;
  const span = Math.max(0.001, end - begin);
  const overlap = intervals.reduce((sum, x) => sum + Math.max(0, Math.min(x.end, end) - Math.max(x.start, begin)), 0);
  return overlap / span >= 0.18;
}
function renderLine(line, index, backingLine, backingEvidence, secondaryIntervals, autoV2, autoBg, lang) {
  const agent = autoV2 && lineHasSecondary(line, secondaryIntervals) ? 'v2' : 'v1';
  let main = renderSyllables(line.syllables);
  let bg = '';
  if (autoBg) {
    let bgUnits = [];
    if (backingLine && backingLine.words && backingLine.words.length) {
      backingLine.words.forEach((word, wi) => {
        if (word.end * 1000 < line.begin - 80 || word.start * 1000 > line.end + 80) return;
        refineWordSyllables(word.text, word.start * 1000, word.end * 1000, lang, backingEvidence.peaks).forEach(u => bgUnits.push({ text: u.text, begin: u.begin, end: u.end, wordIndex: wi }));
      });
    } else if (line.syllables.length) {
      const overlaps = backingEvidence.intervals.filter(x => overlapSeconds(x, { start: line.begin / 1000, end: line.end / 1000 }) > 0.01);
      if (overlaps.length) {
        bgUnits = line.syllables.filter(u => overlaps.some(x => overlapSeconds(x, { start: u.begin / 1000, end: u.end / 1000 }) > 0.01));
      }
    }
    if (bgUnits.length) {
      const bgStart = Math.max(line.begin, Math.min(line.end, Math.min(...bgUnits.map(x => x.begin))));
      const bgEnd = Math.max(bgStart, Math.min(line.end, Math.max(...bgUnits.map(x => x.end))));
      if (bgEnd > bgStart) bg = '\n        <span ttm:role="x-bg" begin="' + toTime(bgStart) + '" end="' + toTime(bgEnd) + '">' + renderSyllables(bgUnits) + '</span>';
    }
  }
  return '      <p begin="' + toTime(line.begin) + '" end="' + toTime(line.end) + '" itunes:key="L' + (index + 1) + '" ttm:agent="' + agent + '">\n        ' + main + bg + '\n      </p>';
}

function alignmentLanguage(lang) {
  return ({ 'en-US': 'english', ja: 'japanese', ko: 'korean', 'zh-Hans': 'chinese', fil: 'tagalog' })[lang];
}
function configureAlignmentRuntime(mod, useGpu, adapter) {
  try {
    const onnx = mod?.env?.backends?.onnx;
    if (onnx?.env) {
      try { onnx.env.logLevel = 'error'; } catch (e) {}
    }
    if (useGpu && adapter && onnx?.webgpu) {
      try { onnx.webgpu.adapter = adapter; } catch (e) {}
    }
  } catch (e) {}
}

function resetAlignmentPipeline() {
  alignmentPipelinePromise = null;
}

async function getAlignmentPipeline() {
  if (alignmentPipelinePromise) return alignmentPipelinePromise;
  alignmentPipelinePromise = (async () => {
    const mod = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm');
    const useGpu = settings.gpu && gpuAdapter;
    configureAlignmentRuntime(mod, useGpu, gpuAdapter);
    return mod.pipeline('automatic-speech-recognition', 'onnx-community/whisper-tiny', {
      device: useGpu ? 'webgpu' : 'wasm',
      dtype: useGpu ? { encoder_model: 'fp32', decoder_model_merged: 'q4' } : 'q8',
      session_options: { logSeverityLevel: 3 }
    });
  })();
  return alignmentPipelinePromise;
}
async function runRealAlignment(file, lines, lang, onProgress, minCoverage, requireEveryLine) {
  const pipe = await getAlignmentPipeline();
  const ctx = await getAudioContext();
  if (!ctx) throw new Error('Web Audio is unavailable.');
  onProgress(8, 'Decoding ' + file.name, 'Using the actual inserted vocal waveform.');
  const buffer = await ctx.decodeAudioData(await file.arrayBuffer());
  const waveform = (() => {
    if (buffer.sampleRate === 16000) return buffer.getChannelData(0);
    const src = buffer.getChannelData(0);
    const len = Math.max(1, Math.round(src.length * 16000 / buffer.sampleRate));
    const out = new Float32Array(len);
    const scale = (src.length - 1) / Math.max(1, len - 1);
    for (let i = 0; i < len; i++) {
      const p = i * scale, a = Math.floor(p), f = p - a;
      const x = src[a] || 0, y = src[Math.min(src.length - 1, a + 1)] || x;
      out[i] = x + (y - x) * f;
    }
    return out;
  })();
  const language = alignmentLanguage(lang);
  onProgress(20, 'Recognizing sung words', 'Running Whisper word timestamps against the real audio.');
  const result = await pipe(waveform, {
    return_timestamps: 'word',
    chunk_length_s: 29,
    stride_length_s: 5,
    ...(language ? { language, task: 'transcribe' } : { task: 'transcribe' })
  });
  onProgress(58, 'Force-aligning existing lyrics', 'Mapping the textbox words to the observed acoustic word sequence.');
  const alignment = alignLyrics(lines, transcriptWords(result && result.chunks), minCoverage, requireEveryLine);
  log('acoustic alignment: ' + file.name + ' • ' + alignment.matchedWords + '/' + alignment.totalWords + ' words • ' + (alignment.coverage * 100).toFixed(1) + '%');
  return alignment;
}

function updateLyricsStats() {
  const raw = String(($('lyrics').value || '')).replace(/\r\n?/g, '\n');
  const lines = raw.split('\n').filter(x => x.trim().length);
  const lang = $('lang').value;
  let words = 0, syllables = 0;
  for (const line of lines) {
    const ws = splitWords(line);
    words += ws.length;
    syllables += ws.reduce((n, w) => n + splitSyllables(w, lang).length, 0);
  }
  $('lyricsStats').textContent = lines.length + ' lyric lines · ' + words + ' words · ' + syllables + ' syllable units';
}
$('lyrics').addEventListener('input', updateLyricsStats);
$('lang').addEventListener('change', updateLyricsStats);
updateLyricsStats();

document.querySelectorAll('.switch[data-toggle]').forEach(button => button.addEventListener('click', () => button.classList.toggle('on')));
bindInput('leadFile', 'leadCard', f => { lead = { file: f }; });
bindInput('bgFile', 'bgCard', f => { backing = { file: f }; });
$('eco').addEventListener('click', () => { $('eco').classList.toggle('on'); refreshProfile(); });

const runtimeNavigator = navigator;
$('webnn').textContent = runtimeNavigator.ml && runtimeNavigator.ml.createContext ? 'available' : 'optional · not exposed';
$('cores').textContent = String(navigator.hardwareConcurrency || '—');
$('memoryHint').textContent = navigator.deviceMemory ? String(navigator.deviceMemory) + ' GB hint' : 'unavailable';
$('gpuStatus').textContent = runtimeNavigator.gpu && runtimeNavigator.gpu.requestAdapter ? 'ready' : 'unavailable';

async function probeWebGpu() {
  const nav = navigator;
  if (!nav.gpu || !nav.gpu.requestAdapter) return null;
  const adapter = await nav.gpu.requestAdapter();
  if (!adapter || !adapter.requestDevice) return null;
  const device = await adapter.requestDevice();
  if (!device) return null;
  return { device, adapter };
}
async function setGpu(on) {
  if (!on) {
    resetAlignmentPipeline();
    if (gpuDevice && gpuDevice.destroy) gpuDevice.destroy();
    gpuDevice = null; gpuAdapter = null; webnnContext = null; settings.gpu = false;
    $('gpuStatus').textContent = 'off'; refreshProfile(); return;
  }
  try {
    const gpu = await probeWebGpu();
    if (!gpu) throw new Error('WebGPU is not available.');
    gpuAdapter = gpu.adapter;
    gpuDevice = gpu.device;
    resetAlignmentPipeline();
    if (navigator.ml && navigator.ml.createContext) {
      try { webnnContext = await navigator.ml.createContext(gpu.device); } catch { webnnContext = null; }
    }
    settings.gpu = true;
    $('gpuStatus').textContent = 'active';
    log('gpu: active WebGPU device' + (webnnContext ? ' + WebNN' : ''));
  } catch (err) {
    if (gpuDevice && gpuDevice.destroy) gpuDevice.destroy();
    gpuDevice = null; webnnContext = null; settings.gpu = false;
    $('gpuToggle').classList.remove('on');
    $('gpuToggle').setAttribute('aria-pressed', 'false');
    $('gpuStatus').textContent = 'unavailable';
    log('gpu: ' + (err instanceof Error ? err.message : String(err)));
  }
  refreshProfile();
}
function refreshProfile() {
  $('profile').textContent = (settings.cpu ? 'CPU DSP' : 'CPU off') + ' · ' + (settings.gpu ? 'GPU acoustic model' : 'CPU acoustic model') + ' · ' + (settings.adaptive ? 'adaptive' : 'fixed') + ' · ' + (settings.cache ? 'cache' : 'no cache');
  $('topProfile').textContent = settings.gpu ? 'GPU-assisted' : 'CPU-first';
  $('cpuStatus').textContent = settings.cpu ? 'optimized' : 'off';
}
$('cpuToggle').addEventListener('click', () => {
  settings.cpu = $('cpuToggle').classList.toggle('on');
  $('cpuToggle').setAttribute('aria-pressed', String(settings.cpu));
  if (!settings.cpu) {
    passed = false;
    $('generate').setAttribute('disabled', 'true');
    setGate('wait', 'CPU DSP is required for the local audio analysis stage.');
  }
  refreshProfile();
});
$('gpuToggle').addEventListener('click', async () => {
  const on = $('gpuToggle').classList.toggle('on');
  $('gpuToggle').setAttribute('aria-pressed', String(on));
  await setGpu(on);
});
$('adaptiveToggle').addEventListener('click', () => { settings.adaptive = $('adaptiveToggle').classList.toggle('on'); $('adaptiveToggle').setAttribute('aria-pressed', String(settings.adaptive)); refreshProfile(); });
$('cacheToggle').addEventListener('click', () => { settings.cache = $('cacheToggle').classList.toggle('on'); $('cacheToggle').setAttribute('aria-pressed', String(settings.cache)); refreshProfile(); });
$('yieldToggle').addEventListener('click', () => { settings.responsive = $('yieldToggle').classList.toggle('on'); $('yieldToggle').setAttribute('aria-pressed', String(settings.responsive)); refreshProfile(); });

$('analyze').addEventListener('click', async () => {
  if (!lead || !backing) { setGate('bad', 'Both Lead Vocals and Backing Vocals are required.'); return; }
  if (lead.file === backing.file || (lead.file.size === backing.file.size && lead.file.lastModified === backing.file.lastModified && lead.file.name === backing.file.name)) {
    setGate('bad', 'Lead vocals and backing vocals must be two different audio files.');
    return;
  }
  $('analyze').setAttribute('disabled', 'true');
  $('generate').setAttribute('disabled', 'true');
  passed = false;
  setProgress(1, 'Starting audio analysis', 'Decoding both inserted stems.');
  setGate('wait', 'Analyzing both stems…');
  try {
    const a = await analyzeSlot(lead, 'lead', 2, 46);
    const b = await analyzeSlot(backing, 'backing', 48, 46);
    const v2Detected = lead.stats.secondaryIntervals.length > 0 && lead.stats.secondaryVoicePeak >= 0.62;
    const bgDetected = backing.stats.vocalCoverage >= 0.045 && backing.stats.vocalActivity >= 0.44 && backing.stats.vocalIntervals.length > 0 && backing.stats.syllablePeaks.length >= 2;
    $('v2Status').textContent = v2Detected ? 'detected • second voice' : 'not detected';
    $('bgStatus').textContent = bgDetected ? 'detected • ' + backing.stats.vocalIntervals.length + ' vocal regions' : 'not detected';
    const leadOk = a >= 0.60, backingOk = b >= 0.60, ok = leadOk && backingOk;
    $('leadCard').classList.toggle('good', leadOk); $('leadCard').classList.toggle('bad', !leadOk);
    $('bgCard').classList.toggle('good', backingOk); $('bgCard').classList.toggle('bad', !backingOk);
    passed = ok;
    if (!ok) {
      const failed = [leadOk ? '' : 'lead ' + (a * 100).toFixed(0) + '%', backingOk ? '' : 'backing ' + (b * 100).toFixed(0) + '%'].filter(Boolean).join(', ');
      setGate('bad', 'Rejected: ' + failed + '. Use cleaner isolated vocal stems and analyze again.');
      log('gate: REJECT');
    } else {
      setGate('ok', 'Both stems passed the vocal-admission gate. Acoustic word alignment will run during Generate.');
      setProgress(100, 'Audio analysis complete', 'Lead and backing vocal evidence is ready.');
      $('generate').removeAttribute('disabled');
      log('gate: PASS');
    }
  } catch (err) {
    setGate('bad', err instanceof Error ? err.message : 'Analysis failed.');
    log('gate: ERROR');
  }
  $('analyze').removeAttribute('disabled');
});

function setProgress(n, label, detail) {
  const v = Math.max(0, Math.min(100, Number(n) || 0));
  $('progressBar').style.width = v + '%';
  $('pct').textContent = Math.round(v) + '%';
  $('progressLabel').textContent = label || 'Working…';
  $('progressDetail').textContent = detail || 'Processing locally…';
}

async function generateTtml() {
  const lines = getLyricLines();
  if (!lead || !backing || !lead.stats || !backing.stats) throw new Error('Analyze both stems before generating.');
  const title = escapeXmlText($('title').value || 'Untitled Session');
  const artist = escapeXmlText($('artist').value || 'Unknown Artist');
  const lang = $('lang').value;
  const v2Detected = lead.stats.secondaryIntervals.length > 0 && lead.stats.secondaryVoicePeak >= 0.62;
  const bgDetected = backing.stats.vocalCoverage >= 0.045 && backing.stats.vocalActivity >= 0.44 && backing.stats.vocalIntervals.length > 0 && backing.stats.syllablePeaks.length >= 2;
  const autoV2 = v2Detected && qs('[data-toggle="v2"]').classList.contains('on');
  const autoBg = bgDetected && qs('[data-toggle="bg"]').classList.contains('on');

  setProgress(2, 'Starting acoustic alignment', 'Recognizing the actual lead vocal waveform.');
  const leadAlignment = await runRealAlignment(lead.file, lines, lang, (v, l, d) => setProgress(v * 0.57, l, d), 0.82, true);
  if (leadAlignment.coverage < 0.90) log('alignment: lead coverage ' + (leadAlignment.coverage * 100).toFixed(1) + '%');

  setProgress(60, 'Aligning backing vocal audio', 'Analyzing the second waveform independently for true BG activity.');
  let backingAlignment;
  try {
    backingAlignment = await runRealAlignment(backing.file, lines, lang, (v, l, d) => setProgress(60 + v * 0.25, l, d), 0.25, false);
  } catch (err) {
    backingAlignment = { lines: lines.map(text => ({ text, begin: null, end: null, words: [] })), matchedWords: 0, totalWords: splitWords(lines.join(' ')).length, coverage: 0, model: 'audio-only BG detection' };
    log('backing alignment: partial text match unavailable; using independent backing-vocal audio detection');
  }

  const leadUnits = alignmentToUnits(leadAlignment, lead.stats.syllablePeaks, lang);
  let partial = documentHead(title, artist, lang, autoV2);
  $('xml').textContent = partial + documentTail();

  for (let i = 0; i < leadUnits.length; i++) {
    const line = leadUnits[i];
    partial += renderLine(line, i, backingAlignment.lines[i], backing.stats, lead.stats.secondaryIntervals, autoV2, autoBg, lang) + '\n';
    $('xml').textContent = partial + documentTail();
    setProgress(86 + (i + 1) / leadUnits.length * 12, 'Writing acoustically aligned line ' + (i + 1) + ' / ' + leadUnits.length, 'Lead timing and independent backing-vocal timing are being written to TTML.');
    await yieldToUi();
  }
  const xml = partial + documentTail();
  validateTtml(xml);
  log('generation: complete • ' + leadAlignment.matchedWords + ' lead words acoustically aligned');
  return xml;
}
function parseTtmlTime(value) {
  if (!value || !/^(?:\d+):[0-5]\d:[0-5]\d\.\d{3}$/.test(value)) return NaN;
  const p = value.split(':');
  return (Number(p[0]) * 3600 + Number(p[1]) * 60 + Number(p[2])) * 1000;
}
function validateTtml(xml) {
  const doc = new DOMParser().parseFromString(xml, 'application/xml');
  if (doc.getElementsByTagName('parsererror').length) throw new Error('Generated TTML failed XML validation.');
  const paragraphs = Array.from(doc.getElementsByTagName('p'));
  if (!paragraphs.length) throw new Error('Generated TTML contains no lyric lines.');
  let previousBegin = -1;
  paragraphs.forEach((p, index) => {
    if (p.getAttribute('itunes:key') !== 'L' + (index + 1)) throw new Error('Lyric line keys are not continuous.');
    const begin = parseTtmlTime(p.getAttribute('begin')), end = parseTtmlTime(p.getAttribute('end'));
    if (!Number.isFinite(begin) || !Number.isFinite(end) || begin < 0 || end < begin) throw new Error('A lyric line has an invalid or out-of-order timestamp.');
    if (index && begin < previousBegin) throw new Error('Lyric line start times are not ordered.');
    previousBegin = begin;
    for (const span of Array.from(p.getElementsByTagName('span'))) {
      const sb = parseTtmlTime(span.getAttribute('begin')), se = parseTtmlTime(span.getAttribute('end'));
      if (!Number.isFinite(sb) || !Number.isFinite(se) || sb < begin || se > end || se < sb) throw new Error('A lyric syllable/BG span is outside its lyric line or has an invalid timestamp.');
    }
  });
}

$('generate').addEventListener('click', async () => {
  if (!passed) return;
  $('generate').setAttribute('disabled', 'true');
  $('download').setAttribute('disabled', 'true');
  try {
    output = await generateTtml();
    $('xml').textContent = output;
    const lineCount = (output.match(/itunes:key="L\d+"/g) || []).length;
    const syllableCount = (output.match(/<span begin=/g) || []).length;
    setProgress(100, 'TTML ready', lineCount + ' lyric lines • ' + syllableCount + ' timed syllables • audio-aligned lead + BG analysis');
    const base = ($('title').value || 'session').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'session';
    $('fileName').textContent = base + '.ttml';
    $('download').removeAttribute('disabled');
    $('result').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    setGate('bad', message);
    log('generation: ERROR • ' + message);
    if (err instanceof Error && err.stack) log('generation stack: ' + err.stack.split('\\n').slice(0, 4).join(' | '));
  } finally {
    $('generate').removeAttribute('disabled');
  }
});
$('copy').addEventListener('click', async () => {
  if (!output) return;
  try { await navigator.clipboard.writeText(output); $('copy').textContent = 'Copied'; } catch { log('copy: clipboard permission unavailable'); }
});
$('download').addEventListener('click', () => {
  if (!output) return;
  const blob = new Blob([output], { type: 'application/ttml+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = $('fileName').textContent || 'session.ttml';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
});
refreshProfile();