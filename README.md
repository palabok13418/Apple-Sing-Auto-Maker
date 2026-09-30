# Apple Sing TTML Forge

A separate, low-end-first project for generating Apple-Music-style Sing TTML from isolated lead-vocal and backing-vocal stems.

## Product flow

1. Require both lead and backing vocal stems.
2. Analyze the actual audio before generation.
3. Reject mixed/instrumental-heavy, duplicate, or ambiguous uploads rather than trusting filenames.
4. Run compact CPU-first audio analysis with optional WebGPU/WebNN acceleration.
5. Detect vocal activity, possible secondary voice (V2) and backing-vocal activity from the analyzed audio.
6. Accept pasted existing lyrics as multiline source text; every Enter-separated line is preserved as a separate lyric line.
7. Break each supplied line into syllable timing units and map them onto the analyzed vocal activity timeline.
8. Build singer agents, BG placements, song-part structure and metadata.
9. Validate the final TTML, including timestamp format and nested timed spans, before export.

## Stack

- TypeScript + JavaScript: browser application and orchestration
- Rust/WASM: low-overhead DSP primitives
- Python: dataset preparation, model training and calibration
- Tiny softmax model: low-end CPU classifier training artifact

## Implementation note

The browser admission gate is a compact calibrated feature ensemble and is intentionally conservative, but it is not a substitute for a real labeled audio dataset. python/train_classifier.py now contains an actual offline trainer that learns a tiny standardized softmax classifier and weights the reject class more heavily. Train it on representative isolated lead stems, isolated backing stems, full mixes, instrumental-heavy files, noise and ambiguous cases before treating the classifier as production-grade.

The TTML generator now derives its lyric-line structure from the pasted lyrics instead of inventing a fixed-length transcript. Audio timing is still signal-derived rather than a full neural forced-alignment model, so syllable timing is an informed local alignment pass, not a guarantee of phoneme-level accuracy.
