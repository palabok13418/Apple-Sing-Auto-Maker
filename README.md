# Apple Sing TTML Forge

A separate, low-end-first project for generating Apple-Music-style Sing TTML from isolated lead-vocal and backing-vocal stems.

## Product flow

1. Require both lead and backing vocal stems.
2. Analyze the actual audio before generation.
3. Reject mixed/instrumental-heavy or ambiguous uploads rather than trusting filenames.
4. Run a compact client model with CPU inference as the baseline and WebNN as an optional accelerator.
5. Build word timing, background-vocal roles, singer agents, song parts and metadata.
6. Validate the XML.
7. Export a `.ttml` file.

## Stack

- TypeScript + JavaScript: browser application and orchestration
- Rust/WASM: low-overhead DSP primitives
- Python: dataset preparation, training and calibration
- Tiny quantized model: client-side stem classification

## Important implementation note

The included browser classifier is a conservative prototype feature gate, not a trained production classifier. The `python/` folder is the offline training/calibration scaffold. A real deployment should be trained on labeled isolated stems, full mixes, instrumental-heavy files and ambiguous cases, with a high-precision reject policy.
