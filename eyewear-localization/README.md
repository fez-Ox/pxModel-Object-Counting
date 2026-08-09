# Robust eyewear brand attribution

This is a standalone implementation of `Localization-Task-Architecture.md`.
It is independent of the SAM3 sunglasses-counting application.

## Current milestone

Milestone 1 and the initial L0/C1 implementation are available:

- JSON-compatible schemas for instances, signs, evidence, and decisions.
- YAML configuration and replaceable model registry.
- Local path, folder, recursive-folder, and URL input handling.
- Class-agnostic SAM3 adapter using only `sunglasses`, `eyeglasses`,
  `glasses`, and `rimless glasses` prompts.
- Independent EasyOCR adapter with upscaling for small text.
- Gazetteer matching with normalization, token containment, and edit distance ≤ 1.
- C1 per-instance crop OCR and C2 geometry-based signage scope hypotheses.
- Reliability-weighted fusion, `unknown` abstention, and gated C3 smoothing.
- Missing optional models fall back to low-reliability no-op adapters.

C2 is intentionally conservative and is not a substitute for the synthetic
renderer/classifier milestone. Poster detection, logo detection, super-
resolution, VLM auditing, calibration, and evaluation harnesses remain
explicit next milestones.

## Setup

```bash
cd eyewear-localization
uv sync --extra ocr --extra sam3
```

EasyOCR downloads its own recognition models on first use. The `sam3` extra
installs the native runtime dependencies. The companion Kaggle notebook
`../eyewear_localization_kaggle.ipynb` retrieves the gated SAM3 checkpoint
through the direct HTTP downloader using the `HF_TOKEN` Kaggle Secret.

To enable class-agnostic localization locally, download or provide the native
SAM3 checkpoint explicitly:

```bash
uv run python infer.py /path/to/image.jpg \
  --sam3-checkpoint ../sam3-verbose-counting/checkpoints/sam3.pt
```

Without `--sam3-checkpoint`, the localizer uses a safe low-reliability fallback
and emits no instances; the OCR and schema stages still run.

## OCR and attribution output

```bash
uv run python infer.py /kaggle/input/images --recursive \
  --brand-file brands.txt --out outputs
```

The OCR stage detects all text independently. The configured gazetteer controls
which strings become `signs[]`; raw OCR remains in `text_detections[]`. The JSON
contains:

- `instances[]`: class-agnostic eyewear detections kept for attribution.
- `excluded_instances[]`: L0 detections filtered out by the scene filter (worn on a person, inside an advertisement, or not on a detected shelf) with their reasons.
- `signs[]`: OCR strings matched to gazetteer brands, with their inferred scope.
- `evidence[]`: C1/C2/C4 cue records, never final labels.
- `outputs[]`: fused per-instance labels and probabilities, including `unknown`.

A sign is not automatically assigned to nearby eyewear. C2 must first produce a
scope hypothesis, and only then can it emit evidence for instances inside that
region.

## C2 bay/row grouping

C2 attributes an entire inferred display group, not just detections that
horizontally overlap the OCR box. When several signs share a header band, their
midpoints are used as inferred bay boundaries (per-image, never a fixed grid);
with a single header the nearest horizontal display group is selected using a
gap threshold learned from detected object widths. Row-end labels attribute
their whole row.

## Scene filtering

With a SAM3 checkpoint present, the pipeline additionally runs class-agnostic
prompts (`people`/`person`/`faces of people`, `advertisements`/`posters`/
`billboards`, `retail shelves`/`display shelves`/`shelf`). Detections whose
center lies inside a person or advertisement region are removed, and when shelf
regions were detected, detections outside them are removed too. Every removal
is recorded in `excluded_instances[]` and drawn in red in the annotated image.
Control with `--shelf-filter` / `--no-shelf-filter` (default: on).

## Kaggle notebook and headless runner

The Kaggle-ready walkthrough is at `../eyewear_localization_kaggle.ipynb`.
It uses Kaggle's preinstalled CUDA/Torch pair rather than running `uv sync`
inside the GPU worker (installing a newer CUDA wheel can make SAM3 fail with
`no kernel image is available`). It verifies CUDA, authenticates HF access,
downloads the gated checkpoint, and refuses to report a false empty full pass.

For headless execution from the repository root:

```bash
# ~/.kaggle/kaggle.json: Kaggle API credentials
# ~/.kaggle/hf_token: approved token for facebook/sam3
uv run python scripts/run_kaggle.py
```

The runner validates notebook syntax, verifies that local `HEAD` matches
`origin/main`, pushes a temporary notebook copy, injects the HF token only in
that temporary copy, waits for the GPU kernel, and downloads results to
`output/kaggle_results/`. It never writes the token into the working tree.

## Diagnosis (offline, no model downloads)

Every run already writes a JSON file per image. Two things make abstentions
explainable without re-running the model:

1. **`outputs[].decision_debug`** — added to each decision. It records the
   winning brand and its probability, the runner-up, and which acceptance
   gates failed (`tau`, `margin`, `beats_unknown`) with their thresholds.
2. **`diagnose.py`** — a pure-standard-library audit. Point it at any result
   JSON and it prints, per instance, the evidence chain, the gate breakdown,
   and a summary of the two most common abstention causes (no evidence reached
   the instance, or evidence was too weak to pass the tau gate):

```bash
python diagnose.py outputs/image.json
python diagnose.py < outputs/image.json
```

The same five questions `diagnose.py` answers, asked directly in the JSON:

1. `excluded_instances[]` — removed by the scene filter (try `--no-shelf-filter`).
2. `instances[]` — present at all? If not, the localizer missed them.
3. `signs[]` — was the brand text found by OCR at all?
4. `signs[].scope` — did C2 assign a region, and does it cover the instance?
5. `outputs[].decision_debug.gates` — if a brand probability sits near 0.5–0.6,
   the evidence exists and is correct but fails the acceptance gate; lower
   `fusion.tau` / `fusion.margin` in `config.yaml`.

## Tests

```bash
uv run python -m pytest
```

The tests use injected fake model adapters and do not download model weights.
