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

- `instances[]`: class-agnostic eyewear detections.
- `signs[]`: OCR strings matched to gazetteer brands, initially with scope `none`.
- `evidence[]`: C1/C2/C4 cue records, never final labels.
- `outputs[]`: fused per-instance labels and probabilities, including `unknown`.

A sign is not automatically assigned to nearby eyewear. C2 must first produce a
scope hypothesis, and only then can it emit evidence for instances inside that
region.

## Notebook

A simple Kaggle-ready walkthrough is at `../eyewear_localization_kaggle.ipynb`.
It runs OCR without SAM3 first, then optionally runs full attribution when an
approved `sam3.pt` is attached as a Kaggle dataset.

## Tests

```bash
uv run python -m pytest
```

The tests use injected fake model adapters and do not download model weights.
