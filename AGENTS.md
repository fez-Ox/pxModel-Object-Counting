# pxModel-localization

## Package management

- Uses **uv** (not pip/poetry). Install deps: `uv sync`
- Run script: `uv run python main.py` (or `python main.py` inside `.venv`)
- Python 3.12 (`.python-version`, `uv.lock`)

## Secrets

- `.env` has a hard-coded `HF_ACCESS_TOKEN`. The file is gitignored but **the token value is exposed in the repo** (it was committed to history / checked in at some point). Do not commit secrets.
- The `dotenv` package loads `.env` at runtime.

## Structure

Single-file project. Entrypoint: `main.py` — tests DinoV2 and DinoV3 via HuggingFace transformers.

## CountAnything (`count-anything/`)

- Cloned from [MengqiLei/CountAnything](https://github.com/MengqiLei/CountAnything) — SAM3-based text-guided object counter
- **Stripped to inference only**: `data/`, `exp/`, `pretrained/`, `assets/`, scripts, docs, CI removed
- **Checkpoint missing**: download `count_anything.pt` from [HuggingFace](https://huggingface.co/MengqiLei/count-anything) and place at `count-anything/checkpoints/count_anything.pt`
- Install CountAnything deps: `cd count-anything && uv pip install -r requirements.txt` (note: uses pip, not uv sync for this sub-project)
- **Inference CLI**: `cd count-anything && python infer.py image.jpg "text query"` — wraps the subprocess-based API with checkpoint validation and result saving
- **Python API**: `from count_anything import CountAnything; model = CountAnything("checkpoints/count_anything.pt"); result = model("img.jpg", "cars")[0]; result.count; result.show(); result.save()`
- **Inference is subprocess-based**: `CountAnything.__call__` spawns `python -m count_anything.train.train` under the hood with a temporary config/annotation — not a direct model call
- **Output dir**: defaults to `exp/count_anything_inference/` (created on demand)

## Native SAM3 verbose counting (`sam3-verbose-counting/`)

- Standalone app, independent of `count-anything/` (no imports or deps shared)
- **Detection tasks are decoupled** under `sam3-verbose-counting/detectors/`:
  - `base.py` — `DetectionTask` base class + `DetectionOptions` dataclass
  - `__init__.py` — registry (`register`, `get_detector`, `list_detectors`)
  - `sunglasses.py` — the sunglasses detection task (the only registered task)
- The generic, detection-agnostic core is `sam3-verbose-counting/infer.py`
  (`Sam3VerboseCounter`, `build_counter`, `annotate`). Brand-label detection was
  removed; **future detections** are added as new `detectors/<name>.py` modules
  subclassing `DetectionTask` and registered with `@register` — no edits to the
  sunglasses task (or any other task) are required.
- **Inference CLI**: `cd sam3-verbose-counting && uv run python infer.py image.jpg "prompt"`
  or `--detector sunglasses`. `--detector <name>` selects a task and `--prompt`
  overrides the detector default (otherwise the final positional arg is the prompt).
  The default task is `sunglasses` (counts displayed pairs, excluding pairs worn on people).
- **Checkpoint missing**: `sam3-verbose-counting/checkpoints/sam3.pt` (gated on
  HuggingFace). Download via `cd sam3-verbose-counting && uv run python download_model.py`
  (requires an approved `HF_TOKEN` / Kaggle secret).
- Kaggle notebook: `sam3_verbose_counting_kaggle.ipynb` (setup, download, sunglasses inference, cleanup).

## Robust eyewear brand attribution (`eyewear-localization/`)

- Standalone brand attribution package following Specification v2 (`Fix.md`).
- **L0 Perception**: Class-agnostic SAM3 localizer + zero-shot OCR (**Florence-2** via `Florence2OCRBackend` with fallback to EasyOCR) + scene filter.
- **C1 On-Product Branding**: Zero-shot Florence-2 OCR running on instance crops (no fine-tuning, no generative super-resolution).
- **C2 Signage Scope**: 3-stage pure deterministic geometry (`Parse` $\rightarrow$ `Attach` $\rightarrow$ `Assign`). Zero VLM calls inside C2.
- **L2 Precision Cascade**: Priority decision list (`C1` physical brand $>` `C2` signage zone scope $>` `C3` spatial continuity $>` `C4` style prior $>` `unknown`) replacing Softmax probability fusion.
- **Label Semantics**: Distinguishes `product_brand` (physical frame) vs `zone_brand` (display bay) on `AttributionOutput`.
- **Inference CLI**: `cd eyewear-localization && uv run python infer.py image.jpg --ocr-backend florence2`
- **Test suite**: `cd eyewear-localization && uv run pytest` (44 unit tests covering schemas, perception, C2 determinism, and precision cascade).
- **Automated Remote GPU Testing**: `uv run python scripts/run_kaggle.py` (pushes notebook, triggers GPU execution headlessly, streams status, and downloads results to `./output/kaggle_results/`).
- **Kaggle notebook**: `eyewear_localization_kaggle.ipynb` (CUDA GPU execution).

## Design doc

`DESIGN.md` tracks architecture decisions. Two phases:
- **Phase 1** (desktop GPU, ≤20GB): CountAnything-based, native SAM3, and eyewear brand attribution pipeline
- **Phase 2** (smartphone): lightweight approach TBD

## Testing & Quality

- Test runner: `uv run pytest` inside `eyewear-localization/tests`
- Standard Python 3.12 syntax with type annotations.
- **Mandatory Preflight Audit Protocol**: BEFORE launching any remote Kaggle GPU execution (`uv run python scripts/run_kaggle.py`), ALWAYS perform:
  1. **Local Preflight Execution**: Run a local CPU/GPU dry run (`uv run python scripts/prototype_single_pass_ocr.py ...`) to verify all imports, syntax, function signatures, and CLI arguments end-to-end.
  2. **Notebook Cell Audit**: Thoroughly inspect all cells in `eyewear_localization_kaggle.ipynb` to verify that argument strings, function calls, and script paths match the codebase exactly.

