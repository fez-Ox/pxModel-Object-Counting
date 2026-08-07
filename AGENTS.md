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

## Design doc

`DESIGN.md` tracks architecture decisions. Two phases:
- **Phase 1** (desktop GPU, ≤20GB): CountAnything-based, then extend to image-based one-shot
- **Phase 2** (smartphone): lightweight approach TBD

## What's missing

No tests, no lint/format config, no CI. Do not assume any test runner or formatter exists.
