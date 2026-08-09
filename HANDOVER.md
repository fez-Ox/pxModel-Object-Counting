# Eyewear Localization Handover

## Objective
Finish auditable, accurate brand attribution for every detected eyewear instance in the nine-image Kaggle dataset. SAM3 must remain class-agnostic; brands are only allowed through OCR/gazetteer evidence. Unknown and excluded decisions must remain observable.

## Repository / commands

```bash
cd /home/faizan/code/pxModel-localization
cd eyewear-localization && uv run pytest
cd ..
python - <<'PY'
import json
from pathlib import Path
nb = json.loads(Path('eyewear_localization_kaggle.ipynb').read_text())
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        compile(''.join(cell['source']), f'cell-{i}', 'exec')
print('notebook valid')
PY
```

Bounded Kaggle run:

```bash
uv run python -u scripts/run_kaggle.py \
  --target-image IMG_0278.jpg --max-images 1 \
  --wait-timeout 600 --poll-seconds 10 --cleanup-grace-seconds 45
```

The runner requires local `HEAD == origin/main`, saves remote logs, now attempts to download partial outputs before deleting failed workers, and never leaves a worker running beyond the timeout.

## Latest commits

`origin/main` currently ends at `5707a4a`:

- `a89b7e0`: RapidOCR/PP-OCR and PaddleOCR adapter scaffolding, closed-set Oakley aliases, notebook OCR package setup.
- `e2878bd`: UV lock update for the high-accuracy OCR extra.
- `5707a4a`: preserve partial Kaggle output on failed validation.

The worktree has **uncommitted** C2 changes currently being tested:

- `eyewear-localization/eyewear_localization/cues.py`: repeated, spatially separated same-brand signs create a corroborated display-span C2 evidence record and upgrade weaker C2 evidence.
- `eyewear-localization/tests/test_c2.py`: regression test for that behavior.

Run tests before committing those changes. `kernel-metadata.json`, sample images, and `output/` are generated/untracked artifacts; do not commit them.

## Remote validation history

- `IMG_0270`: previously verified 40/40 Oakley.
- `IMG_0295`: previously verified 30/30 — 17 Burberry and 13 Michael Kors.
- `IMG_0278`: SAM3 detects 13 instances. Before the latest C2 change, OCR found two Oakley signs but only 10 low-confidence C2 evidence records; fusion abstained on all 13 because `cascade_t2=0.70` and those records were below the threshold.

Latest partial artifacts are in:

```text
output/kaggle_results/ocr_cache/IMG_0278.json
output/kaggle_results/outputs/IMG_0278.json
output/kaggle_results/eyewear-localization-brand-attribution-remote.log
```

The JSON shows signs `DARLEY` and `DARKY`, both resolved to `oakley` through a closed-set OCR alias, but outputs were still `unknown` before the uncommitted C2 span fix. Do not treat that run as successful.

## OCR comparison findings

Local IMG_0278 checks:

- Tesseract: mostly noise; some `OQMetQ`/Meta-like output.
- EasyOCR 1.7.2: `cMeta`/`DoMeto`, missed Oakley.
- RapidOCR 1.2.3 (PP-OCR ONNX): `xMeta`/`.oMeta`, missed Oakley.
- Florence-2: full-frame output recognized Meta; focused lower-display/quarter crops produced `DARKY` and `DARLEY`, which motivated the strict aliases.
- Kaggle setup successfully installed `paddlepaddle==3.2.2` and `paddleocr==3.2.0` on the last worker, but the PaddleOCR backend has not yet been used in a completed comparison run.

Implemented OCR support:

- `tesseract`, `easyocr`, `florence2`, `rapidocr`, `paddleocr`
- `tesseract+florence2`, `tesseract+rapidocr`, `rapidocr+florence2`, `paddleocr+florence2`, `easyocr+florence2`
- Florence input dtype compatibility, Transformers/tokenizers/hub pins, quad-box parsing, lower-display and four-column focused OCR.

## Immediate continuation plan

1. Run `uv run pytest` and notebook compilation after the uncommitted C2 change. Fix any test failure, then commit/push it.
2. Confirm the pushed notebook really passes `rapidocr+florence2` to both OCR-cache and full C1 commands. Add an explicit backend print if the downloaded JSON still reports stale `tesseract+florence2`.
3. Rerun IMG_0278 with the 600-second bound. Inspect the newly downloaded JSON, especially all 13 `outputs`, `decision_path`, C2 support, signs, and annotated JPEG.
4. Test PaddleOCR explicitly by changing notebook `TUNE['ocr_backend']` to `paddleocr+florence2` (the optional install already attempts PP-OCRv5), then run the same bounded target. Compare raw `text_detections`, matched signs, elapsed time, and instance assignments against RapidOCR/Florence.
5. Keep the fastest backend that reliably recovers Oakley without introducing false brands. PaddleOCR must remain optional/fail-safe because its wheel is Python/CUDA-version dependent.
6. Remove temporary verbose `[FLORENCE]` raw-output debug logging after the backend comparison, retaining concise stage timing/backend logging.
7. Once IMG_0278 is genuinely 13/13 attributed, process the remaining images one at a time with the same 600-second limit. Download and inspect each JSON/annotation before proceeding.
8. Run the full local suite, push final changes, and only then claim dataset-wide accuracy.

## Safety / correctness constraints

- Never put brand names into SAM3 prompts.
- Keep OCR matching closed-set; malformed OCR must not create a new brand.
- Preserve raw OCR text and alias/match method in evidence.
- Do not lower cascade thresholds globally just to pass IMG_0278; use explicit corroborated same-brand geometry and audit it.
- Do not claim success from a failed/partial Kaggle run.
- If a remote worker errors or times out, verify it was deleted before launching another run.
- If session context approaches its limit, summarize this document and continue in a new session rather than dropping the remote-output and commit details.
