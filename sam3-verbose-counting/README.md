# Standalone verbose-prompt SAM3 counting

This folder is an independent native-SAM3 image counting application with its own runtime and environment.

SAM3 receives a natural-language text prompt, returns text-grounded bounding boxes and confidence scores, and the count is the number of detections above the configured threshold.

## Kaggle / UV setup

Run these commands from a Kaggle notebook after cloning the repository:

```bash
cd sam3-verbose-counting
uv sync
uv run python download_model.py
```

The downloader uses a direct HTTP URL and does not require Hugging Face CLI login. The official SAM3 checkpoint is gated upstream, so an approved Hugging Face token is still required for that URL. In Kaggle, add a notebook Secret named `HF_TOKEN`; the script detects it automatically:

```bash
uv run python download_model.py
```

No CLI login is needed. You can also pass a token explicitly:

```bash
uv run python download_model.py --token "$HF_TOKEN"
```

Override the URL when using a permitted mirror:

```bash
uv run python download_model.py --url https://example.com/sam3.pt
```

The downloader places the checkpoint at:

```text
sam3-verbose-counting/checkpoints/sam3.pt
```

The checkpoint is not downloaded automatically by inference. If you cannot access the gated URL, attach a permitted `sam3.pt` file as a Kaggle dataset and point inference at it with `--checkpoint`. The default source is:

```text
https://huggingface.co/facebook/sam3/resolve/main/sam3.pt
```

## Inference

Use a verbose prompt as the final positional argument:

```bash
uv run python infer.py image.jpg \
  "all pairs of black sunglasses displayed on the retail rack"
```

Process a folder:

```bash
uv run python infer.py /kaggle/input/my-images \
  "all pairs of black sunglasses displayed on the retail rack"
```

Include nested folders:

```bash
uv run python infer.py /kaggle/input/my-images \
  "the red cars parked beside the building" --recursive
```

Image URLs are also supported:

```bash
uv run python infer.py \
  "https://example.com/image.jpg" \
  "all people wearing blue shirts"
```

Useful options:

```text
--threshold 0.5       Detection confidence threshold
--out results/        Output directory
--device cuda         Inference device
--no-amp              Disable CUDA mixed precision
--show                Display visualizations
--filter-prompt P     Exclude targets overlapping a second detection (e.g. "faces of people")
--no-filter-center    Don't drop a target whose center is inside a filter box
--filter-iou 0.3      Also drop targets whose IoU with a filter box reaches 0.3
--no-box-cleanup      Keep duplicate and multi-instance enclosing boxes
--box-duplicate-iou 0.9  IoU threshold for duplicate-box suppression
--box-min-children 2  Child detections required to remove an enclosing box
--box-min-area-ratio 1.25  Minimum enclosing/child area ratio
```

### Overlap filter

SAM3 cannot distinguish a *worn* pair of sunglasses from a *displayed* pair
using text alone (``"not being worn"`` has no pixel counterpart). To exclude
objects that sit on a person, run a second pass for the person/face and drop
any target overlapping it:

```bash
uv run python infer.py image.jpg \
  "pairs of sunglasses displayed on the retail rack" \
  --filter-prompt "faces of people"
```

A target box is removed when its **center lies inside** a filter detection
(`--filter-center`, on by default) or its **IoU** with one reaches
`--filter-iou`. The JSON output keeps the unfiltered `raw_count` /
`raw_boxes` / `raw_scores` alongside the filtered `count` so the decision can
be audited. Filtering is also available through the library API:

```python
from infer import build_counter
counter = build_counter(threshold=0.5)
result = counter.infer(image_path, "pairs of sunglasses displayed on the retail rack",
                       filter_prompt="faces of people", filter_center=True, filter_iou=0.0)
print(result["count"], "after dropping", result["filtered_count"], "worn pairs")
```

### Redundant-box cleanup

SAM3 can emit one broad detection spanning multiple objects as well as a
separate box for each object. Cleanup is enabled by default: near-identical
boxes are reduced to the highest-scoring box, and an enclosing box is removed
when it contains at least two smaller detections. The result keeps
`raw_boxes` / `raw_scores` / `raw_count`, plus `deduplicated_count` and
`redundant_box_count`, so brand-level localization can be inspected against
the original model output. Tune the thresholds with the `--box-*` options or
disable this behavior with `--no-box-cleanup`.

Each image produces a JSON file and annotated JPG containing:

- object count
- bounding boxes in pixel `x0, y0, x1, y1` format
- confidence scores
- inference time
- peak allocated and reserved VRAM when running on CUDA

The SAM3 model is loaded once and reused for all images in the invocation.
