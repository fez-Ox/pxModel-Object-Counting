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
```

Each image produces a JSON file and annotated JPG containing:

- object count
- bounding boxes in pixel `x0, y0, x1, y1` format
- confidence scores
- inference time
- peak allocated and reserved VRAM when running on CUDA

The SAM3 model is loaded once and reused for all images in the invocation.
