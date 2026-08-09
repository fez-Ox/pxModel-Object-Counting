# pxModel-localization

## CountAnything inference

From a fresh clone (for example on Kaggle):

```bash
cd count-anything
uv sync
uv run python download_checkpoint.py
uv run python infer.py /path/to/image.jpg "cars"
uv run python infer.py /path/to/image_folder "cars"
uv run python infer.py /path/to/image_folder "cars" --recursive
uv run python infer.py "https://example.com/image.jpg" "cars"
```

`count-anything/pyproject.toml` contains the CountAnything inference dependencies for UV. The checkpoint downloader saves the model to `count-anything/checkpoints/count_anything.pt`, which is the default path used by `infer.py`.

## Native SAM3 verbose-prompt counting

For an independent native-SAM3 pipeline that counts objects from verbose natural-language prompts, see [`sam3-verbose-counting/README.md`](sam3-verbose-counting/README.md).

## Kaggle notebooks and headless execution

For the eyewear attribution pipeline, use [`eyewear_localization_kaggle.ipynb`](eyewear_localization_kaggle.ipynb). The validated headless runner is:

```bash
uv run python scripts/run_kaggle.py
```

It requires Kaggle credentials in `~/.kaggle/kaggle.json` and an approved
Hugging Face token in `~/.kaggle/hf_token`. Tokens are never committed. The
runner checks the notebook and remote git revision before pushing, uses
Kaggle's compatible preinstalled CUDA/Torch stack, and downloads results under
`output/kaggle_results/`.

To run the native SAM3 verbose-prompt counter end to end (setup, checkpoint download, and inline inference) in a Kaggle notebook, open [`sam3_verbose_counting_kaggle.ipynb`](sam3_verbose_counting_kaggle.ipynb). It:

- clones this repository (edit `REPO_URL` in cell 1),
- installs only the SAM3 runtime deps missing from the Kaggle kernel,
- downloads the gated `sam3.pt` using the notebook's `HF_TOKEN` Secret,
- builds the persistent counter once and counts objects in sample and custom images with inline annotated results.

It calls the refactored `sam3-verbose-counting/infer.py` library functions directly (no subprocess):

```python
from infer import build_counter, annotate
counter = build_counter(threshold=0.5)               # loads sam3.pt on cuda/cpu
result = counter.infer("image.jpg", "the red cars parked beside the building")
annotated = annotate("image.jpg", "the red cars parked beside the building",
                     result["boxes"], result["scores"])
print(result["count"])                                # number of detections
```
