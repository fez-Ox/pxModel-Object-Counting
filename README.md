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
