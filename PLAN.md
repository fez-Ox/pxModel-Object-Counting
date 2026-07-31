# Plan: verbose-prompt SAM3 counting pipeline

## Context
- Add a separate, self-contained sub-folder for counting objects with native SAM3 text grounding.
- The new pipeline must not import, install, or depend on anything under `count-anything/`.
- Use verbose natural-language prompts such as `all pairs of black sunglasses displayed on the retail rack`, rather than requiring a short class label.

## Approach
- Create a standalone `sam3-verbose-counting/` application containing its own CLI, downloader, UV project metadata, and vendored SAM3 runtime package.
- Download the native `facebook/sam3` `sam3.pt` checkpoint with a dedicated script into the new folder's checkpoint directory.
- Use SAM3's native `Sam3Processor`: preprocess each image once with `set_image`, encode the full prompt with `set_text_prompt`, threshold text-grounded detections, and use the number of returned boxes as the count.
- Keep the model resident for all inputs; support local images, URLs, folders, and recursive folders; save per-image count/score/box JSON and annotated visualizations.
- Vendor only the SAM3 runtime modules/assets needed by the image builder and processor, and remove the existing repository's CountAnything/PDC imports from the standalone copy.

## Files to modify
- New `sam3-verbose-counting/pyproject.toml` with only standalone runtime dependencies.
- New `sam3-verbose-counting/download_model.py` saving `sam3.pt` under `sam3-verbose-counting/checkpoints/`.
- New `sam3-verbose-counting/infer.py` for URL/folder inputs, verbose prompt handling, persistent model inference, thresholding, output, timing, and peak VRAM reporting.
- New `sam3-verbose-counting/sam3/` minimal vendored native runtime package plus tokenizer assets.
- New `sam3-verbose-counting/README.md` with Kaggle/UV setup and examples.

## Reuse
- `count-anything/sam3/model/sam3_image_processor.py` — native SAM3 image preprocessing, text prompt encoding, and box/score extraction API to reproduce in the standalone package.
- `count-anything/sam3/model/sam3_image.py` — SAM3 image grounding model and `forward_grounding` contract.
- `count-anything/sam3/model/data_misc.py` — standalone `FindStage` input structure.
- `count-anything/sam3/model/box_ops.py` — conversion of normalized center boxes to pixel XYXY boxes.
- `count-anything/sam3/model_builder.py` — architecture/checkpoint-loading structure, adapted so it has no CountAnything import or PDC branch.
- `count-anything/infer.py` — URL, folder, recursive-folder, output, and timing CLI behavior as a reference only; no runtime import.

## Steps
- [x] Confirm native SAM3 checkpoint, standalone boundary, and verbose prompt semantics.
- [x] Trace the native processor API: `set_image` -> `set_text_prompt` -> thresholded `state["boxes"]`/`state["scores"]`.
- [x] Vendor/adapt the minimal SAM3 image runtime without CountAnything or training dependencies.
- [x] Add the standalone model downloader and UV environment metadata.
- [x] Add the persistent inference CLI with folder/URL support, count/box outputs, visualizations, timing, and peak VRAM telemetry.
- [x] Add standalone usage documentation and Kaggle commands.
- [x] Verify local-file, URL, folder, recursive-folder, and verbose-prompt inference paths.

## Verification
- Compile all standalone Python modules and run `uv run python infer.py --help`.
- Confirm `download_model.py` places `sam3.pt` at the documented default path and skips/replaces it correctly.
- Run one verbose prompt on one local/URL image and verify count, boxes, scores, timing, and visualization JSON.
- Run a folder and recursive folder while confirming the model/checkpoint is loaded only once.
- Search the new folder to ensure there are no imports or dependency references to `count_anything` or `count-anything`.
