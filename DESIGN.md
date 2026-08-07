# One-Shot Object Counting Pipeline — Design

## Goal

Count all instances of a class in a **query image** given either a text query (zero-shot) or a single **support image** (one-shot).

## Project Phases

The project has two independent implementations targeting different resource levels.

### Phase 1: Unconstrained (desktop GPU)

- **Target VRAM**: ≤20 GB (single consumer GPU)
- **Base**: [CountAnything](https://github.com/MengqiLei/CountAnything) (cloned at `count-anything/`)
- **Status**: Checkpoint missing — `checkpoints/count_anything.pt` needs to be downloaded from [HuggingFace](https://huggingface.co/MengqiLei/count-anything) and placed at `count-anything/checkpoints/count_anything.pt`
- **Plan**: Get CountAnything inference working, then extend it

#### CountAnything architecture (SAM3-based)

```
Image ──┐
         ├── SAM3VLBackbone ──┬── Vision encoder (Hiera/MAE-style ViT)
Text ───┘                     └── Text encoder (tokenizer + transformer)
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │  Transformer encoder             │
                    │  (self-attention on image feats  │
                    │   + cross-attention with text    │
                    │   prompt features)               │
                    └────────┬───────────────┬────────┘
                             │               │
              ┌──────────────┘               └──────────────┐
              ▼                                             ▼
   ┌──────────────────────┐                    ┌──────────────────────┐
   │ RSC (Region-level    │                    │ PDC (Pixel-level     │
   │  Sparse Counter)     │                    │  Dense Counter)      │
   │                      │                    │                      │
   │ Transformer decoder  │                    │ Feature adapter      │
   │ with learned queries │                    │ → dense point regr.  │
   │ → box proposals      │                    │ → point predictions  │
   └──────────┬───────────┘                    └──────────┬───────────┘
              │                                           │
              └───────────┬───────────────────────────────┘
                          ▼
            ┌─────────────────────────┐
            │ CCF (Complementary      │
            │  Count Fusion)          │
            │ parameter-free combine  │
            └───────────┬─────────────┘
                        ▼
                   Count (int) + point visualizations
```

- **Backbone**: SAM3's `SAM3VLBackbone` with a Hiera/MAE-style ViT vision encoder + text encoder
- **RSC**: Region-level Sparse Counter — transformer decoder with learned object queries produces box proposals (center points → count)
- **PDC**: Pixel-level Dense Counter — lightweight CNN head on pixel embeddings from the segmentation decoder, predicts dense point offsets + foreground scores at each grid location
- **CCF**: Complementary Count Fusion — parameter-free rule that merges RSC and PDC counts, suppressing overlap while preserving each branch's strengths
- **Inference API**: `CountAnything(checkpoint)` → `model(image_path, "text query")` → `result.count`, `result.pred_points`, `result.save()`
- **Input**: Text-guided (natural language category or description)
- **Training**: Full model trained on CLOC dataset (~220K images, 619 categories, 15M instances). LoRA fine-tuning possible via config.

#### Phase 1 roadmap

1. Download checkpoint, get single-image inference running
2. Evaluate accuracy on a few test images
3. Extend for **image-based one-shot**: replace the text prompt with visual features extracted from the support image (e.g., encode support, use its patch features as the prompt instead of text embeddings)
4. Remix the pipeline: potentially replace SAM3 backbone with DinoV2/DinoV3 if needed for the one-shot variant

#### Implemented: native SAM3 verbose counting (`sam3-verbose-counting/`)

A standalone native-SAM3 counter is implemented at `sam3-verbose-counting/`,
independent of the `count-anything/` clone. The core (`infer.Sam3VerboseCounter`)
is detection-agnostic: a verbose text prompt yields text-grounded boxes → count,
with overlap filtering and redundant-box cleanup. Concrete detections are
**decoupled** as tasks under `detectors/` (`DetectionTask` base + registry +
`sunglasses.py`). The sunglasses task counts displayed pairs while excluding
pairs worn on people. Brand-label detection was intentionally removed; adding a
future detection is a new `detectors/<name>.py` module registered with
`@register` — no edits to the sunglasses task (or any other task) are needed.

Inference requires the gated `sam3.pt` checkpoint at
`sam3-verbose-counting/checkpoints/sam3.pt` (download with
`cd sam3-verbose-counting && uv run python download_model.py`). Kaggle notebook:
`sam3_verbose_counting_kaggle.ipynb`.

### Phase 2: Resource-constrained (smartphone)

- **Target**: On-device inference (smartphone-class compute)
- **Approach**: TBD after Phase 1 research
- Likely directions: lightweight ViT (MobileViT, EfficientViT), feature correlation with prototype matching, quantized or distilled models

## Key open questions

- **One-shot via CountAnything**: Can we feed visual features (from a support image) as the prompt in place of text? The `_encode_prompt` path in `sam3_image.py` already supports text, geometric, and visual prompts — needs investigation
- **DinoV2/DinoV3 role**: If CountAnything's SAM3 backbone is too heavy for our use case, we may swap in DinoV2/V3 as the encoder and build a simpler comparison + counting head
- **Training data**: CLOC is text-guided. For one-shot, do we need a different training regime or dataset?
