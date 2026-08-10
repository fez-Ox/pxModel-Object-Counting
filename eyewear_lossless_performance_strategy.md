# Lossless Performance Strategy for Eyewear Localization

## Recommendation

Speed up **execution scheduling**, not perception or attribution policy:

1. Batch *discrete*, already-eligible C1 Florence-2 OCR calls.
2. Micro-batch the unchanged discrete SAM3 prompts (never comma-concatenate them).
3. Enable `torch.compile` only for a model path that passes a strict output-parity gate.
4. Keep a sequential fallback for every optimized path.

This retains every required SAM3 prompt, shelf filter, C1 crop, C1 transform, OCR generation setting, gazetteer match, C2 computation, and precision-cascade decision. The implementation is exposed through `--c1-batch-size`, `--sam3-prompt-batch-size`, and `--sam3-compile`; all default to the reference-safe value (`1`/disabled) until the parity corpus has approved a faster setting. Backend errors and malformed batch responses replay the sequential path.

## What the analysis and implementation show

| Finding | Consequence |
|---|---|
| Baseline mean time is 36.95 s: L0 15.22 s, C1 21.66 s, C2/C4/fusion 0.05 s. | Optimizing deterministic C2/C4 cannot materially help. |
| The proposed consolidated SAM3 strings cut L0 to 7.00 s but missed a top shelf and excluded a valid Burberry frame. | Prompt text and prompt semantics are immutable. Do **not** consolidate, reword, or remove prompts. |
| C2-guided crop skipping loses the C1 physical-brand override for misplaced products. | Every localized instance remains eligible for C1. Do **not** use signage to skip C1. |
| `Sam3VerboseCounter._image_state()` already caches one image embedding for all prompts on an image. | Do not re-implement image-feature caching; optimize the remaining per-prompt work. |
| `OnProductBrandingCue.emit()` serially processes instance/scale/variant OCR work. `Florence2OCRBackend._detect_model()` calls `generate()` one image at a time. | This is the primary batching opportunity. |
| `SelectiveOCRBackend` has a shared, ordered fallback-call budget reset per source image. | A batched implementation must reserve and consume budget in the same instance order. |
| There are no benchmark/profile artifacts beyond the supplied analysis. | All projected gains need measurement on the original nine-image P100 corpus; they must not be presented as achieved results. |

The repository already contains a bounded selective OCR cascade. That is different from the analysis document's direct “up to 200 VLM passes” configuration: it can reduce calls conditionally. This strategy does not change that policy. It preserves whichever OCR backend, fallback budget, C1 scales, CLAHE/polarity settings, and early-exit behavior the reference command uses.

## The required performance envelope

Using the supplied mean timings, a lossless implementation must satisfy approximately:

```text
15.22 / L0_speedup + 21.66 / C1_speedup + 0.05 < 10.0 seconds
```

C1 alone cannot reach the target: eliminating all C1 time would still leave about 15.27 s. Both L0 and C1 need substantial acceleration. For example, if unchanged-prompt L0 batching/compilation reaches 3x, C1 must reach about 4.5x. If L0 reaches only 2x, C1 needs roughly 9x. These are planning constraints, not performance claims.

## Strategy A — staged, output-preserving C1 batching (first priority)

### Design

Replace only the *dispatch* of expensive Florence calls with a bounded batched API, for example:

```text
Florence2OCRBackend.detect_preprocessed_batch(images) -> list[list[TextDetection]]
```

For a batch of crops, it should use the existing model and processor once, with:

- one `<OCR_WITH_REGION>` prompt per image;
- the same `max_new_tokens=1024`, `num_beams=3`, `do_sample=False`, dtype conversion, and post-processing as `_detect_model()`;
- one result list per input crop, returned in input order;
- no model, processor, precision, threshold, or parser substitution.

Build a `C1WorkItem` for each reference-eligible call containing the instance ID/index, crop coordinates, scale, variant name, backend method, and original call order. Process work in dependency stages. After each stage, apply the **existing** stopping predicates before scheduling the next one:

- a variant-level direct gazetteer match stops later variants at that scale;
- the current confidence rule controls whether later scales run;
- the paragraph pass remains in its current position;
- fallback eligibility remains based on the same reliable-primary-match test;
- fallback slots are reserved in original instance order before a batch is dispatched.

This means “all crops remain eligible” keeps its specification meaning: the optimizer introduces no new exclusion. It may not run a later variant only when the reference code would already have stopped that same crop.

### Selective-backend detail

`SelectiveOCRBackend` mutates `_fallback_calls`. The implementation exposes ordered batch helpers for the cheap primary member and the stronger fallback, reserving slots in input order. C1 uses the stronger batch helper only for backends that declare the operation order-independent. The shipped selective backend declares its fallback path order-sensitive because its existing polarity/variant loop may consume more than one slot for a crop; C1 therefore replays those fallback crops sequentially. This preserves the reference allocation exactly while still allowing direct stateless Florence C1 calls to batch.

### Safe operating behavior

- Start with `--c1-batch-size 2` and `--sam3-prompt-batch-size 2`, then try 4; measure peak VRAM on the P100 before raising the cap.
- On OOM or a backend error, split the batch and ultimately execute the existing single-image call. Never return an empty OCR result merely because an optimized batch failed.
- Preserve raw `TextDetection` order when replaying results; raw text boxes are consumed by C2 geometry and are part of the result contract.
- Batch CPU crop preparation (resize, sharpen, contrast, inversion) in a bounded worker pool ahead of GPU execution. These are pure PIL operations, so this overlap does not alter the pixels supplied to OCR. Keep the model invocations ordered/batched as above.

### Why this is valid

It runs the same input pixels through the same model and decoding parameters. The only intended change is grouping independent calls. GPU batched kernels can still introduce small floating-point differences, so the parity gate below is mandatory; if it fails, retain sequential generation for that environment.

## Strategy B — batch SAM3 *discrete* prompts, not prompt text (second priority)

Keep this exact prompt inventory and thresholds:

- class prompts: `sunglasses`, `eyeglasses`, `glasses`, `rimless glasses`;
- scene prompts: `people`, `person`, `faces of people`, `advertisements`, `posters`, `billboards`, `retail shelves`, `display shelves`, `shelf`.

The native runtime already embeds the image once and runs `forward_text([prompt])` plus grounding once per prompt. A new native `infer_prompts()` path can encode a small list of these **separate strings**, pair each text feature with the cached image features, and run grounding as a micro-batch. It must then:

- apply each prompt's existing threshold independently (including the rimless and scene thresholds);
- return detections tagged with their original prompt;
- preserve per-prompt box cleanup, cross-prompt de-duplication, and scene-filter order;
- fall back to the present one-prompt `infer()` implementation if parity or memory checks fail.

`Sam3Processor.set_text_prompt()` is currently single-item and its `FindStage` IDs are single-item. Therefore this requires a dedicated batched processor/model path; simply looping through it, reusing a compound string, or blindly using `set_image_batch()` is not a safe optimization.

A small batch (for example, the four eyewear prompts, then three prompt groups for the scene filter) is preferable to duplicating the image into a large batch. It retains prompt grounding semantics while reducing launch/decoder overhead. It must be accepted only after the exact parity tests pass, especially for `IMG_0298.jpg` and its top-shelf exclusion regression.

## Strategy C — compilation and service lifecycle (conditional)

The existing SAM3 runtime already uses `torch.inference_mode()`, evaluation mode, and CUDA AMP (FP16 on a P100). Do not alter those numerical settings.

The native SAM3 builder already has a vision-backbone compile path. `--sam3-compile` now opts into it; build/compiler failures automatically rebuild the uncompiled reference model. Warm-up remains outside steady-state timing, and the option must not be enabled for reporting until the installed Kaggle Torch/Triton/CUDA stack passes parity. Florence compilation remains a later experiment because its remote generation graph is more dynamic.

Keep models loaded in a persistent worker when processing multiple images only if both models fit safely on the P100. The present `release()` behavior deliberately frees SAM3 before C1 to protect VRAM. Retaining SAM3 is a throughput optimization, not a safe default for a 16 GB device. It must have an OOM-safe fallback to the present release/reload lifecycle.

## Smaller, fully execution-only improvements

These are lower impact but have no reason to change inference inputs when implemented carefully:

- Decode and RGB-convert the source image once per pipeline run, passing immutable copies to L0 OCR, C1 cropping, and dimension lookup instead of reopening it. Verify image bytes/pixels match the reference inputs.
- Cache *completed raw stage outputs* for repeat processing only. Key the cache by input SHA-256, crop coordinates and transforms, model/checkpoint and processor revision, backend settings, gazetteer/config, CUDA/Torch version, and AMP mode. Reuse only an exact key; never use perceptual hashes or approximate crop matching.
- Disable visualization only for a JSON-only throughput mode. It does not improve the reported `total_pipeline_seconds`, but it avoids annotation cost without changing the JSON result. Generate the annotation from the unchanged JSON when it is required.

## Explicitly rejected changes

The following are incompatible with the validity requirement:

- comma-separated, consolidated, shortened, reordered, or otherwise changed SAM3 prompts;
- changing SAM3/OCR thresholds or shelf-filter rules;
- reducing C1 scales, crop margin, contrast/polarity variants, fallback budget, beams, or `max_new_tokens`;
- C2 confidence thresholds, signs, or bay membership used to skip C1;
- switching to Florence base, quantization, TF32/FP8, approximate caching, or a different OCR model;
- treating matching final labels as sufficient while boxes, exclusions, signs, evidence, or decision paths differ.

## Parity gate and benchmark protocol

### Reference corpus

Use the same nine images and exact P100 command/configuration that produced the supplied baseline. Include `IMG_0298.jpg` as a mandatory regression: the top shelf must be detected and `inst_0002` must remain eligible for C1/attribution. Freeze checkpoint hashes, HuggingFace revisions, Python/Torch/Transformers/CUDA versions, config, CLI flags, and model precision.

### Equality requirement

For every optimized run, run `uv run python scripts/check_lossless_parity.py reference.json optimized.json` from `eyewear-localization/`. The checker ignores only timing metadata and the recorded performance knobs. The following must match exactly, including order where it is observable:

- `instances`, `excluded_instances` (including reasons and support);
- `text_detections`, `signs`, and `poster_regions`;
- all C1/C2/C4 evidence and support fields;
- all final outputs, including `product_brand`, `zone_brand`, `decision_path`, probabilities, and abstention;
- annotation pixels too, when visualization is part of the deliverable.

A final-brand-only comparison is insufficient. Any mismatch disables that optimization for the relevant backend/device and uses the sequential implementation.

### Measurements and rollout

1. Record warmed sequential p50/p90 per-stage time, GPU peak memory, calls per backend, and canonical outputs.
2. Introduce CPU preparation/decode reuse; compare outputs and measurements.
3. Introduce C1 batches at sizes 2 and 4; require parity for every image before selecting the fastest safe size.
4. Introduce discrete SAM3 prompt micro-batches; repeat the full parity suite, including shelf recall and exclusion reasons.
5. Trial compilation last. Measure warm and cold behavior separately.
6. Ship each path behind a feature flag with automatic sequential fallback and keep the golden corpus in CI.

The code, parity checker, and tests now cover C1 batch parity, ordered fallback accounting, discrete SAM3 prompt batching, malformed-batch replay, configuration/CLI controls, and compile fallback. A real P100 speedup and full nine-image parity result still require running the benchmark protocol with the gated checkpoints; no benchmark gain is claimed here. This gives a practical route to lower latency while making correctness—not a hoped-for speedup—the release criterion.
