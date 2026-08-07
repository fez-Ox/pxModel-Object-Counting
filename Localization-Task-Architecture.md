# Robust Eyewear Brand Attribution — System Specification v1.0

**Instructions to the implementing agent:** Build the system described below. Follow the invariants strictly, implement milestones in order, and write tests for each acceptance criterion. Where a specified model is unavailable, substitute a heuristic fallback with reduced `reliability` — never crash, never hardcode layout assumptions.

---

## 1. Problem Statement

Given a retail display image, output for every eyewear instance: a segmentation mask, a brand label, and a calibrated confidence — with `unknown` as a valid first-class answer.

The system must be robust across all of these conditions, with no layout-specific code paths:

| ID | Condition | Expected dominant evidence |
|----|-----------|---------------------------|
| K1 | Brand signage geometrically associated by **column** (headers above bays) | C2 |
| K2 | Brand signage associated by **row** (row-end labels, shelf-edge tags) | C2 |
| K3 | No signage association; **branding on the product** (temple text, logos) | C1 |
| K4 | **No branding anywhere** | Abstention → `unknown` |

## 2. Invariants (MUST)

- **I1.** SAM 3 is used only as a class-agnostic localizer. Brand names never appear in its prompts.
- **I2.** Every cue emits `(instance_id, brand, confidence, support)`; hard labels are decided only in the fusion stage.
- **I3.** `unknown` is a designed output, not a failure mode. Tests include cases where `unknown` is correct.
- **I4.** Display geometry (columns/rows/grids) is inferred per image, never assumed.
- **I5.** All modules communicate via the JSON schemas in §5.
- **I6.** Graceful degradation: missing models → heuristic fallback with lowered reliability.

## 3. Architecture Overview

```
image
  │
  ├─► L0 Perception ──► instances[], signs[], poster_regions[]
  │
  ├─► C1 On-product branding ──┐
  ├─► C2 Signage scope         ├──► evidence[]
  ├─► C4 Style prior           ──┘
  │
  ▼
L2 Fusion (belief update) ─► C3 continuity smoothing ─► Decision (τ, margin)
  │
  ├─ uncertain? ─► L3 VLM auditor ─► re-fuse ─► Decision
  ▼
outputs[]
```

## 4. Module Specifications

### L0 — Perception Frontend
- **Instances:** SAM 3 with prompts `sunglasses`, `eyeglasses`, `glasses` (+ `rimless glasses` at lower score threshold). Deduplicate masks (IoU-NMS), filter by min area/aspect. Compute `bbox`, `centroid`, mask.
- **Kind/tint:** `lens_tint_score` = mean luminance of lens interior vs. local shelf background (dark/tinted ⇒ sunglasses). Optional, not required for brand.
- **Text:** whole-image text detection + OCR. Brand-match each string: normalize (lowercase, strip diacritics), fuzzy-match against gazetteer (token containment or Levenshtein ≤ 1). Emit `signs[]` (brand text) — note: detection ≠ labeling scope.
- **Posters/ads:** detect large printed panels (faces/graphics, e.g., classifier or VLM query). Emit `poster_regions[]`; used to (a) exclude false-positive instances, (b) force C2 scope = `none` for signs inside them.
- **Fallbacks:** if SAM 3 unavailable → open-vocab detector; if OCR weak → ensemble recognizers.

### C1 — On-product Branding (reliability 0.95)
For each instance: crop with margin → upscale 2–4× (super-resolution) → run sub-cues:
1. Fine-text OCR on temples/lens etchings; gazetteer fuzzy-match.
2. Logo detector (fine-tuned detector or template matching) for brand emblems (e.g., Cartier script/screw motifs, Moncler badge).
3. Optional zoomed VLM query, structured answer only.
Confidence: max of sub-cues; +boost if ≥2 sub-cues agree. `support` = crop bbox + raw text/detection.

### C2 — Signage–Region Association (reliability 0.85)
Purpose: generalize K1/K2 without layout assumptions.
1. **Region parse:** detect shelves (horizontal light bands) and dividers (vertical dark bands) via line/morphology detection; fallback: proximity clustering of instances.
2. **Scope hypotheses per sign:** `bay_header` (region = bay below), `row_label` (region = row band), `shelf_edge_tag` (region = item row adjacent to tag), `hanging_sign`, or `none`.
3. **Score hypotheses** by: alignment/containment of candidate region, attachment (sign flush to region boundary), sign-size vs region-size, distance, and poster overlap (overlap with `poster_regions[]` ⇒ `none`).
4. Implementation options (any one, swappable): (a) heuristic scorer tuned on synthetic data; (b) small classifier trained on synthetic renders (§7); (c) per-sign VLM query returning structured JSON. All must emit `scope.type`, `scope.region_bbox`, `scope.confidence`.
5. **Evidence:** instance centroid ∈ scope region ⇒ `(brand, conf = scope_confidence × containment_confidence)`.
A sign may correctly label *nothing* (`none`) — e.g., posters.

### C4 — Style Prior (reliability 0.35)
Embed instance crops (DINOv2/CLIP, fine-tuned if data allows); retrieve against per-brand reference gallery; emit evidence only if similarity > threshold. Never decisive alone; tie-breaker/booster.

### C3 — Planogram Continuity (smoothing, post-fusion)
Graph over instances; edges = same-shelf adjacency (y-band + distance) and appearance similarity > threshold. Gated smoothing:
`p_i ← (1−λ)·p_i + λ·Σ_j w_ij·p_j` over neighbors **with** `p_unknown(j) < gate`. (Gating prevents fabricating confidence in unbranded clusters.) λ ≈ 0.25.

### L2 — Fusion & Decision
See §6.

### L3 — VLM Auditor (optional; reliability 0.70)
Trigger when final decision is in the uncertainty band, cues conflict (top two brands within margin), or C2 parse confidence is low. Input: full image + instance crop + candidate hypotheses. Output: structured cue evidence. Re-fuse once; if still uncertain → `unknown`. VLM is one voter, never sole truth.

## 5. Data Schemas

```json
// instance
{"id":"inst_0001","bbox":[x,y,w,h],"centroid":[cx,cy],"mask_rle":"...","lens_tint_score":0.83}
// sign
{"sign_id":"s_01","text":"Cartier","brand":"cartier","bbox":[...],
 "scope":{"type":"bay_header","region_bbox":[...],"confidence":0.92}}
// evidence
{"instance_id":"inst_0001","brand":"cartier","confidence":0.91,"cue":"C1","support":{...}}
// output
{"instance_id":"inst_0001","brand":"cartier","abstained":false,
 "probabilities":{"cartier":0.87,"moncler":0.02,"unknown":0.11},"evidence":[...]}
```

## 6. Fusion Math (default; replaceable by learned combiner)

```
scores[b] = Σ_e reliability(cue_e) · confidence(e)        # per brand
scores[unknown] = δ                                       # unknown prior (≈0.35)
probs = softmax(scores / T)
brand* = argmax_b≠unknown probs[b]
accept iff probs[brand*] ≥ τ  AND  probs[brand*] − probs[runner-up] ≥ margin
       AND  probs[brand*] > probs[unknown]
else brand = "unknown", abstained = true
```
With zero evidence, `unknown` wins by construction. Apply C3 smoothing to `probs` before the decision.

```python
def run(image):
    inst, signs, posters = L0(image)
    ev  = C1(inst) + C2(inst, signs, posters) + C4(inst)
    out = decide(smooth_C3(fuse(ev), inst))
    if any(uncertain(o) for o in out):
        ev += L3(image, [o for o in out if uncertain(o)]); out = decide(fuse(ev))
    return out
```

## 7. Configuration & Data Requirements

```yaml
gazetteer: [moncler, cartier, ...]
cue_reliability: {C1: 0.95, C2: 0.85, C4: 0.35, L3: 0.70}
fusion: {temperature: 0.6, unknown_prior: 0.35, tau: 0.6, margin: 0.15}
smoothing: {lambda: 0.25, gate_punknown: 0.5}
cascade: {use_vlm_audit: true, uncertainty_band: [0.45, 0.70]}
```
- **Synthetic display renderer:** parameterized layouts ∈ {columns, rows, grid, mixed, islands}, sign placements/scopes, posters, unbranded fraction; emits ground-truth JSON. Used to train/tune C2 and for integration tests.
- **Logo dataset** per brand (fine-tune detector); **style gallery** per brand; **real-image calibration set** stratified by K1–K4 (fit T, τ via selective accuracy).

## 8. Evaluation Protocol

- Localization: recall @ IoU ≥ 0.5.
- Attribution: **selective accuracy** (accuracy on non-abstained) ≥ 0.90; **coverage** ≥ 0.80 on K1–K3; **abstention rate** ≥ 0.90 on K4.
- Calibration: reliability diagram / ECE on fused probabilities.
- Ablations: each cue disabled, per-condition breakdown.

## 9. Edge Cases

| Case | Handling |
|------|----------|
| Brand text inside poster/ad | C2 scope = `none` via poster detector; instances in posters excluded |
| Multiple brands in one row | Shelf-edge tags; scope region = contiguous aligned item group |
| Rimless/transparent frames missed | Lower SAM 3 threshold + `rimless glasses` prompt |
| Cursive/ornate logos | Recognizer ensemble + VLM confirmation |
| On-product brand not in gazetteer | Mint new label from OCR string; cluster others via C4 |
| Same brand in disjoint zones | Scopes are per-sign, not per-brand — no conflict |
| Reflections / mirrors | Region sanity check; drop instances outside parsed display |
| Truly unbranded item | All cues weak → abstain (correct behavior) |

## 10. Milestones & Acceptance Criteria

1. **Scaffold:** schemas, config, model registry, I/O. ✔ tests: schema validation.
2. **L0:** SAM 3 wrapper, OCR, tint, poster detector. ✔ recall ≥ 0.90 @ IoU 0.5 on sample set; all header signs detected.
3. **C1:** crop/upscale/OCR/logo detector. ✔ ≥ 0.90 precision on logo-bearing crops.
4. **Renderer + C2:** synthetic renderer first; scope scorer. ✔ scope-type accuracy ≥ 0.90 across layout families; poster signs → `none` ≥ 0.95.
5. **C4 + C3.** ✔ propagation improves selective accuracy on K1–K2 without reducing K4 abstention.
6. **Fusion + decision + outputs.** ✔ §8 targets on calibration set.
7. **L3 auditor.** ✔ uncertain-band items resolved or correctly abstained.
8. **Eval harness, calibration, docs.** ✔ full §8 suite green.

## 11. Non-Goals

Counterfeit detection; multi-store inventory sync; real-time latency guarantees; brand attribution for non-eyewear categories (architecture is portable, but out of scope).