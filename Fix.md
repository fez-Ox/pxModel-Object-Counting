# Fix Specification v2 — Cue Calibration & Deterministic Brand Attribution

**Instructions to the implementing agent:** This document amends Specification v1. It **supersedes** v1 §4 (C1, C2, L2), §6 (fusion math), and the L3 usage policy; all other v1 sections remain in force. Implement milestones in order, with tests per acceptance criterion. Where v1 and v2 conflict, v2 wins.

---

## 0. Goals

1. C1 (on-product branding): replace EasyOCR + generative super-resolution with patch localization + closed-set classification; abstain when pixels are unresolvable.
2. C2 (signage): replace the learned/heuristic scoring soup with three **deterministic, decoupled, pure-function** stages (Parse → Attach → Assign), regression-tested on synthetic renders.
3. L2: replace softmax fusion with a **precision cascade** (decision list); keep legacy fusion only in an experimental branch, promotable solely by ablation.
4. Measurement: per-cue standalone eval, forward selection, evidence viewer, error taxonomy — no change ships without measured improvement on the stratified val set.

## 1. Invariants (MUST, additive to v1 I1–I6)

- **I7.** No generative super-resolution anywhere in C1 (SR invents glyph strokes). Lanczos/bicubic ≤ ×3 + CLAHE/unsharp only.
- **I8.** No VLM calls inside C2. C2 is deterministic geometry; same input ⇒ bitwise-identical output.
- **I9.** Every final label carries a `decision_path` trace (which cue fired, with support artifacts).
- **I10.** Cues may only emit evidence; the cascade is the single place where labels are decided.

## 2. Label Semantics

Ground truth and outputs distinguish two notions (resolves misplaced-item ambiguity):
- `product_brand`: brand of the physical object (C1, C4 evidence).
- `zone_brand`: brand of the display zone containing the instance (C2, C3 evidence).
- `final_brand = product_brand if available else zone_brand else "unknown"`.
Evaluation uses `product_brand` as primary GT when on-product branding is present/known, else `zone_brand`.

## 3. C1 — Rebuild (recognition, not OCR)

Pipeline per instance (pure stages, frozen JSON between them):

1. **Patch localization.** Small detector (YOLOv8-n/Faster-RCNN) trained on branding patches (temple logo, lens etching). Fallback heuristic: temple strip = outer 30% of crop width at mid-height; etching = lens corners.
2. **Resolvability gate.** Estimate glyph height (patch bbox / connected components). If `< min_glyph_px` (default 8): emit `{brand: none, conf: 0, reason: "unresolvable"}` and log to the capture-side audit. **Do not guess.**
3. **Enhancement.** Grayscale → CLAHE → unsharp; Lanczos ×2–3 max (I7).
4. **Closed-set classifier.** ViT-small/CNN fine-tuned on **synthetic renders**: brand strings/logos in true fonts (incl. cursive Cartier, Moncler sans), metallic/acetate backgrounds, curvature warp, blur/noise, downsampled to glyph heights 8–24 px. Classes = gazetteer ∪ {none}. Emit top-2 probs.
5. **Optional VLM reader** (cached, structured `{brand|none, raw_text}`), invoked only when classifier prob is in the gray band `[t1_lo, t1_hi]`. Agree ⇒ conf boost; disagree ⇒ accept classifier only if `p ≥ t1_hi`, else `none`.

Standalone acceptance: top-1 ≥ 0.90 on resolvable branded crops; FPR ≤ 0.05 on unbranded crops; resolvability flags match a manual glyph-height audit on 100 crops.

## 4. C2 — Deterministic Three-Stage Rebuild

All stages are pure functions with frozen schemas; all parameters live in config; unit-tested per stage.

### Stage 1 — Parse (regions)
```
gray = grayscale(display_roi)
col_dark[c] = fraction of rows with gray < dark_thr          # vertical dividers
dividers = column runs where col_dark > 0.6, width ∈ [w_min, w_max]
bays = gaps between consecutive dividers ∩ roi
if len(bays) < 2:                                            # row-fixture fallback
    row_light[r] = fraction of cols with gray > light_thr    # lit shelves
    rows = bands between shelf runs where row_light > 0.7
regions = [{region_id, type: bay|row, bbox}]
```
### Stage 2 — Attach (sign → region)
- `bay_header`: sign bottom ≤ bay top + header_tol AND `width(sign ∩ bay) ≥ 0.7·width(sign)`; pick max-overlap bay; `attach_conf` = overlap ratio.
- `row_label`: sign at row-band left/right extremity AND `height(sign ∩ row) ≥ 0.5·height(sign)`.
- `shelf_edge_tag`: sign height < tag_h AND adjacent to a row; scope x-extent = tag x-span extended to mid-gaps of neighboring tags; region = row ∩ that x-extent.
- Sign inside `poster_regions[]`, or failing all rules ⇒ `scope: none`. **A sign may label nothing.**
### Stage 3 — Assign (instance → region)
- Containment of centroid in region (or sub-region). `d` = distance to nearest region boundary;
  `assign_conf = 0.5 + 0.5·min(1, d / (0.15·region_width))`; if centroid falls in two regions ⇒ `assign_conf *= 0.5` and mark ambiguous.
- Evidence: `{instance, brand, conf = attach_conf · assign_conf, support: {region_id, sign_id}}`.
- `unambiguous` ⇔ exactly one sign's scope contains the centroid and `assign_conf ≥ 0.9`.

Regression acceptance (synthetic renders, ≥ 500 per family: columns/rows/mixed/tags/posters): region-count accuracy ≥ 0.95; attach accuracy ≥ 0.95; assign accuracy ≥ 0.98 on unambiguous; **bitwise determinism across runs.**

## 5. L2 — Precision Cascade (replaces softmax fusion)

```python
def decide(inst):
    p, z = product_evidence(inst), zone_evidence(inst)     # C1/C4 vs C2/C3
    if p.conf >= t1:                                        # on-product = truth
        return final(p.brand, path="C1",
                     audit_flag=(z.conf >= t2 and z.brand != p.brand))
    if z.unambiguous and z.conf >= t2:
        return final(z.brand, path="C2")
    if c3_majority(inst, k=2, c_min=0.8):                   # agreeing high-conf neighbors
        return final(brand, path="C3", conf=min_neighbor_conf * 0.8)
    if c4.sim >= t4:                                        # style prior, solo, high bar
        return final(c4.brand, path="C4")
    return UNKNOWN(abstained=True, path="none")
```
Defaults (tune per-cue on val for precision): `t1=0.80`, `t2=0.85`, `t4=0.85` (cosine), C3 `k=2, c_min=0.80, decay=0.80`.
**Legacy fusion** stays in branch `exp/softfusion`; promote only if stratified val shows ≥ 1 pt selective-accuracy gain at equal-or-better coverage and K4 abstention.

## 6. Measurement Protocol (gate for every change)

- **Per-cue standalone eval** on the K1–K4-stratified val: selective accuracy, coverage, K4 abstention rate, ECE. A cue with standalone selective accuracy < 0.90 is barred from the cascade.
- **Forward selection:** add cues in descending standalone accuracy; keep iff val selective accuracy improves ≥ 1 pt without coverage loss