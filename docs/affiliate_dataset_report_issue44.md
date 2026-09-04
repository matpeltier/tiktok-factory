# Affiliate product dataset #1: vegetable chopper / CleanTok (issue #44)

## Decision

**`product-dataset-ready`**

17 strictly product-relevant videos from 6 distinct creators, decompiled and
validated (0 schema errors, 0 final failures) for **$1.39** total
($0.082/video). The dataset reveals strongly convergent creative patterns
across independent creators selling the same product — the direct foundation
for generating affiliate creative.

## Dataset

| Metric | Value |
|---|---|
| Product stratum | vegetable chopper (Fullstar + equivalent push-press choppers), caption-matched |
| Videos decompiled | 17 |
| Creators | 6 (@dailymusthaves46 ×5, @amazonkitchengems ×6, @michoacana2001 ×2, @finders09 ×2, @glenxda ×2, @alden.barstad ×1) |
| Views range | 453 – 7,497 |
| Duration | ~10–30s (mean 8.5 shots/video) |
| Excluded | 37 roundup/listicle videos (multi-product "kitchen finds") — kept on disk as `skipped_product_mismatch`, zero model spend |
| Spend | $1.39 (budget-capped run; 3 JSON repair passes, 0 needed retries after fixes) |

## Convergent creative patterns (what independent creators share for the SAME product)

| Pattern | Frequency | Reading |
|---|---|---|
| Hook = **product_promise** | 16/17 | open on what the product does for you ("chop in seconds") |
| Arc = **problem_proof_cta** | 11/17 | pain (chopping) → visible proof (dice demo) → CTA |
| Device = **personal_experience** | 11/17 | "I tested / I use it daily" framing |
| Device = **before_after** | 8/17 | messy veggies → diced bowl transformation |
| Hook = **visual_novelty** | 10/17 | the satisfying press-and-dice moment as the visual hook |
| Dialogue on camera | 12/17 | consistent with the #40 like_rate finding (+0.70) |
| Mean OCR segments | 9.3/video | dense on-screen text (dimensions, claims, "Amazon's Choice") |
| On-screen CTA text | **0/17** | CTA is verbal/bio ("Comment for Link", "Link in Bio") — NOT rendered text |

**Format split**: product_ad 11 / demonstration 6.
**Top performers** are the shortest (11–13s), demonstration-format,
visual_novelty-hooked videos.

## Affiliate brief implications (for generation)

1. Structure every generated video as **problem → proof → CTA** in ≤ 15s for
   drafts (the top performers are 11–13s).
2. Open on the **product promise + the satisfying visual moment** (the press-
   and-dice), not on the creator's face or a greeting.
3. Include **creator dialogue** (native audio with Veo-class models) — the
   dataset norm and our strongest like_rate lever.
4. Keep the **CTA verbal/bio-side**; do NOT burn a shot on rendered CTA text —
   no top video does.
5. Show **personal experience + before/after** rather than generic product
   beauty shots.

## Engineering notes

- `pilot.py collect/run --keywords`: product-stratum filtering at listing
  level (no download cost for off-product videos) and at run level (no model
  spend); non-matching records persist as `skipped_product_mismatch`.
- New deterministic fill: empty strings violating schema `minLength: 1`
  (model emits `""` when a video has no offer/CTA) are replaced with
  `not_specified`, logged in `creative_ir.coercions.json`.

## Limitations

- 17 videos is enough for pattern extraction, not for statistical modeling
  (the #40 style regression needs ~50+).
- Views in this stratum are modest (max 7.5k) — these are micro-creator
  affiliate videos; the "winning" patterns here are conversion-oriented
  formats, not mass-viral ones.
- Product stratum includes equivalent choppers (Fullstar + generic push-press):
  deliberate, matching the affiliate reality where creators review the
  product family.
