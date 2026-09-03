# Performance feature analysis (issue #40)

## Decision

**`features-informative`**

CreativeIR structural features carry real signal about **engagement rate**
(like rate predicted at CV R² ≈ 0.43 with n=83) even though raw view counts
remain unpredictable from creative structure alone. The extraction pipeline
is deterministic, tested, and runs offline over the full dataset.

## Setup

- Input: 85 validated records from #38 (`dataset/records/`), no model API calls.
- Extraction: `scripts/performance.py::extract_features` — 47 usable features
  (structure, OCR volume, CTA/devices, dialogue/audio ratios, semantic role
  distribution, hook types, mechanisms, arc, format, camera, pacing, transitions).
- Persisted: `dataset/features.parquet` (features + targets per video_id).
- Model: Ridge on standardized features, 5-fold CV, vs a median baseline.

## Results

### Baselines (k-fold CV)

| Target | n | CV R² | CV MAE | Median-baseline MAE |
|---|---|---|---|---|
| log1p(views) | 83 | **−0.18** | 1.066 | 0.974 |
| log1p(like_rate) | 83 | **+0.43** | 0.012 | 0.019 |

### Strongest Spearman correlations (like_rate)

| Feature | ρ | p |
|---|---|---|
| dialogue_present_ratio | +0.70 | 7e-14 |
| sound_effects_ratio | −0.62 | 2e-10 |
| mech_social_proof | +0.61 | 7e-10 |
| role_challenge_ratio | +0.54 | 1e-07 |
| ocr_segment_count | +0.53 | 2e-07 |
| ocr_char_total | +0.51 | 5e-07 |
| duration_seconds | +0.46 (comment_rate) | 8e-06 |
| pacing_mean | −0.40 | 1e-04 |

Positive levers: the creator speaking on camera, social-proof framing,
challenge-style roles, visible text volume, longer runtime (comments).
Negative levers: dense sound-effect mixes, faster average pacing.

## Reading

- **Reach (views) is not a creative-structure problem** in this data — CV R² ≤ 0
  means structure alone carries no generalizable view signal. Views are driven
  by distribution factors absent from CreativeIR (series audience, posting
  cadence, platform traffic).
- **Like rate is a creative-structure outcome**: the regularized model
  generalizes (CV R² 0.43 > 0) with plausible, interpretable coefficients.
  This is the actionable target for the future generator: optimize the
  engagement-rate levers, not view-count fortune telling.

## Limitations

- n=85, one creator, one niche: coefficient stability is unverified across niches.
- Observational data: dialogue presence correlates with creator identity and
  video type; nothing here is causal.
- ~47 features on 83 rows: regularized + CV mitigates but multiple-comparison
  risk remains for the correlation table (p-values are indicative only).
- Rates are computed from public metadata (views/likes), which TikTok rounds.

## Recommendation

`features-informative` — keep the feature extractor in the pipeline
(`extract_features` per new record), collect more creators/niches before
training anything heavier, and treat like-rate structure as the modeling
target for creative generation.
