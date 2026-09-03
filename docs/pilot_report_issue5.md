# Dataset pilot report (issue #5)

## Decision

**`ready-for-niche-scale-collection`**

15/15 decompilable videos succeeded (100%), with zero schema validation
failures in the final run, deterministic provenance for every fact, and a mean
extraction cost of **$0.13 per video** (~62k tokens/video).

## Setup

| Item | Value |
|---|---|
| Micro-niche | Cozy Minecraft village/animal building shorts (same niche as the validated #4 sample) |
| Creator | @gorilloyt (single creator, consistent format) |
| Videos collected | 24 (yt-dlp + browser impersonation) |
| Valid decompilable videos | 15 |
| Non-video posts excluded | 9 (audio-only slideshow posts, detected via ffprobe, categorized `skipped_no_video_stream`) |
| Model | `google/gemini-3.8-flash` via OpenRouter (video inline base64) |
| Decompilation path | Exactly the validated #4 flow (`issue-4-multistep-perception-v0.1`), reused from `scripts/pipeline.py` |
| Prompt versions | `openrouter-gemini-shot-analysis-v0.1` + `openrouter-gemini-global-synthesis-v0.1` |
| Schema | CreativeIR v0.1 (`schemas/creative_ir_v0_1.json`), Draft 2020-12 + temporal integrity checks |

## Results (15 successful records)

- **Success rate**: 15/15 decompilable videos (100%). The single earlier
  failure (model returned 10 shots for 29 detected hard cuts) succeeded on a
  re-run via the built-in full-flow retry.
- **Shot structure**: 19–40 shots per video (mean 28.5) on 62–120s videos,
  boundaries from PySceneDetect, exact media facts from ffprobe.
- **Cost**: $1.97 total, mean $0.131/video, range $0.075–$0.237
  (44k–110k tokens/video).
- **Latency**: mean 5m22s, min 2m31s, max 12m45s (sequential single-worker run).
- **JSON repair passes**: 0 needed in the final run (the repair mechanism from
  #4 remains as a safety net).
- **Quality spot-checks**: exact on-screen OCR captured (build series titles,
  material labels, countdowns, CTAs), audio descriptions conservative
  (`identity_known=false` everywhere, no invented song names), detailed
  reconstruction briefs, hook/narrative/marketing inference consistent with
  captions.

## Recurring error categories observed (and mitigations added during pilot)

| Category | Observed | Mitigation (deterministic, logged) |
|---|---|---|
| Enum drift (`social_proof` as text role, `delayed_payoff` as mechanism) | 2 videos | `coerce_unknown_enum_values` maps to schema fallbacks/aliases; log persisted per video (`creative_ir.coercions.json`) |
| Schema meta echo (`$schema` key) | 1 video (large, 31MB) | `strip_meta_keys` drops `$`-prefixed keys |
| Provider truncation on large uploads | 1 video (31MB) | 720p deterministic proxy (`video.proxy.mp4`) for >20MB uploads; facts still measured on the original |
| Malformed JSON structure | rare (2 runs total) | text-only repair pass from #4 |
| Slideshow posts (no video stream) | 9/24 collected | detected at run time, categorized `skipped_no_video_stream` |

## Cost / throughput estimate for niche-scale collection

At $0.13 and ~5.4 min per video (single worker): 1,000 videos ≈ **$130** and
~90 GPU-free hours (parallelizable; model calls dominate latency, so 8 workers
would bring wall-clock to ~11h). Slideshow posts (~37% of this creator's
recent output) are filtered before any model spend.

## Artifacts

- `dataset/records/{video_id}/` — raw MP4, `metadata.raw.json` (yt-dlp),
  `metadata.json` (normalized), `perception.json`, per-scene frames,
  raw model outputs per pass (incl. retries), `creative_ir.coercions.json`
  when applicable, `creative_ir.parsed.json` (validated CreativeIR),
  `creative_ir.usage.json` (per-pass tokens + cost), `record.json`
  (status/latency/cost/versions).
- `dataset/index.parquet` — 24 rows linking records to source/performance
  metadata (views/likes/comments/shares), status, cost, latency, versions.
- Inspection: `notebooks/04_dataset_pilot.ipynb` (index summary, random
  side-by-side review vs source MP4, failure categories).

## Note on dataset persistence

`dataset/` is git-ignored (652MB of media); records persist on disk for
reprocessing/audits. The index can be rebuilt at any time with
`python scripts/pilot.py index`.
