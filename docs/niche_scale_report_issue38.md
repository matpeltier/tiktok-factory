# Niche-scale collection report (issue #38)

## Decision

**`dataset-ready-for-modeling`**

85 validated CreativeIR v0.1 records from one micro-niche (@gorilloyt, cozy
Minecraft building shorts), 97% success on decompilable videos, total spend
**$9.18** (≈$0.108/video), well within the issue budget cap.

## Collection summary

| Metric | Value |
|---|---|
| Videos collected (yt-dlp, impersonation) | 120 |
| Validated CreativeIR records | **85** |
| Non-video posts excluded (`skipped_no_video_stream`) | 29 (24% — slideshow posts, zero model spend) |
| Failed | 3 (2× payload-too-large from the pre-fix proxy threshold, 1× model quirk) |
| Unprocessed at budget cap | remainder left resume-safe for a later run |
| Views range | 45k – 12.6M (strong variance for future modeling) |
| Duration range | 12s – 121s (mean 73s) |
| Shots per video | 1 – 45 (mean 23.8) |

## Cost & throughput

- Total recorded spend: **$9.18** across the pilot (#5) and this run
  (cap `$9.20` enforced in code by `pilot.py Budget`).
- Mean $0.108/video, max $0.27 (longest/most-shot videos).
- Mean latency 332s sequential-equivalent; batch ran with **4 parallel workers**.

## Engineering findings during scale-up (all fixed deterministically)

1. **Provider payload limit**: OpenRouter/Google AI Studio rejects request
   bodies > 20MB. Because the base64 data URL inflates files by ~4/3, videos
   over ~14MB must be proxied. OpenRouter sometimes reports this as an
   embedded error inside a 200 response — the client now surfaces embedded
   `error.code` (413) instead of reporting an empty model response.
2. **HEVC downloads**: TikTok serves 1080p as HEVC (`bytevc1`); Google fails
   to decode HEVC (empty responses). The deterministic proxy now re-encodes
   any non-H.264 input.
3. **Proxy sizing**: the original 720p proxy for a 90s video could still
   exceed 14MB; proxies now target 540px width (all proxies ≤ 13MB, verified).
4. **ffmpeg filter arg pitfall**: `scale='min(720,iw)':-2` does not downscale
   (comma breaks the filter expression); the target width is now decided in
   Python and passed explicitly.

Facts remain authoritative from the ORIGINAL file (ffprobe/PySceneDetect);
proxies only affect what the model sees.

## Dataset readiness assessment

- 85 records with full provenance per video: raw MP4, raw yt-dlp metadata,
  normalized metadata, perception (ffprobe + PySceneDetect + frames), raw
  model outputs per pass, coercion logs, validated CreativeIR, per-pass
  token/cost usage, status/latency record.
- `dataset/index.parquet`: 117 rows linking records to source/performance
  metadata (views/likes/comments/shares), status, cost, latency, versions.
- View-count variance (45k–12.6M) across structurally diverse videos
  (1–45 shots) is sufficient to start performance-correlation work.
- Remaining backlog: ~3 failed records re-runnable for ~$0.40, plus the rest
  of the creator catalog (1,000 listings) for later expansion.

## Recommendation

Proceed to performance modeling on the 85-record dataset; expand collection
(opportunistic, budget-capped) only if modeling requires more variance.
