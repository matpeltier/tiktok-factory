"""Dataset pilot runner for issue #5: generalize the validated #4 flow.

One micro-niche (cozy Minecraft animation shorts from a fixed creator list),
20-50 videos, full artifact preservation per video and a Parquet index.

Subcommands:
  collect  download videos + public metadata for each configured creator
  run      decompile every collected video through the validated pipeline
  index    build dataset/index.parquet from per-video records
  report   print quality / failure / latency / cost summary
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pipeline import PIPELINE_VERSION, decompile_video

DATASET_DIR = REPO_ROOT / "dataset"
RECORDS_DIR = DATASET_DIR / "records"
SCHEMA_PATH = REPO_ROOT / "schemas" / "creative_ir_v0_1.json"

# One micro-niche: cozy Minecraft village/animal animation shorts (same niche
# as the validated #4 sample video from @gorilloyt).
CREATORS = ["gorilloyt"]
COLLECT_LIMIT = 25


def _ytdlp(args: list[str], timeout: int = 300) -> str:
    # Run yt_dlp with the current interpreter so venv installs are used
    # even when a broken system-wide yt-dlp shadows it on PATH.
    cmd = [sys.executable, "-m", "yt_dlp", "--impersonate", "chrome", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed ({result.returncode}): {result.stderr[-500:]}")
    return result.stdout


def normalize_metadata(info: dict, url: str) -> dict:
    """Reduce raw yt-dlp info to the metadata shape used by the pipeline prompts."""
    hashtags = info.get("hashtags") or []
    publication_date = info.get("upload_date")
    if publication_date and len(publication_date) == 8:
        publication_date = f"{publication_date[:4]}-{publication_date[4:6]}-{publication_date[6:]}"
    return {
        "source_url": url,
        "video_id": info.get("id"),
        "creator": info.get("uploader") or info.get("channel") or info.get("creator"),
        "caption": info.get("description") or info.get("title") or "",
        "hashtags": hashtags,
        "publication_date": publication_date,
        "duration_seconds": info.get("duration"),
        "views": info.get("view_count"),
        "likes": info.get("like_count"),
        "comments": info.get("comment_count"),
        "shares": info.get("repost_count"),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def list_creator_videos(creator: str, limit: int) -> list[str]:
    """Return up to `limit` recent video URLs for a creator profile."""
    out = _ytdlp(["--flat-playlist", "--print", "%(id)s", f"https://www.tiktok.com/@{creator}"], timeout=600)
    ids = [line.strip() for line in out.splitlines() if line.strip()]
    return [f"https://www.tiktok.com/@{creator}/video/{vid}" for vid in ids[:limit]]


def collect_video(url: str, record_dir: Path) -> None:
    """Download one video and its raw public metadata into record_dir."""
    record_dir.mkdir(parents=True, exist_ok=True)
    _ytdlp(
        [
            "-f", "b[ext=mp4]/b",
            "--merge-output-format", "mp4",
            "-o", str(record_dir / "video.mp4"),
            "--write-info-json",
            "--no-playlist",
            "--no-simulate",
            url,
        ],
        timeout=600,
    )
    info_files = [p for p in record_dir.glob("video*.info.json") if "metadata.raw" not in p.name]
    if not info_files:
        raise RuntimeError(f"yt-dlp did not write an info json for {url}")
    info = json.loads(info_files[0].read_text(encoding="utf-8"))
    info_files[0].rename(record_dir / "metadata.raw.json")
    (record_dir / "metadata.json").write_text(
        json.dumps(normalize_metadata(info, url), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def cmd_collect(args: argparse.Namespace) -> None:
    limit = args.limit
    urls: list[str] = []
    for creator in (CREATORS if args.creators is None else args.creators):
        creator_urls = list_creator_videos(creator, limit)
        print(f"{creator}: {len(creator_urls)} recent videos")
        urls.extend(creator_urls)
    urls = urls[:limit]
    print(f"Collecting {len(urls)} videos into {RECORDS_DIR}")
    for i, url in enumerate(urls, 1):
        video_id = url.rsplit("/", 1)[-1]
        record_dir = RECORDS_DIR / video_id
        if (record_dir / "video.mp4").exists() and (record_dir / "metadata.json").exists():
            print(f"[{i}/{len(urls)}] {video_id}: already collected")
            continue
        try:
            collect_video(url, record_dir)
            print(f"[{i}/{len(urls)}] {video_id}: collected")
        except Exception as exc:
            print(f"[{i}/{len(urls)}] {video_id}: FAILED to collect: {exc}")


def load_record_state(record_dir: Path) -> dict:
    record_path = record_dir / "record.json"
    if record_path.exists():
        return json.loads(record_path.read_text(encoding="utf-8"))
    return {}


def has_video_stream(video_path: Path) -> bool:
    """True if the file contains at least one video stream (slideshow posts have none)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video_path)],
        capture_output=True,
        text=True,
    )
    return "video" in out.stdout


def cmd_run(args: argparse.Namespace) -> None:
    from pipeline import DEFAULT_MODEL

    record_dirs = sorted(d for d in RECORDS_DIR.iterdir() if d.is_dir()) if RECORDS_DIR.exists() else []
    if args.limit:
        record_dirs = record_dirs[: args.limit]
    print(f"Processing {len(record_dirs)} records with model {DEFAULT_MODEL}")
    for i, record_dir in enumerate(record_dirs, 1):
        video_id = record_dir.name
        video_path = record_dir / "video.mp4"
        metadata_path = record_dir / "metadata.json"
        parsed_path = record_dir / "creative_ir.parsed.json"
        if parsed_path.exists():
            print(f"[{i}/{len(record_dirs)}] {video_id}: already decompiled")
            continue
        if not video_path.exists() or not metadata_path.exists():
            print(f"[{i}/{len(record_dirs)}] {video_id}: missing inputs, skipping")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not has_video_stream(video_path):
            record = {
                "video_id": video_id,
                "source_url": metadata.get("source_url"),
                "status": "skipped_no_video_stream",
                "failure_reason": "source is an audio-only slideshow post (no video stream)",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            (record_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(f"[{i}/{len(record_dirs)}] {video_id}: skipped (audio-only slideshow post)")
            continue
        state = load_record_state(record_dir)
        attempts = state.get("attempts", 0) + 1
        started = time.monotonic()
        record = {
            "video_id": video_id,
            "source_url": metadata.get("source_url"),
            "schema_version": "0.1",
            "prompt_versions": "openrouter-gemini-shot-analysis-v0.1+openrouter-gemini-global-synthesis-v0.1",
            "pipeline_version": PIPELINE_VERSION,
            "model": DEFAULT_MODEL,
            "attempts": attempts,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            usage = decompile_video(video_path, metadata, record_dir, SCHEMA_PATH, model=DEFAULT_MODEL)
            elapsed = time.monotonic() - started
            record.update(
                {
                    "status": "ok",
                    "latency_seconds": round(elapsed, 1),
                    "cost_usd": usage.get("cost_usd_total"),
                    "tokens_total": sum(
                        (c["usage"].get("total_tokens") or 0) for c in usage.get("calls", [])
                    ),
                    "repair_passes": sum(c.get("repairs", 0) for c in usage.get("calls", [])),
                }
            )
            print(f"[{i}/{len(record_dirs)}] {video_id}: OK in {elapsed:.0f}s, cost=${usage.get('cost_usd_total')}")
        except Exception as exc:
            elapsed = time.monotonic() - started
            record.update(
                {
                    "status": "failed",
                    "latency_seconds": round(elapsed, 1),
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "failure_traceback": traceback.format_exc()[-2000:],
                }
            )
            print(f"[{i}/{len(record_dirs)}] {video_id}: FAILED after {elapsed:.0f}s: {exc}")
        (record_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


INDEX_FIELDS = [
    "video_id",
    "source_url",
    "creator",
    "caption",
    "status",
    "duration_seconds",
    "views",
    "likes",
    "comments",
    "shares",
    "shot_count",
    "n_scenes",
    "latency_seconds",
    "cost_usd",
    "tokens_total",
    "repair_passes",
    "attempts",
    "failure_reason",
    "schema_version",
    "prompt_versions",
    "pipeline_version",
    "model",
    "updated_at",
]


def cmd_index(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for record_dir in sorted(RECORDS_DIR.iterdir()) if RECORDS_DIR.exists() else []:
        if not record_dir.is_dir():
            continue
        record = load_record_state(record_dir)
        if not record:
            continue
        metadata = {}
        metadata_path = record_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        parsed = {}
        parsed_path = record_dir / "creative_ir.parsed.json"
        if parsed_path.exists():
            parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        perception = {}
        perception_path = record_dir / "perception.json"
        if perception_path.exists():
            perception = json.loads(perception_path.read_text(encoding="utf-8"))
        row = {
            "video_id": record.get("video_id"),
            "source_url": record.get("source_url") or metadata.get("source_url"),
            "creator": metadata.get("creator"),
            "caption": metadata.get("caption"),
            "status": record.get("status"),
            "duration_seconds": metadata.get("duration_seconds"),
            "views": metadata.get("views"),
            "likes": metadata.get("likes"),
            "comments": metadata.get("comments"),
            "shares": metadata.get("shares"),
            "shot_count": len(parsed.get("observed", {}).get("shots", [])) or None,
            "n_scenes": len(perception.get("scenes", [])) or None,
            "latency_seconds": record.get("latency_seconds"),
            "cost_usd": record.get("cost_usd"),
            "tokens_total": record.get("tokens_total"),
            "repair_passes": record.get("repair_passes"),
            "attempts": record.get("attempts"),
            "failure_reason": record.get("failure_reason"),
            "schema_version": record.get("schema_version"),
            "prompt_versions": record.get("prompt_versions"),
            "pipeline_version": record.get("pipeline_version"),
            "model": record.get("model"),
            "updated_at": record.get("updated_at"),
        }
        rows.append({field: row.get(field) for field in INDEX_FIELDS})

    table = pa.Table.from_pylist(rows)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, DATASET_DIR / "index.parquet")
    print(f"Wrote {len(rows)} rows to {DATASET_DIR / 'index.parquet'}")


def cmd_report(args: argparse.Namespace) -> None:
    records = []
    for record_dir in sorted(RECORDS_DIR.iterdir()) if RECORDS_DIR.exists() else []:
        if record_dir.is_dir():
            record = load_record_state(record_dir)
            if record:
                records.append(record)
    if not records:
        print("No records found. Run `collect` and `run` first.")
        return
    ok = [r for r in records if r.get("status") == "ok"]
    failed = [r for r in records if r.get("status") == "failed"]
    costs = [r["cost_usd"] for r in ok if isinstance(r.get("cost_usd"), (int, float))]
    latencies = [r["latency_seconds"] for r in records if isinstance(r.get("latency_seconds"), (int, float))]
    repairs = sum(r.get("repair_passes", 0) or 0 for r in records)
    print(f"=== PILOT REPORT ({len(records)} records) ===")
    print(f"ok: {len(ok)}  failed: {len(failed)}  failure rate: {len(failed) / len(records):.0%}")
    if costs:
        print(f"cost: total=${sum(costs):.3f}  mean=${sum(costs) / len(costs):.4f} per video")
    if latencies:
        print(
            f"latency: mean={sum(latencies) / len(latencies):.0f}s  "
            f"min={min(latencies):.0f}s  max={max(latencies):.0f}s"
        )
    print(f"repair passes triggered: {repairs}")
    if failed:
        print("\nFailures by reason:")
        reasons: dict[str, int] = {}
        for r in failed:
            reason = (r.get("failure_reason") or "unknown").split(":")[0]
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {count}x {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="download videos + metadata")
    p_collect.add_argument("--limit", type=int, default=COLLECT_LIMIT)
    p_collect.add_argument("--creators", nargs="*", default=None)
    p_collect.set_defaults(func=cmd_collect)

    p_run = sub.add_parser("run", help="decompile collected videos")
    p_run.add_argument("--limit", type=int, default=None)
    p_run.set_defaults(func=cmd_run)

    p_index = sub.add_parser("index", help="build Parquet index")
    p_index.set_defaults(func=cmd_index)

    p_report = sub.add_parser("report", help="print pilot summary")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
