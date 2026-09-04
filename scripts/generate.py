"""Closed-loop generation: CreativeIR hook shots -> fal video models -> assembly -> Gemini evaluation.

Providers (user-validated stack):
  omni-flash   google/gemini-omni-flash/v1.1/text-to-video   ~$0.15/clip, top-2 text-to-video Elo
  veo3.1-lite  fal-ai/veo3.1/lite                            $0.05/s, native audio (dialogue)

Costs are estimates from published per-output rates (fal responses carry no
price); they are recorded per shot in generation.json for budget tracking.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import fal_client
from openrouter_client import call_openrouter_json

GEN_SCHEMA_DIR = REPO_ROOT / "dataset" / "generations"
ASPECT_RATIO = "9:16"

SHOT_PROMPT_VERSION = "creativeir-shot-gen-v0.1"
EVAL_PROMPT_VERSION = "creativeir-hook-eval-v0.1"


@dataclass(frozen=True)
class Provider:
    key: str
    endpoint_id: str
    seconds_priced: float  # estimated USD per second of video
    flat_per_clip: float = 0.0  # if the model bills flat per clip

    def estimate_cost(self, duration_s: int) -> float:
        return self.flat_per_clip or round(self.seconds_priced * duration_s, 4)


PROVIDERS: dict[str, Provider] = {
    "omni-flash": Provider(
        key="omni-flash",
        endpoint_id="google/gemini-omni-flash/v1.1/text-to-video",
        seconds_priced=0.15 / 8,  # ~$0.15 per clip at default 8s
    ),
    "veo3.1-lite": Provider(
        key="veo3.1-lite",
        endpoint_id="fal-ai/veo3.1/lite",
        seconds_priced=0.05,
    ),
}


def build_payload(provider: Provider, prompt: str, duration_s: int) -> dict:
    duration = max(4, min(8, duration_s))
    if provider.key == "omni-flash":
        return {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": ASPECT_RATIO,
            "resolution": "720p",
        }
    if provider.key == "veo3.1-lite":
        return {
            "prompt": prompt,
            "duration": f"{duration}s",
            "aspect_ratio": ASPECT_RATIO,
            "resolution": "720p",
            "generate_audio": True,
        }
    raise ValueError(f"Unknown provider: {provider.key}")


def build_shot_prompt(shot: dict, ir: dict) -> str:
    """Compose one model prompt from the shot's reconstruction instructions."""
    generation = shot.get("generation", {})
    parts = [
        generation.get("reconstruction_prompt")
        or shot.get("inferred", {}).get("rationale")
        or "Cinematic vertical short-form video shot.",
    ]
    constraints = ir.get("generation", {}).get("global_constraints") or []
    continuity = generation.get("continuity_requirements") or []
    if continuity or constraints:
        parts.append(
            "Maintain continuity: "
            + "; ".join([*map(str, continuity), *map(str, constraints)])[:600]
        )
    parts.append(
        "Vertical 9:16 short-form video, single continuous shot, no captions or watermarks unless explicitly described."
    )
    return "\n".join(parts)


def select_hook_shots(ir: dict, max_shots: int = 4) -> list[dict]:
    """Hook shots in shot_order, capped; falls back to the first shots."""
    shots = ir.get("observed", {}).get("shots", [])
    order = ir.get("generation", {}).get("shot_order") or [s.get("shot_id") for s in shots]
    hook_ids = set(ir.get("observed", {}).get("hook", {}).get("shot_ids") or [])
    by_id = {s.get("shot_id"): s for s in shots}
    ordered = [by_id[sid] for sid in order if sid in by_id]
    hook_first = sorted(ordered, key=lambda s: (s.get("shot_id") not in hook_ids, order.index(s.get("shot_id"))))
    return hook_first[:max_shots]


def _shot_duration(shot: dict) -> int:
    time_range = shot.get("time_range", {})
    raw = time_range.get("end_seconds", 5) - time_range.get("start_seconds", 0)
    return max(4, min(8, round(raw) or 5))


def generate_hook(
    ir: dict,
    *,
    provider_keys: list[str],
    max_shots: int,
    out_dir: Path,
    timeout_seconds: int = fal_client.DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Generate the hook shots with each provider and persist clips + metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shots = select_hook_shots(ir, max_shots)
    if not shots:
        raise ValueError("No shots available for generation")
    generation_record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": SHOT_PROMPT_VERSION,
        "aspect_ratio": ASPECT_RATIO,
        "providers": provider_keys,
        "shots": [],
    }
    for shot in shots:
        prompt = build_shot_prompt(shot, ir)
        duration = _shot_duration(shot)
        for provider_key in provider_keys:
            provider = PROVIDERS[provider_key]
            payload = build_payload(provider, prompt, duration)
            started = time.monotonic()
            result = fal_client.run(provider.endpoint_id, payload, timeout_seconds=timeout_seconds)
            elapsed = round(time.monotonic() - started, 1)
            clip_url = _extract_video_url(result)
            clip_path = out_dir / f"{shot.get('shot_id', 'shot')}_{provider_key}.mp4"
            fal_client.download(clip_url, clip_path)
            generation_record["shots"].append(
                {
                    "shot_id": shot.get("shot_id"),
                    "provider": provider_key,
                    "endpoint_id": provider.endpoint_id,
                    "prompt": prompt,
                    "duration_s": duration,
                    "clip": clip_path.name,
                    "clip_url": clip_url,
                    "latency_seconds": elapsed,
                    "cost_usd_estimate": provider.estimate_cost(duration),
                    "response_id": result.get("request_id") or result.get("video", {}).get("file_name"),
                }
            )
            print(f"  {shot.get('shot_id')} via {provider_key}: {elapsed}s, ~${provider.estimate_cost(duration)}", flush=True)
    generation_record["cost_usd_estimate_total"] = round(
        sum(s["cost_usd_estimate"] for s in generation_record["shots"]), 4
    )
    (out_dir / "generation.json").write_text(json.dumps(generation_record, indent=2) + "\n", encoding="utf-8")
    return generation_record


def _extract_video_url(result: dict) -> str:
    video = result.get("video") or {}
    url = video.get("url") if isinstance(video, dict) else None
    if not url:
        videos = result.get("videos") or []
        url = videos[0].get("url") if videos else None
    if not url:
        raise fal_client.FalError(f"No video URL in fal result: {str(result)[:300]}")
    return url


def assemble_hook(clip_paths: list[Path], out_path: Path) -> Path:
    """Concatenate clips into one vertical preview (re-encoded for uniform streams)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list = out_path.parent / "concat.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in clip_paths), encoding="utf-8")
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
            # note: force_original_aspect_ratio=1 (== increase); this ffmpeg
            # build rejects the string form with the same meaning.
            "-vf", "scale=720:1280:force_original_aspect_ratio=1,pad=720:1280:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            str(out_path),
        ],
        capture_output=True,
        check=True,
    )
    concat_list.unlink()
    return out_path


EVAL_PROMPT = """You are auditing a GENERATED video against the decompilation brief it was generated from.

CREATIVE BRIEF (hook section):
{brief}

SHOT PROMPTS USED:
{prompts}

Score fidelity strictly:
1. per shot: covered / partially_covered / missing, with one-line justification
2. continuity: are style, setting and palette consistent across shots?
3. overall verdict: faithful / partially_faithful / unfaithful
4. top 3 concrete improvements for the prompts

Return ONLY a JSON object with keys: shot_scores (list of {{shot_id, provider, coverage, note}}), continuity, verdict, improvements.
"""


def evaluate_generation(
    hook_video: Path,
    ir: dict,
    generation_record: dict,
    out_path: Path,
    model: str,
) -> dict:
    """Gemini (via OpenRouter) scores the generated hook against the brief."""
    brief = ir.get("generation", {}).get("global_reconstruction_brief", "")
    hook = ir.get("observed", {}).get("hook", {})
    brief_context = f"{str(hook)[:800]}\n\n{str(brief)[:1200]}"
    prompts = "\n\n".join(
        f"[{s['shot_id']} via {s['provider']}]\n{s['prompt']}" for s in generation_record["shots"]
    )
    evaluation, call, _ = call_openrouter_json(
        EVAL_PROMPT.format(brief=brief_context, prompts=prompts),
        video_path=hook_video,
        model=model,
    )
    record = {
        "prompt_version": EVAL_PROMPT_VERSION,
        "model": model,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": evaluation.get("verdict"),
        "continuity": evaluation.get("continuity"),
        "shot_scores": evaluation.get("shot_scores"),
        "improvements": evaluation.get("improvements"),
        "response_id": call.response_id,
        "cost_usd": call.usage.get("cost"),
    }
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video-id", required=True, help="video_id of the dataset record to use")
    parser.add_argument("--providers", nargs="*", default=["omni-flash", "veo3.1-lite"], choices=sorted(PROVIDERS))
    parser.add_argument("--max-shots", type=int, default=4)
    parser.add_argument("--model", default="google/gemini-3.8-flash", help="Gemini model for evaluation")
    args = parser.parse_args()

    record_dir = REPO_ROOT / "dataset" / "records" / args.source_video_id
    ir = json.loads((record_dir / "creative_ir.parsed.json").read_text(encoding="utf-8"))
    out_dir = GEN_SCHEMA_DIR / args.source_video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating hook for {args.source_video_id} with {args.providers}")
    generation_record = generate_hook(
        ir, provider_keys=args.providers, max_shots=args.max_shots, out_dir=out_dir
    )
    print(f"Estimated cost: ${generation_record['cost_usd_estimate_total']}")

    clips = [out_dir / s["clip"] for s in generation_record["shots"]]
    hook_preview = assemble_hook(clips, out_dir / "hook_preview.mp4")
    print(f"Assembled: {hook_preview}")

    evaluation = evaluate_generation(hook_preview, ir, generation_record, out_dir / "evaluation.json", args.model)
    print(f"Verdict: {evaluation['verdict']} | continuity: {evaluation['continuity']}")


if __name__ == "__main__":
    main()
