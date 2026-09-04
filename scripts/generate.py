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

# v2: chained-frames pipeline endpoints
I2V_ENDPOINTS = {
    "omni-flash": "google/gemini-omni-flash/v1.1/image-to-video",
    "veo3.1-lite": "fal-ai/veo3.1/lite/image-to-video",
}
IMAGE_T2I_ENDPOINT = "fal-ai/nano-banana-2"
IMAGE_EDIT_ENDPOINT = "fal-ai/nano-banana-2/edit"
SFX_ENDPOINT = "cassetteai/sound-effects-generator"
SFX_PRICE_PER_CLIP = 0.03  # estimate; cassetteai bills per generation
IMAGE_PRICE = 0.04  # nano-banana-2 estimate per image

REALISM_DIRECTIVE = (
    "Photorealistic, shot on a smartphone, natural window light, real home kitchen, "
    "shallow depth of field, slight handheld micro-shake, no text, no watermark."
)

# The three chained first-frames: same counter/light across shots (user feedback:
# visual coherence), realism directives, per-shot camera motion for dynamism.
FRAME_PROMPTS = {
    "shot_0": (
        "Vertical 9:16 photorealistic photo of a bright modern home kitchen counter. "
        "Two hands chop onions with a chef knife on a cluttered wooden cutting board, "
        "onion skins scattered, a half-cut onion in focus. " + REALISM_DIRECTIVE
    ),
    "shot_1": (
        "Same kitchen counter, same lighting and angle as the reference image, but the "
        "cutting board and onion mess are cleared: a white-and-green push-press vegetable "
        "chopper sits centered on the counter, half an onion on the blade grid, its clear "
        "catch container empty and ready. " + REALISM_DIRECTIVE
    ),
    "shot_2": (
        "Same kitchen counter, same lighting: the chopper is now open with its container "
        "full of perfectly diced onions, and a young home cook stands behind the counter "
        "holding the chopper, smiling at the camera, mid-sentence as if explaining. "
        + REALISM_DIRECTIVE
    ),
}

# Camera motion per shot for the i2v animation pass (dynamism feedback).
MOTION_PROMPTS = {
    "shot_0": (
        "Animate: the hands continue chopping the onion with quick frustrated strokes, "
        "slow push-in toward the cutting board, handheld micro-shake, natural motion. "
        "No scene change."
    ),
    "shot_1": (
        "Animate: a hand presses the chopper lid down firmly in one fast motion, diced "
        "onion cubes drop into the clear container below, fast punch-in on the falling "
        "dice, crisp satisfying movement. No scene change."
    ),
    "shot_2": (
        "Animate: the cook talks to the camera with natural gestures, holds up the "
        "chopper slightly, nods at the diced onion bowl, subtle handheld movement, "
        "natural talking motion with lips synced to casual speech. No scene change."
    ),
}

SFX_PLAN = [
    {"name": "chop", "prompt": "sharp kitchen knife chopping an onion on a wooden board, single quick chop", "at_seconds": 0.8},
    {"name": "dice", "prompt": "fresh diced vegetables falling into a plastic container, crisp crunchy rattle", "at_seconds": 4.2},
    {"name": "whoosh", "prompt": "short clean transition whoosh, subtle", "at_seconds": 7.6},
]
CLIP_SECONDS = 4.0  # generated clip length; trimmed at assembly for pace
KEEP_SECONDS = 3.2  # per-clip duration kept in the final cut (punchier)


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
            clip_path = out_dir / f"{shot.get('shot_id', 'shot')}_{provider_key}.mp4"
            if clip_path.exists():
                # resume support: reuse clips already generated for this shot
                generation_record["shots"].append(
                    {
                        "shot_id": shot.get("shot_id"),
                        "provider": provider_key,
                        "endpoint_id": provider.endpoint_id,
                        "prompt": prompt,
                        "duration_s": duration,
                        "clip": clip_path.name,
                        "clip_url": "(cached from a previous run)",
                        "latency_seconds": 0.0,
                        "cost_usd_estimate": 0.0,
                        "response_id": "cached",
                    }
                )
                print(f"  {shot.get('shot_id')} via {provider_key}: cached", flush=True)
                continue
            started = time.monotonic()
            result = fal_client.run(provider.endpoint_id, payload, timeout_seconds=timeout_seconds)
            elapsed = round(time.monotonic() - started, 1)
            clip_url = _extract_video_url(result)
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
    concat_list = out_path.parent / f"concat_{out_path.stem}.txt"
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


def assemble_per_provider(generation_record: dict, out_dir: Path) -> list[Path]:
    """One assembled preview per provider (mixing providers per shot breaks continuity)."""
    previews = []
    for provider_key in dict.fromkeys(s["provider"] for s in generation_record["shots"]):
        clips = [out_dir / s["clip"] for s in generation_record["shots"] if s["provider"] == provider_key]
        previews.append(assemble_hook(clips, out_dir / f"hook_preview_{provider_key}.mp4"))
    return previews


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


def _extract_image_url(result: dict) -> str:
    images = result.get("images") or []
    if images and isinstance(images[0], dict) and images[0].get("url"):
        return images[0]["url"]
    image = result.get("image") or {}
    if isinstance(image, dict) and image.get("url"):
        return image["url"]
    raise fal_client.FalError(f"No image URL in fal result: {str(result)[:300]}")


def generate_first_frame(prompt: str, name: str, out_dir: Path, edit_from: list[str] | None = None) -> str:
    """Generate (or edit from a reference frame) one photorealistic first-frame image.

    Returns the fal CDN URL, usable directly as the i2v input. Chained edits
    keep the same counter/lighting across shots (user coherence requirement).
    """
    if edit_from:
        payload = {"prompt": prompt, "image_urls": edit_from, "aspect_ratio": "9:16", "num_images": 1, "resolution": "1K"}
        endpoint = IMAGE_EDIT_ENDPOINT
    else:
        payload = {"prompt": prompt, "aspect_ratio": "9:16", "num_images": 1, "resolution": "1K"}
        endpoint = IMAGE_T2I_ENDPOINT
    result = fal_client.run(endpoint, payload)
    url = _extract_image_url(result)
    fal_client.download(url, out_dir / f"{name}.jpg")
    return url


def animate_frame(provider_key: str, frame_url: str, motion_prompt: str, duration_s: int) -> dict:
    """Animate one first-frame with the selected i2v endpoint."""
    duration = max(4, min(8, duration_s))
    if provider_key == "omni-flash":
        payload = {
            "prompt": motion_prompt,
            "image_url": frame_url,
            "duration": duration,
            "aspect_ratio": ASPECT_RATIO,
            "resolution": "720p",
        }
    elif provider_key == "veo3.1-lite":
        payload = {
            "prompt": motion_prompt,
            "image_url": frame_url,
            "duration": f"{duration}s",
            "aspect_ratio": ASPECT_RATIO,
            "resolution": "720p",
            "generate_audio": True,
        }
    else:
        raise ValueError(f"Unknown i2v provider: {provider_key}")
    return fal_client.run(I2V_ENDPOINTS[provider_key], payload)


def generate_sfx_clip(prompt: str, seconds: int, name: str, out_dir: Path) -> Path:
    """Generate one short SFX clip and download it."""
    result = fal_client.run(SFX_ENDPOINT, {"prompt": prompt, "duration": seconds})
    audio = result.get("audio") or result.get("audio_file") or {}
    url = audio.get("url") if isinstance(audio, dict) else None
    if not url:
        raise fal_client.FalError(f"No audio URL in fal result: {str(result)[:300]}")
    return fal_client.download(url, out_dir / f"sfx_{name}.mp3")


def sfx_delays(total_seconds: float) -> list[dict]:
    """Millisecond delays for the SFX plan given the assembled cut timeline."""
    delays = []
    for entry in SFX_PLAN:
        at = min(entry["at_seconds"], max(0.0, total_seconds - 0.4))
        delays.append({**entry, "delay_ms": int(at * 1000)})
    return delays


def mix_sfx(video_path: Path, sfx_dir: Path, total_seconds: float, out_path: Path) -> Path:
    """Hard-cut concat already done; overlay SFX at cut timestamps over the video audio."""
    delays = [e for e in sfx_delays(total_seconds) if (sfx_dir / f"sfx_{e['name']}.mp3").exists()]
    inputs = ["-i", str(video_path)]
    filters = []
    mix_inputs = "[0:a]"
    for i, entry in enumerate(delays, 1):
        inputs += ["-i", str(sfx_dir / f"sfx_{entry['name']}.mp3")]
        filters.append(f"[{i}:a]adelay={entry['delay_ms']}|{entry['delay_ms']}[sfx{i}]")
        mix_inputs += f"[sfx{i}]"
    filters.append(f"{mix_inputs}amix=inputs={len(delays) + 1}:duration=first:normalize=0[aout]")
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out_path)]
    subprocess.run(cmd, capture_output=True, check=True)
    return out_path


def trim_clip(clip_path: Path, out_path: Path, keep_seconds: float) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(clip_path), "-t", f"{keep_seconds}", "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-c:a", "aac", "-b:a", "128k", str(out_path)],
        capture_output=True,
        check=True,
    )
    return out_path


def generate_frames_flow(out_dir: Path, providers: list[str], timeout_seconds: int = fal_client.DEFAULT_TIMEOUT_SECONDS) -> dict:
    """v2 flow: chained frames -> i2v -> assembly with SFX. Resume-aware."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / "generation.json"
    if gen_path.exists():
        # resume: reload frame URLs and prior steps from the previous run
        record = json.loads(gen_path.read_text(encoding="utf-8"))
    else:
        record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "frames-v2",
            "prompt_version": PATTERN_BRIEF_VERSION,
            "steps": [],
        }
    frame_urls: dict[str, str] = {}
    for shot_id, prompt in FRAME_PROMPTS.items():
        frame_path = out_dir / f"{shot_id}_frame.jpg"
        if frame_path.exists():
            prev_url = None
            for s in record["steps"]:
                if s.get("frame") == frame_path.name:
                    prev_url = s.get("frame_url")
            if prev_url:
                frame_urls[shot_id] = prev_url
                print(f"  {shot_id} frame: cached", flush=True)
                continue
            raise fal_client.FalError(f"{frame_path} exists but no cached URL; delete it and re-run")
        edit_from = [frame_urls["shot_0"]] if shot_id == "shot_1" else ([frame_urls["shot_1"]] if shot_id == "shot_2" else None)
        url = generate_first_frame(prompt, shot_id + "_frame", out_dir, edit_from=edit_from)
        frame_urls[shot_id] = url
        record["steps"].append({"step": "frame", "shot_id": shot_id, "frame": frame_path.name, "frame_url": url, "cost_usd_estimate": IMAGE_PRICE})
        print(f"  {shot_id} frame: {url[:80]}", flush=True)

    clips: list[dict] = []
    for shot_id in FRAME_PROMPTS:
        for provider_key in providers:
            clip_path = out_dir / f"{shot_id}_{provider_key}_i2v.mp4"
            if clip_path.exists():
                clips.append({"shot_id": shot_id, "provider": provider_key, "clip": clip_path.name, "cost_usd_estimate": 0.0})
                print(f"  {shot_id} via {provider_key}: cached", flush=True)
                continue
            result = animate_frame(provider_key, frame_urls[shot_id], MOTION_PROMPTS[shot_id], int(CLIP_SECONDS))
            clip_url = _extract_video_url(result)
            fal_client.download(clip_url, clip_path)
            cost = PROVIDERS[provider_key].estimate_cost(int(CLIP_SECONDS))
            clips.append({"shot_id": shot_id, "provider": provider_key, "clip": clip_path.name, "cost_usd_estimate": cost})
            record["steps"].append({"step": "animate", "shot_id": shot_id, "provider": provider_key, "clip": clip_path.name, "cost_usd_estimate": cost})
            print(f"  {shot_id} via {provider_key}: animated, ~${cost}", flush=True)

    for entry in SFX_PLAN:
        sfx_path = out_dir / f"sfx_{entry['name']}.mp3"
        if sfx_path.exists():
            continue
        generate_sfx_clip(entry["prompt"], 2, entry["name"], out_dir)
        record["steps"].append({"step": "sfx", "name": entry["name"], "cost_usd_estimate": SFX_PRICE_PER_CLIP})
        print(f"  sfx {entry['name']}: generated", flush=True)

    record["cost_usd_estimate_total"] = round(sum(s.get("cost_usd_estimate", 0.0) for s in record["steps"]), 4)
    (out_dir / "generation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


# Convergent patterns measured across 6 independent creators on the same
# product (issue #44, docs/affiliate_dataset_report_issue44.md):
# hook=product_promise (16/17), arc=problem_proof_cta (11/17),
# personal_experience (11/17), creator dialogue (12/17),
# before/after (8/17), CTA verbal/bio only (0/17 on-screen).
PATTERN_SOURCE = "docs/affiliate_dataset_report_issue44.md"
PATTERN_BRIEF_VERSION = "pattern-informed-hook-v0.1"

PRODUCT_FACTS = (
    "Push-press 4-in-1 vegetable chopper (Fullstar-style): heavy base, interchangeable "
    "dice/slice/spiralize blades, diced pieces fall into a lidded catch container, ~$25-30, "
    "sold on Amazon with 100k+ reviews."
)


def build_pattern_brief() -> dict:
    """Synthesize a 3-shot affiliate hook IR from the measured dataset patterns.

    Deterministic: the patterns and product facts above fully determine the
    shots; no model call is involved in building the brief.
    """
    return {
        "creative_ir_version": "0.1",
        "source": {"observed": {"platform": "tiktok", "video_id": "pattern-brief-chopper"}},
        "observed": {
            "shots": [
                {
                    "shot_id": "shot_0",
                    "time_range": {"start_seconds": 0.0, "end_seconds": 4.0},
                    "generation": {
                        "reconstruction_prompt": (
                            "POV hands in a home kitchen chopping onions with a knife on a cluttered "
                            "cutting board, teary squinting eyes implied by slow frustrated cutting, "
                            "messy onion skins around: the tedious prep problem, framed as the pain "
                            "the product removes."
                        ),
                        "continuity_requirements": [
                            "Same home kitchen as later shots",
                            "Warm daylight",
                            "Vertical 9:16 phone-shot realism",
                        ],
                    },
                    "inferred": {"semantic_role": "setup"},
                },
                {
                    "shot_id": "shot_1",
                    "time_range": {"start_seconds": 4.0, "end_seconds": 8.0},
                    "generation": {
                        "reconstruction_prompt": (
                            "Top-down close-up: half an onion placed on the chopper blade grid, "
                            "the lid pressed down firmly in one satisfying motion, perfect dice "
                            "cubes falling into the clear catch container below. The visually "
                            "novel press-and-dice payoff moment, sharp focus on the cubes."
                        ),
                        "continuity_requirements": [
                            "Same kitchen counter as shot_0",
                            "Product: " + PRODUCT_FACTS,
                        ],
                    },
                    "inferred": {"semantic_role": "reveal", "attention_mechanisms": ["visual_novelty"]},
                },
                {
                    "shot_id": "shot_2",
                    "time_range": {"start_seconds": 8.0, "end_seconds": 12.0},
                    "generation": {
                        "reconstruction_prompt": (
                            "Creator facing camera holding the chopper with the diced onion bowl, "
                            "talking to camera with genuine enthusiasm about using it every day "
                            "for meal prep, ends on a short verbal call to action pointing to the "
                            "link in bio. No on-screen text overlays."
                        ),
                        "continuity_requirements": [
                            "Same creator outfit and kitchen as previous shots",
                            "Personal-experience tone, conversational",
                            "CTA is verbal only, no rendered text",
                        ],
                    },
                    "inferred": {"semantic_role": "cta", "attention_mechanisms": ["personal_experience", "social_proof"]},
                },
            ],
            "hook": {"shot_ids": ["shot_0", "shot_1", "shot_2"]},
            "marketing": {"engagement_devices": ["personal_experience", "before_after"]},
        },
        "generation": {
            "shot_order": ["shot_0", "shot_1", "shot_2"],
            "global_constraints": [
                "Vertical 9:16 TikTok-native framing",
                "Total hook <= 12 seconds",
                "Arc: problem -> proof -> verbal CTA (problem_proof_cta)",
                "Hook type: product_promise + visual_novelty",
                "No on-screen captions or watermarks",
            ],
            "global_reconstruction_brief": (
                "Affiliate hook for " + PRODUCT_FACTS + " Structure: problem (knife prep pain) "
                "-> proof (satisfying press-and-dice) -> personal experience with verbal CTA. "
                "Creator dialogue throughout, warm home kitchen, vertical."
            ),
        },
    }


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


def assemble_frames_flow(record: dict, out_dir: Path, providers: list[str]) -> list[Path]:
    """Trim clips for pace, hard-cut concat per provider, then mix SFX at cut points."""
    previews = []
    total = round(KEEP_SECONDS * len(FRAME_PROMPTS), 2)
    for provider_key in providers:
        trimmed = []
        for shot_id in FRAME_PROMPTS:
            trimmed.append(trim_clip(out_dir / f"{shot_id}_{provider_key}.mp4", out_dir / f"{shot_id}_{provider_key}_trim.mp4", KEEP_SECONDS))
        concat_list = out_dir / f"concat_{provider_key}.txt"
        concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in trimmed), encoding="utf-8")
        base = out_dir / f"hook_base_{provider_key}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-c:a", "aac", "-b:a", "128k", str(base)],
            capture_output=True,
            check=True,
        )
        concat_list.unlink()
        preview = mix_sfx(base, out_dir, total, out_dir / f"hook_preview_{provider_key}.mp4")
        previews.append(preview)
        print(f"  preview {provider_key}: {preview.name} ({total}s, SFX mixed)", flush=True)
    return previews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video-id", default=None, help="video_id of the dataset record to use")
    parser.add_argument("--pattern-mode", action="store_true", help="generate from the synthetic #44 pattern brief instead of a dataset record")
    parser.add_argument("--mode", choices=["t2v", "frames"], default="t2v", help="t2v = text-to-video v1; frames = chained frames + i2v + SFX (v2)")
    parser.add_argument("--providers", nargs="*", default=["omni-flash", "veo3.1-lite"], choices=sorted(PROVIDERS))
    parser.add_argument("--max-shots", type=int, default=4)
    parser.add_argument("--model", default="google/gemini-3.8-flash", help="Gemini model for evaluation")
    args = parser.parse_args()

    out_dir = GEN_SCHEMA_DIR / (args.source_video_id or "pattern-brief-chopper")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "frames":
        record = generate_frames_flow(out_dir, args.providers)
        print(f"Estimated cost: ${record['cost_usd_estimate_total']}")
        for preview in assemble_frames_flow(record, out_dir, args.providers):
            print(f"Assembled: {preview}")
        return

    if args.pattern_mode:
        ir = build_pattern_brief()
        (out_dir / "pattern_brief.json").write_text(json.dumps(ir, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        generation_record = generate_hook(
            ir, provider_keys=args.providers, max_shots=args.max_shots, out_dir=out_dir
        )
    else:
        if not args.source_video_id:
            parser.error("--source-video-id is required unless --pattern-mode is set")
        record_dir = REPO_ROOT / "dataset" / "records" / args.source_video_id
        ir = json.loads((record_dir / "creative_ir.parsed.json").read_text(encoding="utf-8"))
        print(f"Generating hook for {args.source_video_id} with {args.providers}")
        generation_record = generate_hook(
            ir, provider_keys=args.providers, max_shots=args.max_shots, out_dir=out_dir
        )
    print(f"Estimated cost: ${generation_record['cost_usd_estimate_total']}")

    previews = assemble_per_provider(generation_record, out_dir)
    for preview in previews:
        print(f"Assembled: {preview}")

    if not args.pattern_mode:
        evaluation = evaluate_generation(hook_preview := previews[0], ir, generation_record, out_dir / "evaluation.json", args.model)
        print(f"Verdict: {evaluation['verdict']} | continuity: {evaluation['continuity']}")


if __name__ == "__main__":
    main()
