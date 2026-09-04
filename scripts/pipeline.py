"""Validated multi-step CreativeIR decompilation pipeline (issue #4 flow).

Single source of truth shared by the inspection notebook and the dataset pilot:
deterministic perception (ffprobe + PySceneDetect), OpenRouter Gemini shot
analysis, global creative synthesis, deterministic fact injection, merge and
validation. Do not redesign the decompilation path here: issue #5 reuses it
as-is.
"""
from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator

from openrouter_client import call_openrouter_json
from perceive import PerceptionResult, detect_scenes, probe_media

SHOT_PROMPT_VERSION = "openrouter-gemini-shot-analysis-v0.1"
SYNTH_PROMPT_VERSION = "openrouter-gemini-global-synthesis-v0.1"
PIPELINE_VERSION = "issue-4-multistep-perception-v0.1"
DEFAULT_MODEL = "google/gemini-3.8-flash"
METADATA_KEYS = ("source_url", "video_id", "creator", "caption", "hashtags", "publication_date")


@dataclass
class PassMeta:
    """Bookkeeping for one model pass, including any repair call."""

    response_id: str
    finish_reason: str | None
    usage: dict = field(default_factory=dict)
    repair_path: Path | None = None


def extract_scene_frames(video_path: Path, scenes: list[dict], frame_dir: Path) -> list[Path]:
    """Extract one representative frame per scene at its midpoint with ffmpeg."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for i, scene in enumerate(scenes):
        mid = (scene["start_seconds"] + scene["end_seconds"]) / 2
        frame_path = frame_dir / f"scene_{i:03d}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(mid), "-i", str(video_path), "-frames:v", "1", "-q:v", "2", str(frame_path)],
            capture_output=True,
            check=True,
        )
        frame_paths.append(frame_path)
    return frame_paths


# OpenRouter/Google AI Studio rejects request bodies over 20MB; the base64
# data URL inflates the file by ~4/3, so the file must stay below ~14.5MB.
MODEL_VIDEO_MAX_BYTES = 14_000_000


def _video_codec(video_path: Path) -> str | None:
    """Codec of the first video stream, or None if absent/unreadable."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name", "-of", "json", str(video_path)],
        capture_output=True,
        text=True,
    )
    try:
        streams = json.loads(out.stdout).get("streams", [])
        for stream in streams:
            if stream.get("codec_type") == "video":
                return stream.get("codec_name")
    except json.JSONDecodeError:
        pass
    return None


def ensure_model_proxy(video_path: Path, source_dir: Path) -> Path:
    """Return a path suitable for the model call: the original, or a proxy.

    The proxy is needed when the file is large (> 14MB) or not H.264:
    TikTok serves HEVC (`bytevc1`/h265) for 1080p downloads and the provider
    fails to decode those, and bodies over 20MB (base64-inflated) are rejected.
    The proxy downscales to 540px width and re-encodes H.264 so the request
    body stays well under the provider limit. It is only what the model SEES:
    every deterministic fact is still measured on the original file by
    ffprobe/PySceneDetect. Idempotent per video; oversized stale proxies are
    regenerated.
    """
    proxy_path = source_dir / "video.proxy.mp4"
    needs_proxy = video_path.stat().st_size > MODEL_VIDEO_MAX_BYTES or _video_codec(video_path) not in (None, "h264")
    if not needs_proxy:
        return video_path
    if proxy_path.exists():
        if 0 < proxy_path.stat().st_size <= MODEL_VIDEO_MAX_BYTES:
            return proxy_path
        proxy_path.unlink()

    width = None
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width", "-select_streams", "v:0", "-of", "csv=p=0", str(video_path)],
        capture_output=True,
        text=True,
    )
    try:
        width = int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        width = None

    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if width is None or width > 540:
        # scale=720:-2 style expressions with commas break ffmpeg filter args;
        # use an explicit target width decided in Python instead.
        cmd += ["-vf", "scale=540:-2"]
    cmd += ["-c:v", "libx264", "-crf", "30", "-preset", "fast", "-c:a", "aac", "-b:a", "96k", str(proxy_path)]
    subprocess.run(cmd, capture_output=True, check=True)
    return proxy_path


def perceive_video(video_path: Path, source_dir: Path) -> tuple[PerceptionResult, list[dict], dict]:
    """Deterministic perception: ffprobe facts + PySceneDetect boundaries + frames.

    Persists `perception.json` into source_dir and returns (media, scenes, perception).
    """
    media = probe_media(video_path)
    media_dict = media.to_dict()
    scenes = detect_scenes(video_path)
    frame_dir = source_dir / "frames"
    frame_paths = extract_scene_frames(video_path, scenes, frame_dir)
    perception = {
        "media_facts": media_dict,
        "scenes": scenes,
        "frame_paths": [str(p.relative_to(source_dir)) for p in frame_paths],
        "probe_raw": {"streams": media.raw_streams, "format": media.raw_format},
    }
    (source_dir / "perception.json").write_text(json.dumps(perception, indent=2) + "\n", encoding="utf-8")
    return media, scenes, perception


def build_shot_prompt(media_dict: dict, scenes: list[dict], metadata: dict, schema: dict, model_name: str) -> str:
    metadata_context = json.dumps({key: metadata.get(key) for key in METADATA_KEYS}, ensure_ascii=False)
    scene_context = json.dumps(scenes, indent=2)
    media_context = json.dumps(media_dict, indent=2)
    return f"""You are a meticulous audiovisual decompiler performing shot-level analysis.

DETERMINISTIC FACTS (from ffprobe, do NOT override):
{media_context}

DETECTED SHOT BOUNDARIES (from PySceneDetect, do NOT override timestamps):
{scene_context}

SOURCE METADATA:
{metadata_context}

Analyze each detected shot in the video. For every shot provide:
- Visual: exact description, subjects, environment, palette
- Camera: framing, angle, motion, composition
- Text: every legible on-screen text segment with exact OCR, timing, placement, role
- Dialogue: presence (present/absent/uncertain), exact words if present
- Audio: music/original sound presence and label, sound effects, mix notes. Keep audio descriptions CONSERVATIVE — do not invent specific sound identities, song names, or effects you cannot confirm. Use identity_known=false unless you can identify the audio source.
- Editing: transition_in, transition_out, pacing, notes
- Evidence: at least one evidence entry per shot referencing the video time range

Also provide:
- Per-shot semantic_role, attention_mechanisms, confidence, rationale
- Per-shot reconstruction_prompt and continuity_requirements

The complete repository CreativeIR v0.1 schema is authoritative for every nested field:
{json.dumps(schema, ensure_ascii=False, separators=(",", ":"))}

CRITICAL RULES:
- Use the DETECTED shot boundaries above exactly — do not invent new boundaries
- Use the DETECTED media facts exactly — do not guess duration, resolution, or fps
- Record exact visible OCR text. If you see text that is readable, transcribe it faithfully.
- Keep uncertain audio/OCR claims explicitly uncertain
- Do not copy caption text into on-screen OCR unless those exact words are visibly rendered
- Ensure decompilation.model is exactly {model_name!r}, prompt_version is {SHOT_PROMPT_VERSION!r}, schema_version is "0.1", annotator_type is "automated"
- Return ONLY the CreativeIR JSON object matching the repository schema
"""


def run_shot_analysis(
    video_path: Path,
    media_dict: dict,
    scenes: list[dict],
    metadata: dict,
    schema: dict,
    source_dir: Path,
    model: str = DEFAULT_MODEL,
    name_suffix: str = "",
) -> tuple[dict, PassMeta]:
    """Pass 1: per-shot semantic analysis of the full video."""
    shot_prompt = build_shot_prompt(media_dict, scenes, metadata, schema, model)
    shot_result, shot_call, shot_repair = call_openrouter_json(shot_prompt, model=model, video_path=video_path)
    (source_dir / f"creative_ir.shot_analysis.raw{name_suffix}.json").write_text(shot_call.content, encoding="utf-8")
    usage = dict(shot_call.usage)
    repair_path = None
    if shot_repair is not None:
        repair_path = source_dir / f"creative_ir.shot_analysis.repaired{name_suffix}.json"
        repair_path.write_text(shot_repair.content, encoding="utf-8")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] = (usage.get(key) or 0) + (shot_repair.usage.get(key) or 0)
        if isinstance(usage.get("cost"), (int, float)) and isinstance(shot_repair.usage.get("cost"), (int, float)):
            usage["cost"] = usage["cost"] + shot_repair.usage["cost"]
        usage["repair_calls"] = 1
    return shot_result, PassMeta(shot_call.response_id, shot_call.finish_reason, usage, repair_path)


def build_synth_prompt(
    media_dict: dict,
    scenes: list[dict],
    shot_injected: dict,
    metadata: dict,
    schema: dict,
    model_name: str,
) -> str:
    metadata_context = json.dumps({key: metadata.get(key) for key in METADATA_KEYS}, ensure_ascii=False)
    shot_summaries = []
    for shot in shot_injected.get("observed", {}).get("shots", []):
        shot_summaries.append(
            {
                "shot_id": shot["shot_id"],
                "time_range": shot["time_range"],
                "visual": shot.get("observed", {}).get("visual", {}).get("description", ""),
                "subjects": shot.get("observed", {}).get("visual", {}).get("subjects", []),
                "text_segments": [
                    {"text": seg["text"], "role": seg["role"], "placement": seg["placement"]}
                    for seg in shot.get("observed", {}).get("text", {}).get("segments", [])
                ],
                "dialogue_presence": shot.get("observed", {}).get("dialogue", {}).get("presence", "unknown"),
                "audio_label": shot.get("observed", {}).get("audio", {}).get("music_or_original_sound", {}).get("label", ""),
                "semantic_role": shot.get("inferred", {}).get("semantic_role", "other"),
            }
        )
    return f"""You are a creative strategist performing global synthesis of a decompiled video.

DETERMINISTIC MEDIA FACTS:
{json.dumps(media_dict, indent=2)}

DETECTED SHOT STRUCTURE ({len(scenes)} shots):
{json.dumps(scenes, indent=2)}

SHOT-LEVEL ANALYSIS (from prior Gemini pass):
{json.dumps(shot_summaries, indent=2)}

SOURCE METADATA:
{metadata_context}

Based on the above factual shot analysis and the original video, produce the GLOBAL sections of the CreativeIR:
1. observed.context (visible_subject, caption_signal, evidence)
2. observed.hook (shot_ids, text_ids, visual_summary, evidence with time ranges)
3. observed.narrative (beats with beat_id, label, shot_ids, time_range, visible_event)
4. observed.marketing (call_to_action_text_ids, engagement_devices, evidence)
5. observed.commercial (product_presence, evidence)
6. inferred.overall_concept (premise, format, viewer_action, confidence, rationale)
7. inferred.target_audience (primary_audience, interest_clusters, confidence, rationale)
8. inferred.hook (hook_types, promise, confidence, rationale)
9. inferred.narrative (story_summary, arc, payoff, confidence, rationale)
10. inferred.marketing (mechanisms, confidence, rationale)
11. inferred.commercial (status, problem, desire, promise, offer, proof_type, trust_signals, objections_addressed, cta_type, confidence, rationale)
12. generation.global_reconstruction_brief (detailed: timeline, shot duration, composition, text treatment, transitions, pacing, continuity, payoff timing)
13. generation.shot_order
14. generation.global_constraints

CRITICAL:
- Use the exact shot IDs and time ranges from the shot analysis above
- The reconstruction brief must be detailed enough to reproduce the creative without seeing the original
- Include shot-level timing, composition details, text styling, transitions, and pacing notes
- Keep audio descriptions conservative
- Return ONLY a JSON object with these global fields (not the shots themselves — those are already provided)
- Ensure decompilation.model is exactly {model_name!r}, prompt_version is {SYNTH_PROMPT_VERSION!r}

The complete repository CreativeIR v0.1 schema:
{json.dumps(schema, ensure_ascii=False, separators=(",", ":"))}
"""


def run_global_synthesis(
    video_path: Path,
    media_dict: dict,
    scenes: list[dict],
    shot_injected: dict,
    metadata: dict,
    schema: dict,
    source_dir: Path,
    model: str = DEFAULT_MODEL,
    name_suffix: str = "",
) -> tuple[dict, PassMeta]:
    """Pass 2: global creative synthesis over the injected shot analysis."""
    synth_prompt = build_synth_prompt(media_dict, scenes, shot_injected, metadata, schema, model)
    synth_result, synth_call, synth_repair = call_openrouter_json(synth_prompt, model=model, video_path=video_path)
    (source_dir / f"creative_ir.global_synth.raw{name_suffix}.json").write_text(synth_call.content, encoding="utf-8")
    usage = dict(synth_call.usage)
    repair_path = None
    if synth_repair is not None:
        repair_path = source_dir / f"creative_ir.global_synth.repaired{name_suffix}.json"
        repair_path.write_text(synth_repair.content, encoding="utf-8")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] = (usage.get(key) or 0) + (synth_repair.usage.get(key) or 0)
        if isinstance(usage.get("cost"), (int, float)) and isinstance(synth_repair.usage.get("cost"), (int, float)):
            usage["cost"] = usage["cost"] + synth_repair.usage["cost"]
        usage["repair_calls"] = 1
    return synth_result, PassMeta(synth_call.response_id, synth_call.finish_reason, usage, repair_path)


def _parse_shot_index(shot_id: str) -> int | None:
    """Safely extract numeric index from shot_N id; returns None if malformed."""
    try:
        return int(shot_id.split("_")[1])
    except (IndexError, ValueError):
        return None


def _shot_span(shot_ids: list[str], scenes: list[dict]) -> dict | None:
    """Time range spanning the referenced shots, robust to unordered ids."""
    indices = [_parse_shot_index(s) for s in shot_ids]
    indices = [i for i in indices if i is not None and 0 <= i < len(scenes)]
    if not indices:
        return None
    first_idx, last_idx = min(indices), max(indices)
    return {
        "start_seconds": scenes[first_idx]["start_seconds"],
        "end_seconds": scenes[last_idx]["end_seconds"],
    }


def inject_deterministic_facts(ir: dict, media: PerceptionResult, scenes: list[dict]) -> dict:
    """Override model-guessed values with authoritative deterministic facts."""
    ir = copy.deepcopy(ir)

    # Source media facts
    src = ir.setdefault("source", {}).setdefault("observed", {})
    src["duration_seconds"] = media.duration_seconds
    src["frame_size"] = {"width": media.width, "height": media.height}
    src["aspect_ratio"] = media.aspect_ratio_label
    src.setdefault("evidence", []).append(
        {
            "kind": "metadata",
            "note": f"Exact values from ffprobe (duration={media.duration_seconds:.3f}s, {media.width}x{media.height}, fps={media.fps:.2f}, codec={media.video_codec})",
        }
    )

    # Shot time ranges from detected scenes: every scene must map to exactly one shot
    shots = ir.get("observed", {}).get("shots", [])
    if len(shots) != len(scenes):
        raise ValueError(
            f"Shot count mismatch: model returned {len(shots)} shots but PySceneDetect detected {len(scenes)}. "
            "Re-run the shot analysis pass before continuing."
        )
    for i, shot in enumerate(shots):
        shot["time_range"] = {
            "start_seconds": scenes[i]["start_seconds"],
            "end_seconds": scenes[i]["end_seconds"],
        }
        shot.setdefault("observed", {}).setdefault("evidence", []).append(
            {"kind": "timing", "note": f"Exact boundary from PySceneDetect scene {i}"}
        )

    # Update narrative beat time ranges if they reference shots
    beats = ir.get("observed", {}).get("narrative", {}).get("beats", [])
    for beat in beats:
        beat_shots = beat.get("shot_ids", [])
        if beat_shots:
            span = _shot_span(beat_shots, scenes)
            if span:
                beat["time_range"] = span

    # Update hook time range (append timing evidence, keep model evidence)
    hook = ir.get("observed", {}).get("hook", {})
    hook_shots = hook.get("shot_ids", [])
    if hook_shots:
        span = _shot_span(hook_shots, scenes)
        if span:
            hook.setdefault("evidence", []).append(
                {
                    "kind": "timing",
                    "note": "Hook span recomputed from PySceneDetect boundaries",
                    "time_range": span,
                }
            )

    return ir


def strip_meta_keys(ir: dict) -> dict:
    """Remove JSON-schema meta keys (`$schema`, `$id`, ...) the model may echo.

    The CreativeIR schema disallows additional properties, so any `$`-prefixed
    key emitted by the model must be dropped before validation.
    """
    cleaned = copy.deepcopy(ir)
    stack = [cleaned]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key in [k for k in node if isinstance(k, str) and k.startswith("$")]:
                del node[key]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return cleaned


def canonicalize_ids(ir: dict, stage: str) -> dict:
    """Deterministically renumber entity ids to the schema patterns and remap references.

    Gemini sometimes invents ids like `text_watermark_0` that violate the
    CreativeIR `^text_[0-9]+$` style patterns. Renumbering ids and remapping
    references is a deterministic canonicalization step, not a redesign of the
    model flow. Idempotent: canonical inputs pass through unchanged.

    stage="shots": shot_id (references: hook.shot_ids, beats.shot_ids, shot_order)
    stage="merged": text_id, dialogue_id, beat_id (references: hook.text_ids,
        marketing.call_to_action_text_ids)
    """
    ir = copy.deepcopy(ir)

    def _remap_list(values: list | None, mapping: dict) -> list:
        if not values:
            return []
        return [mapping[v] for v in values if v in mapping]

    if stage == "shots":
        shots = ir.get("observed", {}).get("shots", [])
        shot_mapping = {shot.get("shot_id", f"shot_{i}"): f"shot_{i}" for i, shot in enumerate(shots)}
        for i, shot in enumerate(shots):
            shot["shot_id"] = f"shot_{i}"
        observed = ir.get("observed", {})
        hook = observed.get("hook", {})
        if hook.get("shot_ids"):
            hook["shot_ids"] = _remap_list(hook["shot_ids"], shot_mapping)
        for beat in observed.get("narrative", {}).get("beats", []):
            if beat.get("shot_ids"):
                beat["shot_ids"] = _remap_list(beat["shot_ids"], shot_mapping)
        generation = ir.get("generation", {})
        if generation.get("shot_order"):
            generation["shot_order"] = _remap_list(generation["shot_order"], shot_mapping)
        return ir

    if stage == "merged":
        text_mapping: dict[str, str] = {}
        counter = 0
        for shot in ir.get("observed", {}).get("shots", []):
            segments = shot.get("observed", {}).get("text", {}).get("segments", [])
            for segment in segments:
                old_id = segment.get("text_id")
                new_id = f"text_{counter}"
                counter += 1
                if old_id is not None:
                    text_mapping[old_id] = new_id
                segment["text_id"] = new_id
            dialogues = shot.get("observed", {}).get("dialogue", {}).get("segments", [])
            for j, dialogue in enumerate(dialogues):
                dialogue["dialogue_id"] = f"dialogue_{j}"
        observed = ir.get("observed", {})
        hook = observed.get("hook", {})
        if hook.get("text_ids"):
            hook["text_ids"] = _remap_list(hook["text_ids"], text_mapping)
        marketing = observed.get("marketing", {})
        if marketing.get("call_to_action_text_ids"):
            marketing["call_to_action_text_ids"] = _remap_list(
                marketing["call_to_action_text_ids"], text_mapping
            )
        for i, beat in enumerate(observed.get("narrative", {}).get("beats", [])):
            beat["beat_id"] = f"beat_{i}"
        return ir

    raise ValueError(f"Unknown canonicalization stage: {stage}")


ENUM_FALLBACKS = ("other", "uncertain", "unknown", "not_applicable", "absent")
ENUM_ALIASES = {
    # model output -> intended schema value (used only when it exists in the enum)
    "delayed_payoff": "payoff_reveal",
}


def resolve_local_ref(root: dict, node):
    """Resolve `$ref: #/$defs/...` nodes against the schema root (recursive copy)."""
    if isinstance(node, dict) and set(node) == {"$ref"}:
        ref = node["$ref"]
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"Unsupported schema reference: {ref}")
        target = root
        for part in ref[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        return resolve_local_ref(root, copy.deepcopy(target))
    if isinstance(node, dict):
        return {key: resolve_local_ref(root, value) for key, value in node.items()}
    if isinstance(node, list):
        return [resolve_local_ref(root, value) for value in node]
    return node


def _coerce_enum_drift(instance, schema: dict, path: str, coercions: list):
    if isinstance(schema, dict):
        enum = schema.get("enum")
        if enum is not None and isinstance(instance, str) and instance not in enum:
            replacement = None
            alias = ENUM_ALIASES.get(instance)
            if alias is not None and alias in enum:
                replacement = alias
            else:
                for fallback in ENUM_FALLBACKS:
                    if fallback in enum:
                        replacement = fallback
                        break
            if replacement is not None:
                coercions.append({"path": path, "from": instance, "to": replacement})
                return replacement
            return instance
        properties = schema.get("properties", {})
        if properties and isinstance(instance, dict):
            result = dict(instance)
            for key, subschema in properties.items():
                if key in result:
                    result[key] = _coerce_enum_drift(result[key], subschema, f"{path}.{key}", coercions)
            return result
        items = schema.get("items")
        if items is not None and isinstance(instance, list):
            return [_coerce_enum_drift(element, items, f"{path}[{i}]", coercions) for i, element in enumerate(instance)]
    return instance


def coerce_unknown_enum_values(ir: dict, schema: dict) -> tuple[dict, list]:
    """Deterministically map enum-drift values to the schema's own fallback values.

    Gemini sometimes emits a semantically adjacent string that is not in the
    CreativeIR enum (e.g. `social_proof` for `text_segment.role`). This walk
    replaces such values with the closest value the schema itself allows:
    an alias when present in the enum, else the schema's fallback
    (`other`/`uncertain`/`unknown`/...). Fields without a fallback (e.g.
    `confidence`) are left untouched so they surface as validation failures.

    Returns ``(coerced_ir, coercion_log)``; the log is persisted for provenance.
    """
    expanded = resolve_local_ref(schema, schema)
    coercions: list = []
    coerced = _coerce_enum_drift(copy.deepcopy(ir), expanded, "$", coercions)
    return coerced, coercions


def _fill_empty_strings(instance, schema: dict, path: str, fills: list):
    if isinstance(schema, dict):
        if instance == "" and schema.get("type") == "string" and schema.get("minLength", 0) >= 1:
            fills.append({"path": path, "from": "", "to": "not_specified"})
            return "not_specified"
        properties = schema.get("properties", {})
        if properties and isinstance(instance, dict):
            result = dict(instance)
            for key, subschema in properties.items():
                if key in result:
                    result[key] = _fill_empty_strings(result[key], subschema, f"{path}.{key}", fills)
            return result
        items = schema.get("items")
        if items is not None and isinstance(instance, list):
            return [_fill_empty_strings(element, items, f"{path}[{i}]", fills) for i, element in enumerate(instance)]
    return instance


def fill_empty_strings(ir: dict, schema: dict) -> tuple[dict, list]:
    """Replace empty strings that violate schema `minLength: 1` with 'not_specified'.

    Gemini emits "" for fields with nothing to report (e.g. an offer the video
    does not make); the CreativeIR schema forbids empty strings and offers no
    null. The substitution is deterministic and logged for provenance.

    Returns ``(filled_ir, fill_log)``.
    """
    expanded = resolve_local_ref(schema, schema)
    fills: list = []
    filled = _fill_empty_strings(copy.deepcopy(ir), expanded, "$", fills)
    return filled, fills


def merge_creative_ir(shot_ir: dict, synth_global: dict) -> dict:
    """Merge shot analysis and global synthesis into one CreativeIR."""
    merged = copy.deepcopy(shot_ir)

    # Replace global sections from synthesis
    for section in ("context", "hook", "narrative", "marketing", "commercial"):
        if section in synth_global.get("observed", {}):
            merged.setdefault("observed", {})[section] = synth_global["observed"][section]

    # Replace inferred sections
    if "inferred" in synth_global:
        merged["inferred"] = synth_global["inferred"]

    # Replace generation sections
    if "generation" in synth_global:
        merged["generation"] = synth_global["generation"]

    # Ensure version
    merged["creative_ir_version"] = "0.1"

    return merged


def assert_temporal_integrity(ir: dict) -> None:
    duration = ir["source"]["observed"]["duration_seconds"]
    shots = ir["observed"]["shots"]
    assert shots, "at least one shot is required"
    previous_end = 0.0
    for shot in shots:
        start = shot["time_range"]["start_seconds"]
        end = shot["time_range"]["end_seconds"]
        assert 0 <= start < end <= duration + 0.05, (shot["shot_id"], start, end, duration)
        assert start >= previous_end - 0.05, "shot ranges must be ordered"
        previous_end = end
    assert abs(shots[0]["time_range"]["start_seconds"]) <= 0.05
    assert abs(shots[-1]["time_range"]["end_seconds"] - duration) <= 0.05
    assert ir["generation"]["shot_order"] == [shot["shot_id"] for shot in shots]


def write_usage_record(
    usage_path: Path,
    model: str,
    video_id: str,
    passes: list[tuple[str, PassMeta]],
) -> dict:
    """Persist one usage record entry per model pass (including retries)."""

    def _usage_summary(usage: dict) -> dict:
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost_usd": usage.get("cost"),
        }

    calls = []
    for pass_name, meta in passes:
        summary = _usage_summary(meta.usage or {})
        calls.append(
            {
                "pass": pass_name,
                "prompt_version": SHOT_PROMPT_VERSION if "shot" in pass_name else SYNTH_PROMPT_VERSION,
                "response_id": meta.response_id,
                "finish_reason": meta.finish_reason,
                "repairs": (meta.usage or {}).get("repair_calls", 0),
                "usage": summary,
            }
        )
    costs = [
        c["usage"]["cost_usd"]
        for c in calls
        if isinstance(c["usage"].get("cost_usd"), (int, float))
    ]
    usage_record = {
        "provider": "openrouter",
        "model": model,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "video_id": video_id,
        "calls": calls,
        "cost_usd_total": round(sum(costs), 6) if costs else None,
    }
    usage_path.write_text(json.dumps(usage_record, indent=2) + "\n", encoding="utf-8")
    return usage_record


def decompile_video(
    video_path: Path,
    metadata: dict,
    source_dir: Path,
    schema_path: Path,
    model: str = DEFAULT_MODEL,
    max_attempts: int = 2,
) -> dict:
    """Run the validated #4 flow on one video and persist every artifact.

    On schema/temporal validation failure the whole validated flow is retried
    once (same path, fresh model calls — thinking models are non-deterministic
    across calls). Usage from every attempt is persisted even on failure.

    Returns the usage record; writes `creative_ir.parsed.json` on success.
    Raises the last validation error after max_attempts.
    """
    video_path = Path(video_path)
    source_dir = Path(source_dir)
    schema_path = Path(schema_path)
    source_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    media, scenes, _ = perceive_video(video_path, source_dir)
    media_dict = media.to_dict()
    model_video = ensure_model_proxy(video_path, source_dir)
    passes: list[tuple[str, PassMeta]] = []
    usage_path = source_dir / "creative_ir.usage.json"
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            suffix = f"_retry{attempt}" if attempt > 1 else ""
            shot_result, shot_pass = run_shot_analysis(
                model_video, media_dict, scenes, metadata, schema, source_dir, model, name_suffix=suffix
            )
            passes.append((f"shot_analysis{suffix}", shot_pass))
            shot_result = canonicalize_ids(shot_result, "shots")
            shot_injected = inject_deterministic_facts(shot_result, media, scenes)
            synth_result, synth_pass = run_global_synthesis(
                model_video, media_dict, scenes, shot_injected, metadata, schema, source_dir, model, name_suffix=suffix
            )
            passes.append((f"global_synthesis{suffix}", synth_pass))

            final = strip_meta_keys(canonicalize_ids(merge_creative_ir(shot_injected, synth_result), "merged"))
            final, coercions = coerce_unknown_enum_values(final, schema)
            final, empty_fills = fill_empty_strings(final, schema)
            coercions = coercions + empty_fills
            if coercions:
                (source_dir / "creative_ir.coercions.json").write_text(
                    json.dumps(coercions, indent=2) + "\n", encoding="utf-8"
                )
            final["decompilation"] = {
                "model": model,
                "prompt_version": f"{SHOT_PROMPT_VERSION}+{SYNTH_PROMPT_VERSION}",
                "schema_version": "0.1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "pipeline_version": PIPELINE_VERSION,
                "annotator_type": "automated",
            }

            Draft202012Validator(schema).validate(final)
            assert_temporal_integrity(final)

            (source_dir / "creative_ir.parsed.json").write_text(
                json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return write_usage_record(usage_path, model, metadata.get("video_id", ""), passes)
        except Exception as exc:
            last_error = exc
            # Preserve usage for every attempt even when the attempt fails.
            write_usage_record(usage_path, model, metadata.get("video_id", ""), passes)

    raise last_error  # type: ignore[misc]
