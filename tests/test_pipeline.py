"""Tests for the shared decompilation pipeline (scripts/pipeline.py)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pipeline
from perceive import PerceptionResult
from pipeline import (
    assert_temporal_integrity,
    decompile_video,
    inject_deterministic_facts,
    merge_creative_ir,
    write_usage_record,
    PassMeta,
)


REPO = Path(__file__).resolve().parent.parent
SAMPLE_IR = REPO / ".orca" / "drops" / "creative_ir.parsed.json"
SAMPLE_VIDEO = REPO / ".orca" / "drops" / "video.mp4"
SAMPLE_METADATA = REPO / ".orca" / "drops" / "metadata.json"
SCHEMA_PATH = REPO / "schemas" / "creative_ir_v0_1.json"

FAKE_MEDIA = PerceptionResult(
    duration_seconds=3.0,
    width=576,
    height=1024,
    fps=30.0,
    nb_frames=90,
    video_codec="h264",
    audio_codec="aac",
    aspect_ratio_label="vertical_9_16",
    file_size_bytes=1000,
    bit_rate=None,
)


def _fake_scenes():
    return [
        {"start_seconds": 0.0, "end_seconds": 1.0},
        {"start_seconds": 1.0, "end_seconds": 2.0},
        {"start_seconds": 2.0, "end_seconds": 3.0},
    ]


def _minimal_ir():
    """A CreativeIR skeleton with three shots aligned to _fake_scenes."""
    shots = []
    for i in range(3):
        shots.append(
            {
                "shot_id": f"shot_{i}",
                "time_range": {"start_seconds": float(i), "end_seconds": float(i + 1)},
                "observed": {"evidence": [{"kind": "visual", "note": "fake"}]},
                "inferred": {},
            }
        )
    return {
        "creative_ir_version": "0.1",
        "source": {"observed": {"platform": "tiktok", "source_url": "u", "video_id": "v", "evidence": []}},
        "observed": {
            "shots": shots,
            "narrative": {"beats": [{"beat_id": "b0", "shot_ids": ["shot_2", "shot_0"]}]},
            "hook": {"shot_ids": ["shot_0"], "evidence": [{"kind": "visual", "note": "model evidence"}]},
        },
        "inferred": {},
        "generation": {"shot_order": ["shot_0", "shot_1", "shot_2"]},
    }


def test_inject_facts_overrides_boundaries_and_media():
    ir = _minimal_ir()
    ir["source"]["observed"]["duration_seconds"] = 99.0
    ir["source"]["observed"]["frame_size"] = {"width": 1, "height": 1}
    injected = inject_deterministic_facts(ir, FAKE_MEDIA, _fake_scenes())

    src = injected["source"]["observed"]
    assert src["duration_seconds"] == 3.0
    assert src["frame_size"] == {"width": 576, "height": 1024}
    assert src["aspect_ratio"] == "vertical_9_16"
    assert any(e["kind"] == "metadata" for e in src["evidence"])
    for i, shot in enumerate(injected["observed"]["shots"]):
        assert shot["time_range"] == _fake_scenes()[i]
        assert any(e["kind"] == "timing" for e in shot["observed"]["evidence"])


def test_inject_facts_recomputes_beat_span_unordered_ids():
    ir = _minimal_ir()
    injected = inject_deterministic_facts(ir, FAKE_MEDIA, _fake_scenes())
    beat = injected["observed"]["narrative"]["beats"][0]
    # shot_2 + shot_0 -> span from scene 0 start to scene 2 end
    assert beat["time_range"] == {"start_seconds": 0.0, "end_seconds": 3.0}


def test_inject_facts_appends_hook_evidence_and_keeps_model_evidence():
    ir = _minimal_ir()
    original_evidence = copy.deepcopy(ir["observed"]["hook"]["evidence"])
    injected = inject_deterministic_facts(ir, FAKE_MEDIA, _fake_scenes())
    hook_evidence = injected["observed"]["hook"]["evidence"]
    assert hook_evidence[0] == original_evidence[0]
    assert len(hook_evidence) == 2
    assert hook_evidence[1]["kind"] == "timing"


def test_inject_facts_shot_count_mismatch_raises():
    ir = _minimal_ir()
    ir["observed"]["shots"] = ir["observed"]["shots"][:2]
    with pytest.raises(ValueError, match="Shot count mismatch"):
        inject_deterministic_facts(ir, FAKE_MEDIA, _fake_scenes())


def test_merge_replaces_global_sections_and_keeps_shots():
    shot_ir = _minimal_ir()
    synth_global = {
        "observed": {"hook": {"shot_ids": ["shot_1"]}, "marketing": {"evidence": []}},
        "inferred": {"overall_concept": {"premise": "x"}},
        "generation": {"shot_order": ["shot_0", "shot_1", "shot_2"]},
    }
    merged = merge_creative_ir(shot_ir, synth_global)
    assert merged["observed"]["hook"] == {"shot_ids": ["shot_1"]}
    assert "marketing" in merged["observed"]
    assert merged["inferred"] == synth_global["inferred"]
    assert len(merged["observed"]["shots"]) == 3
    assert merged["creative_ir_version"] == "0.1"


def test_temporal_integrity_accepts_valid_ir():
    ir = _minimal_ir()
    ir["source"]["observed"]["duration_seconds"] = 3.0
    assert_temporal_integrity(ir)


def test_temporal_integrity_rejects_gaps_and_order_mismatch():
    ir = _minimal_ir()
    ir["source"]["observed"]["duration_seconds"] = 3.0
    ir["observed"]["shots"][1]["time_range"]["end_seconds"] = 0.5
    with pytest.raises(AssertionError):
        assert_temporal_integrity(ir)

    ir = _minimal_ir()
    ir["source"]["observed"]["duration_seconds"] = 3.0
    ir["generation"]["shot_order"] = ["shot_2", "shot_1", "shot_0"]
    with pytest.raises(AssertionError):
        assert_temporal_integrity(ir)


def test_write_usage_record_aggregates_costs(tmp_path):
    shot_pass = PassMeta("id-1", "stop", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01})
    synth_pass = PassMeta("id-2", "stop", {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25, "cost": 0.02})
    record = write_usage_record(tmp_path / "usage.json", "m", "vid", [("shot_analysis", shot_pass), ("global_synthesis", synth_pass)])
    assert record["cost_usd_total"] == 0.03
    assert record["calls"][0]["response_id"] == "id-1"
    assert record["calls"][1]["pass"] == "global_synthesis"
    assert json.loads((tmp_path / "usage.json").read_text()) == record


def test_write_usage_record_includes_retry_passes(tmp_path):
    shot_pass = PassMeta("id-1", "stop", {"total_tokens": 15, "cost": 0.01})
    synth_pass = PassMeta("id-2", "stop", {"total_tokens": 25, "cost": 0.02})
    shot_retry = PassMeta("id-3", "stop", {"total_tokens": 15, "cost": 0.01})
    synth_retry = PassMeta("id-4", "stop", {"total_tokens": 25, "cost": 0.02})
    record = write_usage_record(
        tmp_path / "usage.json",
        "m",
        "vid",
        [("shot_analysis", shot_pass), ("global_synthesis", synth_pass), ("shot_analysis_retry", shot_retry), ("global_synthesis_retry", synth_retry)],
    )
    assert [c["pass"] for c in record["calls"]] == ["shot_analysis", "global_synthesis", "shot_analysis_retry", "global_synthesis_retry"]
    assert record["cost_usd_total"] == 0.06


@pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="No sample video fixture")
def test_decompile_video_end_to_end_with_mocked_passes(tmp_path, monkeypatch):
    full_ir = json.loads(SAMPLE_IR.read_text(encoding="utf-8"))
    metadata = json.loads(SAMPLE_METADATA.read_text(encoding="utf-8"))

    shot_ir = {k: v for k, v in full_ir.items() if k != "decompilation"}
    synth_global = {
        "observed": {k: full_ir["observed"][k] for k in ("context", "hook", "narrative", "marketing", "commercial") if k in full_ir["observed"]},
        "inferred": full_ir["inferred"],
        "generation": full_ir["generation"],
    }

    def fake_shot(*a, **k):
        return copy.deepcopy(shot_ir), PassMeta("mock-shot", "stop", {"total_tokens": 1, "cost": 0.0})

    def fake_synth(*a, **k):
        return copy.deepcopy(synth_global), PassMeta("mock-synth", "stop", {"total_tokens": 1, "cost": 0.0})

    monkeypatch.setattr(pipeline, "run_shot_analysis", fake_shot)
    monkeypatch.setattr(pipeline, "run_global_synthesis", fake_synth)

    usage = decompile_video(SAMPLE_VIDEO, metadata, tmp_path, SCHEMA_PATH)

    parsed = json.loads((tmp_path / "creative_ir.parsed.json").read_text(encoding="utf-8"))
    assert parsed["decompilation"]["model"] == "google/gemini-3.8-flash"
    assert parsed["decompilation"]["pipeline_version"] == pipeline.PIPELINE_VERSION
    assert (tmp_path / "creative_ir.usage.json").exists()
    assert (tmp_path / "perception.json").exists()
    assert (tmp_path / "frames").is_dir()
    assert usage["video_id"] == metadata["video_id"]


def _completion(content: str):
    from openrouter_client import CallResult

    call = CallResult(content, "google/gemini-3.8-flash", "gen-x", "stop", {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "cost": 0.001}, {})
    return json.loads(content), call, None


def _repair_completion(broken: str, fixed: str):
    from openrouter_client import CallResult

    call = CallResult(broken, "google/gemini-3.8-flash", "gen-x", "stop", {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "cost": 0.001}, {})
    repair = CallResult(fixed, "google/gemini-3.8-flash", "gen-y", "stop", {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40, "cost": 0.002}, {})
    return json.loads(fixed), call, repair


def test_run_shot_analysis_writes_raw_and_repairs(tmp_path, monkeypatch):
    if not SAMPLE_VIDEO.exists():
        pytest.skip("No sample video fixture")
    broken = '{"creative_ir_version": "0.1"'
    fixed = json.dumps(_minimal_ir())
    monkeypatch.setattr(pipeline, "call_openrouter_json", lambda *a, **k: _repair_completion(broken, fixed))

    schema = json.loads(SCHEMA_PATH.read_text())
    result, meta = pipeline.run_shot_analysis(
        SAMPLE_VIDEO, {}, _fake_scenes(), {}, schema, tmp_path
    )
    assert meta.repair_path is not None and meta.repair_path.exists()
    assert (tmp_path / "creative_ir.shot_analysis.raw.json").read_text() == broken
    assert meta.usage["total_tokens"] == 60
    assert meta.usage["repair_calls"] == 1
    assert result == json.loads(fixed)


def test_run_global_synthesis_writes_raw(tmp_path, monkeypatch):
    if not SAMPLE_VIDEO.exists():
        pytest.skip("No sample video fixture")
    payload = json.dumps({"inferred": {"overall_concept": {"premise": "x"}}})
    monkeypatch.setattr(pipeline, "call_openrouter_json", lambda *a, **k: _completion(payload))

    schema = json.loads(SCHEMA_PATH.read_text())
    result, meta = pipeline.run_global_synthesis(
        SAMPLE_VIDEO, {}, _fake_scenes(), {}, {}, schema, tmp_path
    )
    assert meta.repair_path is None
    assert result == {"inferred": {"overall_concept": {"premise": "x"}}}
    assert (tmp_path / "creative_ir.global_synth.raw.json").read_text() == payload


def test_extract_scene_frames_creates_one_frame_per_scene(tmp_path):
    if not SAMPLE_VIDEO.exists():
        pytest.skip("No sample video fixture")
    from pipeline import extract_scene_frames

    frames = extract_scene_frames(SAMPLE_VIDEO, _fake_scenes()[:2], tmp_path / "frames")
    assert len(frames) == 2
    assert all(f.exists() for f in frames)


def test_canonicalize_ids_renumbers_and_remaps_references():
    from pipeline import canonicalize_ids

    ir = _minimal_ir()
    ir["observed"]["shots"][0]["observed"]["text"] = {"segments": [
        {"text_id": "text_watermark_0", "text": "wm"},
        {"text_id": "text_5", "text": "ocr"},
    ]}
    ir["observed"]["shots"][1]["observed"] = {"text": {"segments": [{"text_id": "text_hook", "text": "h"}]}}
    ir["observed"]["hook"]["text_ids"] = ["text_hook", "text_missing_ref"]
    ir["observed"]["marketing"] = {"call_to_action_text_ids": ["text_5"]}

    merged = canonicalize_ids(ir, "merged")
    segs0 = merged["observed"]["shots"][0]["observed"]["text"]["segments"]
    segs1 = merged["observed"]["shots"][1]["observed"]["text"]["segments"]
    assert [s["text_id"] for s in segs0] == ["text_0", "text_1"]
    assert segs1[0]["text_id"] == "text_2"
    # dangling reference dropped, valid ones remapped
    assert merged["observed"]["hook"]["text_ids"] == ["text_2"]
    assert merged["observed"]["marketing"]["call_to_action_text_ids"] == ["text_1"]


def test_canonicalize_ids_shots_renumbers_and_remaps():
    from pipeline import canonicalize_ids

    ir = _minimal_ir()
    ir["observed"]["shots"][0]["shot_id"] = "shot_opening"
    ir["observed"]["shots"][2]["shot_id"] = "shot_9"
    ir["generation"]["shot_order"] = ["shot_opening", "shot_1", "shot_9"]
    ir["observed"]["narrative"]["beats"][0]["shot_ids"] = ["shot_9", "shot_opening"]
    ir["observed"]["narrative"]["beats"].append({"beat_id": "beat_x", "shot_ids": ["shot_gone"]})

    out = canonicalize_ids(ir, "shots")
    assert [s["shot_id"] for s in out["observed"]["shots"]] == ["shot_0", "shot_1", "shot_2"]
    assert out["generation"]["shot_order"] == ["shot_0", "shot_1", "shot_2"]
    assert out["observed"]["narrative"]["beats"][0]["shot_ids"] == ["shot_2", "shot_0"]
    # dangling reference to a non-existent shot is dropped
    assert out["observed"]["narrative"]["beats"][1]["shot_ids"] == []


def test_canonicalize_ids_is_idempotent():
    from pipeline import canonicalize_ids

    ir = _minimal_ir()
    once = canonicalize_ids(ir, "merged")
    twice = canonicalize_ids(once, "merged")
    assert once == twice


def test_coerce_unknown_enum_values_maps_to_fallback():
    from pipeline import coerce_unknown_enum_values

    ir = _minimal_ir()
    ir["observed"]["shots"][0]["observed"]["text"] = {"segments": [{"text_id": "text_0", "role": "social_proof", "text": "x"}]}
    ir["inferred"] = {"marketing": {"mechanisms": ["delayed_payoff"], "confidence": "high"}}
    ir["source"]["observed"]["aspect_ratio"] = "horizontal_16_9"

    schema = json.loads(SCHEMA_PATH.read_text())
    coerced, log = coerce_unknown_enum_values(ir, schema)

    role = coerced["observed"]["shots"][0]["observed"]["text"]["segments"][0]["role"]
    assert role == "other"
    # alias resolves to payoff_reveal when the enum contains it
    assert coerced["inferred"]["marketing"]["mechanisms"] == ["payoff_reveal"]
    # valid values untouched
    assert coerced["inferred"]["marketing"]["confidence"] == "high"
    assert coerced["source"]["observed"]["aspect_ratio"] == "horizontal_16_9"
    assert {"path": "$.observed.shots[0].observed.text.segments[0].role", "from": "social_proof", "to": "other"} in log


def test_coerce_unknown_enum_values_leaves_fields_without_fallback():
    from pipeline import coerce_unknown_enum_values

    ir = _minimal_ir()
    ir["inferred"] = {"overall_concept": {"confidence": "certain"}}
    schema = json.loads(SCHEMA_PATH.read_text())
    coerced, log = coerce_unknown_enum_values(ir, schema)
    # confidence enum has no fallback value: drift must surface, not be masked
    assert coerced["inferred"]["overall_concept"]["confidence"] == "certain"
    assert log == []


def test_ensure_model_proxy_transcodes_large_files(tmp_path):
    from pipeline import ensure_model_proxy

    if not SAMPLE_VIDEO.exists():
        pytest.skip("No sample video fixture")
    # sample video is small: should pass through unchanged
    result = ensure_model_proxy(SAMPLE_VIDEO, tmp_path)
    assert result == SAMPLE_VIDEO

    big = tmp_path / "big.mp4"
    big.write_bytes(b"\x00" * (21 * 1024 * 1024))
    with pytest.raises(Exception):
        # invalid media must fail loudly (ffmpeg cannot transcode it)
        ensure_model_proxy(big, tmp_path)


def test_strip_meta_keys_removes_schema_echo():
    from pipeline import strip_meta_keys

    ir = _minimal_ir()
    ir["$schema"] = "https://example.org/creative_ir_v0_1.json"
    ir["observed"]["shots"][0]["$comment"] = "meta"
    cleaned = strip_meta_keys(ir)
    assert "$schema" not in cleaned
    assert "$comment" not in cleaned["observed"]["shots"][0]
    assert cleaned["observed"]["shots"][0]["shot_id"] == "shot_0"


def test_budget_cap_stops_when_spent(tmp_path, monkeypatch):
    import pilot as pilot_mod

    records = tmp_path / "records"
    (records / "a").mkdir(parents=True)
    (records / "a" / "record.json").write_text(json.dumps({"cost_usd": 0.5}))

    monkeypatch.setattr(pilot_mod, "RECORDS_DIR", records)
    budget = pilot_mod.Budget(1.0)
    assert budget._spent == 0.5
    assert budget.reserve("m") is True
    budget.commit(0.6)
    # cap reached: no further reservation
    assert budget.reserve("m") is False
    assert budget.reserve("m") is False


def test_budget_none_is_unlimited(tmp_path, monkeypatch):
    import pilot as pilot_mod

    monkeypatch.setattr(pilot_mod, "RECORDS_DIR", tmp_path)
    budget = pilot_mod.Budget(None)
    for _ in range(5):
        assert budget.reserve("m") is True
        budget.commit(100.0)


def test_collect_video_keyword_filter(tmp_path, monkeypatch):
    import pilot as pilot_mod

    info = {"id": "vid1", "description": "Making guacamole with my new gadget!", "uploader": "chef",
            "hashtags": ["cooktok"], "duration": 15}
    (tmp_path / "vid1").mkdir(parents=True)
    (tmp_path / "vid1" / "video.info.json").write_text(json.dumps(info))
    # yt-dlp creates the media file during download; simulate
    def fake_ytdlp(args, timeout=300):
        for a in args:
            if str(a).endswith("video.mp4"):
                Path(a).write_bytes(b"media")
        return ""
    monkeypatch.setattr(pilot_mod, "_ytdlp", fake_ytdlp)

    outcome = pilot_mod.collect_video("https://www.tiktok.com/@chef/video/vid1", tmp_path / "vid1", keywords=["chopper", "fullstar"])
    assert outcome == "skipped_product_mismatch"
    record = json.loads((tmp_path / "vid1" / "record.json").read_text())
    assert record["status"] == "skipped_product_mismatch"
    assert not (tmp_path / "vid1" / "video.mp4").exists()
    assert (tmp_path / "vid1" / "metadata.json").exists()

    info2 = dict(info, id="vid2", description="Fullstar vegetable chopper review")
    (tmp_path / "vid2").mkdir(parents=True)
    (tmp_path / "vid2" / "video.info.json").write_text(json.dumps(info2))
    def fake_ytdlp2(args, timeout=300):
        for a in args:
            if str(a).endswith("video.mp4"):
                Path(a).write_bytes(b"media")
        return ""
    monkeypatch.setattr(pilot_mod, "_ytdlp", fake_ytdlp2)
    outcome2 = pilot_mod.collect_video("https://www.tiktok.com/@chef/video/vid2", tmp_path / "vid2", keywords=["chopper", "fullstar"])
    assert outcome2 == "collected"
    assert (tmp_path / "vid2" / "video.mp4").exists()


def test_cmd_run_keyword_filter_skips_mismatch(tmp_path, monkeypatch):
    import pilot as pilot_mod

    for vid, caption in (("match1", "Fullstar vegetable chopper review"), ("nope1", "random vlog")):
        d = tmp_path / vid
        d.mkdir(parents=True)
        (d / "video.mp4").write_bytes(b"media")
        (d / "metadata.json").write_text(json.dumps({"video_id": vid, "caption": caption, "source_url": "u"}))

    monkeypatch.setattr(pilot_mod, "RECORDS_DIR", tmp_path)
    monkeypatch.setattr(pilot_mod, "_recorded_cost", lambda: 0.0)
    monkeypatch.setattr(pilot_mod, "has_video_stream", lambda path: True)
    monkeypatch.setattr(pilot_mod, "decompile_video", lambda *a, **k: {"cost_usd_total": 0.1, "calls": []})

    args = type("Args", (), {"limit": None, "workers": 1, "max_cost": 5.0, "keywords": ["chopper", "fullstar"]})()
    pilot_mod.cmd_run(args)

    assert json.loads((tmp_path / "match1" / "record.json").read_text())["status"] == "ok"
    assert json.loads((tmp_path / "nope1" / "record.json").read_text())["status"] == "skipped_product_mismatch"


def test_fill_empty_strings_replaces_violations_and_logs():
    from pipeline import fill_empty_strings

    ir = _minimal_ir()
    ir["observed"]["commercial"] = {"offer_text": "", "cta_text": "link in bio", "product_mentions": ["chopper"]}
    schema = json.loads(SCHEMA_PATH.read_text())
    filled, log = fill_empty_strings(ir, schema)
    assert filled["observed"]["commercial"]["offer_text"] == "not_specified"
    assert filled["observed"]["commercial"]["cta_text"] == "link in bio"
    assert {"path": "$.observed.commercial.offer_text", "from": "", "to": "not_specified"} in log
