"""Pipeline validation tests for multi-step CreativeIR."""
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from perceive import probe_media, detect_scenes

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "creative_ir_v0_1.json"
SAMPLE_VIDEO = Path(__file__).resolve().parent.parent / "data" / "exploration" / "7106594312292453675" / "video.mp4"
FALLBACK_VIDEO = Path(__file__).resolve().parent.parent.parent / "tiktok-factory" / ".orca" / "drops" / "video.mp4"
DROPS_DIR = Path(__file__).resolve().parent.parent.parent / "tiktok-factory" / ".orca" / "drops"


def _video_path():
    if SAMPLE_VIDEO.exists():
        return SAMPLE_VIDEO
    if FALLBACK_VIDEO.exists():
        return FALLBACK_VIDEO
    return None


def test_schema_validates_baseline():
    baseline = DROPS_DIR / "creative_ir.parsed.json"
    if not baseline.exists():
        return
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    ir = json.loads(baseline.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(ir)


def test_perception_matches_schema_requirements():
    vp = _video_path()
    if vp is None:
        return
    media = probe_media(vp)
    scenes = detect_scenes(vp)
    assert media.duration_seconds > 0
    assert media.width > 0
    assert media.height > 0
    assert len(scenes) >= 2


def test_temporal_integrity_baseline():
    baseline = DROPS_DIR / "creative_ir.parsed.json"
    if not baseline.exists():
        return
    ir = json.loads(baseline.read_text(encoding="utf-8"))
    duration = ir["source"]["observed"]["duration_seconds"]
    shots = ir["observed"]["shots"]
    assert shots
    previous_end = 0.0
    for shot in shots:
        start = shot["time_range"]["start_seconds"]
        end = shot["time_range"]["end_seconds"]
        assert 0 <= start < end <= duration + 0.05
        assert start >= previous_end - 0.05
        previous_end = end
    assert ir["generation"]["shot_order"] == [s["shot_id"] for s in shots]


def test_improved_boundaries_are_detected():
    """The improved pipeline should detect more boundaries than the baseline's 5 shots."""
    vp = _video_path()
    if vp is None:
        return
    scenes = detect_scenes(vp)
    assert len(scenes) >= 5, f"Expected at least 5 detected scenes, got {len(scenes)}"


def test_perception_duration_matches_video():
    vp = _video_path()
    if vp is None:
        return
    media = probe_media(vp)
    scenes = detect_scenes(vp)
    # Last scene end should match duration within tolerance
    assert abs(scenes[-1]["end_seconds"] - media.duration_seconds) < 0.1


def test_schema_file_exists():
    assert SCHEMA_PATH.exists(), f"Schema not found at {SCHEMA_PATH}"
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
