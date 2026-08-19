"""Unit tests for deterministic perception module."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from perceive import probe_media, detect_scenes, PerceptionResult

SAMPLE_VIDEO = Path(__file__).resolve().parent.parent / "data" / "exploration" / "7106594312292453675" / "video.mp4"
FALLBACK_VIDEO = Path(__file__).resolve().parent.parent.parent / "tiktok-factory" / ".orca" / "drops" / "video.mp4"


def _video_path():
    if SAMPLE_VIDEO.exists():
        return SAMPLE_VIDEO
    if FALLBACK_VIDEO.exists():
        return FALLBACK_VIDEO
    return None


def test_probe_media_returns_exact_facts():
    vp = _video_path()
    if vp is None:
        return
    result = probe_media(vp)
    assert isinstance(result, PerceptionResult)
    assert result.duration_seconds > 0
    assert result.width > 0
    assert result.height > 0
    assert result.fps > 0
    assert result.video_codec in ("h264", "hevc", "vp9", "av1", "mpeg4")
    assert result.audio_codec is not None
    assert result.aspect_ratio_label in ("vertical_9_16", "vertical_other", "square", "horizontal_16_9", "horizontal_other", "unknown")
    assert result.nb_frames > 0
    assert result.file_size_bytes > 0


def test_probe_media_sample_video_specifics():
    vp = _video_path()
    if vp is None:
        return
    result = probe_media(vp)
    assert abs(result.duration_seconds - 24.39) < 0.1, f"Expected ~24.39s, got {result.duration_seconds}"
    assert result.width == 576
    assert result.height == 1024
    assert result.aspect_ratio_label == "vertical_9_16"


def test_probe_media_to_dict():
    vp = _video_path()
    if vp is None:
        return
    result = probe_media(vp)
    d = result.to_dict()
    assert isinstance(d, dict)
    assert "duration_seconds" in d
    assert "width" in d
    assert "height" in d
    assert "fps" in d
    assert "video_codec" in d


def test_probe_media_missing_file():
    import pytest
    with pytest.raises(FileNotFoundError):
        probe_media(Path("/nonexistent/video.mp4"))


def test_detect_scenes_returns_boundaries():
    vp = _video_path()
    if vp is None:
        return
    scenes = detect_scenes(vp)
    assert len(scenes) >= 4, f"Expected >=4 scenes, got {len(scenes)}"
    for i, scene in enumerate(scenes):
        assert "start_seconds" in scene
        assert "end_seconds" in scene
        assert scene["end_seconds"] > scene["start_seconds"]
        if i > 0:
            assert abs(scene["start_seconds"] - scenes[i - 1]["end_seconds"]) < 0.01
    assert abs(scenes[0]["start_seconds"]) < 0.01


def test_detect_scenes_sample_video_count():
    vp = _video_path()
    if vp is None:
        return
    scenes = detect_scenes(vp)
    assert len(scenes) >= 5, f"Expected at least 5 scenes, got {len(scenes)}"


def test_detect_scenes_missing_file():
    import pytest
    with pytest.raises(FileNotFoundError):
        detect_scenes(Path("/nonexistent/video.mp4"))


def test_probe_media_to_dict_serializable():
    vp = _video_path()
    if vp is None:
        return
    result = probe_media(vp)
    d = result.to_dict()
    # Must be JSON-serializable
    serialized = json.dumps(d)
    assert len(serialized) > 10
