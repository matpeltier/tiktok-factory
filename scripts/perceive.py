"""Deterministic video perception: ffprobe probing + PySceneDetect scene detection."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


@dataclass(frozen=True)
class PerceptionResult:
    duration_seconds: float
    width: int
    height: int
    fps: float
    nb_frames: int
    video_codec: str
    audio_codec: str | None
    aspect_ratio_label: str
    file_size_bytes: int
    bit_rate: int | None
    raw_streams: list[dict] = field(default_factory=list)
    raw_format: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "duration_seconds": self.duration_seconds,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "nb_frames": self.nb_frames,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "aspect_ratio_label": self.aspect_ratio_label,
            "file_size_bytes": self.file_size_bytes,
            "bit_rate": self.bit_rate,
        }


def _classify_aspect(width: int, height: int) -> str:
    ratio = width / height if height else 0
    if abs(ratio - 9 / 16) < 0.05:
        return "vertical_9_16"
    if abs(ratio - 1) < 0.05:
        return "square"
    if abs(ratio - 16 / 9) < 0.05:
        return "horizontal_16_9"
    if ratio < 1:
        return "vertical_other"
    return "horizontal_other"


def probe_media(video_path: Path) -> PerceptionResult:
    """Extract exact media facts from a video file using ffprobe."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    raw = json.loads(_run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path),
    ]))

    file_size = video_path.stat().st_size

    video_stream = None
    audio_stream = None
    for s in raw.get("streams", []):
        if s.get("codec_type") == "video" and video_stream is None:
            video_stream = s
        elif s.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = s

    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    # Duration: prefer format, fallback to stream
    duration_str = raw.get("format", {}).get("duration")
    if duration_str is None:
        duration_str = video_stream.get("duration")
    if duration_str is None:
        raise ValueError(f"Cannot determine duration for {video_path}")
    duration = float(duration_str)

    width = int(video_stream["width"])
    height = int(video_stream["height"])

    # Frame rate: parse r_frame_rate "num/den"
    fps_str = video_stream.get("r_frame_rate", "0/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0

    nb_frames = int(video_stream.get("nb_frames", 0))
    video_codec = video_stream.get("codec_name", "unknown")
    audio_codec = audio_stream.get("codec_name") if audio_stream else None
    bit_rate = int(raw.get("format", {}).get("bit_rate", 0)) or None

    return PerceptionResult(
        duration_seconds=duration,
        width=width,
        height=height,
        fps=fps,
        nb_frames=nb_frames,
        video_codec=video_codec,
        audio_codec=audio_codec,
        aspect_ratio_label=_classify_aspect(width, height),
        file_size_bytes=file_size,
        bit_rate=bit_rate,
        raw_streams=raw.get("streams", []),
        raw_format=raw.get("format", {}),
    )


def detect_scenes(video_path: Path, threshold: float = 27.0) -> list[dict]:
    """Detect hard shot boundaries using PySceneDetect ContentDetector."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        # Fallback: single scene spanning entire video
        probe = probe_media(video_path)
        return [{"start_seconds": 0.0, "end_seconds": probe.duration_seconds}]

    return [
        {
            "start_seconds": round(start.seconds, 3),
            "end_seconds": round(end.seconds, 3),
        }
        for start, end in scene_list
    ]
