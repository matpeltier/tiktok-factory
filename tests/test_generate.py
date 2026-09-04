"""Tests for the fal generation loop (issue #42)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fal_client
import generate
from generate import (
    PROVIDERS,
    assemble_hook,
    build_payload,
    build_shot_prompt,
    select_hook_shots,
)


def _ir(hook_ids=None):
    shots = []
    for i in range(5):
        shots.append(
            {
                "shot_id": f"shot_{i}",
                "time_range": {"start_seconds": float(i * 3), "end_seconds": float((i + 1) * 3)},
                "generation": {
                    "reconstruction_prompt": f"Shot {i}: place logs on the meadow.",
                    "continuity_requirements": ["Spruce wood theme"],
                },
                "inferred": {"semantic_role": "hook" if i < 2 else "payoff"},
            }
        )
    return {
        "creative_ir_version": "0.1",
        "source": {"observed": {"video_id": "v1", "duration_seconds": 15.0}},
        "observed": {"shots": shots, "hook": {"shot_ids": hook_ids or ["shot_0"]}},
        "generation": {
            "shot_order": [s["shot_id"] for s in shots],
            "global_constraints": ["Warm golden sunlight"],
            "global_reconstruction_brief": "Reconstruct a vertical Minecraft tutorial.",
        },
    }


def test_build_payload_omni_flash():
    payload = build_payload(PROVIDERS["omni-flash"], "a prompt", 6)
    assert payload["prompt"] == "a prompt"
    assert payload["duration"] == 6
    assert payload["aspect_ratio"] == "9:16"
    assert payload["resolution"] == "720p"


def test_build_payload_veo_lite_audio_on():
    payload = build_payload(PROVIDERS["veo3.1-lite"], "a prompt", 12)
    assert payload["duration"] == "8s"
    assert payload["aspect_ratio"] == "9:16"
    assert payload["generate_audio"] is True


def test_build_payload_clamps_duration():
    assert build_payload(PROVIDERS["omni-flash"], "p", 30)["duration"] == 8
    assert build_payload(PROVIDERS["omni-flash"], "p", 1)["duration"] == 4


def test_build_shot_prompt_includes_reconstruction_and_constraints():
    ir = _ir()
    prompt = build_shot_prompt(ir["observed"]["shots"][0], ir)
    assert "Shot 0: place logs" in prompt
    assert "Spruce wood theme" in prompt
    assert "Warm golden sunlight" in prompt
    assert "9:16" in prompt


def test_select_hook_shots_prioritizes_hook_and_caps():
    ir = _ir(hook_ids=["shot_2"])
    selected = select_hook_shots(ir, max_shots=2)
    assert [s["shot_id"] for s in selected] == ["shot_2", "shot_0"]


def test_select_hook_shots_falls_back_to_first_shots():
    ir = _ir(hook_ids=[])
    ir["observed"]["hook"]["shot_ids"] = []
    selected = select_hook_shots(ir, max_shots=3)
    assert [s["shot_id"] for s in selected] == ["shot_0", "shot_1", "shot_2"]


def test_provider_cost_estimates():
    assert PROVIDERS["veo3.1-lite"].estimate_cost(5) == pytest.approx(0.25)
    assert PROVIDERS["omni-flash"].estimate_cost(8) == pytest.approx(0.15)


def test_extract_video_url_variants():
    assert generate._extract_video_url({"video": {"url": "http://x"}}) == "http://x"
    assert generate._extract_video_url({"videos": [{"url": "http://y"}]}) == "http://y"
    with pytest.raises(fal_client.FalError, match="No video URL"):
        generate._extract_video_url({})


def test_fal_client_missing_key(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(fal_client.FalError, match="FAL_KEY"):
        fal_client.submit("some/endpoint", {})


def test_generate_hook_end_to_end_mocked(tmp_path, monkeypatch):
    calls = []

    def fake_run(endpoint_id, payload, timeout_seconds=600):
        calls.append((endpoint_id, payload))
        return {"video": {"url": f"http://cdn/{len(calls)}.mp4"}, "request_id": f"req-{len(calls)}"}

    def fake_download(url, destination, **kwargs):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-mp4")
        return destination

    monkeypatch.setattr(fal_client, "run", fake_run)
    monkeypatch.setattr(fal_client, "download", fake_download)

    ir = _ir(hook_ids=["shot_0", "shot_1"])
    record = generate.generate_hook(
        ir, provider_keys=["omni-flash", "veo3.1-lite"], max_shots=2, out_dir=tmp_path
    )
    assert len(calls) == 4  # 2 shots x 2 providers
    assert [s["provider"] for s in record["shots"]] == ["omni-flash", "veo3.1-lite", "omni-flash", "veo3.1-lite"]
    assert all((tmp_path / s["clip"]).exists() for s in record["shots"])
    assert record["cost_usd_estimate_total"] > 0
    saved = json.loads((tmp_path / "generation.json").read_text())
    assert saved["prompt_version"] == generate.SHOT_PROMPT_VERSION
    assert len(saved["shots"]) == 4


def test_assemble_hook_concatenates(tmp_path):
    if not (Path(__file__).parent.parent / ".orca" / "drops" / "video.mp4").exists():
        pytest.skip("No sample video fixture")
    clips = []
    for i in range(2):
        clip = tmp_path / f"shot_{i}.mp4"
        import subprocess

        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=blue:size=320x568:duration=0.5",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
            capture_output=True, check=True,
        )
        clips.append(clip)
    out = assemble_hook(clips, tmp_path / "hook_preview.mp4")
    assert out.exists() and out.stat().st_size > 0


def test_build_pattern_brief_encodes_dataset_patterns():
    brief = generate.build_pattern_brief()
    shots = brief["observed"]["shots"]
    assert [s["shot_id"] for s in shots] == ["shot_0", "shot_1", "shot_2"]
    roles = [s["inferred"]["semantic_role"] for s in shots]
    assert roles == ["setup", "reveal", "cta"]
    # problem -> proof -> verbal CTA arc
    assert "problem" in shots[0]["generation"]["reconstruction_prompt"].lower()
    assert "press" in shots[1]["generation"]["reconstruction_prompt"].lower()
    assert "verbal call to action" in shots[2]["generation"]["reconstruction_prompt"].lower()
    assert "no on-screen text" in shots[2]["generation"]["reconstruction_prompt"].lower()
    # product facts and pattern provenance are present
    assert "chopper" in brief["generation"]["global_reconstruction_brief"].lower()
    assert any("problem_proof_cta" in c for c in brief["generation"]["global_constraints"])
    # works through the standard generation path
    prompt = build_shot_prompt(shots[1], brief)
    assert "press-and-dice" in prompt


def test_generate_hook_pattern_brief_mocked(tmp_path, monkeypatch):
    def fake_run(endpoint_id, payload, timeout_seconds=600):
        return {"video": {"url": f"http://cdn/{payload['prompt'][:10]}.mp4"}}

    monkeypatch.setattr(fal_client, "run", fake_run)
    monkeypatch.setattr(fal_client, "download", lambda url, dest, **k: (Path(dest).write_bytes(b"x") or Path(dest)))

    ir = generate.build_pattern_brief()
    record = generate.generate_hook(ir, provider_keys=["omni-flash"], max_shots=3, out_dir=tmp_path)
    assert len(record["shots"]) == 3
    assert all((tmp_path / s["clip"]).exists() for s in record["shots"])


def test_assemble_per_provider_builds_one_preview_each(tmp_path):
    if not (Path(__file__).parent.parent / ".orca" / "drops" / "video.mp4").exists():
        pytest.skip("No sample video fixture")
    import subprocess

    clips = []
    for provider in ("omni-flash", "veo3.1-lite"):
        for i in range(2):
            clip = tmp_path / f"shot_{i}_{provider}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=green:size=320x568:duration=0.4",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
                capture_output=True, check=True,
            )
            clips.append({"shot_id": f"shot_{i}", "provider": provider, "clip": clip.name})
    previews = generate.assemble_per_provider({"shots": clips}, tmp_path)
    assert len(previews) == 2
    assert all(p.exists() and p.stat().st_size > 0 for p in previews)


def test_build_payload_i2v_variants(monkeypatch):
    payload = generate.animate_frame.__globals__["build_payload"]  # sanity: t2v untouched
    assert callable(payload)
    # i2v payloads are built inline in animate_frame; test via mocked fal run
    captured = {}

    def fake_run(endpoint_id, payload, timeout_seconds=600):
        captured["endpoint"] = endpoint_id
        captured["payload"] = payload
        return {"video": {"url": "http://cdn/v.mp4"}}

    monkeypatch.setattr(fal_client, "run", fake_run)
    monkeypatch.setattr(fal_client, "download", lambda url, dest, **k: Path(dest).write_bytes(b"x") or Path(dest))
    generate.animate_frame("omni-flash", "http://img", "zoom in", 4)
    assert captured["endpoint"] == generate.I2V_ENDPOINTS["omni-flash"]
    assert captured["payload"]["image_url"] == "http://img"
    assert captured["payload"]["duration"] == 4
    generate.animate_frame("veo3.1-lite", "http://img", "talk", 4)
    assert captured["payload"]["duration"] == "4s"
    assert captured["payload"]["generate_audio"] is True


def test_sfx_delays_clamped_to_timeline():
    delays = generate.sfx_delays(9.6)
    assert [d["delay_ms"] for d in delays] == [800, 4200, 7600]
    short = generate.sfx_delays(6.0)
    assert all(d["delay_ms"] <= (6.0 - 0.4) * 1000 for d in short)


def test_generate_frames_flow_mocked(tmp_path, monkeypatch):
    urls = iter([f"http://cdn/frame{i}.jpg" for i in range(3)])

    def fake_run(endpoint_id, payload, timeout_seconds=600):
        if endpoint_id in (generate.IMAGE_T2I_ENDPOINT, generate.IMAGE_EDIT_ENDPOINT):
            return {"images": [{"url": next(urls)}]}
        if endpoint_id == generate.SFX_ENDPOINT:
            return {"audio": {"url": "http://cdn/sfx.mp3"}}
        return {"video": {"url": f"http://cdn/clip{payload['prompt'][:8]}.mp4"}}

    def fake_download(url, dest, **k):
        Path(dest).write_bytes(b"data")
        return Path(dest)

    monkeypatch.setattr(fal_client, "run", fake_run)
    monkeypatch.setattr(fal_client, "download", fake_download)
    monkeypatch.setattr(generate, "mix_sfx", lambda base, sfx_dir, total, out: out)

    record = generate.generate_frames_flow(tmp_path, ["omni-flash", "veo3.1-lite"])
    assert len([s for s in record["steps"] if s["step"] == "frame"]) == 3
    assert len([s for s in record["steps"] if s["step"] == "sfx"]) == 3
    assert record["cost_usd_estimate_total"] > 0
    # shots animated for both providers
    assert len([s for s in record["steps"] if s["step"] == "animate"]) == 6


def test_trim_and_mix_real_ffmpeg(tmp_path):
    import subprocess

    if not (Path(__file__).parent.parent / ".orca" / "drops" / "video.mp4").exists():
        pytest.skip("No sample video fixture")
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:size=320x568:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(clip)],
        capture_output=True, check=True,
    )
    trimmed = generate.trim_clip(clip, tmp_path / "trim.mp4", 1.0)
    assert trimmed.exists()
    sfx = tmp_path / "sfx_chop.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=880:duration=0.5", str(sfx)],
        capture_output=True, check=True,
    )
    (tmp_path / "sfx_chop.mp3").rename(tmp_path / "sfx_chop.mp3")
    mixed = generate.mix_sfx(trimmed, tmp_path, 1.0, tmp_path / "mixed.mp4")
    assert mixed.exists() and mixed.stat().st_size > 0
