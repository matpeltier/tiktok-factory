"""Tests for deterministic performance feature extraction (issue #40)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import performance


def _ir():
    return {
        "creative_ir_version": "0.1",
        "source": {"observed": {"video_id": "v1", "duration_seconds": 4.0}},
        "observed": {
            "shots": [
                {
                    "shot_id": "shot_0",
                    "time_range": {"start_seconds": 0.0, "end_seconds": 1.0},
                    "observed": {
                        "camera": {"framing": "close_up", "motion": "static"},
                        "text": {"segments": [{"text_id": "text_0", "text": "DAY 1"}]},
                        "dialogue": {"presence": "present", "segments": []},
                        "audio": {"sound_effects": "present"},
                        "editing": {"pacing": "fast", "transition_in": "cut"},
                    },
                    "inferred": {"semantic_role": "hook", "attention_mechanisms": ["curiosity_gap"]},
                },
                {
                    "shot_id": "shot_1",
                    "time_range": {"start_seconds": 1.0, "end_seconds": 4.0},
                    "observed": {
                        "camera": {"framing": "wide", "motion": "pan"},
                        "text": {"segments": []},
                        "dialogue": {"presence": "absent", "segments": []},
                        "audio": {"sound_effects": "absent"},
                        "editing": {"pacing": "slow", "transition_in": "fade"},
                    },
                    "inferred": {"semantic_role": "payoff", "attention_mechanisms": []},
                },
            ],
            "hook": {"shot_ids": ["shot_0"]},
            "marketing": {"call_to_action_text_ids": ["text_0"], "engagement_devices": ["explicit_question", "countdown"]},
        },
        "inferred": {
            "overall_concept": {"format": "tutorial"},
            "hook": {"hook_types": ["curiosity_gap", "visual_novelty"]},
            "narrative": {"arc": "setup_payoff"},
            "marketing": {"mechanisms": ["payoff_reveal"]},
        },
        "generation": {"shot_order": ["shot_0", "shot_1"]},
    }


def _metadata():
    return {"video_id": "v1", "views": 100000, "likes": 10000, "comments": 100, "shares": 50}


def test_extract_features_structure():
    features = performance.extract_features(_ir(), _metadata())
    assert features["video_id"] == "v1"
    assert features["duration_seconds"] == 4.0
    assert features["shot_count"] == 2
    assert features["mean_shot_duration"] == 2.0
    assert features["max_shot_duration"] == 3.0
    assert features["cut_rate_per_10s"] == 5.0
    assert features["ocr_segment_count"] == 1
    assert features["ocr_char_total"] == 5
    assert features["cta_present"] == 1
    assert features["engagement_device_count"] == 2
    assert features["dialogue_present_ratio"] == 0.5
    assert features["sound_effects_ratio"] == 0.5
    assert features["static_shot_ratio"] == 0.5
    assert features["closeup_ratio"] == 0.5
    assert features["pacing_mean"] == 2.0
    assert features["pacing_fast_ratio"] == 0.5
    assert features["cut_transition_ratio"] == 0.5
    assert features["motion_variety"] == 1.0
    assert features["role_hook_ratio"] == 0.5
    assert features["role_payoff_ratio"] == 0.5
    assert features["hook_curiosity_gap"] == 1
    assert features["hook_countdown"] == 0
    assert features["mech_curiosity_gap"] == 1
    assert features["mech_payoff_reveal"] == 1
    assert features["arc_setup_payoff"] == 1
    assert features["format_tutorial"] == 1


def test_extract_features_targets():
    features = performance.extract_features(_ir(), _metadata())
    assert features["views"] == 100000
    assert features["log1p_views"] == pytest.approx(__import__("math").log1p(100000))
    assert features["like_rate"] == pytest.approx(0.1)
    assert features["comment_rate"] == pytest.approx(0.001)
    assert features["share_rate"] == pytest.approx(0.0005)


def test_extract_features_handles_missing_metadata():
    features = performance.extract_features(_ir(), {"video_id": "v1"})
    assert features["log1p_views"] is None
    assert features["like_rate"] is None


def test_build_features_table_skips_non_ok_records(tmp_path):
    for vid, status in (("ok1", "ok"), ("bad", "failed")):
        d = tmp_path / vid
        d.mkdir()
        (d / "creative_ir.parsed.json").write_text(json.dumps(_ir()))
        (d / "metadata.json").write_text(json.dumps({**_metadata(), "video_id": vid}))
        (d / "record.json").write_text(json.dumps({"status": status}))
    df = performance.build_features_table(tmp_path)
    assert list(df["video_id"]) == ["ok1"]


def test_fit_baseline_returns_cv_and_coefficients():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "shot_count": rng.integers(5, 40, n).astype(float),
        "duration_seconds": rng.uniform(10, 120, n),
        "log1p_views": rng.normal(12, 1, n),
    })
    result = performance.fit_baseline(df, ["shot_count", "duration_seconds"])
    assert result["n"] == 60
    assert -10 <= result["cv_r2"] <= 1
    assert len(result["top_coefficients"]) == 2
