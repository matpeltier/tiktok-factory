"""Performance feature extraction and exploratory baseline (issue #40).

Deterministically converts each validated CreativeIR record into structural
features and correlates them with public engagement metrics. Includes a
regularized CV baseline on log1p(views). Runs fully offline on existing
artifacts; no model API calls.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDS_DIR = REPO_ROOT / "dataset" / "records"
DATASET_DIR = REPO_ROOT / "dataset"

PACING_ORDINAL = {"very_fast": 4.0, "fast": 3.0, "moderate": 2.0, "slow": 1.0}
SEMANTIC_ROLES = ["hook", "challenge", "search", "reveal", "payoff", "setup", "product_demo", "product_proof", "cta"]
HOOK_TYPES = ["curiosity_gap", "challenge", "countdown", "narrative_promise", "visual_novelty", "product_promise"]
MECHANISMS = ["curiosity_gap", "active_participation", "time_pressure", "payoff_reveal", "visual_novelty", "social_proof", "personal_experience", "before_after"]
ARCS = ["setup_challenge_search_reveal", "setup_payoff", "problem_solution", "problem_proof_cta"]
FORMATS = ["challenge", "tutorial", "story", "demonstration", "list", "reaction", "product_ad"]

TARGETS = ["log1p_views", "log1p_like_rate", "like_rate", "comment_rate", "share_rate"]


def extract_features(ir: dict, metadata: dict) -> dict:
    """Convert one validated CreativeIR + metadata pair into flat features."""
    shots = ir.get("observed", {}).get("shots", [])
    duration = ir.get("source", {}).get("observed", {}).get("duration_seconds") or 0.0
    shot_durations = [
        max(0.0, s.get("time_range", {}).get("end_seconds", 0) - s.get("time_range", {}).get("start_seconds", 0))
        for s in shots
    ]
    text_segments = [
        seg
        for s in shots
        for seg in s.get("observed", {}).get("text", {}).get("segments", [])
    ]
    pacing_values = [
        PACING_ORDINAL[e]
        for s in shots
        if (e := s.get("observed", {}).get("editing", {}).get("pacing")) in PACING_ORDINAL
    ]
    transitions = [
        s.get("observed", {}).get("editing", {}).get("transition_in")
        for s in shots
        if s.get("observed", {}).get("editing", {}).get("transition_in") not in (None, "start")
    ]
    motions = [s.get("observed", {}).get("camera", {}).get("motion") for s in shots]
    framings = [s.get("observed", {}).get("camera", {}).get("framing") for s in shots]
    roles = [s.get("inferred", {}).get("semantic_role") for s in shots]
    dialogue_presence = [s.get("observed", {}).get("dialogue", {}).get("presence") for s in shots]
    sound_effects = [s.get("observed", {}).get("audio", {}).get("sound_effects") for s in shots]

    hook_types = ir.get("inferred", {}).get("hook", {}).get("hook_types", []) or []
    shot_mechanisms = {m for s in shots for m in (s.get("inferred", {}).get("attention_mechanisms") or [])}
    global_mechanisms = set(ir.get("inferred", {}).get("marketing", {}).get("mechanisms") or [])
    devices = ir.get("observed", {}).get("marketing", {}).get("engagement_devices", []) or []

    features: dict = {
        "video_id": metadata.get("video_id") or ir.get("source", {}).get("observed", {}).get("video_id"),
        "duration_seconds": duration,
        "shot_count": len(shots),
        "mean_shot_duration": float(np.mean(shot_durations)) if shot_durations else None,
        "median_shot_duration": float(np.median(shot_durations)) if shot_durations else None,
        "max_shot_duration": max(shot_durations) if shot_durations else None,
        "cut_rate_per_10s": (len(shots) / duration * 10.0) if duration > 0 else None,
        "ocr_segment_count": len(text_segments),
        "ocr_char_total": sum(len(seg.get("text", "")) for seg in text_segments),
        "cta_present": int(bool(ir.get("observed", {}).get("marketing", {}).get("call_to_action_text_ids"))),
        "engagement_device_count": len(set(devices)),
        "dialogue_present_ratio": dialogue_presence.count("present") / len(shots) if shots else None,
        "sound_effects_ratio": sound_effects.count("present") / len(shots) if shots else None,
        "static_shot_ratio": motions.count("static") / len(shots) if shots else None,
        "closeup_ratio": sum(1 for f in framings if f in ("close_up", "extreme_close_up")) / len(shots) if shots else None,
        "pacing_mean": float(np.mean(pacing_values)) if pacing_values else None,
        "pacing_fast_ratio": sum(1 for p in pacing_values if p >= 3.0) / len(pacing_values) if pacing_values else None,
        "cut_transition_ratio": transitions.count("cut") / len(transitions) if transitions else None,
        "motion_variety": len({m for m in motions if m}) / len(shots) if shots else None,
    }
    for role in SEMANTIC_ROLES:
        features[f"role_{role}_ratio"] = roles.count(role) / len(shots) if shots else None
    for hook_type in HOOK_TYPES:
        features[f"hook_{hook_type}"] = int(hook_type in hook_types)
    for mechanism in MECHANISMS:
        features[f"mech_{mechanism}"] = int(mechanism in shot_mechanisms or mechanism in global_mechanisms)
    arc = ir.get("inferred", {}).get("narrative", {}).get("arc")
    for candidate in ARCS:
        features[f"arc_{candidate}"] = int(arc == candidate)
    fmt = ir.get("inferred", {}).get("overall_concept", {}).get("format")
    for candidate in FORMATS:
        features[f"format_{candidate}"] = int(fmt == candidate)

    views = metadata.get("views")
    likes = metadata.get("likes")
    comments = metadata.get("comments")
    shares = metadata.get("shares")
    features["views"] = views
    features["log1p_views"] = math.log1p(views) if isinstance(views, (int, float)) and views is not None else None
    like_rate = likes / views if isinstance(views, (int, float)) and views and isinstance(likes, (int, float)) else None
    features["like_rate"] = like_rate
    features["log1p_like_rate"] = math.log1p(like_rate) if like_rate is not None else None
    features["comment_rate"] = comments / views if isinstance(views, (int, float)) and views and isinstance(comments, (int, float)) else None
    features["share_rate"] = shares / views if isinstance(views, (int, float)) and views and isinstance(shares, (int, float)) else None
    return features


def build_features_table(records_dir: Path | None = None) -> pd.DataFrame:
    """Extract features for every decompiled record (status ok)."""
    records_dir = records_dir or RECORDS_DIR
    rows = []
    for record_dir in sorted(records_dir.iterdir()):
        if not record_dir.is_dir():
            continue
        parsed_path = record_dir / "creative_ir.parsed.json"
        metadata_path = record_dir / "metadata.json"
        record_path = record_dir / "record.json"
        if not parsed_path.exists() or not metadata_path.exists():
            continue
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("status") != "ok":
                continue
        ir = json.loads(parsed_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows.append(extract_features(ir, metadata))
    return pd.DataFrame(rows)


def save_features_parquet(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or (DATASET_DIR / "features.parquet")
    df.to_parquet(path, index=False)
    return path


def spearman_report(df: pd.DataFrame, feature_cols: list[str], targets: list[str]) -> pd.DataFrame:
    """Spearman correlation of each feature with each target, with p-values."""
    from scipy.stats import spearmanr

    rows = []
    for feature in feature_cols:
        for target in targets:
            sub = df[[feature, target]].dropna()
            if len(sub) < 5 or sub[feature].nunique() < 2:
                rows.append({"feature": feature, "target": target, "spearman": None, "p_value": None, "n": len(sub)})
                continue
            rho, p = spearmanr(sub[feature], sub[target])
            rows.append({"feature": feature, "target": target, "spearman": rho, "p_value": p, "n": len(sub)})
    return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False, na_position="last")


def fit_baseline(df: pd.DataFrame, feature_cols: list[str], target: str = "log1p_views", n_splits: int = 5, seed: int = 7) -> dict:
    """Ridge regression on standardized features vs a median baseline, k-fold CV."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sub = df[feature_cols + [target]].dropna()
    X = sub[feature_cols].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)

    median_pred = np.full_like(y, y.mean())
    baseline_r2 = 1 - np.sum((y - median_pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    baseline_mae = float(np.mean(np.abs(y - median_pred)))

    kfold = KFold(n_splits=min(n_splits, max(2, len(sub) // 10)), shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    pred = cross_val_predict(model, X, y, cv=kfold)
    cv_r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    cv_mae = float(np.mean(np.abs(y - pred)))

    model.fit(X, y)
    coefficients = model.named_steps["ridge"].coef_
    ranked = sorted(zip(feature_cols, coefficients), key=lambda pair: -abs(pair[1]))

    return {
        "n": int(len(sub)),
        "cv_r2": float(cv_r2),
        "cv_mae_log_views": cv_mae,
        "baseline_r2": float(baseline_r2),
        "baseline_mae_log_views": baseline_mae,
        "top_coefficients": [(name, float(coef)) for name, coef in ranked[:15]],
    }


def main() -> None:
    df = build_features_table()
    path = save_features_parquet(df)
    print(f"Features for {len(df)} records saved to {path}")

    feature_cols = [
        c
        for c in df.columns
        if c not in {"video_id", "views", *TARGETS} and df[c].notna().sum() >= 10 and df[c].nunique() >= 2
    ]
    print(f"{len(feature_cols)} usable features")

    print("\n=== Top Spearman correlations with engagement ===")
    report = spearman_report(df, feature_cols, TARGETS)
    print(report.head(20).to_string(index=False))

    print("\n=== Ridge CV baseline: log1p(views) ===")
    for target in ("log1p_views", "log1p_like_rate"):
        result = fit_baseline(df, feature_cols, target=target)
        print(f"--- target: {target} ---")
        print(json.dumps({k: v for k, v in result.items() if k != "top_coefficients"}, indent=1))
        print("Top |coefficients|:")
        for name, coef in result["top_coefficients"][:10]:
            print(f"  {coef:+.3f}  {name}")

    print("\nCaveats: n=85 single-creator observational sample; correlations are "
          "exploratory, not causal; many features tested -> multiple-comparison risk.")


if __name__ == "__main__":
    main()
