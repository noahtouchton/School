"""Ablation: do the usage features actually earn their place?

Runs the identical walk-forward validation three times per position, changing
nothing but the feature set:

  v3       pre-usage features only
  v4-all   every usage feature, regardless of whether it means anything
  v4-pos   only the usage features that apply to that position (what ships)

Folds, hyperparameters and blend search are held constant, which is the only way
the comparison means anything. The v4-all arm exists because handing QB a pile of
structurally-zero receiving columns measurably hurt it -- that's what motivated
the per-position split.

Usage:
    ./.venv/bin/python scripts/ablate_usage_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.ml.features import (
    BASELINE_COLUMN,
    CORE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    POSITIONS,
    build_training_frame,
    features_for,
)
from app.ml.train import BLEND_GRID, VALIDATION_SEASONS

V3_FEATURES = CORE_FEATURE_COLUMNS


def _fit(train_df: pd.DataFrame, features: list[str]) -> XGBRegressor | None:
    if len(train_df) < 50:
        return None
    last_season = sorted(train_df.season.unique())[-1]
    fit_df = train_df[train_df.season < last_season]
    eval_df = train_df[train_df.season == last_season]
    if len(fit_df) < 50 or len(eval_df) < 10:
        fit_df, eval_df = train_df, train_df
    model = XGBRegressor(
        n_estimators=1000, max_depth=3, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=20,
        early_stopping_rounds=40, random_state=42,
    )
    model.fit(
        fit_df[features], fit_df["residual"],
        eval_set=[(eval_df[features], eval_df["residual"])], verbose=False,
    )
    return model


def evaluate(df: pd.DataFrame, position: str, features: list[str], validation_season: int) -> dict:
    pos_df = df[df.position == position].dropna(subset=features + ["points"]).copy()
    pos_df["residual"] = pos_df["points"] - pos_df[BASELINE_COLUMN]
    seasons = sorted(int(s) for s in pos_df.season.unique())
    folds = [s for s in seasons if validation_season - VALIDATION_SEASONS < s <= validation_season]

    actual, baseline, residual = [], [], []
    for season in folds:
        train_df = pos_df[pos_df.season < season]
        val_df = pos_df[pos_df.season == season]
        model = _fit(train_df, features)
        if model is None or len(val_df) < 10:
            continue
        actual.append(val_df["points"])
        baseline.append(val_df[BASELINE_COLUMN])
        residual.append(model.predict(val_df[features]))

    actual = pd.concat(actual).to_numpy()
    baseline = pd.concat(baseline).to_numpy()
    residual = np.concatenate(residual)

    base_mae = mean_absolute_error(actual, baseline)
    best_blend, best_mae = 0.0, base_mae
    curve = {}
    for blend in BLEND_GRID:
        mae = mean_absolute_error(actual, baseline + blend * residual)
        curve[blend] = round(float(mae), 4)
        if mae < best_mae:
            best_blend, best_mae = blend, mae
    return {
        "baseline_mae": round(float(base_mae), 4),
        "mae": round(float(best_mae), 4),
        "blend": best_blend,
        "gain_pct": round(100 * (base_mae - best_mae) / base_mae, 2),
        "curve": curve,
        "rows": len(actual),
    }


def main() -> None:
    settings = get_settings()
    validation_season = settings.ingestion_end_year
    db = SessionLocal()
    try:
        print("Building feature frame once (shared by both arms)...")
        df = build_training_frame(db, settings.ingestion_start_year, settings.ingestion_end_year)
    finally:
        db.close()

    print(f"\nWalk-forward folds: last {VALIDATION_SEASONS} seasons through {validation_season}")
    print(f"{'pos':<5}{'arm':<8}{'baseline':>10}{'MAE':>9}{'blend':>7}{'gain%':>8}{'rows':>8}")
    print("-" * 55)

    results = {}
    for position in POSITIONS:
        arms = {
            "v3": evaluate(df, position, V3_FEATURES, validation_season),
            "v4-all": evaluate(df, position, FEATURE_COLUMNS, validation_season),
            "v4-pos": evaluate(df, position, features_for(position), validation_season),
        }
        for name, r in arms.items():
            print(
                f"{position:<5}{name:<8}{r['baseline_mae']:>10}{r['mae']:>9}"
                f"{r['blend']:>7}{r['gain_pct']:>8}{r['rows']:>8}"
            )
        print(f"     v4-pos curve: {arms['v4-pos']['curve']}")
        print()
        results[position] = arms

    print("SUMMARY (winner = lowest pooled walk-forward MAE)")
    for position, arms in results.items():
        winner = min(arms, key=lambda k: arms[k]["mae"])
        shipped = arms["v4-pos"]
        v3 = arms["v3"]
        delta = v3["mae"] - shipped["mae"]
        print(
            f"  {position}: winner={winner:<7} shipped(v4-pos) {shipped['mae']} "
            f"vs v3 {v3['mae']} ({100 * delta / v3['mae']:+.2f}%), blend={shipped['blend']}"
        )


if __name__ == "__main__":
    main()
