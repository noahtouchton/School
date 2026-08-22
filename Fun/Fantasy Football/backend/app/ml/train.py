"""Trains one XGBoost regressor per position to predict weekly fantasy points.

Approach: predict a *residual* on top of the rolling-8-game average rather than raw points.
Predicting points directly was tried first and only clearly beat the naive rolling-average
baseline for QB; for RB/WR/TE (especially WR/TE, which are notoriously boom-bust/TD-variance
driven week to week) it landed at or slightly worse than just using the rolling average. The
residual framing fixes this structurally: worst case, the model learns to predict ~0 residual
and you get the baseline back. On top of that, a blend weight between "pure baseline" and "full
model" is grid-searched per position on the held-out validation season and stored with the
model, so a position only gets the ML correction applied if it actually earns it on validation.

Validation strategy: hold out the most recent season as validation (out-of-time split, not
random) so the reported MAE reflects genuine forward-looking accuracy.
"""

from __future__ import annotations

import datetime as dt

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sqlalchemy.orm import Session
from xgboost import XGBRegressor

from app.core.config import DATA_DIR
from app.db.models import TrainedModelArtifact
from app.ml.features import (
    BASELINE_COLUMN,
    FEATURE_COLUMNS,
    POSITIONS,
    build_training_frame,
    features_for,
)

MODEL_DIR = DATA_DIR / "models"
# v4 adds opportunity/usage features (target share, WOPR, carries, air yards).
# The version is part of the model filename and the projections primary key, so
# bumping it keeps v3 artifacts and projections intact for comparison instead of
# silently loading a model trained on a different feature set.
MODEL_VERSION = "xgb_v4_usage"

BLEND_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

# Number of most-recent seasons used as walk-forward validation folds.
#
# A single held-out season is too small to choose a blend weight on: one season of
# WR rows put the model's edge over the rolling-average baseline at ~0.01 points of
# MAE, which is comfortably inside season-to-season noise, so the "best" blend was
# being picked off a coin flip. Pooling several rolling-origin folds multiplies the
# validation data and makes both the blend choice and the reported MAE mean something.
VALIDATION_SEASONS = 3


def _new_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        min_child_weight=20,
        early_stopping_rounds=40,
        random_state=42,
    )


def _fit(train_df: pd.DataFrame, features: list[str]) -> XGBRegressor | None:
    """Fit on train_df, holding its most recent season out for early stopping only."""
    if len(train_df) < 50:
        return None
    last_season = sorted(train_df.season.unique())[-1]
    fit_df = train_df[train_df.season < last_season]
    eval_df = train_df[train_df.season == last_season]
    if len(fit_df) < 50 or len(eval_df) < 10:
        fit_df, eval_df = train_df, train_df

    model = _new_model()
    model.fit(
        fit_df[features],
        fit_df["residual"],
        eval_set=[(eval_df[features], eval_df["residual"])],
        verbose=False,
    )
    return model


def train_position_model(df: pd.DataFrame, position: str, validation_season: int) -> dict:
    """Walk-forward validation to pick the blend weight, then a production refit.

    For each of the last VALIDATION_SEASONS seasons we train only on seasons strictly
    before it and predict it -- never training on the future of what we score. The
    fold predictions are pooled and the blend weight is chosen once on that pool.
    The shipped model is then refit on everything through validation_season, since
    for real 2026 projections there's no reason to throw away the most recent season.
    """
    features = features_for(position)
    pos_df = df[df.position == position].dropna(subset=features + ["points"]).copy()
    pos_df["residual"] = pos_df["points"] - pos_df[BASELINE_COLUMN]

    # int() matters: these land in metadata_json, and numpy int64 isn't JSON serializable.
    seasons = sorted(int(s) for s in pos_df.season.unique())
    fold_seasons = [s for s in seasons if validation_season - VALIDATION_SEASONS < s <= validation_season]
    if not fold_seasons or len(pos_df) < 200:
        return {"position": position, "status": "skipped", "reason": "insufficient data"}

    pooled_actual: list[pd.Series] = []
    pooled_baseline: list[pd.Series] = []
    pooled_residual: list[np.ndarray] = []
    per_fold: dict[int, float] = {}

    for season in fold_seasons:
        train_df = pos_df[pos_df.season < season]
        val_df = pos_df[pos_df.season == season]
        if len(train_df) < 50 or len(val_df) < 10:
            continue
        model = _fit(train_df, features)
        if model is None:
            continue
        residual_pred = model.predict(val_df[features])
        pooled_actual.append(val_df["points"])
        pooled_baseline.append(val_df[BASELINE_COLUMN])
        pooled_residual.append(residual_pred)
        per_fold[int(season)] = round(
            float(mean_absolute_error(val_df["points"], val_df[BASELINE_COLUMN])), 3
        )

    if not pooled_actual:
        return {"position": position, "status": "skipped", "reason": "no usable folds"}

    actual = pd.concat(pooled_actual).to_numpy()
    baseline = pd.concat(pooled_baseline).to_numpy()
    residual = np.concatenate(pooled_residual)

    baseline_mae = mean_absolute_error(actual, baseline)
    best_blend, best_mae = 0.0, baseline_mae
    blend_curve = {}
    for blend in BLEND_GRID:
        mae = mean_absolute_error(actual, baseline + blend * residual)
        blend_curve[blend] = round(float(mae), 4)
        if mae < best_mae:
            best_blend, best_mae = blend, mae

    # Production refit on all data through validation_season.
    final_model = _fit(pos_df[pos_df.season <= validation_season], features)
    if final_model is None:
        return {"position": position, "status": "skipped", "reason": "final fit failed"}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{position.lower()}_{MODEL_VERSION}.joblib"
    joblib.dump(final_model, model_path)

    return {
        "position": position,
        "status": "trained",
        "folds": fold_seasons,
        "val_rows": int(len(actual)),
        "mae": round(float(best_mae), 3),
        "baseline_mae": round(float(baseline_mae), 3),
        "blend_weight": best_blend,
        "improvement_pct": round(100 * (baseline_mae - best_mae) / baseline_mae, 2),
        "blend_curve": blend_curve,
        "baseline_mae_by_fold": per_fold,
        "model_path": str(model_path),
    }


def train_all(db: Session, start_season: int, end_season: int, validation_season: int) -> list[dict]:
    df = build_training_frame(db, start_season, end_season)

    results = []
    for position in POSITIONS:
        result = train_position_model(df, position, validation_season)
        results.append(result)
        if result["status"] != "trained":
            print(f"  {position}: {result}")
            continue
        print(
            f"  {position}: MAE {result['mae']} vs baseline {result['baseline_mae']} "
            f"({result['improvement_pct']:+.2f}%), blend={result['blend_weight']}, "
            f"folds={result['folds']}, val_rows={result['val_rows']}"
        )
        print(f"        blend curve: {result['blend_curve']}")

        if result["status"] == "trained":
            db.merge(
                TrainedModelArtifact(
                    name=f"projection_{position.lower()}",
                    version=MODEL_VERSION,
                    kind="projection_model",
                    path=result["model_path"],
                    metadata_json={
                        "mae": result["mae"],
                        "baseline_mae": result["baseline_mae"],
                        "blend_weight": result["blend_weight"],
                        "improvement_pct": result["improvement_pct"],
                        "trained_at": dt.datetime.utcnow().isoformat(),
                        "val_rows": result["val_rows"],
                        "validation_folds": result["folds"],
                        "blend_curve": result["blend_curve"],
                        "feature_columns": features_for(position),
                    },
                )
            )
    db.commit()
    return results
