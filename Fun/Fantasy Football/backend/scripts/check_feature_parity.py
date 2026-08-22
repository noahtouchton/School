"""Guard: the inference feature frame must supply every column each model was trained on.

Training (ml/features.py) and inference (ml/predict.py) build their rolling windows
in separate code paths -- training walks each row's prior games, inference only needs
each player's latest state. That split is deliberate but it means a feature added to
one side can silently go missing on the other, and the failure shows up as a KeyError
in the middle of generating projections rather than at the point of the mistake.

Run this after touching either module.

Usage:
    ./.venv/bin/python scripts/check_feature_parity.py
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import SessionLocal
from app.ml.features import POSITIONS, features_for
from app.ml.predict import build_current_feature_frame


def main() -> int:
    season = dt.date.today().year
    db = SessionLocal()
    try:
        df = build_current_feature_frame(db, season, 1)
    finally:
        db.close()

    if df.empty:
        print("FAIL: inference frame is empty -- no data ingested?")
        return 1

    failures = []
    for position in POSITIONS:
        required = features_for(position)
        missing = [c for c in required if c not in df.columns]
        if missing:
            failures.append((position, missing))
            print(f"FAIL {position}: inference frame missing {missing}")
        else:
            print(f"ok   {position}: all {len(required)} features present")

    if failures:
        print("\nInference frame cannot satisfy every model. Fix before generating projections.")
        return 1

    print("\nFeature parity OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
