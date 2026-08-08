"""Generates weekly projections for the upcoming (season, week) using the trained models.

Builds a "current state" feature row per player: rolling stats computed from all real games
played so far (across seasons), opponent-defense strength estimated from the most relevant
available window (this season's prior weeks if any exist, otherwise last season's full-season
rate), and the new-team flag from season-level team history. This mirrors the training feature
pipeline (app/ml/features.py) as closely as possible so inference-time features are distributed
like training-time features.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.scoring import calculate_fantasy_points
from app.db.models import Player, PlayerSeasonTeam, PlayerWeeklyProjection, PlayerWeeklyStat
from app.ml.features import (
    BASELINE_COLUMN,
    FEATURE_COLUMNS,
    POSITIONS,
    _team_schedule_map,
)
from app.ml.train import MODEL_DIR, MODEL_VERSION


def _load_full_history(db: Session, through_season: int) -> pd.DataFrame:
    stats_rows = db.query(PlayerWeeklyStat).filter(PlayerWeeklyStat.season <= through_season).all()
    players = {p.id: p for p in db.query(Player).all()}

    records = []
    for row in stats_rows:
        player = players.get(row.player_id)
        if player is None or player.position not in POSITIONS:
            continue
        records.append(
            {
                "player_id": row.player_id,
                "position": player.position,
                "season": row.season,
                "week": row.week,
                "points": calculate_fantasy_points(row.stats),
            }
        )
    return pd.DataFrame.from_records(records)


def _player_current_rolling_stats(history: pd.DataFrame) -> pd.DataFrame:
    """For each player, the rolling stats *as of right now* (i.e. including all real games played)."""
    history = history.sort_values(["player_id", "season", "week"])

    def summarize(group: pd.DataFrame) -> pd.Series:
        points = group["points"].to_numpy()
        return pd.Series(
            {
                "rolling_avg_8": points[-8:].mean() if len(points) else np.nan,
                "rolling_avg_3": points[-3:].mean() if len(points) else np.nan,
                "last_game_points": points[-1] if len(points) else np.nan,
                "games_played_prior": len(points),
            }
        )

    return history.groupby("player_id").apply(summarize, include_groups=False).reset_index()


def _opponent_defense_now(
    history: pd.DataFrame, target_season: int, target_week: int, season_team: dict[tuple[str, int], str]
) -> pd.DataFrame:
    """Defense-allowed-by-position estimate: this season's prior weeks if any exist, else last
    season's full-season rate. Returns columns [def_team, position, allowed]."""
    this_season_prior = history[(history.season == target_season) & (history.week < target_week)]
    if not this_season_prior.empty:
        source = this_season_prior.copy()
    else:
        source = history[history.season == target_season - 1].copy()

    source["team"] = source.apply(lambda r: season_team.get((r.player_id, r.season)), axis=1)
    schedule = _team_schedule_map(source["season"].min(), source["season"].max())
    source = source.merge(schedule, on=["season", "week", "team"], how="left")

    return (
        source.dropna(subset=["opponent"])
        .groupby(["opponent", "position"])["points"]
        .mean()
        .reset_index()
        .rename(columns={"opponent": "def_team", "points": "allowed"})
    )


def build_current_feature_frame(db: Session, target_season: int, target_week: int) -> pd.DataFrame:
    history = _load_full_history(db, through_season=target_season)
    if history.empty:
        return pd.DataFrame()

    players = db.query(Player).filter(Player.position.in_(POSITIONS)).all()
    base = pd.DataFrame(
        [
            {
                "player_id": p.id,
                "position": p.position,
                "name": p.name,
                "team": p.nfl_team,
                "injury_status": p.injury_status,
            }
            for p in players
        ]
    )

    rolling = _player_current_rolling_stats(history)
    df = base.merge(rolling, on="player_id", how="left")

    position_baseline = history.groupby("position")["points"].mean().to_dict()
    df["rolling_avg_8"] = df.apply(
        lambda r: r.rolling_avg_8 if pd.notna(r.rolling_avg_8) else position_baseline.get(r.position, 5.0), axis=1
    )
    df["rolling_avg_3"] = df.apply(
        lambda r: r.rolling_avg_3 if pd.notna(r.rolling_avg_3) else position_baseline.get(r.position, 5.0), axis=1
    )
    df["last_game_points"] = df["last_game_points"].fillna(df["rolling_avg_3"])
    df["games_played_prior"] = df["games_played_prior"].fillna(0)
    df["trend"] = df["rolling_avg_3"] - df["rolling_avg_8"]

    season_team_rows = db.query(PlayerSeasonTeam).filter(PlayerSeasonTeam.season >= target_season - 1).all()
    season_team = {(r.player_id, r.season): r.nfl_team for r in season_team_rows}

    def new_team_flag(row) -> int:
        this_team = season_team.get((row.player_id, target_season), row.team)
        last_team = season_team.get((row.player_id, target_season - 1))
        if last_team is None:
            return 0
        return int(this_team != last_team)

    df["new_team"] = df.apply(new_team_flag, axis=1)

    schedule = _team_schedule_map(target_season, target_season)
    schedule = schedule[schedule.week == target_week][["team", "opponent", "implied_total"]]
    df = df.merge(schedule, on="team", how="left")

    # No row in the schedule for this team/week = bye week (or a team code we don't recognize).
    # Flagged here and zeroed out at the end of generate_projections rather than silently
    # falling back to a baseline value, which would otherwise invent a nonsense projection for
    # a player who isn't playing that week.
    df["is_bye"] = df["opponent"].isna()

    defense = _opponent_defense_now(history, target_season, target_week, season_team)
    df = df.merge(
        defense.rename(columns={"def_team": "opponent"}), on=["opponent", "position"], how="left"
    )
    df["opp_def_allowed"] = df.apply(
        lambda r: r.allowed if pd.notna(r.allowed) else position_baseline.get(r.position, 8.0), axis=1
    )
    df["implied_total"] = df["implied_total"].fillna(df["implied_total"].mean())

    return df


def _load_models() -> dict:
    models = {}
    for position in POSITIONS:
        path = MODEL_DIR / f"{position.lower()}_{MODEL_VERSION}.joblib"
        if path.exists():
            models[position] = joblib.load(path)
    return models


# Report statuses treated as "will not play" for redistribution purposes. "Questionable" is
# deliberately excluded -- it's too uncertain (historically ~75% of Questionable players do
# play) to zero out a projection over.
OUT_STATUSES = {"Out", "Doubtful", "IR"}
# How much of the sidelined player's recent role-level production the next-man-up inherits.
# Not 1.0: a backup stepping into a larger role rarely replicates the starter's efficiency
# outright, but does absorb most of the volume.
BACKUP_INHERITANCE = 0.65


def _apply_injury_redistribution(df: pd.DataFrame, projected: dict[str, float]) -> tuple[dict[str, float], dict[str, str]]:
    """Zeroes out sidelined players and boosts the next-healthiest same-team/position teammate.

    "Next man up" is determined by recent usage (rolling_avg_8), not the official depth chart --
    nflverse's depth-chart export models offensive personnel packages, not a clean WR1/WR2/WR3
    ranking, so actual recent production is the more reliable signal for "who plays more if this
    guy is out."
    """
    adjusted = dict(projected)
    boost_source: dict[str, str] = {}

    rolling_by_player = df.set_index("player_id")["rolling_avg_8"].to_dict()
    name_by_player = df.set_index("player_id")["name"].to_dict()

    for (_team, _position), group in df.groupby(["team", "position"]):
        injured = group[group["injury_status"].isin(OUT_STATUSES)]
        if injured.empty:
            continue
        healthy = group[~group["injury_status"].isin(OUT_STATUSES)].sort_values(
            "rolling_avg_8", ascending=False
        )
        if healthy.empty:
            continue

        available = list(healthy.itertuples())
        for inj_row in injured.sort_values("rolling_avg_8", ascending=False).itertuples():
            adjusted[inj_row.player_id] = 0.0
            if not available:
                continue
            backup = available.pop(0)

            backup_own = adjusted.get(backup.player_id, 0.0)
            inherited = (
                BACKUP_INHERITANCE * rolling_by_player.get(inj_row.player_id, 0.0)
                + (1 - BACKUP_INHERITANCE) * backup_own
            )
            if inherited > backup_own:
                adjusted[backup.player_id] = round(float(inherited), 2)
                boost_source[backup.player_id] = name_by_player.get(inj_row.player_id, "a teammate")

    return adjusted, boost_source


def generate_projections(db: Session, target_season: int, target_week: int) -> dict:
    from app.db.models import TrainedModelArtifact

    df = build_current_feature_frame(db, target_season, target_week)
    if df.empty:
        return {"status": "no_data"}

    models = _load_models()
    blend_weights = {}
    for position in POSITIONS:
        artifact = db.get(TrainedModelArtifact, {"name": f"projection_{position.lower()}", "version": MODEL_VERSION})
        blend_weights[position] = artifact.metadata_json.get("blend_weight", 0.0) if artifact else 0.0

    projected_by_player: dict[str, float] = {}
    valid_rows = []
    for position in POSITIONS:
        pos_df = df[df.position == position].dropna(subset=FEATURE_COLUMNS)
        if pos_df.empty:
            continue
        valid_rows.append(pos_df)

        baseline = pos_df[BASELINE_COLUMN]
        model = models.get(position)
        residual = model.predict(pos_df[FEATURE_COLUMNS]) if model is not None else np.zeros(len(pos_df))

        blend = blend_weights.get(position, 0.0)
        projected = (baseline + blend * residual).clip(lower=0.0)
        projected = projected.where(~pos_df["is_bye"], 0.0)

        for player_id, points in zip(pos_df.player_id, projected):
            projected_by_player[player_id] = round(float(points), 2)

    if not valid_rows:
        return {"status": "no_data"}

    valid_df = pd.concat(valid_rows, ignore_index=True)
    adjusted, boost_source = _apply_injury_redistribution(valid_df, projected_by_player)

    written = 0
    for player_id, points in adjusted.items():
        pk = {
            "player_id": player_id,
            "season": target_season,
            "week": target_week,
            "model_version": MODEL_VERSION,
        }
        stats = {"boosted_due_to": boost_source[player_id]} if player_id in boost_source else {}
        existing = db.get(PlayerWeeklyProjection, pk)
        if existing:
            existing.projected_points = points
            existing.stats = stats
        else:
            db.add(PlayerWeeklyProjection(**pk, projected_points=points, stats=stats))
        written += 1

    db.commit()
    return {"status": "ok", "projections_written": written, "injury_boosts": len(boost_source)}
