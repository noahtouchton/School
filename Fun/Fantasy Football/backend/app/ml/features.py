"""Feature engineering for the weekly fantasy-points projection models.

The old app's "projection" was a rolling average of past points and nothing else -- it had no
way to react to a team change, opponent strength, or anything besides recent box scores. This
module builds a richer, leakage-free feature set:

- rolling_avg_3 / rolling_avg_8: the player's own recent scoring, computed only from games
  strictly before the target row (chronological, across season boundaries).
- games_played_prior: how much history we have for this player (a natural stand-in for
  experience/rookie uncertainty without needing precise historical age/experience-at-the-time,
  which nfl_data_py doesn't cleanly expose per season).
- opp_def_allowed: a rolling estimate of how many fantasy points the upcoming opponent has
  allowed to this position, computed only from that season's prior weeks.
- new_team: 1 if the player's team this season differs from last season (see
  ingestion/season_teams.py) -- directly targets the "didn't know about team changes" bug.

Team assignment is season-level (see PlayerSeasonTeam), not week-level, so in-season trades are
attributed to the pre-trade team for the whole season. That's a known simplification -- in-season
trades are rare relative to full-season team stability, and the season-level signal still catches
the far more common case (free agency / offseason trades), which was the user-reported failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import nfl_data_py as nfl
from sqlalchemy.orm import Session

from app.core.scoring import calculate_fantasy_points
from app.db.models import Player, PlayerSeasonTeam, PlayerWeeklyStat

BASELINE_COLUMN = "rolling_avg_8"

# The model predicts a *residual* on top of BASELINE_COLUMN (a strong rolling-average predictor
# on its own) rather than the raw point total -- see ml/train.py for why: predicting points
# directly let a modest feature set land at or below the naive baseline for some positions,
# while residual-on-baseline reliably matches-or-beats it (worst case the model just learns to
# predict ~0 residual).
# Raw per-game usage stats pulled out of the stats blob (see ingestion/nfl_data.py).
# Each becomes a leakage-free rolling average over the player's prior games.
USAGE_STATS = [
    "targets",
    "carries",
    "attempts",
    "target_share",
    "air_yards_share",
    "wopr",
    "receiving_air_yards",
]

USAGE_FEATURE_COLUMNS = [f"{stat}_8" for stat in USAGE_STATS] + ["opportunity_8", "usage_trend"]

# Features every position gets.
CORE_FEATURE_COLUMNS = [
    "rolling_avg_8",
    "trend",
    "last_game_points",
    "games_played_prior",
    "opp_def_allowed",
    "implied_total",
    "new_team",
]

# Opportunity features, assigned per position rather than wholesale.
#
# Fantasy points are noisy (touchdowns are close to a coin flip week to week)
# while volume is not, so knowing a player's role has changed before his points
# catch up is worth something -- but only where the metric means anything, and
# only where it's measurably better. These lists are the output of
# scripts/ablate_usage_features.py (pooled 3-fold walk-forward MAE, 2023-2025):
#
#   pos   no usage   all usage   position-aware   shipped
#   QB      6.2421      6.2672           6.2552   no usage
#   RB      4.2916      4.2803           4.2786   position-aware
#   WR      4.1563      4.1381           4.1368   position-aware
#   TE      3.2433      3.2225           3.2238   position-aware
#
# QB gets nothing: target share and air-yards share are structurally zero for a
# quarterback, and even after trimming to attempts/carries the usage arm still
# lost to plain rolling-average features. Volume tells you nothing about a QB
# that his own passing production hasn't already told you. Re-run the ablation
# before adding to any of these lists.
USAGE_BY_POSITION = {
    "QB": [],
    "RB": [
        "carries_8",
        "targets_8",
        "target_share_8",
        "opportunity_8",
        "usage_trend",
    ],
    "WR": [
        "targets_8",
        "target_share_8",
        "air_yards_share_8",
        "wopr_8",
        "receiving_air_yards_8",
        "opportunity_8",
        "usage_trend",
    ],
    "TE": [
        "targets_8",
        "target_share_8",
        "air_yards_share_8",
        "wopr_8",
        "receiving_air_yards_8",
        "opportunity_8",
        "usage_trend",
    ],
}

POSITIONS = ["QB", "RB", "WR", "TE"]


def features_for(position: str) -> list[str]:
    """The feature list a given position's model is trained and scored on.

    Position-specific by design -- see USAGE_BY_POSITION. Training and inference
    must both call this or the model gets columns in a different order/shape.
    """
    return CORE_FEATURE_COLUMNS + USAGE_BY_POSITION.get(position, [])


# Superset of every column any position needs; used for frame construction and
# NaN handling, never fed to a model directly.
FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + USAGE_FEATURE_COLUMNS


def _load_base_frame(db: Session, start_season: int, end_season: int) -> pd.DataFrame:
    stats_rows = (
        db.query(PlayerWeeklyStat)
        .filter(PlayerWeeklyStat.season >= start_season, PlayerWeeklyStat.season <= end_season)
        .all()
    )
    players = {p.id: p for p in db.query(Player).all()}

    records = []
    for row in stats_rows:
        player = players.get(row.player_id)
        if player is None or player.position not in POSITIONS:
            continue
        stats = row.stats or {}
        records.append(
            {
                "player_id": row.player_id,
                "position": player.position,
                "season": row.season,
                "week": row.week,
                "points": calculate_fantasy_points(stats),
                # Seasons ingested before usage metrics existed simply have no such
                # keys; 0.0 keeps those rows usable rather than dropping them.
                **{stat: float(stats.get(stat, 0.0) or 0.0) for stat in USAGE_STATS},
            }
        )
    return pd.DataFrame.from_records(records)


def _load_season_team_map(db: Session) -> dict[tuple[str, int], str]:
    rows = db.query(PlayerSeasonTeam).all()
    return {(r.player_id, r.season): r.nfl_team for r in rows}


def _add_new_team_flag(df: pd.DataFrame, season_team: dict[tuple[str, int], str]) -> pd.DataFrame:
    def flag(row):
        this_team = season_team.get((row.player_id, row.season))
        last_team = season_team.get((row.player_id, row.season - 1))
        if this_team is None or last_team is None:
            return 0
        return int(this_team != last_team)

    df["new_team"] = df.apply(flag, axis=1)
    return df


def _add_rolling_player_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    def per_player(group: pd.DataFrame) -> pd.DataFrame:
        points = group["points"].to_numpy()
        n = len(points)
        roll3 = np.zeros(n)
        roll8 = np.zeros(n)
        last_game = np.full(n, np.nan)
        prior_count = np.zeros(n, dtype=int)
        for i in range(n):
            prior = points[max(0, i - 8) : i]
            prior3 = points[max(0, i - 3) : i]
            roll8[i] = prior.mean() if len(prior) else np.nan
            roll3[i] = prior3.mean() if len(prior3) else np.nan
            if i > 0:
                last_game[i] = points[i - 1]
            prior_count[i] = i
        group = group.copy()
        group["rolling_avg_3"] = roll3
        group["rolling_avg_8"] = roll8
        group["last_game_points"] = last_game
        group["games_played_prior"] = prior_count

        # Same strictly-prior windows for every usage stat, plus a short window so
        # a recent role change is visible against the longer baseline.
        for stat in USAGE_STATS:
            values = group[stat].to_numpy(dtype=float)
            long_avg = np.full(n, np.nan)
            short_avg = np.full(n, np.nan)
            for i in range(n):
                prior = values[max(0, i - 8) : i]
                prior3 = values[max(0, i - 3) : i]
                long_avg[i] = prior.mean() if len(prior) else np.nan
                short_avg[i] = prior3.mean() if len(prior3) else np.nan
            group[f"{stat}_8"] = long_avg
            group[f"{stat}_3"] = short_avg
        return group

    original = df
    df = df.groupby("player_id", group_keys=False).apply(per_player)
    # pandas excludes the grouping column from the result; restore it via index alignment
    # (group_keys=False preserves the original row index).
    df["player_id"] = original.loc[df.index, "player_id"]

    position_baseline = df.groupby("position")["points"].transform("mean")
    df["rolling_avg_3"] = df["rolling_avg_3"].fillna(position_baseline)
    df["rolling_avg_8"] = df["rolling_avg_8"].fillna(position_baseline)
    # last_game_points has no meaningful value for a player's first-ever game -- use their
    # rolling_avg_3 fallback (which is already position-baseline-filled) rather than a second
    # independent fallback.
    df["last_game_points"] = df["last_game_points"].fillna(df["rolling_avg_3"])

    return finalize_usage_columns(df)


def finalize_usage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill usage gaps and add the derived usage columns.

    Shared by training (build_training_frame) and inference (ml/predict.py), which
    compute their rolling windows separately -- training walks each row's history,
    inference only needs the latest state. Anything derived from those windows has
    to live here or the two paths silently disagree about what a feature means.
    """
    # A debut has no prior usage. Zero is the honest fill here (unlike points,
    # where a position baseline is the better guess): we have no evidence this
    # player touches the ball at all, and that's exactly what the model should see.
    for stat in USAGE_STATS:
        for window in (8, 3):
            column = f"{stat}_{window}"
            if column in df.columns:
                df[column] = df[column].fillna(0.0)

    # Total touches+targets: one volume number that means the same thing for a
    # runner and a receiver, which per-position columns can't express.
    df["opportunity_8"] = df["carries_8"] + df["targets_8"]
    opportunity_3 = df["carries_3"] + df["targets_3"]
    # Positive = role is expanding faster than the 8-game average reflects.
    df["usage_trend"] = opportunity_3 - df["opportunity_8"]
    return df


def _team_schedule_map(start_season: int, end_season: int) -> pd.DataFrame:
    """Team-per-row schedule with the game's Vegas-implied team point total attached.

    implied_total is derived from the closing spread/total lines nfl_data_py already carries
    (spread_line is the home team's line; negative means home favored) -- a live, market-driven
    "game script" signal: a team implied for 27+ points is expected to be productive on offense
    regardless of what any individual player did earlier this season, and it reacts to real-time
    news (injuries, weather, etc.) that the box-score-based features can't see yet.
    """
    schedules = nfl.import_schedules(list(range(start_season, end_season + 1)))
    schedules = schedules.copy()
    schedules["home_implied_total"] = schedules["total_line"] / 2 - schedules["spread_line"] / 2
    schedules["away_implied_total"] = schedules["total_line"] / 2 + schedules["spread_line"] / 2

    home = schedules[["season", "week", "home_team", "away_team", "home_implied_total"]].rename(
        columns={"home_team": "team", "away_team": "opponent", "home_implied_total": "implied_total"}
    )
    away = schedules[["season", "week", "away_team", "home_team", "away_implied_total"]].rename(
        columns={"away_team": "team", "home_team": "opponent", "away_implied_total": "implied_total"}
    )
    return pd.concat([home, away], ignore_index=True)


def _add_opponent_defense_feature(
    df: pd.DataFrame, season_team: dict[tuple[str, int], str], start_season: int, end_season: int
) -> pd.DataFrame:
    df = df.copy()
    df["team"] = df.apply(lambda r: season_team.get((r.player_id, r.season)), axis=1)

    schedule = _team_schedule_map(start_season, end_season)
    df = df.merge(schedule, on=["season", "week", "team"], how="left")

    # Points allowed by (season, week, opponent-as-defense, position) = points scored by
    # players of that position who played the opponent that week.
    allowed = (
        df.dropna(subset=["opponent"])
        .groupby(["season", "week", "opponent", "position"])["points"]
        .mean()
        .reset_index()
        .rename(columns={"opponent": "def_team", "points": "allowed"})
    )

    position_baseline = df.groupby("position")["points"].mean().to_dict()

    def rolling_allowed(row) -> float:
        if pd.isna(row.opponent):
            return position_baseline.get(row.position, 10.0)
        prior = allowed[
            (allowed.def_team == row.opponent)
            & (allowed.position == row.position)
            & (allowed.season == row.season)
            & (allowed.week < row.week)
        ]
        if prior.empty:
            return position_baseline.get(row.position, 10.0)
        return prior["allowed"].mean()

    df["opp_def_allowed"] = df.apply(rolling_allowed, axis=1)
    df["implied_total"] = df["implied_total"].fillna(df["implied_total"].mean())
    return df


def build_training_frame(db: Session, start_season: int, end_season: int) -> pd.DataFrame:
    df = _load_base_frame(db, start_season, end_season)
    if df.empty:
        return df

    season_team = _load_season_team_map(db)
    df = _add_new_team_flag(df, season_team)
    df = _add_rolling_player_features(df)
    df = _add_opponent_defense_feature(df, season_team, start_season, end_season)
    df["trend"] = df["rolling_avg_3"] - df["rolling_avg_8"]
    return df
