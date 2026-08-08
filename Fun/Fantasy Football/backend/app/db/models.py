from __future__ import annotations

import datetime as dt

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Player(Base):
    """A real NFL player. Team/position reflect the current-season live refresh, not stale cached history."""

    __tablename__ = "players"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, index=True)
    position: Mapped[str] = mapped_column(String, index=True)
    nfl_team: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, default="Active")
    injury_status: Mapped[str | None] = mapped_column(String, nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    experience: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow
    )


class PlayerWeeklyStat(Base):
    """Actual accumulated stats for a player in a given season/week."""

    __tablename__ = "player_weekly_stats"

    player_id: Mapped[str] = mapped_column(ForeignKey("players.id"), primary_key=True)
    season: Mapped[int] = mapped_column(Integer, primary_key=True)
    week: Mapped[int] = mapped_column(Integer, primary_key=True)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)


class PlayerWeeklyProjection(Base):
    """ML-projected stats/points for a player in a given season/week, tagged by model version."""

    __tablename__ = "player_weekly_projections"

    player_id: Mapped[str] = mapped_column(ForeignKey("players.id"), primary_key=True)
    season: Mapped[int] = mapped_column(Integer, primary_key=True)
    week: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str] = mapped_column(String, primary_key=True, default="baseline")
    projected_points: Mapped[float] = mapped_column(Float)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)


class PlayerSeasonTeam(Base):
    """Season-level team assignment history, independent of the single 'current team' on Player.

    Used to compute a new-team flag for the ML feature pipeline (did this player change teams
    since last season?) without needing full week-by-week roster tracking.
    """

    __tablename__ = "player_season_teams"

    player_id: Mapped[str] = mapped_column(ForeignKey("players.id"), primary_key=True)
    season: Mapped[int] = mapped_column(Integer, primary_key=True)
    nfl_team: Mapped[str] = mapped_column(String)


class ConsensusAdp(Base):
    """Public consensus average-draft-position data, used only as a calibration/sanity-check
    signal against our own model's rankings -- not copied into projections.
    """

    __tablename__ = "consensus_adp"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[int] = mapped_column(Integer, index=True)
    source: Mapped[str] = mapped_column(String, default="ffc")
    scoring_format: Mapped[str] = mapped_column(String, default="half-ppr")
    player_id: Mapped[str | None] = mapped_column(ForeignKey("players.id"), nullable=True)
    name_raw: Mapped[str] = mapped_column(String)
    position: Mapped[str] = mapped_column(String)
    nfl_team: Mapped[str | None] = mapped_column(String, nullable=True)
    adp: Mapped[float] = mapped_column(Float)


class PlayerInjuryReport(Base):
    """Weekly injury report status (Out/Doubtful/Questionable), used both for display and to
    drive the backup-redistribution logic in ml/predict.py.
    """

    __tablename__ = "player_injury_reports"

    player_id: Mapped[str] = mapped_column(ForeignKey("players.id"), primary_key=True)
    season: Mapped[int] = mapped_column(Integer, primary_key=True)
    week: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_status: Mapped[str] = mapped_column(String)
    primary_injury: Mapped[str | None] = mapped_column(String, nullable=True)


class IngestionCacheManifest(Base):
    """Tracks which (season, data_type) pairs have been ingested.

    Historical data types (weekly_stats, schedules) are cache-once. `current_roster` is
    intentionally re-ingested on every run regardless of manifest state -- see ingestion/rosters.py.
    """

    __tablename__ = "ingestion_cache_manifest"

    season: Mapped[int] = mapped_column(Integer, primary_key=True)
    data_type: Mapped[str] = mapped_column(String, primary_key=True)
    refreshed_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    source: Mapped[str] = mapped_column(String, default="sandbox")  # sandbox | espn
    settings: Mapped[dict] = mapped_column(JSON, default=dict)

    teams: Mapped[list["FantasyTeam"]] = relationship(back_populates="league")


class FantasyTeam(Base):
    __tablename__ = "fantasy_teams"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    league_id: Mapped[str] = mapped_column(ForeignKey("leagues.id"))
    name: Mapped[str] = mapped_column(String)
    owner_type: Mapped[str] = mapped_column(String, default="ai")  # human | ai
    persona: Mapped[str | None] = mapped_column(String, nullable=True)
    roster: Mapped[dict] = mapped_column(JSON, default=dict)  # {starters, bench, ir}: [player_id]
    faab_balance: Mapped[int] = mapped_column(Integer, default=100)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    points_for: Mapped[float] = mapped_column(Float, default=0.0)

    league: Mapped["League"] = relationship(back_populates="teams")


class EspnLeagueLink(Base):
    __tablename__ = "espn_league_links"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    league_id: Mapped[int] = mapped_column(Integer)
    year: Mapped[int] = mapped_column(Integer)
    espn_s2: Mapped[str | None] = mapped_column(Text, nullable=True)
    swid: Mapped[str | None] = mapped_column(String, nullable=True)
    nickname: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class TrainedModelArtifact(Base):
    """Registry of trained ML projection models and evolved agent-parameter sets."""

    __tablename__ = "trained_model_artifacts"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[str] = mapped_column(String)  # projection_model | agent_params
    path: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class ChatMessage(Base):
    """History for the LLM tuning-chat feature."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    role: Mapped[str] = mapped_column(String)  # user | assistant | tool
    content: Mapped[str] = mapped_column(Text)
    provider: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
