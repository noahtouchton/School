from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import ConsensusAdp, Player, PlayerWeeklyProjection
from app.db.session import get_db
from app.ml.train import MODEL_VERSION
from app.schemas import ProjectionListResponse, ProjectionOut

router = APIRouter(prefix="/projections", tags=["projections"])


def _latest_projection_week(db: Session) -> tuple[int | None, int | None]:
    season = db.execute(
        select(func.max(PlayerWeeklyProjection.season)).where(
            PlayerWeeklyProjection.model_version == MODEL_VERSION
        )
    ).scalar_one_or_none()
    if season is None:
        return None, None
    week = db.execute(
        select(func.max(PlayerWeeklyProjection.week)).where(
            PlayerWeeklyProjection.model_version == MODEL_VERSION,
            PlayerWeeklyProjection.season == season,
        )
    ).scalar_one_or_none()
    return season, week


@router.get("", response_model=ProjectionListResponse)
def list_projections(
    season: int | None = Query(default=None),
    week: int | None = Query(default=None),
    position: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=300, le=1000),
    db: Session = Depends(get_db),
) -> ProjectionListResponse:
    if season is None or week is None:
        season, week = _latest_projection_week(db)
    if season is None:
        return ProjectionListResponse(season=0, week=0, count=0, players=[])

    stmt = (
        select(Player, PlayerWeeklyProjection.projected_points, PlayerWeeklyProjection.stats, ConsensusAdp.adp)
        .join(
            PlayerWeeklyProjection,
            (PlayerWeeklyProjection.player_id == Player.id)
            & (PlayerWeeklyProjection.season == season)
            & (PlayerWeeklyProjection.week == week)
            & (PlayerWeeklyProjection.model_version == MODEL_VERSION),
        )
        .join(
            ConsensusAdp,
            (ConsensusAdp.player_id == Player.id) & (ConsensusAdp.season == season),
            isouter=True,
        )
    )
    if position:
        stmt = stmt.where(Player.position == position.upper())
    if search:
        stmt = stmt.where(Player.name.ilike(f"%{search}%"))

    total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()

    stmt = stmt.order_by(PlayerWeeklyProjection.projected_points.desc()).limit(limit)
    rows = db.execute(stmt).all()

    players = [
        ProjectionOut(
            player_id=player.id,
            name=player.name,
            position=player.position,
            nfl_team=player.nfl_team,
            status=player.status,
            injury_status=player.injury_status,
            projected_points=projected_points,
            adp=adp,
            boosted_due_to=(stats or {}).get("boosted_due_to"),
        )
        for player, projected_points, stats, adp in rows
    ]
    return ProjectionListResponse(season=season, week=week, count=total, players=players)
