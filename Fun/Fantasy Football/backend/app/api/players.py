from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Player
from app.db.session import get_db
from app.schemas import PlayerListResponse, PlayerOut

router = APIRouter(prefix="/players", tags=["players"])


def _filtered_stmt(position: str | None, team: str | None, search: str | None):
    stmt = select(Player)
    if position:
        stmt = stmt.where(Player.position == position.upper())
    if team:
        stmt = stmt.where(Player.nfl_team == team.upper())
    if search:
        stmt = stmt.where(Player.name.ilike(f"%{search}%"))
    return stmt


@router.get("", response_model=PlayerListResponse)
def list_players(
    position: str | None = Query(default=None, description="Filter by position, e.g. RB"),
    team: str | None = Query(default=None, description="Filter by NFL team abbreviation"),
    search: str | None = Query(default=None, description="Case-insensitive name search"),
    limit: int = Query(default=200, le=1000),
    db: Session = Depends(get_db),
) -> PlayerListResponse:
    base_stmt = _filtered_stmt(position, team, search)

    total = db.execute(select(func.count()).select_from(base_stmt.subquery())).scalar_one()
    players = db.execute(base_stmt.order_by(Player.name).limit(limit)).scalars().all()

    return PlayerListResponse(count=total, players=[PlayerOut.model_validate(p) for p in players])


@router.get("/{player_id}", response_model=PlayerOut)
def get_player(player_id: str, db: Session = Depends(get_db)) -> PlayerOut:
    player = db.get(Player, player_id)
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return PlayerOut.model_validate(player)
