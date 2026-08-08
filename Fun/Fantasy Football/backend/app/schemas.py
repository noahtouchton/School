from pydantic import BaseModel, ConfigDict


class PlayerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    position: str
    nfl_team: str
    status: str
    injury_status: str | None = None
    age: int | None = None
    experience: int | None = None


class PlayerListResponse(BaseModel):
    count: int
    players: list[PlayerOut]


class ProjectionOut(BaseModel):
    player_id: str
    name: str
    position: str
    nfl_team: str
    status: str
    injury_status: str | None = None
    projected_points: float
    adp: float | None = None
    boosted_due_to: str | None = None


class ProjectionListResponse(BaseModel):
    season: int
    week: int
    count: int
    players: list[ProjectionOut]
