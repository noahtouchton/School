from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.draft import router as draft_router
from app.api.espn import router as espn_router
from app.api.players import router as players_router
from app.api.projections import router as projections_router
from app.api.yahoo import router as yahoo_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(title="Fantasy Football API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(players_router)
app.include_router(projections_router)
app.include_router(draft_router)
app.include_router(espn_router)
app.include_router(yahoo_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
