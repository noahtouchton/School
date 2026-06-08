from ...config import LeagueSettings
from ..base_agent import BaseAgent

# Original personas (kept for backwards compatibility)
from .free_agent_demon import FreeAgentDemon
from .trade_demon import TradeDemon
from .matchup_all_star import MatchupAllStar
from .conservative import ConservativeAgent
from .zero_rb import ZeroRBAgent
from .hero_rb import HeroRBAgent
from .high_risk import HighRiskAgent
from .balanced import BalancedAgent
from .late_qb import LateRoundQBAgent
from .robust_rb import RobustRBAgent

# The 10 training personas
from .big_trader import BigTrader
from .waiver_wolf import WaiverWolf
from .projection_truster import ProjectionTruster
from .player_loyalist import PlayerLoyalist
from .all_rounder import AllRounder
from .trade_refuser import TradeRefuser
from .weird_bench import WeirdBench
from .cheap_qb import CheapQB
from .stack_builder import StackBuilder
from .rookie_hunter import RookieHunter

# Ordered list of the 10 canonical training personas
TRAINING_PERSONAS = [
    "big_trader",
    "waiver_wolf",
    "projection_truster",
    "player_loyalist",
    "all_rounder",
    "trade_refuser",
    "weird_bench",
    "cheap_qb",
    "stack_builder",
    "rookie_hunter",
]

PERSONA_DISPLAY_NAMES = {
    "big_trader":         "Big Trader",
    "waiver_wolf":        "Waiver Wolf",
    "projection_truster": "Projection Truster",
    "player_loyalist":    "Player Loyalist",
    "all_rounder":        "All Rounder",
    "trade_refuser":      "Trade Refuser",
    "weird_bench":        "Weird Bench",
    "cheap_qb":           "Cheap QB",
    "stack_builder":      "Stack Builder",
    "rookie_hunter":      "Rookie Hunter",
}

PERSONA_DESCRIPTIONS = {
    "big_trader":         "Always looking for a deal. Proposes multiple trades per week, drafts RBs as trade chips.",
    "waiver_wolf":        "Lives on the wire. Churns the bench aggressively with high FAAB bids every week.",
    "projection_truster": "The numbers guy. Starts whoever the model says to start, no gut-feel overrides.",
    "player_loyalist":    "Believes in their guys. Almost never drops or trades a core player once drafted.",
    "all_rounder":        "No glaring weaknesses. Solid VORP drafting, fair trades, light matchup awareness.",
    "trade_refuser":      "Won't deal. Refuses every trade and compensates with aggressive waiver work.",
    "weird_bench":        "Ignores conventional wisdom. Drafts 2 QBs early, picks K/DST ahead of schedule.",
    "cheap_qb":           "Skips QB in the draft entirely. Loads up on skill positions and streams QBs all season.",
    "stack_builder":      "Drafts a QB, then targets that QB's WRs and TE for correlated upside.",
    "rookie_hunter":      "Obsessed with youth. Heavy rookie boost, steep age penalties, trades for young players.",
}


def get_agent_by_persona(name: str, team_id: str, settings: LeagueSettings) -> BaseAgent:
    """Factory function to get an agent instance based on their persona name."""
    from ...data import db
    from ..base_agent import AgentParameters

    # Check if this is a custom permanently trained model from DB
    saved_params = db.get_trained_model(name)
    if saved_params:
        params = AgentParameters(**saved_params)
        return BaseAgent(team_id, settings, params)

    n = name.lower().replace("_", "").replace(" ", "")

    # --- 10 canonical training personas ---
    if n == "bigtrader":
        return BigTrader(team_id, settings)
    elif n == "waiverwolf":
        return WaiverWolf(team_id, settings)
    elif n == "projectiontruster":
        return ProjectionTruster(team_id, settings)
    elif n == "playerloyalist":
        return PlayerLoyalist(team_id, settings)
    elif n == "allrounder":
        return AllRounder(team_id, settings)
    elif n == "traderefuser":
        return TradeRefuser(team_id, settings)
    elif n == "weirdbench":
        return WeirdBench(team_id, settings)
    elif n == "cheapqb":
        return CheapQB(team_id, settings)
    elif n == "stackbuilder":
        return StackBuilder(team_id, settings)
    elif n == "rookiehunter":
        return RookieHunter(team_id, settings)

    # --- Legacy persona aliases ---
    elif "freeagent" in n or "waiver" in n:
        return FreeAgentDemon(team_id, settings)
    elif "trade" in n and "refus" not in n:
        return TradeDemon(team_id, settings)
    elif "matchup" in n or "benchallstar" in n:
        return MatchupAllStar(team_id, settings)
    elif "conservative" in n or "riskaverse" in n:
        return ConservativeAgent(team_id, settings)
    elif "zerorb" in n:
        return ZeroRBAgent(team_id, settings)
    elif "herorb" in n:
        return HeroRBAgent(team_id, settings)
    elif "highrisk" in n or "boombust" in n:
        return HighRiskAgent(team_id, settings)
    elif "lateqb" in n or "lateroundqb" in n:
        return LateRoundQBAgent(team_id, settings)
    elif "robustrb" in n:
        return RobustRBAgent(team_id, settings)
    else:
        return BalancedAgent(team_id, settings)
