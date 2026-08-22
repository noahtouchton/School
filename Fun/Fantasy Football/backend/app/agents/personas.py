"""Named strategy presets used as Arena opponents and as the evolutionary baseline.

Ported from the original persona classes (src/ai/personas/*). Each is just a
parameter vector -- there's no per-persona code -- which is what lets the trained
agent be compared against them on equal terms.
"""

from __future__ import annotations

from app.agents.params import AgentParameters

PERSONAS: dict[str, tuple[str, AgentParameters]] = {
    "balanced": (
        "Takes the best player available and lets the roster shape itself.",
        AgentParameters(),
    ),
    "zero_rb": (
        "Refuses running backs early, loading up on receivers instead.",
        AgentParameters(early_rb_limit=0, vorp_decay_wr=0.55, vorp_decay_te=0.4),
    ),
    "hero_rb": (
        "One elite running back early, then receivers the rest of the way.",
        AgentParameters(early_rb_limit=1, vorp_decay_wr=0.5),
    ),
    "robust_rb": (
        "Hammers running backs with the first three picks.",
        AgentParameters(early_rb_minimum=3, vorp_decay_rb=0.55),
    ),
    "late_qb": (
        "Waits on quarterback no matter how the board falls.",
        AgentParameters(early_qb_limit=0, vorp_decay_rb=0.5, vorp_decay_wr=0.5),
    ),
    "stack_builder": (
        "Pairs a quarterback with his own pass catchers for correlated upside.",
        AgentParameters(early_qb_limit=1, vorp_decay_qb=0.3, qb_wr_stack_boost=1.35),
    ),
    "rookie_hunter": (
        "Chases young breakouts and fades anyone on the wrong side of 27.",
        AgentParameters(
            rookie_boost=1.5, age_penalty_threshold=27, age_penalty_factor=0.10,
            waiver_min_improvement=1.8,
        ),
    ),
    "veteran_believer": (
        "Trusts proven production and distrusts unproven rookies.",
        AgentParameters(rookie_boost=0.75, age_penalty_factor=0.0, waiver_min_improvement=4.0),
    ),
    "waiver_wolf": (
        "Churns the waiver wire relentlessly and spends FAAB freely.",
        AgentParameters(
            waiver_min_improvement=0.4, waiver_max_faab_pct=0.25, faab_urgency_factor=1.6,
            rookie_boost=1.2,
        ),
    ),
    "conservative": (
        "Sits on the roster it drafted and rarely makes a move.",
        AgentParameters(
            waiver_min_improvement=5.0, waiver_max_faab_pct=0.02, faab_urgency_factor=0.6,
        ),
    ),
    "tier_hawk": (
        "Drafts on tier breaks, reaching to avoid being left behind.",
        AgentParameters(tier_scarcity_weight=2.2, bye_stack_penalty=0.15),
    ),
    "matchup_chaser": (
        "Sets lineups almost entirely on who the opponent's defense is.",
        AgentParameters(matchup_adjustment=0.9, waiver_min_improvement=1.5),
    ),
}


def persona_params(name: str) -> AgentParameters:
    entry = PERSONAS.get(name)
    return entry[1] if entry else AgentParameters()


def persona_description(name: str) -> str:
    entry = PERSONAS.get(name)
    return entry[0] if entry else "Custom parameter set."


def default_opponents(count: int) -> list[str]:
    """A spread of distinct strategies to fill a league, cycling if needed."""
    names = [n for n in PERSONAS if n != "balanced"]
    ordered = ["balanced", *names]
    return [ordered[i % len(ordered)] for i in range(count)]
