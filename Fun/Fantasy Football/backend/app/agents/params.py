"""The genome: every number that decides how an AI manager behaves.

Deliberately a flat vector of plain floats/ints with declared bounds, because the
evolutionary trainer (app/evolution) mutates and crosses these blindly. Anything
that isn't expressible as a bounded scalar doesn't belong here.

The vocabulary is carried over from the original Streamlit-era agent
(src/ai/base_agent.py) so the hand-written personas still mean what they meant.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, fields


@dataclass
class AgentParameters:
    """Numerical knobs driving draft, lineup, and waiver decisions."""

    # --- Draft: how fast a position's value decays once starters are filled.
    # Higher = keeps taking that position for depth rather than moving on.
    vorp_decay_qb: float = 0.5
    vorp_decay_rb: float = 0.4
    vorp_decay_wr: float = 0.4
    vorp_decay_te: float = 0.3

    # --- Draft: hard positional shape constraints. These are what make a
    # strategy recognisable -- zero-RB is early_rb_limit=0, hero-RB is 1.
    early_rb_limit: int = 99       # max RBs in the first 6 rounds
    early_qb_limit: int = 99       # max QBs in the first 6 rounds
    early_rb_minimum: int = 0      # forced RBs inside the first 3 rounds

    # --- Draft: risk appetite.
    rookie_boost: float = 1.0      # multiplier on players with no NFL history
    age_penalty_threshold: int = 30
    age_penalty_factor: float = 0.0  # fractional penalty per year past threshold
    tier_scarcity_weight: float = 1.0  # how hard to chase a thinning tier
    bye_stack_penalty: float = 0.0   # penalty for piling starters onto one bye week

    # --- Draft: correlation.
    qb_wr_stack_boost: float = 1.0  # bonus for pass-catchers on a drafted QB's team

    # --- Waivers.
    waiver_min_improvement: float = 2.5  # projected weekly gain needed to act
    waiver_max_faab_pct: float = 0.05    # max share of remaining budget per claim
    faab_urgency_factor: float = 1.0     # scales bids up as the season runs out

    # --- Lineup.
    matchup_adjustment: float = 0.0  # 0 = ignore opponent defense, 1 = full weight

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentParameters":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# Mutation bounds. Anything absent is treated as fixed. Ints are declared here
# too so the GA rounds them rather than producing a fractional "max 2.4 RBs".
PARAM_BOUNDS: dict[str, tuple[float, float, bool]] = {
    # name: (low, high, is_int)
    "vorp_decay_qb": (0.0, 1.0, False),
    "vorp_decay_rb": (0.0, 1.0, False),
    "vorp_decay_wr": (0.0, 1.0, False),
    "vorp_decay_te": (0.0, 1.0, False),
    "early_rb_limit": (0, 6, True),
    "early_qb_limit": (0, 6, True),
    "early_rb_minimum": (0, 3, True),
    "rookie_boost": (0.6, 1.8, False),
    "age_penalty_threshold": (26, 34, True),
    "age_penalty_factor": (0.0, 0.15, False),
    "tier_scarcity_weight": (0.0, 2.5, False),
    "bye_stack_penalty": (0.0, 0.3, False),
    "qb_wr_stack_boost": (0.8, 1.5, False),
    "waiver_min_improvement": (0.2, 6.0, False),
    "waiver_max_faab_pct": (0.01, 0.30, False),
    "faab_urgency_factor": (0.5, 2.0, False),
    "matchup_adjustment": (0.0, 1.0, False),
}


def random_params(rng: random.Random) -> AgentParameters:
    """A uniformly random genome inside the declared bounds."""
    values = {}
    for name, (low, high, is_int) in PARAM_BOUNDS.items():
        values[name] = rng.randint(int(low), int(high)) if is_int else rng.uniform(low, high)
    return AgentParameters(**values)


def mutate(params: AgentParameters, rng: random.Random, rate: float, scale: float) -> AgentParameters:
    """Gaussian jitter on a random subset of genes, clamped to bounds.

    scale is expressed as a fraction of each gene's own range so one sigma means
    the same thing for a 0-1 decay weight and a 0-6 round limit.
    """
    values = params.to_dict()
    for name, (low, high, is_int) in PARAM_BOUNDS.items():
        if rng.random() > rate:
            continue
        sigma = (high - low) * scale
        mutated = values[name] + rng.gauss(0.0, sigma)
        mutated = max(low, min(high, mutated))
        values[name] = int(round(mutated)) if is_int else mutated
    return AgentParameters.from_dict(values)


def crossover(a: AgentParameters, b: AgentParameters, rng: random.Random) -> AgentParameters:
    """Uniform crossover -- each gene independently from one parent.

    Uniform rather than single-point because these genes have no meaningful
    ordering; adjacency in the dataclass says nothing about how they interact.
    """
    a_values, b_values = a.to_dict(), b.to_dict()
    return AgentParameters.from_dict(
        {name: (a_values[name] if rng.random() < 0.5 else b_values[name]) for name in a_values}
    )
