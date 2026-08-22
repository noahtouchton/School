"""Genetic algorithm over AgentParameters.

Each candidate is dropped into a league of hand-written personas and made to play
real seasons; fitness is wins*100 + points-for, so winning games dominates and
total scoring breaks ties. Over generations the population drifts toward whatever
actually wins against that field.

Two things guard against fooling ourselves:

*Multiple seasons per candidate.* A single 14-game season is mostly luck. Every
candidate plays the same set of (season, seed) pairs, so a lucky schedule helps
everyone equally and can't crown a bad genome.

*A persona benchmark.* The trained agent's fitness means nothing on its own -- the
number only says something next to what a plain "balanced" manager scores in the
same leagues, which is reported alongside it.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy.orm import Session

from app.agents.params import AgentParameters, crossover, mutate, random_params
from app.agents.personas import default_opponents, persona_params
from app.simulation.season import (
    DEFAULT_ROSTER_POSITIONS,
    PreparedSeason,
    fitness,
    prepare_season,
    simulate_season,
)


@dataclass
class TrainingConfig:
    seasons: list[int] = field(default_factory=list)  # empty -> resolved from data
    population: int = 20
    generations: int = 8
    seasons_per_candidate: int = 3
    elite: int = 3
    mutation_rate: float = 0.3
    mutation_scale: float = 0.15
    tournament_size: int = 3
    num_teams: int = 10
    seed: int = 1


@dataclass
class GenerationReport:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_wins: float
    baseline_fitness: float
    best_params: dict
    elapsed_s: float


@dataclass
class HoldoutResult:
    """The only number worth quoting.

    Training fitness is measured on the same handful of (season, seed) worlds the
    GA searched against, so it is guaranteed to drift upward whether or not the
    agent learned anything transferable. This re-runs the winner and the baseline
    persona on worlds neither has seen.
    """

    seeds: int
    agent_fitness: float
    agent_wins: float
    baseline_fitness: float
    baseline_wins: float

    @property
    def lift(self) -> float:
        return round(self.agent_fitness - self.baseline_fitness, 1)

    @property
    def beat_baseline(self) -> bool:
        return self.agent_fitness > self.baseline_fitness

    def to_dict(self) -> dict:
        return {
            "seeds": self.seeds,
            "agent_fitness": round(self.agent_fitness, 1),
            "agent_wins": round(self.agent_wins, 2),
            "baseline_fitness": round(self.baseline_fitness, 1),
            "baseline_wins": round(self.baseline_wins, 2),
            "lift": self.lift,
            "beat_baseline": self.beat_baseline,
        }


def _evaluate(
    db: Session,
    params: AgentParameters,
    prepared: dict[int, PreparedSeason],
    matchups: list[tuple[int, int]],
    opponents: list[str],
) -> tuple[float, float]:
    """Mean fitness and mean wins across the shared (season, seed) set."""
    total, wins = 0.0, 0.0
    for season, seed in matchups:
        result = simulate_season(
            db,
            season=season,
            candidate=params,
            opponents=opponents,
            seed=seed,
            capture_detail=False,
            prepared=prepared[season],
        )
        total += fitness(result)
        wins += next(
            (r["wins"] for r in result.standings if r["team_id"] == "candidate"), 0
        )
    n = len(matchups)
    return total / n, wins / n


def train(
    db: Session,
    config: TrainingConfig,
    on_generation: Callable[[GenerationReport], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[AgentParameters, list[GenerationReport]]:
    rng = random.Random(config.seed)
    opponents = default_opponents(config.num_teams - 1)

    prepared = {
        season: prepare_season(db, season, DEFAULT_ROSTER_POSITIONS, config.num_teams)
        for season in config.seasons
    }
    # Every candidate faces exactly this schedule of worlds.
    matchups = [
        (season, rng.randrange(10_000))
        for season in config.seasons
        for _ in range(max(1, config.seasons_per_candidate // max(1, len(config.seasons))))
    ]

    # What a stock manager scores in these same leagues -- the number that makes
    # the trained agent's fitness interpretable.
    baseline_fitness, _ = _evaluate(
        db, persona_params("balanced"), prepared, matchups, opponents
    )

    # Seed the population with the hand-written personas before going random, so
    # generation 0 starts from strategies known to be coherent.
    population: list[AgentParameters] = [persona_params("balanced")]
    for name in ("robust_rb", "zero_rb", "hero_rb", "late_qb", "tier_hawk"):
        if len(population) < config.population:
            population.append(persona_params(name))
    while len(population) < config.population:
        population.append(random_params(rng))

    reports: list[GenerationReport] = []
    best_overall: tuple[float, AgentParameters] | None = None

    for generation in range(1, config.generations + 1):
        if should_stop and should_stop():
            break
        started = time.time()

        scored: list[tuple[float, float, AgentParameters]] = []
        for candidate in population:
            score, wins = _evaluate(db, candidate, prepared, matchups, opponents)
            scored.append((score, wins, candidate))
        scored.sort(key=lambda row: row[0], reverse=True)

        if best_overall is None or scored[0][0] > best_overall[0]:
            best_overall = (scored[0][0], scored[0][2])

        report = GenerationReport(
            generation=generation,
            best_fitness=round(scored[0][0], 1),
            mean_fitness=round(sum(s[0] for s in scored) / len(scored), 1),
            best_wins=round(scored[0][1], 2),
            baseline_fitness=round(baseline_fitness, 1),
            best_params=scored[0][2].to_dict(),
            elapsed_s=round(time.time() - started, 1),
        )
        reports.append(report)
        if on_generation:
            on_generation(report)

        if generation == config.generations:
            break

        # Elitism keeps the best genomes verbatim; the rest are bred by
        # tournament selection so a merely-good genome still gets a chance.
        survivors = [row[2] for row in scored[: config.elite]]
        next_population = list(survivors)
        while len(next_population) < config.population:
            a = _tournament(scored, rng, config.tournament_size)
            b = _tournament(scored, rng, config.tournament_size)
            child = crossover(a, b, rng)
            child = mutate(child, rng, config.mutation_rate, config.mutation_scale)
            next_population.append(child)
        population = next_population

    best = best_overall[1] if best_overall else AgentParameters()
    return best, reports


def _tournament(
    scored: list[tuple[float, float, AgentParameters]], rng: random.Random, size: int
) -> AgentParameters:
    contenders = rng.sample(scored, min(size, len(scored)))
    return max(contenders, key=lambda row: row[0])[2]
