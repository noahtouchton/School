from dataclasses import dataclass


@dataclass
class ScoringRules:
    """Fantasy points awarded per statistical category. Default: half-PPR."""

    pass_yard: float = 0.04
    pass_td: float = 4.0
    pass_int: float = -2.0
    rush_yard: float = 0.1
    rush_td: float = 6.0
    rec_yard: float = 0.1
    rec_td: float = 6.0
    rec_reception: float = 0.5
    fumble_lost: float = -2.0
    two_point_conversion: float = 2.0

    @classmethod
    def standard(cls) -> "ScoringRules":
        return cls(rec_reception=0.0)

    @classmethod
    def half_ppr(cls) -> "ScoringRules":
        return cls(rec_reception=0.5)

    @classmethod
    def ppr(cls) -> "ScoringRules":
        return cls(rec_reception=1.0)


DEFAULT_SCORING = ScoringRules.half_ppr()


def calculate_fantasy_points(stats: dict, rules: ScoringRules = DEFAULT_SCORING) -> float:
    """Computes fantasy points from a raw stat dict (see ingestion/nfl_data.py for keys)."""
    pts = 0.0
    pts += stats.get("passing_yards", 0.0) * rules.pass_yard
    pts += stats.get("passing_tds", 0.0) * rules.pass_td
    pts += stats.get("interceptions", 0.0) * rules.pass_int
    pts += stats.get("rushing_yards", 0.0) * rules.rush_yard
    pts += stats.get("rushing_tds", 0.0) * rules.rush_td
    pts += stats.get("receiving_yards", 0.0) * rules.rec_yard
    pts += stats.get("receiving_tds", 0.0) * rules.rec_td
    pts += stats.get("receptions", 0.0) * rules.rec_reception
    pts += stats.get("fumbles_lost", 0.0) * rules.fumble_lost
    pts += stats.get("two_point_conversions", 0.0) * rules.two_point_conversion
    return round(pts, 2)
