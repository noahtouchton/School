from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class ScoringRules:
    """Defines fantasy points awarded for various statistical categories."""
    pass_yard: float = 0.04
    pass_td: float = 4.0
    pass_int: float = -2.0
    pass_2pt: float = 2.0
    
    rush_yard: float = 0.1
    rush_td: float = 6.0
    rush_2pt: float = 2.0
    
    rec_yard: float = 0.1
    rec_td: float = 6.0
    rec_reception: float = 0.5  # Default Half-PPR
    rec_2pt: float = 2.0
    
    te_premium_bonus: float = 0.0  # Extra PPR points for tight ends
    
    fumble_lost: float = -2.0
    fumble_rec_td: float = 6.0

    @classmethod
    def standard(cls):
        """Standard scoring (0.0 PPR)."""
        return cls(rec_reception=0.0)

    @classmethod
    def half_ppr(cls):
        """Half PPR scoring (0.5 PPR)."""
        return cls(rec_reception=0.5)

    @classmethod
    def ppr(cls):
        """Full PPR scoring (1.0 PPR)."""
        return cls(rec_reception=1.0)

    @classmethod
    def te_premium(cls, bonus: float = 0.5):
        """PPR scoring with Tight End Premium bonus."""
        return cls(rec_reception=1.0, te_premium_bonus=bonus)


@dataclass
class RosterSettings:
    """Defines how many players can start at each position and total bench spots."""
    qb: int = 1
    rb: int = 2
    wr: int = 2
    te: int = 1
    flex: int = 2        # Flex spots (RB/WR/TE)
    superflex: int = 0   # Superflex spots (QB/RB/WR/TE)
    bench: int = 6
    ir: int = 1

    def total_roster_spots(self) -> int:
        return self.qb + self.rb + self.wr + self.te + self.flex + self.superflex + self.bench + self.ir

    def starting_positions(self) -> Dict[str, int]:
        """Returns starter counts per position."""
        return {
            "QB": self.qb,
            "RB": self.rb,
            "WR": self.wr,
            "TE": self.te,
            "FLEX": self.flex,
            "SUPERFLEX": self.superflex,
        }


@dataclass
class LeagueSettings:
    """Overall settings for the simulated league."""
    name: str = "Sandbox League"
    teams_count: int = 10
    scoring: ScoringRules = field(default_factory=ScoringRules.half_ppr)
    roster: RosterSettings = field(default_factory=RosterSettings)
    draft_type: str = "snake"  # snake or linear
    faab_budget: int = 100    # Free Agent Acquisition Budget for waiver bids
