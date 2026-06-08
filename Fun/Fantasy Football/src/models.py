from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum

class Position(str, Enum):
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DST = "DST"

@dataclass
class Player:
    """Represents a real NFL player."""
    id: str  # Unique identifier (e.g. Sleeper ID or GSIS ID)
    name: str
    position: Position
    nfl_team: str
    status: str = "Active"  # Active, IR, Out, Suspended, etc.
    injury_status: Optional[str] = None
    age: Optional[int] = None
    experience: Optional[int] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Player):
            return False
        return self.id == other.id


@dataclass
class PlayerWeeklyStats:
    """Represents actual NFL statistics accumulated by a player in a specific week."""
    player_id: str
    year: int
    week: int
    stats: Dict[str, float] = field(default_factory=dict)
    
    def get(self, key: str, default: float = 0.0) -> float:
        return self.stats.get(key, default)


@dataclass
class PlayerWeeklyProjection:
    """Represents projected NFL statistics / fantasy points for a player in a specific week."""
    player_id: str
    year: int
    week: int
    projected_points: float
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class Roster:
    """Represents a fantasy team's roster of players, split into starters and bench."""
    starters: List[Player] = field(default_factory=list)
    bench: List[Player] = field(default_factory=list)
    ir: List[Player] = field(default_factory=list)

    def all_players(self) -> List[Player]:
        return self.starters + self.bench + self.ir

    def add_player(self, player: Player):
        self.bench.append(player)

    def remove_player(self, player: Player):
        if player in self.bench:
            self.bench.remove(player)
        elif player in self.starters:
            self.starters.remove(player)
        elif player in self.ir:
            self.ir.remove(player)
        else:
            raise ValueError(f"Player {player.name} not found on roster.")


@dataclass
class Team:
    """Represents a fantasy franchise owned by a user or an AI persona."""
    id: str
    name: str
    owner_persona: str  # "human", "free_agent_demon", "trade_demon", etc.
    roster: Roster = field(default_factory=Roster)
    faab_balance: int = 100
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    points_against: float = 0.0

    @property
    def record_str(self) -> str:
        return f"{self.wins}-{self.losses}-{self.ties}"


@dataclass
class Matchup:
    """Represents a head-to-head matchup between two teams in a given week."""
    week: int
    team_a_id: str
    team_b_id: str
    team_a_score: float = 0.0
    team_b_score: float = 0.0
    completed: bool = False
    
    # Store the actual rosters locked in for this matchup
    team_a_starters: List[Player] = field(default_factory=list)
    team_b_starters: List[Player] = field(default_factory=list)


@dataclass
class DraftState:
    """Represents the real-time state of a fantasy draft."""
    year: int
    rounds: int
    teams: List[Team]
    draft_order: List[str]  # List of team_ids in first-round order
    picks: List[Dict[str, Any]] = field(default_factory=list) # List of dicts: {pick_num, round, team_id, player_id}
    drafted_player_ids: Set[str] = field(default_factory=set)

    @property
    def current_pick_index(self) -> int:
        return len(self.picks)

    @property
    def current_round(self) -> int:
        return (self.current_pick_index // len(self.draft_order)) + 1

    def get_current_team_id(self) -> str:
        """Determines whose turn it is using snake draft logic."""
        if self.current_pick_index >= self.rounds * len(self.draft_order):
            raise ValueError("Draft is already completed.")
            
        round_idx = self.current_pick_index // len(self.draft_order)
        pick_within_round = self.current_pick_index % len(self.draft_order)
        
        # If snake draft and even round (0-indexed), reverse the order
        if round_idx % 2 == 1:
            return self.draft_order[len(self.draft_order) - 1 - pick_within_round]
        else:
            return self.draft_order[pick_within_round]

    def draft_player(self, team_id: str, player: Player):
        if player.id in self.drafted_player_ids:
            raise ValueError(f"Player {player.name} has already been drafted.")
            
        pick_num = self.current_pick_index + 1
        round_num = self.current_round
        
        self.picks.append({
            "pick_number": pick_num,
            "round": round_num,
            "team_id": team_id,
            "player": player
        })
        self.drafted_player_ids.add(player.id)


@dataclass
class TransactionType(str, Enum):
    ADD = "ADD"
    DROP = "DROP"
    TRADE = "TRADE"


@dataclass
class WaiverClaim:
    """Represents an AI/human team's bid for a free agent player."""
    team_id: str
    player_to_add: Player
    player_to_drop: Optional[Player] = None
    bid_amount: int = 0  # 0 for normal waiver priority, >0 for FAAB leagues
    priority_order: int = 0  # To rank multiple claims by the same team


@dataclass
class TradeProposal:
    """Represents a trade offer between two teams."""
    id: str
    proposer_team_id: str
    receiver_team_id: str
    proposer_sends: List[Player]
    receiver_sends: List[Player]
    status: str = "Proposed"  # Proposed, Accepted, Rejected, Cancelled, Executed
    week: int = 1
