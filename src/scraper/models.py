from dataclasses import dataclass, fields
from typing import Dict, Any

@dataclass
class BattingStat:
    player_name: str
    country: str
    runs: int
    not_out: int
    mins: int
    bf: int
    fours: int
    sixes: int
    sr: float
    inns: int
    opposition: str
    ground: str
    start_date: str

    @classmethod
    def from_row(cls, data: Dict[str, Any]):
        return cls(**data)

@dataclass
class BowlingStat:
    player_name: str
    country: str
    overs: float
    maidens: int
    runs: int
    wickets: int
    economy: float
    inns: int
    opposition: str
    ground: str
    start_date: str

    @classmethod
    def from_row(cls, data: Dict[str, Any]):
        return cls(**data)
