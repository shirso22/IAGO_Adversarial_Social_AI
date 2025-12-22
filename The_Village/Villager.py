"""Individual villager representation"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    # Medieval Europe didn't acknowledge other options,
    # which is one of many reasons it sucked

class MaritalStatus(Enum):
    SINGLE = "single"
    MARRIED = "married"
    WIDOWED = "widowed"
    DIVORCED = "divorced"  # rare but existed

class Occupation(Enum):
    FARMER = "farmer"
    HEALER = "healer"
    MIDWIFE = "midwife"
    MERCHANT = "merchant"
    CLERGY = "clergy"
    JUDGE = "judge"  #presides trials
    CRAFTSPERSON = "craftsperson"
    CIVIL_SERVANT = "civilservant"
    BEGGAR = "beggar"
    NOBILITY = "nobility"
    POLICE = "police" #law enforcement
    # Add more as your historical accuracy obsession demands

@dataclass
class Villager:
    """A node in your graph of suffering"""

    # Immutable attributes (set at initialization)
    id: int
    name: str  # For debugging, because "Node 47 accused Node 89" is depressing
    gender: Gender
    age: int
    occupation: Occupation
    beauty: float #Physical attractiveness, on 0-1 scale

    # Semi-stable attributes (change slowly or rarely)
    social_status: float  # 0-1, where 1 = you can get away with murder (literally)
    wealth: float  # 0-1, relative wealth
    marital_status: MaritalStatus
    num_children: int = 0
    num_children_deceased: int = 0  # Because medieval childhood mortality was ~30%
    pain_tolerance: float = 0.5  # 0-1, how much ´pain´ they can withstand
    rationality: float = 0.5  # 0-1, how rational/intelligent they are

    # Dynamic attributes (change frequently during simulation)
    stress: float = 0.5  # 0-1, accumulated long term stress, models systemic suffering (e.g: a peasant from famine, general feudal oppression, etc)
    conformity_score: float = 0.5  # 0-1, how "normal" they appear
    reputation: float = 0.5  # 0-1, community standing

    # Accusation tracking (the fun stuff)
    accusations_made: List[int] = field(default_factory=list)  # IDs of people they accused
    accusations_received: List[int] = field(default_factory=list)  # IDs of accusers
    accusation_credibility: float = 0.5  # How believable their accusations are
    times_accused_total: int = 0  # Historical count
    times_accused_recent: int = 0  # Rolling window

    # Physical/social markers
    has_suspicious_marks: bool = False  # Birthmarks, whatever
    owns_cat: bool = False  # RIP if true
    lives_alone: bool = False
    church_attendance: float = 0.5  # 0-1, how often they show up

    # State tracking
    is_alive: bool = True
    is_accused_currently: bool = False
    is_on_trial: bool = False
    is_imprisoned: bool = False
    days_since_last_accusation: int = 999

    # Network-derived attributes (computed from graph, not stored)
    # These you'll calculate on-demand:
    # - degree_centrality
    # - betweenness_centrality
    # - clustering_coefficient
    # - number_of_high_status_connections

    @property
    def net_pain(self) -> float:
        """Calculates net pain dynamically based on fear and stress."""
        # Assumes PainCalculator is defined and accessible
        return calc_net_pain(self)

    def __repr__(self):
        return f"Villager({self.name}, {self.age}yo {self.gender.value} {self.occupation.value}, status={self.social_status:.2f}, alive={self.is_alive})"
