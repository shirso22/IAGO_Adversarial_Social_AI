"""Individual villager representation"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

from Personality_and_emotions import Personality, EmotionalState


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
    JUDGE = "judge"  # presides trials
    CRAFTSPERSON = "craftsperson"
    CIVIL_SERVANT = "civilservant"
    BEGGAR = "beggar"
    NOBILITY = "nobility"
    POLICE = "police"  # law enforcement
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
    beauty: float = 0.5  # Physical attractiveness, on 0-1 scale

    # Personality and emotional state (THE KEY ADDITIONS)
    personality: Personality = field(default_factory=Personality)
    emotional_state: EmotionalState = field(default_factory=EmotionalState)

    # Semi-stable attributes (change slowly or rarely)
    social_status: float = 0.5  # 0-1, where 1 = you can get away with murder (literally)
    wealth: float = 0.5  # 0-1, relative wealth
    marital_status: MaritalStatus = MaritalStatus.SINGLE
    num_children: int = 0
    num_children_deceased: int = 0  # Because medieval childhood mortality was ~30%
    pain_tolerance: float = 0.5  # 0-1, how much pain they can withstand
    rationality: float = 0.5  # 0-1, how rational/intelligent they are

    # Dynamic attributes (change frequently during simulation)
    stress: float = 0.5  # 0-1, accumulated long term stress
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
    
    # Memory and trauma (for hysteresis)
    trauma_score: float = 0.0  # Accumulated psychological damage, decays very slowly
    witnessed_executions: int = 0  # Count of executions witnessed
    family_executions: int = 0  # Executions of family members (near-permanent trauma)
    
    # Alliance tracking (for diminishing returns)
    alliance_history: Dict[int, int] = field(default_factory=dict)  # target_id -> interaction count
    
    # Patronage
    patron_id: Optional[int] = None  # ID of protecting patron, if any
    dependents: List[int] = field(default_factory=list)  # IDs of those under protection

    def __repr__(self):
        status = "ALIVE" if self.is_alive else "DEAD"
        accused = " [ACCUSED]" if self.is_accused_currently else ""
        return f"Villager({self.name}, {self.age}yo {self.gender.value} {self.occupation.value}{accused}, {status})"
