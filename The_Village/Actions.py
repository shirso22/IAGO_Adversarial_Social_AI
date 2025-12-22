#Defines different possible actions an individual villager can take

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict

class ActionType(Enum):
    # Aggressive
    ACCUSE_WITCHCRAFT = auto()
    SPREAD_RUMOR = auto()
    COUNTER_ACCUSE = auto()
    SEIZE_PROPERTY = auto()

    # Social Bonding
    FORM_ALLIANCE = auto()
    BREAK_ALLIANCE = auto()
    DEFEND_PERSON = auto()
    RECONCILE_WITH = auto()

    # Information
    SEEK_ADVICE = auto()
    SPY_ON = auto()
    GATHER_EVIDENCE = auto()

    # Economic
    SHARE_RESOURCES = auto()
    REQUEST_HELP = auto()
    WITHHOLD_RESOURCES = auto()
    DEMAND_PAYMENT = auto()

    # Institutional
    ATTEND_CHURCH = auto()
    APPEAL_TO_AUTHORITY = auto()
    SEEK_PROTECTION = auto()
    OFFER_PATRONAGE = auto()

    # Romantic
    COURT_PERSON = auto()
    REJECT_ADVANCES = auto()
    MARRIAGE = auto()
    DIVORCE = auto()

    # Trial/Legal
    TESTIFY_AGAINST = auto()
    TESTIFY_FOR = auto()
    CONFESS = auto()

    # Self-Preservation
    HIDE_ACTIVITY = auto()
    FAKE_PIETY = auto()
    ISOLATE_FROM = auto()

    # Social Signaling
    PUBLIC_DISPLAY = auto() #to signal conformity
    AVOID_PERSON = auto()

    # Passive
    PASS = auto() #Mind your own business
    DISPLAY_EMOTION = auto() #Internal state leaking, some personalities can hide or even fake this

@dataclass
class Action:
    """An action taken by a villager"""
    actor_id: int
    action_type: ActionType
    target_id: Optional[int] = None  # Some actions have targets
    parameters: Dict = field(default_factory=dict)  # Additional params
    timestep: int = 0

    def is_valid(self, village_state) -> bool:
        """Check if this action is valid given current state"""
        # Different actions have different validity conditions
        # e.g., can't accuse someone who's already dead
        # can't counter-accuse if you haven't been accused
        pass

    def get_cost(self, village_state) -> float:
        """Some actions have costs (economic, social, risk)"""
        pass

    def get_expected_utility(self, village_state, personality, emotions) -> float:
        """How much does this action benefit the actor?
        Used for NPC decision-making"""
        pass
