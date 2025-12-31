"""
Actions module - The full behavioral repertoire of medieval villainy and survival.

Now with actual implementation instead of aspirational enums.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from Villager import Villager
    from Village import VillageState


class ActionCategory(Enum):
    """Broad categories for action filtering"""
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    SOCIAL = "social"
    ECONOMIC = "economic"
    ROMANTIC = "romantic"
    INSTITUTIONAL = "institutional"
    INFORMATION = "information"
    PASSIVE = "passive"


class ActionType(Enum):
    # Aggressive
    ACCUSE_WITCHCRAFT = auto()
    SPREAD_RUMOR = auto()
    COUNTER_ACCUSE = auto()
    SEIZE_PROPERTY = auto()
    INSULT_PUBLICLY = auto()

    # Social Bonding
    FORM_ALLIANCE = auto()
    BREAK_ALLIANCE = auto()
    DEFEND_PERSON = auto()
    RECONCILE_WITH = auto()
    SOCIALIZE = auto()  # Generic positive interaction

    # Information
    SEEK_ADVICE = auto()
    SPY_ON = auto()
    GATHER_EVIDENCE = auto()
    SHARE_GOSSIP = auto()

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
    PROPOSE_MARRIAGE = auto()

    # Trial/Legal
    TESTIFY_AGAINST = auto()
    TESTIFY_FOR = auto()
    CONFESS = auto()

    # Self-Preservation
    HIDE_ACTIVITY = auto()
    FAKE_PIETY = auto()
    ISOLATE_FROM = auto()
    FLEE_VILLAGE = auto()

    # Social Signaling
    PUBLIC_DISPLAY = auto()
    AVOID_PERSON = auto()

    # Passive
    PASS = auto()


# Action metadata - categories, base costs, requirements
ACTION_METADATA = {
    # Aggressive actions
    ActionType.ACCUSE_WITCHCRAFT: {
        "category": ActionCategory.AGGRESSIVE,
        "base_cost": 0.08,  # Reduced from 0.15 - accusations are cheap
        "requires_target": True,
        "risk": 0.15,  # Reduced from 0.3 - less personal risk (historically accurate)
    },
    ActionType.SPREAD_RUMOR: {
        "category": ActionCategory.INFORMATION,
        "base_cost": 0.05,
        "requires_target": True,
        "risk": 0.1,
    },
    ActionType.COUNTER_ACCUSE: {
        "category": ActionCategory.DEFENSIVE,
        "base_cost": 0.2,
        "requires_target": True,
        "risk": 0.4,
        "requires_accused": True,  # Can only do if currently accused
    },
    ActionType.INSULT_PUBLICLY: {
        "category": ActionCategory.AGGRESSIVE,
        "base_cost": 0.1,
        "requires_target": True,
        "risk": 0.2,
    },
    ActionType.SEIZE_PROPERTY: {
        "category": ActionCategory.ECONOMIC,
        "base_cost": 0.25,
        "requires_target": True,
        "risk": 0.3,
        "requires_target_dead": True,
    },

    # Social actions
    ActionType.FORM_ALLIANCE: {
        "category": ActionCategory.SOCIAL,
        "base_cost": 0.1,
        "requires_target": True,
        "risk": 0.05,
    },
    ActionType.BREAK_ALLIANCE: {
        "category": ActionCategory.SOCIAL,
        "base_cost": 0.1,
        "requires_target": True,
        "risk": 0.15,
    },
    ActionType.DEFEND_PERSON: {
        "category": ActionCategory.SOCIAL,
        "base_cost": 0.2,
        "requires_target": True,
        "risk": 0.25,  # Guilt by association
    },
    ActionType.RECONCILE_WITH: {
        "category": ActionCategory.SOCIAL,
        "base_cost": 0.1,
        "requires_target": True,
        "risk": 0.05,
    },
    ActionType.SOCIALIZE: {
        "category": ActionCategory.SOCIAL,
        "base_cost": 0.02,
        "requires_target": True,
        "risk": 0.0,
    },

    # Information actions
    ActionType.SEEK_ADVICE: {
        "category": ActionCategory.INFORMATION,
        "base_cost": 0.05,
        "requires_target": True,
        "risk": 0.0,
    },
    ActionType.SPY_ON: {
        "category": ActionCategory.INFORMATION,
        "base_cost": 0.1,
        "requires_target": True,
        "risk": 0.2,
    },
    ActionType.GATHER_EVIDENCE: {
        "category": ActionCategory.INFORMATION,
        "base_cost": 0.15,
        "requires_target": True,
        "risk": 0.15,
    },
    ActionType.SHARE_GOSSIP: {
        "category": ActionCategory.INFORMATION,
        "base_cost": 0.03,
        "requires_target": True,
        "risk": 0.05,
    },

    # Economic actions
    ActionType.SHARE_RESOURCES: {
        "category": ActionCategory.ECONOMIC,
        "base_cost": 0.15,
        "requires_target": True,
        "risk": 0.0,
    },
    ActionType.REQUEST_HELP: {
        "category": ActionCategory.ECONOMIC,
        "base_cost": 0.05,
        "requires_target": True,
        "risk": 0.05,
    },
    ActionType.WITHHOLD_RESOURCES: {
        "category": ActionCategory.ECONOMIC,
        "base_cost": 0.05,
        "requires_target": True,
        "risk": 0.1,
    },

    # Institutional actions
    ActionType.ATTEND_CHURCH: {
        "category": ActionCategory.INSTITUTIONAL,
        "base_cost": 0.05,
        "requires_target": False,
        "risk": 0.0,
    },
    ActionType.APPEAL_TO_AUTHORITY: {
        "category": ActionCategory.INSTITUTIONAL,
        "base_cost": 0.1,
        "requires_target": False,
        "risk": 0.1,
    },
    ActionType.SEEK_PROTECTION: {
        "category": ActionCategory.INSTITUTIONAL,
        "base_cost": 0.15,
        "requires_target": True,  # From a high-status patron
        "risk": 0.05,
    },
    ActionType.OFFER_PATRONAGE: {
        "category": ActionCategory.INSTITUTIONAL,
        "base_cost": 0.2,
        "requires_target": True,
        "risk": 0.05,
        "requires_high_status": True,
    },

    # Romantic actions
    ActionType.COURT_PERSON: {
        "category": ActionCategory.ROMANTIC,
        "base_cost": 0.1,
        "requires_target": True,
        "risk": 0.1,
    },
    ActionType.PROPOSE_MARRIAGE: {
        "category": ActionCategory.ROMANTIC,
        "base_cost": 0.2,
        "requires_target": True,
        "risk": 0.15,
    },

    # Trial actions
    ActionType.TESTIFY_AGAINST: {
        "category": ActionCategory.AGGRESSIVE,
        "base_cost": 0.1,
        "requires_target": True,
        "risk": 0.1,
        "requires_target_on_trial": True,
    },
    ActionType.TESTIFY_FOR: {
        "category": ActionCategory.DEFENSIVE,
        "base_cost": 0.15,
        "requires_target": True,
        "risk": 0.2,
        "requires_target_on_trial": True,
    },
    ActionType.CONFESS: {
        "category": ActionCategory.DEFENSIVE,
        "base_cost": 0.0,
        "requires_target": False,
        "risk": 0.8,
        "requires_accused": True,
    },

    # Self-preservation
    ActionType.HIDE_ACTIVITY: {
        "category": ActionCategory.DEFENSIVE,
        "base_cost": 0.05,
        "requires_target": False,
        "risk": 0.1,
    },
    ActionType.FAKE_PIETY: {
        "category": ActionCategory.DEFENSIVE,
        "base_cost": 0.1,
        "requires_target": False,
        "risk": 0.15,
    },
    ActionType.AVOID_PERSON: {
        "category": ActionCategory.DEFENSIVE,
        "base_cost": 0.02,
        "requires_target": True,
        "risk": 0.0,
    },
    ActionType.FLEE_VILLAGE: {
        "category": ActionCategory.DEFENSIVE,
        "base_cost": 0.5,
        "requires_target": False,
        "risk": 0.3,
        "desperate": True,
    },

    # Passive
    ActionType.PASS: {
        "category": ActionCategory.PASSIVE,
        "base_cost": 0.0,
        "requires_target": False,
        "risk": 0.0,
    },
}


@dataclass
class Action:
    """An action taken by a villager"""
    actor_id: int
    action_type: ActionType
    target_id: Optional[int] = None
    parameters: Dict = field(default_factory=dict)
    timestep: int = 0
    utility_score: float = 0.0  # Computed expected utility

    @property
    def metadata(self) -> dict:
        return ACTION_METADATA.get(self.action_type, {})

    @property
    def category(self) -> ActionCategory:
        return self.metadata.get("category", ActionCategory.PASSIVE)

    @property
    def base_cost(self) -> float:
        return self.metadata.get("base_cost", 0.0)

    @property
    def risk(self) -> float:
        return self.metadata.get("risk", 0.0)

    def is_valid(self, actor: 'Villager', state: 'VillageState') -> bool:
        """Check if this action is valid given current state"""
        meta = self.metadata

        # Dead or imprisoned actors can't act
        if not actor.is_alive or actor.is_imprisoned:
            return False

        # Check target requirements
        if meta.get("requires_target", False):
            if self.target_id is None:
                return False
            target = state.villagers.get(self.target_id)
            if target is None:
                return False

            # Target must be alive (unless action requires dead target)
            if meta.get("requires_target_dead", False):
                if target.is_alive:
                    return False
            else:
                if not target.is_alive:
                    return False

            # Target must be on trial
            if meta.get("requires_target_on_trial", False):
                if not target.is_on_trial:
                    return False

        # Actor must be accused for certain actions
        if meta.get("requires_accused", False):
            if not actor.is_accused_currently:
                return False

        # Actor must be high status for certain actions
        if meta.get("requires_high_status", False):
            if actor.social_status < 0.6:
                return False

        # Action-specific validation
        if self.action_type == ActionType.COUNTER_ACCUSE:
            if not actor.accusations_received:
                return False

        elif self.action_type == ActionType.PROPOSE_MARRIAGE:
            from Villager import MaritalStatus
            if actor.marital_status == MaritalStatus.MARRIED:
                return False
            target = state.villagers.get(self.target_id)
            if target and target.marital_status == MaritalStatus.MARRIED:
                return False

        elif self.action_type == ActionType.BREAK_ALLIANCE:
            # Must have an existing alliance/loyalty
            if actor.emotional_state.loyalties.get(self.target_id, 0) < 0.3:
                return False

        return True


def get_available_actions(actor: 'Villager', state: 'VillageState') -> List[ActionType]:
    """Get list of action types available to this actor given current state"""
    available = [ActionType.PASS]  # Always available

    if not actor.is_alive or actor.is_imprisoned:
        return available

    # Self-targeted actions (no target needed)
    available.extend([
        ActionType.ATTEND_CHURCH,
        ActionType.HIDE_ACTIVITY,
        ActionType.FAKE_PIETY,
        ActionType.APPEAL_TO_AUTHORITY,
    ])

    # If accused, add defensive options
    if actor.is_accused_currently:
        available.append(ActionType.COUNTER_ACCUSE)
        available.append(ActionType.CONFESS)
        available.append(ActionType.FLEE_VILLAGE)

    # If high status, can offer patronage
    if actor.social_status > 0.6:
        available.append(ActionType.OFFER_PATRONAGE)

    # Target-requiring actions - these will be paired with targets later
    available.extend([
        ActionType.ACCUSE_WITCHCRAFT,
        ActionType.SPREAD_RUMOR,
        ActionType.INSULT_PUBLICLY,
        ActionType.FORM_ALLIANCE,
        ActionType.SOCIALIZE,
        ActionType.SHARE_GOSSIP,
        ActionType.COURT_PERSON,
        ActionType.SPY_ON,
        ActionType.GATHER_EVIDENCE,
        ActionType.SEEK_ADVICE,
        ActionType.SHARE_RESOURCES,
        ActionType.REQUEST_HELP,
        ActionType.SEEK_PROTECTION,
        ActionType.AVOID_PERSON,
        ActionType.RECONCILE_WITH,
    ])

    # If someone is on trial, can testify
    if state.trial_queue:
        available.append(ActionType.TESTIFY_FOR)
        available.append(ActionType.TESTIFY_AGAINST)

    # If has alliances, can break them
    if any(v > 0.3 for v in actor.emotional_state.loyalties.values()):
        available.append(ActionType.BREAK_ALLIANCE)

    return list(set(available))  # Deduplicate
