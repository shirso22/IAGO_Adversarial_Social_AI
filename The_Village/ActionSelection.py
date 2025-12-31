"""
Action Selection Module - Where personality meets decision-making.

This is the connective tissue between the interaction vectors,
personality traits, emotional states, and actual behavioral choices.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from Villager import Villager, Gender, MaritalStatus, Occupation
from Village import VillageState
from Relationships import Relationship, RelationshipType
from Actions import ActionType, ActionCategory, Action, ACTION_METADATA, get_available_actions
from Utils import (
    get_vulnerability, 
    get_base_interaction_vector,
    get_alliance_diminishing_factor,
    get_patron_protection_value,
    get_protection_for_dependent,
    StressorType,
    STRESSOR_EFFECTS,
)


@dataclass
class ScoredAction:
    """An action with its computed utility score"""
    action_type: ActionType
    target_id: Optional[int]
    utility: float
    reasoning: str = ""  # For debugging


def compute_action_utility(
    actor: Villager,
    action_type: ActionType,
    target_id: Optional[int],
    state: VillageState
) -> float:
    """
    Compute expected utility of an action for an actor.
    
    This is where personality, emotions, relationships, and context
    all combine to produce a single number representing "how much
    does this actor want to do this action right now?"
    """
    meta = ACTION_METADATA.get(action_type, {})
    base_cost = meta.get("base_cost", 0.0)
    risk = meta.get("risk", 0.0)
    
    # Start with base utility (some actions are inherently more rewarding)
    utility = 0.1
    
    # Risk aversion based on personality
    risk_tolerance = (
        actor.personality.psychopathy * 0.3 +
        (1 - actor.personality.neuroticism) * 0.3 +
        actor.personality.machiavellianism * 0.2 +
        (1 - actor.personality.conscientiousness) * 0.2
    )
    risk_penalty = risk * (1 - risk_tolerance)
    utility -= risk_penalty
    
    # Cost sensitivity based on wealth and greed
    cost_sensitivity = (1 - actor.wealth) * 0.5 + actor.personality.greed * 0.5
    utility -= base_cost * cost_sensitivity
    
    # Get target if applicable
    target = state.villagers.get(target_id) if target_id is not None else None
    
    # Get relationship state if target exists
    relationship_quality = 0.0
    if target:
        loyalty = actor.emotional_state.loyalties.get(target_id, 0)
        hatred = actor.emotional_state.hatreds.get(target_id, 0)
        relationship_quality = loyalty - hatred
    
    # === ACTION-SPECIFIC UTILITY CALCULATIONS ===
    
    # --- AGGRESSIVE ACTIONS ---
    if action_type == ActionType.ACCUSE_WITCHCRAFT:
        if target:
            # Motivation: hatred, fear, machiavellianism, rivalry
            # Balanced coefficients - accusations happen but don't cascade
            utility += actor.emotional_state.hatreds.get(target_id, 0) * 0.35
            utility += actor.personality.machiavellianism * 0.25
            utility += actor.emotional_state.fear * 0.18  # Scapegoating
            
            # Target vulnerability makes it more attractive
            vuln = get_vulnerability(target, state)
            utility += vuln * 0.25
            
            # Rivalry bonus
            if (actor.id, target_id) in state.relationships:
                rel = state.relationships[(actor.id, target_id)]
                if RelationshipType.RIVALRY in rel.relationship_types:
                    utility += 0.2
            
            # PATRON TARGETING - taking out a patron is high-value for Machiavellians
            if target.dependents and len(target.dependents) > 2:
                utility += actor.personality.machiavellianism * 0.2
                utility += min(len(target.dependents) * 0.03, 0.15)
            
            # STRESSOR EFFECTS
            if StressorType.PLAGUE in state.active_stressors:
                if target.occupation in [Occupation.HEALER, Occupation.MIDWIFE]:
                    utility += STRESSOR_EFFECTS[StressorType.PLAGUE].get('healer_suspicion', 0.3) * 0.7
            
            if StressorType.INQUISITOR_VISIT in state.active_stressors:
                utility += 0.12
            
            # Historical violence makes accusations seem more "normal"
            utility += state.historical_violence * 0.12
            
            # Panic drives accusations
            utility += state.panic_level * 0.15
            
            # SOCIAL COST OF ACCUSING
            utility -= actor.personality.agreeableness * 0.2
            
            # Serial accusers face diminishing returns
            recent_accusations_by_actor = len([a for a in actor.accusations_made 
                                               if a in [acc[2] for acc in state.recent_accusations 
                                                       if acc[0] > state.timestep - 10]])
            utility -= recent_accusations_by_actor * 0.12
            
            # Family/loyalty penalty
            utility -= actor.emotional_state.loyalties.get(target_id, 0) * 0.5
            if (actor.id, target_id) in state.relationships:
                rel = state.relationships[(actor.id, target_id)]
                if RelationshipType.FAMILY in rel.relationship_types:
                    utility -= 0.4
            
            # Agreeableness suppresses accusations
            utility -= actor.personality.agreeableness * 0.3
            
            # Panic increases accusation utility
            utility += state.panic_level * 0.2
            
    elif action_type == ActionType.SPREAD_RUMOR:
        if target:
            # Gossip-driven personalities love this
            utility += actor.personality.extraversion * 0.2
            utility += (1 - actor.personality.conscientiousness) * 0.15
            utility += actor.emotional_state.hatreds.get(target_id, 0) * 0.2
            utility += actor.personality.envy * 0.15
            
            # Rumor saturation diminishes returns
            utility -= state.rumor_saturation * 0.3
            
    elif action_type == ActionType.COUNTER_ACCUSE:
        if target and actor.is_accused_currently:
            # Desperation-driven
            utility += actor.emotional_state.fear * 0.4
            utility += actor.emotional_state.anger * 0.3
            utility += actor.personality.machiavellianism * 0.2
            
            # Target is ideally an accuser
            if target_id in actor.accusations_received:
                utility += 0.3
                
    elif action_type == ActionType.INSULT_PUBLICLY:
        if target:
            # Wrath and low agreeableness drive this
            utility += actor.personality.wrath * 0.3
            utility += (1 - actor.personality.agreeableness) * 0.25
            utility += actor.emotional_state.anger * 0.25
            utility += actor.emotional_state.hatreds.get(target_id, 0) * 0.3
            
            # Status differential matters
            if target.social_status < actor.social_status:
                utility += 0.1  # Safer to punch down
            else:
                utility -= 0.2  # Risky to punch up
                
    # --- SOCIAL ACTIONS ---
    elif action_type == ActionType.FORM_ALLIANCE:
        if target:
            # Driven by strategic thinking and social needs
            utility += actor.personality.extraversion * 0.2
            utility += actor.personality.machiavellianism * 0.15
            
            # Target's status makes alliance more valuable
            utility += target.social_status * 0.25
            
            # Existing positive relationship helps
            utility += max(0, relationship_quality) * 0.2
            
            # Fear drives alliance-seeking
            utility += actor.emotional_state.fear * 0.2
            
            # Can't ally with enemies
            if actor.emotional_state.hatreds.get(target_id, 0) > 0.5:
                utility -= 0.5
            
            # DIMINISHING RETURNS - repeated alliances with same person less valuable
            diminishing_factor = get_alliance_diminishing_factor(actor, target_id)
            utility *= diminishing_factor
                
    elif action_type == ActionType.BREAK_ALLIANCE:
        if target:
            # Driven by betrayal, better opportunities, or self-preservation
            utility += actor.personality.machiavellianism * 0.3
            utility += actor.emotional_state.hatreds.get(target_id, 0) * 0.3
            
            # If target is accused, self-preservation kicks in
            if target.is_accused_currently:
                utility += (1 - actor.personality.agreeableness) * 0.3
                utility += actor.emotional_state.fear * 0.2
                
            # Loyalty suppresses this
            utility -= actor.emotional_state.loyalties.get(target_id, 0) * 0.4
            
    elif action_type == ActionType.SOCIALIZE:
        if target:
            # Basic social interaction - make this more attractive
            utility += actor.personality.extraversion * 0.35  # Was 0.3
            utility += actor.personality.agreeableness * 0.25  # Was 0.2
            utility += max(0, relationship_quality) * 0.25  # Was 0.2
            
            # Socializing reduces stress - intrinsic reward
            utility += actor.stress * 0.15
            
            # Avoid enemies
            utility -= actor.emotional_state.hatreds.get(target_id, 0) * 0.4
            
    elif action_type == ActionType.RECONCILE_WITH:
        if target:
            # Trying to repair a damaged relationship
            utility += actor.personality.agreeableness * 0.35  # Was 0.3
            utility += (1 - actor.personality.pride) * 0.25  # Was 0.2
            
            # Stress and fear motivate reconciliation (safety in numbers)
            utility += actor.stress * 0.15
            utility += actor.emotional_state.fear * 0.1
            
            # Only makes sense if relationship is damaged
            if relationship_quality < 0:
                utility += abs(relationship_quality) * 0.25  # Was 0.2
            else:
                utility -= 0.3  # Why reconcile if not damaged?
                
    elif action_type == ActionType.DEFEND_PERSON:
        if target and target.is_accused_currently:
            # Loyalty-driven
            utility += actor.emotional_state.loyalties.get(target_id, 0) * 0.5
            utility += actor.personality.agreeableness * 0.2
            
            # Family bonds compel defense
            if (actor.id, target_id) in state.relationships:
                rel = state.relationships[(actor.id, target_id)]
                if RelationshipType.FAMILY in rel.relationship_types:
                    utility += 0.4
                    
            # But fear of association suppresses it
            utility -= actor.emotional_state.fear * 0.3
            utility -= state.panic_level * 0.2
            
    # --- INFORMATION ACTIONS ---
    elif action_type == ActionType.SPY_ON:
        if target:
            # Machiavellian prep work
            utility += actor.personality.machiavellianism * 0.35
            utility += actor.emotional_state.suspicions.get(target_id, 0) * 0.25
            utility += actor.emotional_state.hatreds.get(target_id, 0) * 0.2
            
    elif action_type == ActionType.GATHER_EVIDENCE:
        if target:
            # Building a case against someone
            utility += actor.personality.machiavellianism * 0.3
            utility += actor.emotional_state.hatreds.get(target_id, 0) * 0.3
            utility += actor.personality.conscientiousness * 0.15  # Methodical
            
    elif action_type == ActionType.SHARE_GOSSIP:
        if target:
            # Social bonding through information sharing
            utility += actor.personality.extraversion * 0.25
            utility += (1 - actor.personality.conscientiousness) * 0.15
            utility += max(0, relationship_quality) * 0.15
            utility += state.rumor_saturation * 0.1  # More gossip = more to share
            
    elif action_type == ActionType.SEEK_ADVICE:
        if target:
            # Seeking wisdom from higher-status individuals
            if target.social_status > actor.social_status:
                utility += 0.2
            utility += actor.emotional_state.fear * 0.15
            utility += (1 - actor.personality.narcissism) * 0.2
            
    # --- ECONOMIC ACTIONS ---
    elif action_type == ActionType.SHARE_RESOURCES:
        if target:
            # Generosity and alliance-building
            utility += actor.personality.agreeableness * 0.3
            utility += (1 - actor.personality.greed) * 0.2
            utility += actor.emotional_state.loyalties.get(target_id, 0) * 0.25
            
            # Only if you have resources to spare
            utility += (actor.wealth - 0.5) * 0.3
            
            # DIMINISHING RETURNS - repeatedly sharing with same person less impactful
            diminishing_factor = get_alliance_diminishing_factor(actor, target_id)
            utility *= diminishing_factor
            
    elif action_type == ActionType.REQUEST_HELP:
        if target:
            # Desperation-driven
            utility += (1 - actor.wealth) * 0.3
            utility += actor.stress * 0.2
            utility += actor.emotional_state.loyalties.get(target_id, 0) * 0.2
            
    # --- INSTITUTIONAL ACTIONS ---
    elif action_type == ActionType.ATTEND_CHURCH:
        # Self-preservation through conformity
        utility += actor.personality.conscientiousness * 0.2
        utility += (1 - actor.conformity_score) * 0.15  # More need if non-conformist
        utility += actor.emotional_state.fear * 0.2
        utility += state.panic_level * 0.15
        
        # Stressor effects - inquisitor makes church attendance urgent
        if StressorType.INQUISITOR_VISIT in state.active_stressors:
            utility += 0.3
        if StressorType.HARSH_WINTER in state.active_stressors:
            utility += 0.1  # Seeking comfort
        
        # Conformity pressure increases utility
        utility += state.conformity_pressure * 0.2
        
        # Already pious? Less urgent
        utility -= actor.church_attendance * 0.2
        
    elif action_type == ActionType.FAKE_PIETY:
        # Desperate conformity signaling
        utility += actor.personality.machiavellianism * 0.3
        utility += actor.emotional_state.fear * 0.3
        utility += (1 - actor.church_attendance) * 0.2
        
        # Stressor effects
        if StressorType.INQUISITOR_VISIT in state.active_stressors:
            utility += 0.25
        
        utility += state.conformity_pressure * 0.15
        
        # Riskier if you're not good at lying
        utility -= actor.personality.conscientiousness * 0.15
    
    elif action_type == ActionType.OFFER_PATRONAGE:
        if target and actor.social_status > target.social_status:
            # Building a power base
            utility += actor.personality.machiavellianism * 0.2
            utility += actor.personality.pride * 0.15  # Ego gratification
            
            # Target loyalty makes them better dependents
            utility += target.emotional_state.loyalties.get(actor.id, 0) * 0.2
            
            # COSTS OF PATRONAGE - wealthy can afford more
            current_dependents = len(actor.dependents)
            if current_dependents > 0:
                # Each additional dependent is less valuable
                utility -= current_dependents * 0.1
            
            # If resources are scarce, patronage is expensive
            utility -= state.resource_scarcity * 0.3
            
            # Wealthy can sustain more dependents
            utility += actor.wealth * 0.25
            
            # During famine, taking on dependents is very costly
            if StressorType.FAMINE in state.active_stressors:
                utility -= 0.3
        
    elif action_type == ActionType.SEEK_PROTECTION:
        if target and target.social_status > actor.social_status:
            # Finding a patron
            utility += actor.emotional_state.fear * 0.35
            utility += (1 - actor.social_status) * 0.2
            utility += target.social_status * 0.2
            
            # Already have a patron? Less urgent
            if actor.patron_id is not None:
                utility -= 0.3
            
            # Check patron's current capacity - crowded patrons less attractive
            current_protection = get_patron_protection_value(target, state)
            num_dependents = len(target.dependents)
            
            if num_dependents > 0:
                # Dilution penalty - less attractive if already crowded
                crowding_penalty = min(0.3, num_dependents * 0.05)
                utility -= crowding_penalty
            
            # Wealthy patrons more attractive (can sustain more dependents)
            utility += target.wealth * 0.15
            
            # DIMINISHING RETURNS if repeatedly seeking from same patron
            diminishing_factor = get_alliance_diminishing_factor(actor, target_id)
            utility *= diminishing_factor
            
    elif action_type == ActionType.APPEAL_TO_AUTHORITY:
        utility += actor.emotional_state.fear * 0.25
        utility += state.trust_in_authority * 0.2
        
        # If accused, more desperate
        if actor.is_accused_currently:
            utility += 0.3
            
    # --- ROMANTIC ACTIONS ---
    elif action_type == ActionType.COURT_PERSON:
        if target and actor.gender != target.gender:
            # Driven by attraction and lust
            utility += actor.personality.lust * 0.3
            utility += actor.emotional_state.desire * 0.25
            utility += actor.emotional_state.romantic_attractions.get(target_id, 0) * 0.3
            
            # Beauty of target matters
            utility += target.beauty * 0.2
            
            # Status considerations
            utility -= abs(actor.social_status - target.social_status) * 0.15
            
            # Already married? Penalty
            if actor.marital_status == MaritalStatus.MARRIED:
                utility -= 0.3
                utility += actor.personality.machiavellianism * 0.1  # Unless scheming
                
    elif action_type == ActionType.PROPOSE_MARRIAGE:
        if target:
            # Requires existing attraction
            attraction = actor.emotional_state.romantic_attractions.get(target_id, 0)
            if attraction > 0.5:
                utility += attraction * 0.4
                utility += actor.emotional_state.loyalties.get(target_id, 0) * 0.2
                
                # Strategic marriages
                utility += actor.personality.machiavellianism * target.social_status * 0.2
            else:
                utility -= 0.5  # Don't propose to strangers
                
    # --- TRIAL ACTIONS ---
    elif action_type == ActionType.TESTIFY_AGAINST:
        if target and target.is_on_trial:
            utility += actor.emotional_state.hatreds.get(target_id, 0) * 0.4
            utility += actor.personality.machiavellianism * 0.2
            utility += state.panic_level * 0.15  # Mob mentality
            
            # Rivalry bonus
            if (actor.id, target_id) in state.relationships:
                rel = state.relationships[(actor.id, target_id)]
                if RelationshipType.RIVALRY in rel.relationship_types:
                    utility += 0.25
                    
    elif action_type == ActionType.TESTIFY_FOR:
        if target and target.is_on_trial:
            utility += actor.emotional_state.loyalties.get(target_id, 0) * 0.4
            utility += actor.personality.agreeableness * 0.2
            
            # Family compels testimony
            if (actor.id, target_id) in state.relationships:
                rel = state.relationships[(actor.id, target_id)]
                if RelationshipType.FAMILY in rel.relationship_types:
                    utility += 0.35
                    
            # But fear of association
            utility -= actor.emotional_state.fear * 0.25
            utility -= state.panic_level * 0.2
            
    elif action_type == ActionType.CONFESS:
        if actor.is_accused_currently:
            # Last resort - might reduce sentence (historically sometimes true)
            utility += actor.emotional_state.despair * 0.4
            utility += (1 - actor.emotional_state.hope) * 0.3
            
            # Pain tolerance affects willingness to endure trial
            utility -= actor.pain_tolerance * 0.2
            
    # --- SELF-PRESERVATION ---
    elif action_type == ActionType.HIDE_ACTIVITY:
        utility += actor.emotional_state.fear * 0.3
        utility += actor.personality.neuroticism * 0.2
        utility += (1 - actor.conformity_score) * 0.2  # Non-conformists have more to hide
        
    elif action_type == ActionType.AVOID_PERSON:
        if target:
            utility += actor.emotional_state.hatreds.get(target_id, 0) * 0.2
            utility += actor.emotional_state.fear * 0.2
            
            # Avoid accusers
            if target_id in actor.accusations_received:
                utility += 0.3
                
            # Avoid the accused (guilt by association)
            if target.is_accused_currently:
                utility += 0.25
                
    elif action_type == ActionType.FLEE_VILLAGE:
        if actor.is_accused_currently:
            # Extreme desperation
            utility += actor.emotional_state.fear * 0.5
            utility += actor.emotional_state.despair * 0.3
            
            # But it means losing everything
            utility -= actor.wealth * 0.3
            utility -= actor.social_status * 0.2
        else:
            utility -= 0.5  # Why flee if not accused?
            
    # --- PASSIVE ---
    elif action_type == ActionType.PASS:
        # Default option - attractive to cautious, introverted types
        utility = 0.15
        utility += (1 - actor.personality.extraversion) * 0.1
        utility += actor.personality.conscientiousness * 0.05
        utility -= actor.emotional_state.anger * 0.1  # Angry people want to act
        utility -= actor.emotional_state.fear * 0.05  # Fearful people want to protect
    
    return utility


def select_action(
    actor: Villager,
    state: VillageState,
    rng: np.random.RandomState,
    top_k: int = 5
) -> Tuple[ActionType, Optional[int]]:
    """
    Select an action for a villager using utility-weighted random selection.
    
    Process:
    1. Get available action types
    2. Generate (action, target) pairs for each action type
    3. Score each pair
    4. Soft-select from top options
    """
    if not actor.is_alive or actor.is_imprisoned:
        return (ActionType.PASS, None)
    
    available_actions = get_available_actions(actor, state)
    scored_actions: List[ScoredAction] = []
    
    # Get list of potential targets
    alive_others = [v for v in state.villagers.values() 
                    if v.id != actor.id and v.is_alive]
    
    for action_type in available_actions:
        meta = ACTION_METADATA.get(action_type, {})
        
        if meta.get("requires_target", False):
            # Score this action with each potential target
            for target in alive_others:
                # Quick validity check
                if meta.get("requires_target_on_trial", False) and not target.is_on_trial:
                    continue
                if meta.get("requires_high_status", False) and actor.social_status < 0.6:
                    continue
                    
                utility = compute_action_utility(actor, action_type, target.id, state)
                
                if utility > 0:  # Only consider positive utility actions
                    scored_actions.append(ScoredAction(
                        action_type=action_type,
                        target_id=target.id,
                        utility=utility
                    ))
        else:
            # No target needed
            utility = compute_action_utility(actor, action_type, None, state)
            scored_actions.append(ScoredAction(
                action_type=action_type,
                target_id=None,
                utility=utility
            ))
    
    if not scored_actions:
        return (ActionType.PASS, None)
    
    # Sort by utility and take top k
    scored_actions.sort(key=lambda x: x.utility, reverse=True)
    top_actions = scored_actions[:top_k]
    
    # Convert utilities to probabilities (softmax-ish)
    utilities = np.array([max(a.utility, 0.01) for a in top_actions])
    
    # Temperature parameter - higher = more random, lower = more greedy
    temperature = 0.5
    utilities = utilities ** (1 / temperature)
    probs = utilities / utilities.sum()
    
    # Select
    idx = rng.choice(len(top_actions), p=probs)
    selected = top_actions[idx]
    
    return (selected.action_type, selected.target_id)


def get_testimony_actions(
    actor: Villager,
    accused_id: int,
    state: VillageState,
    rng: np.random.RandomState
) -> Optional[Tuple[ActionType, int]]:
    """
    During a trial, determine if this villager will testify and how.
    Called separately from main action selection.
    """
    accused = state.villagers.get(accused_id)
    if not accused or not accused.is_on_trial:
        return None
    
    if not actor.is_alive or actor.is_imprisoned:
        return None
    
    if actor.id == accused_id:
        return None
    
    # Calculate utilities for testimony
    for_utility = compute_action_utility(actor, ActionType.TESTIFY_FOR, accused_id, state)
    against_utility = compute_action_utility(actor, ActionType.TESTIFY_AGAINST, accused_id, state)
    abstain_utility = 0.2  # Base utility for staying quiet
    
    # Normalize
    total = for_utility + against_utility + abstain_utility
    if total <= 0:
        return None
    
    probs = np.array([for_utility, against_utility, abstain_utility]) / total
    probs = np.clip(probs, 0, 1)
    probs = probs / probs.sum()
    
    choice = rng.choice(3, p=probs)
    
    if choice == 0 and for_utility > 0.1:
        return (ActionType.TESTIFY_FOR, accused_id)
    elif choice == 1 and against_utility > 0.1:
        return (ActionType.TESTIFY_AGAINST, accused_id)
    
    return None
