"""
Action Execution Module - Where decisions become consequences.

This module handles the actual state changes when villagers take actions.
Every action has effects on emotions, relationships, reputation, and village state.
"""

import numpy as np
from typing import Optional, Tuple, List

from Villager import Villager, Gender, MaritalStatus
from Village import VillageState
from Relationships import Relationship, RelationshipType
from Actions import ActionType
from Utils import record_alliance_interaction, handle_patron_collapse


def execute_action(
    actor: Villager,
    action_type: ActionType,
    target_id: Optional[int],
    state: VillageState,
    rng: np.random.RandomState,
    verbose: bool = True
) -> bool:
    """
    Execute an action and update all relevant state.
    Returns True if action was successful, False otherwise.
    """
    target = state.villagers.get(target_id) if target_id else None
    
    # === AGGRESSIVE ACTIONS ===
    
    if action_type == ActionType.ACCUSE_WITCHCRAFT:
        return _execute_accuse(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.SPREAD_RUMOR:
        return _execute_spread_rumor(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.COUNTER_ACCUSE:
        return _execute_counter_accuse(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.INSULT_PUBLICLY:
        return _execute_insult(actor, target, state, rng, verbose)
    
    # === SOCIAL ACTIONS ===
    
    elif action_type == ActionType.FORM_ALLIANCE:
        return _execute_form_alliance(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.BREAK_ALLIANCE:
        return _execute_break_alliance(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.DEFEND_PERSON:
        return _execute_defend(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.SOCIALIZE:
        return _execute_socialize(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.RECONCILE_WITH:
        return _execute_reconcile(actor, target, state, rng, verbose)
    
    # === INFORMATION ACTIONS ===
    
    elif action_type == ActionType.SPY_ON:
        return _execute_spy(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.GATHER_EVIDENCE:
        return _execute_gather_evidence(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.SHARE_GOSSIP:
        return _execute_share_gossip(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.SEEK_ADVICE:
        return _execute_seek_advice(actor, target, state, rng, verbose)
    
    # === ECONOMIC ACTIONS ===
    
    elif action_type == ActionType.SHARE_RESOURCES:
        return _execute_share_resources(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.REQUEST_HELP:
        return _execute_request_help(actor, target, state, rng, verbose)
    
    # === INSTITUTIONAL ACTIONS ===
    
    elif action_type == ActionType.ATTEND_CHURCH:
        return _execute_attend_church(actor, state, rng, verbose)
    
    elif action_type == ActionType.FAKE_PIETY:
        return _execute_fake_piety(actor, state, rng, verbose)
    
    elif action_type == ActionType.SEEK_PROTECTION:
        return _execute_seek_protection(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.APPEAL_TO_AUTHORITY:
        return _execute_appeal_authority(actor, state, rng, verbose)
    
    # === ROMANTIC ACTIONS ===
    
    elif action_type == ActionType.COURT_PERSON:
        return _execute_court(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.PROPOSE_MARRIAGE:
        return _execute_propose_marriage(actor, target, state, rng, verbose)
    
    # === TRIAL ACTIONS ===
    
    elif action_type == ActionType.TESTIFY_AGAINST:
        return _execute_testify_against(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.TESTIFY_FOR:
        return _execute_testify_for(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.CONFESS:
        return _execute_confess(actor, state, rng, verbose)
    
    # === SELF-PRESERVATION ===
    
    elif action_type == ActionType.HIDE_ACTIVITY:
        return _execute_hide_activity(actor, state, rng, verbose)
    
    elif action_type == ActionType.AVOID_PERSON:
        return _execute_avoid_person(actor, target, state, rng, verbose)
    
    elif action_type == ActionType.FLEE_VILLAGE:
        return _execute_flee(actor, state, rng, verbose)
    
    elif action_type == ActionType.PASS:
        return True  # Doing nothing always succeeds
    
    return False


# === AGGRESSIVE ACTION IMPLEMENTATIONS ===

def _execute_accuse(actor: Villager, target: Villager, state: VillageState, 
                    rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    # Record accusation
    actor.accusations_made.append(target.id)
    target.accusations_received.append(actor.id)
    target.times_accused_total += 1
    target.times_accused_recent += 1
    target.is_accused_currently = True
    target.days_since_last_accusation = 0
    
    state.recent_accusations.append((state.timestep, actor.id, target.id))
    state.trial_queue.append(target.id)
    
    # Village effects
    state.panic_level = min(1.0, state.panic_level + 0.05)
    state.rumor_saturation = min(1.0, state.rumor_saturation + 0.1)
    
    # Target emotional impact
    target.emotional_state.fear = min(1.0, target.emotional_state.fear + 0.4)
    target.stress = min(1.0, target.stress + 0.3)
    target.reputation = max(0.0, target.reputation - 0.2)
    target.emotional_state.hatreds[actor.id] = min(1.0, 
        target.emotional_state.hatreds.get(actor.id, 0) + 0.5)
    
    # If target is a patron, their dependents are alarmed
    if target.dependents:
        for dep_id in target.dependents:
            dep = state.villagers.get(dep_id)
            if dep and dep.is_alive:
                dep.emotional_state.fear = min(1.0, dep.emotional_state.fear + 0.2)
                dep.stress = min(1.0, dep.stress + 0.1)
    
    # Accuser may feel guilt (if agreeable) or satisfaction (if not)
    if actor.personality.agreeableness > 0.6:
        actor.emotional_state.shame = min(1.0, actor.emotional_state.shame + 0.1)
    else:
        actor.emotional_state.joy = min(1.0, actor.emotional_state.joy + 0.1)
    
    # Village-wide trauma from witnessing accusation
    for v in state.villagers.values():
        if v.is_alive and v.id != actor.id and v.id != target.id:
            v.witnessed_executions += 0  # Just accusation, not execution yet
            v.emotional_state.fear = min(1.0, v.emotional_state.fear + 0.02)
    
    if verbose:
        print(f"  🔥 {actor.name} ACCUSES {target.name} of witchcraft!")
    
    return True


def _execute_spread_rumor(actor: Villager, target: Villager, state: VillageState,
                          rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    # Damage target's reputation
    target.reputation = max(0.0, target.reputation - 0.05)
    target.conformity_score = max(0.0, target.conformity_score - 0.02)
    
    state.rumor_saturation = min(1.0, state.rumor_saturation + 0.02)
    
    # Increase community suspicion of target
    for v in state.villagers.values():
        if v.id != actor.id and v.id != target.id and v.is_alive:
            current = v.emotional_state.suspicions.get(target.id, 0)
            v.emotional_state.suspicions[target.id] = min(1.0, current + 0.03)
    
    return True


def _execute_counter_accuse(actor: Villager, target: Villager, state: VillageState,
                            rng: np.random.RandomState, verbose: bool) -> bool:
    if not target or not actor.is_accused_currently:
        return False
    
    # Essentially accuse the accuser back
    actor.accusations_made.append(target.id)
    target.accusations_received.append(actor.id)
    target.times_accused_total += 1
    target.times_accused_recent += 1
    target.is_accused_currently = True
    
    state.recent_accusations.append((state.timestep, actor.id, target.id))
    state.trial_queue.append(target.id)
    
    # High drama increases panic
    state.panic_level = min(1.0, state.panic_level + 0.08)
    
    # Mutual hatred escalates
    target.emotional_state.hatreds[actor.id] = min(1.0,
        target.emotional_state.hatreds.get(actor.id, 0) + 0.4)
    actor.emotional_state.hatreds[target.id] = min(1.0,
        actor.emotional_state.hatreds.get(target.id, 0) + 0.3)
    
    if verbose:
        print(f"  ⚔️ {actor.name} COUNTER-ACCUSES {target.name}!")
    
    return True


def _execute_insult(actor: Villager, target: Villager, state: VillageState,
                    rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    # Target effects
    target.emotional_state.anger = min(1.0, target.emotional_state.anger + 0.3)
    target.emotional_state.shame = min(1.0, target.emotional_state.shame + 0.15)
    target.reputation = max(0.0, target.reputation - 0.02)
    
    # Relationship damage
    target.emotional_state.hatreds[actor.id] = min(1.0,
        target.emotional_state.hatreds.get(actor.id, 0) + 0.2)
    target.emotional_state.loyalties[actor.id] = max(0.0,
        target.emotional_state.loyalties.get(actor.id, 0) - 0.2)
    
    # Actor might face backlash if target is high status
    if target.social_status > actor.social_status + 0.2:
        actor.reputation = max(0.0, actor.reputation - 0.05)
    
    return True


# === SOCIAL ACTION IMPLEMENTATIONS ===

def _execute_form_alliance(actor: Villager, target: Villager, state: VillageState,
                           rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    # Record this interaction for diminishing returns
    record_alliance_interaction(actor, target.id)
    
    # Check if target accepts (based on their disposition toward actor)
    target_disposition = (
        target.emotional_state.loyalties.get(actor.id, 0.3) -
        target.emotional_state.hatreds.get(actor.id, 0) +
        target.personality.agreeableness * 0.2
    )
    
    acceptance_prob = np.clip(target_disposition + 0.3, 0.1, 0.9)
    
    if rng.random() < acceptance_prob:
        # Alliance formed - mutual loyalty boost
        actor.emotional_state.loyalties[target.id] = min(1.0,
            actor.emotional_state.loyalties.get(target.id, 0) + 0.3)
        target.emotional_state.loyalties[actor.id] = min(1.0,
            target.emotional_state.loyalties.get(actor.id, 0) + 0.3)
        
        # Add or strengthen friendship relationship
        if (actor.id, target.id) not in state.relationships:
            rel = Relationship(source=actor.id, target=target.id)
            rel.add_relationship_type(RelationshipType.FRIENDSHIP, strength=0.5)
            state.relationships[(actor.id, target.id)] = rel
            state.relationships[(target.id, actor.id)] = rel
        else:
            state.relationships[(actor.id, target.id)].add_relationship_type(
                RelationshipType.FRIENDSHIP, strength=0.5)
        
        if verbose:
            print(f"  🤝 {actor.name} forms an alliance with {target.name}")
        return True
    else:
        # Rejection - slight shame for actor
        actor.emotional_state.shame = min(1.0, actor.emotional_state.shame + 0.05)
        return False


def _execute_break_alliance(actor: Villager, target: Villager, state: VillageState,
                            rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    # Reduce mutual loyalty
    actor.emotional_state.loyalties[target.id] = max(0.0,
        actor.emotional_state.loyalties.get(target.id, 0) - 0.4)
    
    # Target feels betrayed
    target.emotional_state.loyalties[actor.id] = max(0.0,
        target.emotional_state.loyalties.get(actor.id, 0) - 0.5)
    target.emotional_state.hatreds[actor.id] = min(1.0,
        target.emotional_state.hatreds.get(actor.id, 0) + 0.3)
    target.emotional_state.anger = min(1.0, target.emotional_state.anger + 0.2)
    
    if verbose:
        print(f"  💔 {actor.name} breaks alliance with {target.name}")
    
    return True


def _execute_defend(actor: Villager, target: Villager, state: VillageState,
                    rng: np.random.RandomState, verbose: bool) -> bool:
    if not target or not target.is_accused_currently:
        return False
    
    # Risk to defender - guilt by association
    actor.reputation = max(0.0, actor.reputation - 0.05)
    actor.emotional_state.suspicions[actor.id] = min(1.0,
        sum(v.emotional_state.suspicions.get(actor.id, 0) for v in state.villagers.values()) / len(state.villagers) + 0.02)
    
    # Target benefits
    target.reputation = min(1.0, target.reputation + 0.03)
    target.emotional_state.hope = min(1.0, target.emotional_state.hope + 0.15)
    
    # Strengthen bond
    target.emotional_state.loyalties[actor.id] = min(1.0,
        target.emotional_state.loyalties.get(actor.id, 0) + 0.3)
    
    # Store for trial calculation
    if 'defenders' not in state.__dict__:
        state.__dict__['defenders'] = {}
    if target.id not in state.__dict__['defenders']:
        state.__dict__['defenders'][target.id] = []
    state.__dict__['defenders'][target.id].append(actor.id)
    
    if verbose:
        print(f"  🛡️ {actor.name} defends {target.name}")
    
    return True


def _execute_socialize(actor: Villager, target: Villager, state: VillageState,
                       rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    # Mild mutual benefits
    actor.emotional_state.joy = min(1.0, actor.emotional_state.joy + 0.05)
    target.emotional_state.joy = min(1.0, target.emotional_state.joy + 0.05)
    
    actor.stress = max(0.0, actor.stress - 0.02)
    target.stress = max(0.0, target.stress - 0.02)
    
    # Small loyalty boost
    actor.emotional_state.loyalties[target.id] = min(1.0,
        actor.emotional_state.loyalties.get(target.id, 0) + 0.05)
    target.emotional_state.loyalties[actor.id] = min(1.0,
        target.emotional_state.loyalties.get(actor.id, 0) + 0.05)
    
    return True


def _execute_reconcile(actor: Villager, target: Villager, state: VillageState,
                       rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    # Success depends on target's hatred level and actor's social skills
    hatred = target.emotional_state.hatreds.get(actor.id, 0)
    success_prob = (1 - hatred * 0.7) * (0.3 + actor.personality.agreeableness * 0.4)
    
    if rng.random() < success_prob:
        # Reduce mutual hatred
        target.emotional_state.hatreds[actor.id] = max(0.0, hatred - 0.3)
        actor.emotional_state.hatreds[target.id] = max(0.0,
            actor.emotional_state.hatreds.get(target.id, 0) - 0.2)
        
        # Small loyalty gain
        target.emotional_state.loyalties[actor.id] = min(1.0,
            target.emotional_state.loyalties.get(actor.id, 0) + 0.1)
        
        if verbose:
            print(f"  🕊️ {actor.name} reconciles with {target.name}")
        return True
    
    return False


# === INFORMATION ACTION IMPLEMENTATIONS ===

def _execute_spy(actor: Villager, target: Villager, state: VillageState,
                 rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    # Chance of discovery based on conscientiousness and luck
    discovery_prob = 0.2 - actor.personality.conscientiousness * 0.1
    
    if rng.random() < discovery_prob:
        # Caught! Reputation damage and target hatred
        actor.reputation = max(0.0, actor.reputation - 0.1)
        target.emotional_state.hatreds[actor.id] = min(1.0,
            target.emotional_state.hatreds.get(actor.id, 0) + 0.3)
        return False
    
    # Success - gain suspicion information
    actor.emotional_state.suspicions[target.id] = min(1.0,
        actor.emotional_state.suspicions.get(target.id, 0) + 0.2)
    
    # Store "evidence" for future accusations
    if 'gathered_evidence' not in actor.__dict__:
        actor.__dict__['gathered_evidence'] = {}
    actor.__dict__['gathered_evidence'][target.id] = \
        actor.__dict__['gathered_evidence'].get(target.id, 0) + 0.15
    
    return True


def _execute_gather_evidence(actor: Villager, target: Villager, state: VillageState,
                             rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    # Similar to spy but more methodical, higher yield
    if 'gathered_evidence' not in actor.__dict__:
        actor.__dict__['gathered_evidence'] = {}
    actor.__dict__['gathered_evidence'][target.id] = \
        actor.__dict__['gathered_evidence'].get(target.id, 0) + 0.25
    
    return True


def _execute_share_gossip(actor: Villager, target: Villager, state: VillageState,
                          rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    # Social bonding through information sharing
    actor.emotional_state.loyalties[target.id] = min(1.0,
        actor.emotional_state.loyalties.get(target.id, 0) + 0.05)
    target.emotional_state.loyalties[actor.id] = min(1.0,
        target.emotional_state.loyalties.get(actor.id, 0) + 0.05)
    
    # Spread suspicions
    if actor.emotional_state.suspicions:
        # Share actor's suspicions with target
        for suspect_id, level in actor.emotional_state.suspicions.items():
            if suspect_id != target.id and level > 0.2:
                current = target.emotional_state.suspicions.get(suspect_id, 0)
                target.emotional_state.suspicions[suspect_id] = min(1.0, current + level * 0.3)
    
    return True


def _execute_seek_advice(actor: Villager, target: Villager, state: VillageState,
                         rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    # Actor gains small stress reduction, target gains small status boost
    actor.stress = max(0.0, actor.stress - 0.05)
    actor.emotional_state.loyalties[target.id] = min(1.0,
        actor.emotional_state.loyalties.get(target.id, 0) + 0.05)
    
    # Target may share their suspicions
    if target.emotional_state.suspicions and rng.random() < 0.5:
        for suspect_id, level in target.emotional_state.suspicions.items():
            if level > 0.3:
                current = actor.emotional_state.suspicions.get(suspect_id, 0)
                actor.emotional_state.suspicions[suspect_id] = min(1.0, current + level * 0.2)
    
    return True


# === ECONOMIC ACTION IMPLEMENTATIONS ===

def _execute_share_resources(actor: Villager, target: Villager, state: VillageState,
                             rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    # Record interaction for diminishing returns
    record_alliance_interaction(actor, target.id)
    
    # Transfer wealth and build loyalty
    transfer = min(0.1, actor.wealth * 0.2)
    actor.wealth = max(0.0, actor.wealth - transfer)
    target.wealth = min(1.0, target.wealth + transfer)
    
    target.emotional_state.loyalties[actor.id] = min(1.0,
        target.emotional_state.loyalties.get(actor.id, 0) + 0.2)
    target.emotional_state.joy = min(1.0, target.emotional_state.joy + 0.1)
    
    if verbose:
        print(f"  💰 {actor.name} shares resources with {target.name}")
    
    return True


def _execute_request_help(actor: Villager, target: Villager, state: VillageState,
                          rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    # Success based on target's wealth, agreeableness, and relationship
    loyalty = target.emotional_state.loyalties.get(actor.id, 0)
    help_prob = (
        target.personality.agreeableness * 0.3 +
        loyalty * 0.4 +
        target.wealth * 0.2
    )
    
    if rng.random() < help_prob:
        transfer = min(0.05, target.wealth * 0.1)
        target.wealth = max(0.0, target.wealth - transfer)
        actor.wealth = min(1.0, actor.wealth + transfer)
        
        actor.emotional_state.loyalties[target.id] = min(1.0,
            actor.emotional_state.loyalties.get(target.id, 0) + 0.1)
        return True
    
    return False


# === INSTITUTIONAL ACTION IMPLEMENTATIONS ===

def _execute_attend_church(actor: Villager, state: VillageState,
                           rng: np.random.RandomState, verbose: bool) -> bool:
    # Increase church attendance and conformity
    actor.church_attendance = min(1.0, actor.church_attendance + 0.1)
    actor.conformity_score = min(1.0, actor.conformity_score + 0.05)
    
    # Slight stress reduction (spiritual comfort)
    actor.stress = max(0.0, actor.stress - 0.03)
    
    return True


def _execute_fake_piety(actor: Villager, state: VillageState,
                        rng: np.random.RandomState, verbose: bool) -> bool:
    # More dramatic conformity boost but with risk
    discovery_risk = 0.1 + (1 - actor.personality.machiavellianism) * 0.1
    
    if rng.random() < discovery_risk:
        # Caught being insincere - reputation hit
        actor.reputation = max(0.0, actor.reputation - 0.1)
        actor.conformity_score = max(0.0, actor.conformity_score - 0.1)
        return False
    
    actor.church_attendance = min(1.0, actor.church_attendance + 0.15)
    actor.conformity_score = min(1.0, actor.conformity_score + 0.1)
    
    return True


def _execute_seek_protection(actor: Villager, target: Villager, state: VillageState,
                             rng: np.random.RandomState, verbose: bool) -> bool:
    if not target or target.social_status <= actor.social_status:
        return False
    
    # Record interaction for diminishing returns
    record_alliance_interaction(actor, target.id)
    
    # Already has a patron?
    if actor.patron_id is not None:
        return False
    
    # Ask high-status individual for patronage
    acceptance_prob = (
        target.personality.agreeableness * 0.3 +
        target.emotional_state.loyalties.get(actor.id, 0) * 0.3 +
        (1 - actor.social_status) * 0.2  # They prefer helping the truly needy
    )
    
    # Patrons with many dependents less likely to accept more
    current_dependents = len(target.dependents)
    acceptance_prob -= current_dependents * 0.05
    
    # Wealthy patrons more likely to accept
    acceptance_prob += target.wealth * 0.2
    
    if rng.random() < np.clip(acceptance_prob, 0.1, 0.8):
        # Protection granted - establish patron-dependent relationship
        actor.patron_id = target.id
        target.dependents.append(actor.id)
        
        # Add patronage relationship
        if (target.id, actor.id) not in state.relationships:
            rel = Relationship(source=target.id, target=actor.id)
            rel.add_relationship_type(RelationshipType.PATRONAGE, strength=0.5)
            state.relationships[(target.id, actor.id)] = rel
        else:
            state.relationships[(target.id, actor.id)].add_relationship_type(
                RelationshipType.PATRONAGE, strength=0.5)
        
        actor.emotional_state.fear = max(0.0, actor.emotional_state.fear - 0.1)
        actor.emotional_state.loyalties[target.id] = min(1.0,
            actor.emotional_state.loyalties.get(target.id, 0) + 0.3)
        
        if verbose:
            print(f"  👑 {target.name} takes {actor.name} under their protection")
        return True
    
    return False


def _execute_appeal_authority(actor: Villager, state: VillageState,
                              rng: np.random.RandomState, verbose: bool) -> bool:
    # Generic appeal to institutional authority
    if state.trust_in_authority > 0.5:
        actor.emotional_state.fear = max(0.0, actor.emotional_state.fear - 0.05)
        actor.stress = max(0.0, actor.stress - 0.05)
        
        # If accused, slight boost to trial outcome
        if actor.is_accused_currently:
            actor.reputation = min(1.0, actor.reputation + 0.02)
        return True
    
    return False


# === ROMANTIC ACTION IMPLEMENTATIONS ===

def _execute_court(actor: Villager, target: Villager, state: VillageState,
                   rng: np.random.RandomState, verbose: bool) -> bool:
    if not target or actor.gender == target.gender:
        return False
    
    # Courting success based on various factors
    success_factors = (
        actor.beauty * 0.25 +
        actor.social_status * 0.2 +
        actor.wealth * 0.15 +
        target.personality.lust * 0.2 +
        target.emotional_state.romantic_attractions.get(actor.id, 0) * 0.3 -
        target.emotional_state.hatreds.get(actor.id, 0) * 0.5
    )
    
    if target.marital_status == MaritalStatus.MARRIED:
        success_factors -= 0.3
    
    if rng.random() < np.clip(success_factors, 0.1, 0.8):
        # Successful courtship - mutual attraction increases
        target.emotional_state.romantic_attractions[actor.id] = min(1.0,
            target.emotional_state.romantic_attractions.get(actor.id, 0) + 0.2)
        actor.emotional_state.romantic_attractions[target.id] = min(1.0,
            actor.emotional_state.romantic_attractions.get(target.id, 0) + 0.15)
        
        target.emotional_state.joy = min(1.0, target.emotional_state.joy + 0.1)
        actor.emotional_state.joy = min(1.0, actor.emotional_state.joy + 0.1)
        
        if verbose:
            print(f"  💕 {actor.name} courts {target.name}")
        return True
    else:
        actor.emotional_state.shame = min(1.0, actor.emotional_state.shame + 0.05)
        return False


def _execute_propose_marriage(actor: Villager, target: Villager, state: VillageState,
                              rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    if actor.marital_status == MaritalStatus.MARRIED or target.marital_status == MaritalStatus.MARRIED:
        return False
    
    # Requires high mutual attraction
    actor_attraction = actor.emotional_state.romantic_attractions.get(target.id, 0)
    target_attraction = target.emotional_state.romantic_attractions.get(actor.id, 0)
    
    if actor_attraction < 0.5 or target_attraction < 0.4:
        actor.emotional_state.shame = min(1.0, actor.emotional_state.shame + 0.15)
        return False
    
    acceptance_prob = target_attraction * 0.6 + target.emotional_state.loyalties.get(actor.id, 0) * 0.3
    
    if rng.random() < acceptance_prob:
        # Marriage!
        actor.marital_status = MaritalStatus.MARRIED
        target.marital_status = MaritalStatus.MARRIED
        
        # Add marriage relationship
        rel = Relationship(source=actor.id, target=target.id)
        rel.add_relationship_type(RelationshipType.MARRIAGE, strength=0.9)
        rel.add_relationship_type(RelationshipType.FAMILY, strength=0.8)
        state.relationships[(actor.id, target.id)] = rel
        state.relationships[(target.id, actor.id)] = rel
        
        # Emotional effects
        actor.emotional_state.joy = min(1.0, actor.emotional_state.joy + 0.3)
        target.emotional_state.joy = min(1.0, target.emotional_state.joy + 0.3)
        actor.lives_alone = False
        target.lives_alone = False
        
        if verbose:
            print(f"  💒 {actor.name} and {target.name} are married!")
        return True
    else:
        actor.emotional_state.shame = min(1.0, actor.emotional_state.shame + 0.2)
        actor.emotional_state.grief = min(1.0, actor.emotional_state.grief + 0.1)
        return False


# === TRIAL ACTION IMPLEMENTATIONS ===

def _execute_testify_against(actor: Villager, target: Villager, state: VillageState,
                             rng: np.random.RandomState, verbose: bool) -> bool:
    if not target or not target.is_on_trial:
        return False
    
    # Store testimony for trial calculation
    if 'testimony_against' not in state.__dict__:
        state.__dict__['testimony_against'] = {}
    if target.id not in state.__dict__['testimony_against']:
        state.__dict__['testimony_against'][target.id] = []
    state.__dict__['testimony_against'][target.id].append(actor.id)
    
    # Target knows and hates
    target.emotional_state.hatreds[actor.id] = min(1.0,
        target.emotional_state.hatreds.get(actor.id, 0) + 0.4)
    
    if verbose:
        print(f"  ⚖️ {actor.name} testifies AGAINST {target.name}")
    
    return True


def _execute_testify_for(actor: Villager, target: Villager, state: VillageState,
                         rng: np.random.RandomState, verbose: bool) -> bool:
    if not target or not target.is_on_trial:
        return False
    
    # Risk to defender
    actor.reputation = max(0.0, actor.reputation - 0.03)
    
    # Store testimony
    if 'testimony_for' not in state.__dict__:
        state.__dict__['testimony_for'] = {}
    if target.id not in state.__dict__['testimony_for']:
        state.__dict__['testimony_for'][target.id] = []
    state.__dict__['testimony_for'][target.id].append(actor.id)
    
    # Target grateful
    target.emotional_state.loyalties[actor.id] = min(1.0,
        target.emotional_state.loyalties.get(actor.id, 0) + 0.3)
    
    if verbose:
        print(f"  ⚖️ {actor.name} testifies FOR {target.name}")
    
    return True


def _execute_confess(actor: Villager, state: VillageState,
                     rng: np.random.RandomState, verbose: bool) -> bool:
    if not actor.is_accused_currently:
        return False
    
    # Confession - historically sometimes meant quicker death or mercy
    actor.is_accused_currently = False
    actor.is_alive = False  # Executed
    state.recent_deaths.append((state.timestep, actor.id))
    
    if verbose:
        print(f"  📜 {actor.name} CONFESSES and is executed")
    
    return True


# === SELF-PRESERVATION IMPLEMENTATIONS ===

def _execute_hide_activity(actor: Villager, state: VillageState,
                           rng: np.random.RandomState, verbose: bool) -> bool:
    # Reduce visibility - makes accusations slightly less likely
    actor.conformity_score = min(1.0, actor.conformity_score + 0.03)
    
    # But suspicion might increase if noticed
    if rng.random() < 0.1:
        for v in state.villagers.values():
            if v.id != actor.id and v.is_alive:
                current = v.emotional_state.suspicions.get(actor.id, 0)
                v.emotional_state.suspicions[actor.id] = min(1.0, current + 0.02)
    
    return True


def _execute_avoid_person(actor: Villager, target: Villager, state: VillageState,
                          rng: np.random.RandomState, verbose: bool) -> bool:
    if not target:
        return False
    
    # Reduce interaction frequency (modeled as reduced relationship effects)
    actor.emotional_state.loyalties[target.id] = max(0.0,
        actor.emotional_state.loyalties.get(target.id, 0) - 0.05)
    
    # Slight stress reduction from avoiding threat
    if target.is_accused_currently or target.id in actor.accusations_received:
        actor.stress = max(0.0, actor.stress - 0.05)
    
    return True


def _execute_flee(actor: Villager, state: VillageState,
                  rng: np.random.RandomState, verbose: bool) -> bool:
    # Desperate escape - removes from simulation
    actor.is_alive = False  # Not dead, but gone
    
    # Flee might fail (caught trying to escape)
    if rng.random() < 0.3:
        actor.is_imprisoned = True
        actor.is_alive = True
        if verbose:
            print(f"  🚨 {actor.name} tried to flee but was caught!")
        return False
    
    if verbose:
        print(f"  🏃 {actor.name} FLEES the village!")
    
    return True
