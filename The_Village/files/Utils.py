"""Utility functions for the witch trial simulation - where the math of misery lives"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from Villager import Villager, Gender, MaritalStatus, Occupation
from Village import VillageState
from Relationships import Relationship, RelationshipType


# ============================================================================
# VULNERABILITY AND CREDIBILITY CALCULATIONS
# ============================================================================

def get_vulnerability(villager: Villager, state: VillageState) -> float:
    """Calculate how vulnerable a villager is to accusation.
    
    Higher values = more likely to be accused.
    This is where the historical injustice gets quantified.
    """
    v = 0.0

    # Gender (the depressing coefficient)
    v += 0.3 if villager.gender == Gender.FEMALE else 0.0

    # Age effects (U-shaped - young and old are vulnerable)
    age_factor = abs(villager.age - 40) / 40
    v += age_factor * 0.2

    # Marital status
    if villager.marital_status == MaritalStatus.WIDOWED:
        v += 0.3
    elif villager.marital_status == MaritalStatus.SINGLE and villager.age > 30:
        v += 0.2

    # Social status (inverse relationship - the powerful are protected)
    v += (1 - villager.social_status) * 0.4

    # Conformity (inverse - weirdos get targeted)
    v += (1 - villager.conformity_score) * 0.3
    
    # Conformity pressure amplifies non-conformity vulnerability
    if state.conformity_pressure > 0.6:
        v += (1 - villager.conformity_score) * state.conformity_pressure * 0.2

    # Network isolation
    connections = sum(1 for (s, t) in state.relationships if s == villager.id or t == villager.id)
    avg_connections = len(state.relationships) / max(len(state.villagers), 1) * 2
    isolation = max(0, (avg_connections - connections) / max(avg_connections, 1))
    v += isolation * 0.3

    # Suspicious attributes
    if villager.owns_cat:
        v += 0.2
    if villager.lives_alone:
        v += 0.15
    if villager.has_suspicious_marks:
        v += 0.1

    # Previous accusations (feedback loop!)
    v += min(villager.times_accused_recent * 0.2, 0.5)

    # Occupation risk
    if villager.occupation in [Occupation.HEALER, Occupation.MIDWIFE]:
        v += 0.25

    # Low church attendance
    v += (1 - villager.church_attendance) * 0.15
    
    # Stressor-specific suspicion modifiers
    v += get_stressor_suspicion_modifier(villager, state)
    
    # PATRON PROTECTION - reduces vulnerability
    protection = get_protection_for_dependent(villager, state)
    v -= protection * 0.4  # Protection reduces vulnerability
    
    # Historical violence makes everyone slightly more vulnerable
    v += state.historical_violence * 0.1

    return np.clip(v, 0, 1)


def get_accusation_credibility(accuser: Villager, accused: Villager, state: VillageState) -> float:
    """How believable is this accusation?
    
    Combines accuser's social standing with accused's vulnerability.
    """
    credibility = 0.5

    # Accuser's social status helps
    credibility += accuser.social_status * 0.3

    # Accuser's gender (men more believed, sadly accurate for the era)
    if accuser.gender == Gender.MALE:
        credibility += 0.2

    # Accused's vulnerability makes accusations more believable
    credibility += get_vulnerability(accused, state) * 0.3

    # Rivalry makes it less credible (obvious motive)
    if (accuser.id, accused.id) in state.relationships:
        rel = state.relationships[(accuser.id, accused.id)]
        if RelationshipType.RIVALRY in rel.relationship_types:
            credibility -= 0.1

    # Diminishing returns on serial accusers
    num_previous = len(accuser.accusations_made)
    credibility -= min(num_previous * 0.05, 0.3)

    # Panic level makes all accusations more credible
    credibility += state.panic_level * 0.2

    return np.clip(credibility, 0, 1)


# ============================================================================
# INTERACTION VECTORS - How villagers interact based on personality
# ============================================================================

def get_base_interaction_vector(villager: Villager) -> np.ndarray:
    """
    Personality determines natural interaction style.
    Returns unnormalized weights for [chill, gossip, formal, flirt, insult, respect]
    """
    base = np.zeros(6)

    # Chill/Friendly - driven by agreeableness, extraversion
    base[0] = (
        villager.personality.agreeableness * 0.6 +
        villager.personality.extraversion * 0.4
    )

    # Gossip - driven by extraversion, openness, low conscientiousness
    base[1] = (
        villager.personality.extraversion * 0.4 +
        villager.personality.openness * 0.3 +
        (1 - villager.personality.conscientiousness) * 0.3
    )

    # Neutral/Formal - driven by conscientiousness, low extraversion
    base[2] = (
        villager.personality.conscientiousness * 0.5 +
        (1 - villager.personality.extraversion) * 0.3 +
        (1 - villager.personality.agreeableness) * 0.2
    )

    # Flirt - driven by extraversion, lust, low conscientiousness
    base[3] = (
        villager.personality.extraversion * 0.3 +
        villager.personality.lust * 0.4 +
        (1 - villager.personality.conscientiousness) * 0.2 +
        villager.emotional_state.desire * 0.1
    )

    # Insult - driven by low agreeableness, wrath, psychopathy
    base[4] = (
        (1 - villager.personality.agreeableness) * 0.3 +
        villager.personality.wrath * 0.3 +
        villager.personality.psychopathy * 0.2 +
        villager.emotional_state.anger * 0.2
    )

    # Respect - driven by conscientiousness, low narcissism
    base[5] = (
        villager.personality.conscientiousness * 0.4 +
        (1 - villager.personality.narcissism) * 0.3 +
        villager.personality.agreeableness * 0.3
    )

    return base


def compute_relational_modifiers(
    actor: Villager,
    target: Villager,
    relationship_state: float  # -1 to 1, from actor's perspective
) -> np.ndarray:
    """
    How you interact depends on who they are and how you feel about them.
    Returns multiplicative modifiers for each interaction dimension.
    """
    modifiers = np.ones(6)

    # Status differential
    status_diff = target.social_status - actor.social_status

    # CHILL: More likely with equals, less with high-status targets
    modifiers[0] *= (1 - abs(status_diff) * 0.5)

    # GOSSIP: More with equals, suppressed with high-status targets
    modifiers[1] *= (1 - status_diff * 0.7) if status_diff > 0 else 1.0

    # FORMAL: Increases with status differential
    modifiers[2] *= (1 + abs(status_diff) * 0.8)

    # FLIRT: Suppressed by large status differential (unless narcissist)
    if actor.gender != target.gender:
        status_penalty = abs(status_diff) * (1 - actor.personality.narcissism)
        modifiers[3] *= max(0.1, 1 - status_penalty)
    else:
        modifiers[3] = 0  # Same-sex flirting disabled per original design

    # INSULT: Suppressed toward high-status unless angry
    if status_diff > 0:
        safety_factor = (
            actor.personality.psychopathy * 0.3 +
            actor.emotional_state.anger * 0.4
        )
        modifiers[4] *= safety_factor
    else:
        cruelty_factor = (
            (1 - actor.personality.agreeableness) * 0.5 +
            actor.personality.narcissism * 0.3
        )
        modifiers[4] *= (1 + cruelty_factor)

    # RESPECT: High for high-status targets
    base_respect = max(0, status_diff)
    narcissism_penalty = actor.personality.narcissism * 0.7
    resentment_penalty = actor.emotional_state.resentment * 0.5
    modifiers[5] *= (base_respect * (1 - narcissism_penalty - resentment_penalty) + 0.2)

    # Relationship quality affects everything
    if relationship_state > 0.5:
        modifiers[0] *= 1.5  # More chill
        modifiers[2] *= 0.5  # Less formal
        modifiers[4] *= 0.2  # Less insult
    elif relationship_state < -0.5:
        modifiers[0] *= 0.3  # Less chill
        modifiers[2] *= 1.3  # More formal
        modifiers[4] *= 2.0  # More insult

    return modifiers


def compute_contextual_modifiers(
    village_state: VillageState,
    actor: Villager,
    target: Villager
) -> np.ndarray:
    """
    Village-level context affects interaction style.
    High panic = less chill, more formal/fearful.
    """
    modifiers = np.ones(6)

    # High panic suppresses chill, increases formality
    modifiers[0] *= (1 - village_state.panic_level * 0.6)
    modifiers[2] *= (1 + village_state.panic_level * 0.5)

    # High panic increases gossip (spreading fear/rumors)
    modifiers[1] *= (1 + village_state.panic_level * 0.8)

    # Low institutional trust makes people more guarded
    trust_factor = village_state.trust_in_authority
    modifiers[2] *= (1 + (1 - trust_factor) * 0.3)
    modifiers[5] *= trust_factor

    # If target is currently accused, interaction changes dramatically
    if target.is_accused_currently:
        modifiers[0] *= 0.2  # Avoid being friendly (guilt by association)
        modifiers[2] *= 1.5  # More formal/distant
        modifiers[4] *= 1.8  # More acceptable to insult
        modifiers[5] *= 0.3  # Less respect

    # If actor is currently accused, they're desperate/defensive
    if actor.is_accused_currently:
        modifiers[0] *= 1.5  # Try to be friendly (seeking allies)
        modifiers[5] *= 1.8  # Show excessive respect (appeasement)
        modifiers[4] *= 0.1  # Avoid insults (can't afford more enemies)

    return modifiers


def compute_interaction_vector(
    actor: Villager,
    target: Villager,
    village_state: VillageState
) -> np.ndarray:
    """
    Combines all factors to produce final interaction vector.
    Returns normalized 6D vector: [chill, gossip, formal, flirt, insult, respect]
    """
    # Get actor's base interaction tendencies
    base_vector = get_base_interaction_vector(actor)

    # Get relationship from actor's perspective
    relationship = actor.emotional_state.loyalties.get(target.id, 0) - \
                   actor.emotional_state.hatreds.get(target.id, 0)

    # Apply relational modifiers
    relational_mods = compute_relational_modifiers(actor, target, relationship)

    # Apply contextual modifiers
    contextual_mods = compute_contextual_modifiers(village_state, actor, target)

    # Combine: base × relational × contextual
    interaction_vector = base_vector * relational_mods * contextual_mods

    # Normalize to sum to 1 (proper probability distribution)
    total = interaction_vector.sum()
    if total > 0:
        interaction_vector = interaction_vector / total

    return interaction_vector


# ============================================================================
# STATE UPDATES AFTER INTERACTION
# ============================================================================

def apply_interaction_effects(
    actor: Villager,
    target: Villager,
    interaction_vector: np.ndarray,
    village_state: VillageState
):
    """
    Update villager states based on interaction vector.
    vector = [chill, gossip, formal, flirt, insult, respect]
    """
    chill, gossip, formal, flirt, insult, respect = interaction_vector

    # === Target's emotional updates ===

    # CHILL increases joy, decreases stress slightly
    target.emotional_state.joy = min(1.0, target.emotional_state.joy + chill * 0.1)
    target.stress = max(0, target.stress - chill * 0.05)

    # GOSSIP increases fear/anxiety
    target.emotional_state.fear = min(1.0, target.emotional_state.fear + gossip * 0.05 * village_state.panic_level)
    target.stress = min(1.0, target.stress + gossip * 0.03)

    # FORMAL doesn't affect emotions much
    target.emotional_state.joy = max(0, target.emotional_state.joy - formal * 0.02)

    # FLIRT affects romantic attraction, desire, joy
    if flirt > 0.1:
        target_receptiveness = (
            target.personality.lust * 0.4 +
            target.emotional_state.desire * 0.3 +
            (0.3 if target.marital_status != MaritalStatus.MARRIED else 0.0)
        )
        if target_receptiveness > 0.4:
            current_attraction = target.emotional_state.romantic_attractions.get(actor.id, 0)
            target.emotional_state.romantic_attractions[actor.id] = min(1.0, current_attraction + flirt * 0.2)
            target.emotional_state.joy = min(1.0, target.emotional_state.joy + flirt * 0.1)
            target.emotional_state.desire = min(1.0, target.emotional_state.desire + flirt * 0.15)
        else:
            target.emotional_state.shame = min(1.0, target.emotional_state.shame + flirt * 0.1)

    # INSULT increases anger, shame, decreases joy
    target.emotional_state.anger = min(1.0, target.emotional_state.anger + insult * 0.3)
    target.emotional_state.shame = min(1.0, target.emotional_state.shame + insult * 0.2)
    target.emotional_state.joy = max(0, target.emotional_state.joy - insult * 0.2)
    target.reputation = max(0.0, target.reputation - insult * 0.02)

    # RESPECT decreases shame, reduces stress
    target.emotional_state.shame = max(0, target.emotional_state.shame - respect * 0.1)
    target.stress = max(0, target.stress - respect * 0.05)

    # === Relationship updates ===

    # Chill and respect increase loyalty
    loyalty_change = (chill * 0.1 + respect * 0.15)
    current_loyalty = target.emotional_state.loyalties.get(actor.id, 0)
    target.emotional_state.loyalties[actor.id] = min(1.0, current_loyalty + loyalty_change)

    # Insult increases hatred, decreases loyalty
    hatred_change = insult * 0.2
    current_hatred = target.emotional_state.hatreds.get(actor.id, 0)
    target.emotional_state.hatreds[actor.id] = min(1.0, current_hatred + hatred_change)
    target.emotional_state.loyalties[actor.id] = max(0, target.emotional_state.loyalties.get(actor.id, 0) - insult * 0.15)

    # === Actor's updates (smaller, self-reflective) ===

    # Insulting someone might make high-agreeableness people feel guilt
    if insult > 0.3:
        guilt = insult * actor.personality.agreeableness
        actor.emotional_state.shame = min(1.0, actor.emotional_state.shame + guilt * 0.1)

    # Successful flirting increases actor's joy too
    if flirt > 0.3:
        actor.emotional_state.joy = min(1.0, actor.emotional_state.joy + flirt * 0.08)
        current_attraction = actor.emotional_state.romantic_attractions.get(target.id, 0)
        actor.emotional_state.romantic_attractions[target.id] = min(1.0, current_attraction + flirt * 0.15)

    # Showing respect to high-status individuals might make you feel secure
    if respect > 0.5 and target.social_status > actor.social_status:
        actor.stress = max(0, actor.stress - respect * 0.03)


# ============================================================================
# GROUP INTERACTIONS
# ============================================================================

def compute_group_interaction(
    participants: List[Villager],
    village_state: VillageState
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Compute pairwise interaction vectors for all participants.
    Returns dict mapping (actor_id, target_id) -> interaction_vector
    """
    interactions = {}

    for actor in participants:
        for target in participants:
            if actor.id != target.id:
                vector = compute_interaction_vector(actor, target, village_state)
                interactions[(actor.id, target.id)] = vector

    return interactions


def apply_group_interaction_effects(
    participants: List[Villager],
    interactions: Dict[Tuple[int, int], np.ndarray],
    village_state: VillageState
):
    """
    Apply all pairwise interactions with dampening for group settings.
    """
    dampening_factor = 1.0 / np.sqrt(len(participants))

    for (actor_id, target_id), vector in interactions.items():
        actor = village_state.villagers[actor_id]
        target = village_state.villagers[target_id]

        dampened_vector = vector * dampening_factor
        apply_interaction_effects(actor, target, dampened_vector, village_state)


# ============================================================================
# MEMORY DECAY RATES (for hysteresis)
# ============================================================================

DECAY_RATES = {
    'suspicions': 0.90,           # Fast decay - rumors fade quickly
    'rumors': 0.88,               # Very fast - gossip is ephemeral
    'acquittal_memory': 0.97,     # Medium - "wasn't she accused once?"
    'accusation_stigma': 0.99,    # Slow - accusations stick (was 0.995)
    'trauma': 0.995,              # Slow but recoverable (was 0.998)
    'family_trauma': 0.998,       # Near-permanent - generational wounds (was 0.9995)
    'historical_violence': 0.99,  # Village-level trauma - recoverable over time (was 0.997)
    'hatred': 0.95,               # Hatred fades faster (was 0.97)
    'loyalty': 0.99,              # Loyalty more persistent than hatred
}


# ============================================================================
# EXTERNAL STRESSOR SYSTEM
# ============================================================================

class StressorType:
    FAMINE = "famine"
    WAR_RUMOR = "war_rumor"
    PLAGUE = "plague"
    HARSH_WINTER = "harsh_winter"
    INQUISITOR_VISIT = "inquisitor"


STRESSOR_EFFECTS = {
    StressorType.FAMINE: {
        'resource_scarcity_boost': 0.3,
        'patron_drain_multiplier': 2.0,
        'panic_boost': 0.1,
        'trust_drain': 0.05,
        'duration_range': (20, 40),
        'description': "Famine grips the village. Crops have failed.",
    },
    StressorType.WAR_RUMOR: {
        'panic_boost': 0.15,
        'conformity_boost': 0.2,
        'outsider_suspicion': 0.3,
        'trust_drain': 0.03,
        'duration_range': (10, 25),
        'description': "Rumors of approaching armies spread fear.",
    },
    StressorType.PLAGUE: {
        'random_death_rate': 0.02,  # Per day while active
        'fear_boost': 0.25,
        'healer_suspicion': 0.4,
        'panic_boost': 0.2,
        'duration_range': (15, 35),
        'description': "A mysterious illness sweeps through the village.",
    },
    StressorType.HARSH_WINTER: {
        'resource_scarcity_boost': 0.2,
        'isolation_boost': 0.15,
        'church_attendance_boost': 0.1,
        'duration_range': (30, 60),
        'description': "The harshest winter in memory descends.",
    },
    StressorType.INQUISITOR_VISIT: {
        'conformity_boost': 0.3,
        'accusation_credibility_boost': 0.2,
        'panic_boost': 0.1,
        'fear_boost': 0.15,
        'duration_range': (5, 15),
        'description': "An inquisitor has arrived to investigate.",
    },
}

# Stressor probability weights (some are rarer than others)
STRESSOR_PROBABILITIES = {
    StressorType.FAMINE: 0.25,
    StressorType.WAR_RUMOR: 0.25,
    StressorType.PLAGUE: 0.15,
    StressorType.HARSH_WINTER: 0.25,
    StressorType.INQUISITOR_VISIT: 0.10,
}


def maybe_trigger_stressor(state: VillageState, rng: np.random.RandomState, 
                           base_probability: float = 0.005) -> Optional[str]:
    """
    Potentially trigger an external stressor. Probability increases with
    existing instability (compounding crises).
    """
    # Don't stack too many stressors
    if len(state.active_stressors) >= 2:
        return None
    
    # Base probability modified by current state
    trigger_prob = base_probability
    
    # High resource scarcity makes famine more likely
    trigger_prob += state.resource_scarcity * 0.005
    
    # Historical violence makes everything more fragile
    trigger_prob += state.historical_violence * 0.01
    
    # Low trust makes crises more likely
    trigger_prob += (1 - state.trust_in_authority) * 0.005
    
    if rng.random() > trigger_prob:
        return None
    
    # Select which stressor
    stressor_types = list(STRESSOR_PROBABILITIES.keys())
    probs = np.array(list(STRESSOR_PROBABILITIES.values()))
    
    # Condition probabilities on state
    if state.resource_scarcity > 0.5:
        # Famine more likely when already scarce
        probs[stressor_types.index(StressorType.FAMINE)] *= 1.5
    
    probs = probs / probs.sum()
    
    selected = rng.choice(stressor_types, p=probs)
    
    # Don't repeat active stressors
    if selected in state.active_stressors:
        return None
    
    # Set duration
    effects = STRESSOR_EFFECTS[selected]
    duration = rng.randint(effects['duration_range'][0], effects['duration_range'][1])
    state.active_stressors[selected] = duration
    
    return selected


def apply_stressor_effects(state: VillageState, rng: np.random.RandomState):
    """Apply ongoing effects of active stressors."""
    
    stressors_to_remove = []
    
    for stressor_type, days_remaining in state.active_stressors.items():
        effects = STRESSOR_EFFECTS[stressor_type]
        
        # Apply daily effects
        if 'resource_scarcity_boost' in effects:
            # Don't keep adding, maintain elevated level
            target = state.base_resource_scarcity + effects['resource_scarcity_boost']
            state.resource_scarcity = max(state.resource_scarcity, min(1.0, target))
        
        if 'panic_boost' in effects:
            # Small daily panic increment
            state.panic_level = min(1.0, state.panic_level + effects['panic_boost'] * 0.05)
        
        if 'conformity_boost' in effects:
            state.conformity_pressure = min(1.0, 
                state.conformity_pressure + effects['conformity_boost'] * 0.02)
        
        if 'trust_drain' in effects:
            state.trust_in_authority = max(0.0, 
                state.trust_in_authority - effects['trust_drain'] * 0.1)
        
        # Plague causes random deaths
        if stressor_type == StressorType.PLAGUE:
            death_rate = effects.get('random_death_rate', 0.02)
            for villager in state.villagers.values():
                if villager.is_alive and rng.random() < death_rate:
                    villager.is_alive = False
                    state.recent_deaths.append((state.timestep, villager.id))
                    
                    # Plague deaths increase fear across village
                    for v in state.villagers.values():
                        if v.is_alive:
                            v.emotional_state.fear = min(1.0, v.emotional_state.fear + 0.1)
                            v.emotional_state.grief = min(1.0, v.emotional_state.grief + 0.05)
        
        # Decrement duration
        state.active_stressors[stressor_type] = days_remaining - 1
        if days_remaining <= 1:
            stressors_to_remove.append(stressor_type)
    
    # Remove expired stressors
    for stressor_type in stressors_to_remove:
        del state.active_stressors[stressor_type]
        # Resource scarcity slowly returns to baseline
        state.resource_scarcity = max(state.base_resource_scarcity, 
                                       state.resource_scarcity - 0.05)


def get_stressor_suspicion_modifier(target: 'Villager', state: VillageState) -> float:
    """
    During certain stressors, specific occupations become more suspicious.
    """
    modifier = 0.0
    
    if StressorType.PLAGUE in state.active_stressors:
        # Healers and midwives blamed for plague
        if target.occupation in [Occupation.HEALER, Occupation.MIDWIFE]:
            modifier += STRESSOR_EFFECTS[StressorType.PLAGUE].get('healer_suspicion', 0.3)
    
    if StressorType.WAR_RUMOR in state.active_stressors:
        # Anyone "different" becomes suspicious
        if target.conformity_score < 0.4:
            modifier += STRESSOR_EFFECTS[StressorType.WAR_RUMOR].get('outsider_suspicion', 0.2)
    
    if StressorType.INQUISITOR_VISIT in state.active_stressors:
        # Low church attendance is dangerous
        if target.church_attendance < 0.4:
            modifier += 0.2
    
    return modifier


# ============================================================================
# PATRON PROTECTION MECHANICS
# ============================================================================

def get_patron_protection_value(patron: 'Villager', state: VillageState) -> float:
    """
    Calculate how much protection a patron provides.
    Dilutes with more dependents (sqrt scaling).
    """
    if not patron.is_alive:
        return 0.0
    
    num_dependents = len(patron.dependents)
    
    if num_dependents == 0:
        return 0.0
    
    # Base protection from patron's status
    base_protection = patron.social_status * 0.5 + patron.reputation * 0.3
    
    # Dilution: protection spread across dependents
    # First dependent gets full value, 4th gets half, 9th gets third
    diluted_protection = base_protection / np.sqrt(num_dependents)
    
    # Patron's own stress reduces protection quality
    stress_penalty = patron.stress * 0.3
    
    # Patron's wealth affects ability to protect (resources for legal defense, etc)
    wealth_factor = 0.5 + patron.wealth * 0.5
    
    return max(0.0, diluted_protection * wealth_factor - stress_penalty)


def get_protection_for_dependent(dependent: 'Villager', state: VillageState) -> float:
    """Calculate protection level for a specific dependent."""
    if dependent.patron_id is None:
        return 0.0
    
    patron = state.villagers.get(dependent.patron_id)
    if not patron or not patron.is_alive:
        # Patron dead - clear the reference
        dependent.patron_id = None
        return 0.0
    
    base_protection = get_patron_protection_value(patron, state)
    
    # Earlier dependents get slightly better protection (they're more "core")
    try:
        position = patron.dependents.index(dependent.id)
        seniority_bonus = max(0, 0.1 - position * 0.02)  # First 5 get a bonus
    except ValueError:
        seniority_bonus = 0.0
    
    return base_protection + seniority_bonus


def calculate_patron_costs(patron: 'Villager', state: VillageState, rng: np.random.RandomState):
    """
    Calculate and apply ongoing costs of being a patron.
    Called each timestep for each patron.
    """
    num_dependents = len(patron.dependents)
    
    if num_dependents == 0:
        return
    
    # Resource drain: each dependent costs wealth
    base_drain = 0.015 * num_dependents
    
    # Stressor multiplier
    drain_multiplier = 1.0
    if StressorType.FAMINE in state.active_stressors:
        drain_multiplier = STRESSOR_EFFECTS[StressorType.FAMINE].get('patron_drain_multiplier', 2.0)
    
    wealth_drain = base_drain * drain_multiplier
    patron.wealth = max(0.0, patron.wealth - wealth_drain)
    
    # Stress from responsibility
    patron.stress = min(1.0, patron.stress + 0.005 * num_dependents)
    
    # If any dependent is accused, patron reputation at risk
    accused_dependents = sum(1 for d_id in patron.dependents 
                            if state.villagers.get(d_id, None) 
                            and state.villagers[d_id].is_accused_currently)
    if accused_dependents > 0:
        patron.reputation = max(0.0, patron.reputation - 0.02 * accused_dependents)
    
    # If patron is broke, they can't maintain dependents
    if patron.wealth < 0.1:
        # Start losing dependents (they seek other patrons)
        if num_dependents > 0 and rng.random() < 0.1:
            lost_dependent_id = patron.dependents.pop()
            lost_dependent = state.villagers.get(lost_dependent_id)
            if lost_dependent:
                lost_dependent.patron_id = None


def handle_patron_collapse(patron: 'Villager', state: VillageState):
    """
    When a patron dies or is executed, cascade effects to dependents.
    This is the delayed failure mode.
    """
    for dependent_id in patron.dependents:
        dependent = state.villagers.get(dependent_id)
        if not dependent or not dependent.is_alive:
            continue
        
        # Loss of protection
        dependent.patron_id = None
        
        # Psychological impact
        dependent.emotional_state.fear = min(1.0, dependent.emotional_state.fear + 0.3)
        dependent.stress = min(1.0, dependent.stress + 0.2)
        dependent.emotional_state.despair = min(1.0, dependent.emotional_state.despair + 0.15)
        
        # If patron was executed, even worse
        if not patron.is_alive:
            dependent.trauma_score = min(1.0, dependent.trauma_score + 0.2)
            # Guilt by association - reputation hit
            dependent.reputation = max(0.0, dependent.reputation - 0.1)
    
    # Clear patron's dependent list
    patron.dependents = []


# ============================================================================
# DIMINISHING RETURNS FOR REPEATED INTERACTIONS
# ============================================================================

def get_alliance_diminishing_factor(actor: 'Villager', target_id: int) -> float:
    """
    Calculate diminishing returns for repeated alliance/resource sharing.
    Returns a multiplier (0 to 1) for utility.
    """
    times_interacted = actor.alliance_history.get(target_id, 0)
    
    # Exponential decay: each repeat reduces utility by 20%
    # 0 interactions: 1.0, 1: 0.8, 2: 0.64, 3: 0.51, 5: 0.33, 10: 0.11
    return 0.8 ** times_interacted


def record_alliance_interaction(actor: 'Villager', target_id: int):
    """Record that an alliance-type interaction occurred."""
    actor.alliance_history[target_id] = actor.alliance_history.get(target_id, 0) + 1


# ============================================================================
# STATE DECAY FUNCTIONS (Updated with hysteresis)
# ============================================================================

def update_village_state(state: VillageState, rng: np.random.RandomState):
    """Update village-level state between timesteps"""

    # Check for recent violence (affects panic decay)
    recent_violence = any(t > state.timestep - 10 for t, _ in state.recent_deaths)
    recent_accusations = any(t > state.timestep - 5 for t, _, _ in state.recent_accusations)
    
    # Panic decays, but historical violence slows the recovery
    if recent_violence or recent_accusations:
        # Active crisis - panic decays slowly
        base_panic_decay = 0.97
    else:
        # No recent violence - panic can decay faster
        base_panic_decay = 0.92
    
    # Trauma still provides resistance, but less extreme
    trauma_resistance = 1 + state.historical_violence * 0.3  # Was 0.5
    panic_decay = base_panic_decay ** (1 / trauma_resistance)
    state.panic_level = max(0.0, state.panic_level * panic_decay)

    # Rumors decay fast
    state.rumor_saturation = max(0.0, state.rumor_saturation * DECAY_RATES['rumors'])

    # Historical violence decays - villages can heal over time
    state.historical_violence = max(0.0, state.historical_violence * DECAY_RATES['historical_violence'])
    
    # Conformity pressure returns to baseline
    state.conformity_pressure = max(0.5, state.conformity_pressure * 0.97)

    # Social cohesion slowly recovers (faster if no recent violence)
    if not recent_violence:
        state.social_cohesion = min(1.0, state.social_cohesion + 0.008)
    
    # Trust in authority slowly recovers (unless stressors active)
    if not state.active_stressors:
        state.trust_in_authority = min(1.0, state.trust_in_authority + 0.003)
    
    # Resource scarcity returns to baseline when no stressors
    if not any(s in state.active_stressors for s in [StressorType.FAMINE, StressorType.HARSH_WINTER]):
        state.resource_scarcity = max(state.base_resource_scarcity,
                                      state.resource_scarcity * 0.98)

    # Apply ongoing stressor effects
    apply_stressor_effects(state, rng)
    
    # Maybe trigger new stressor
    new_stressor = maybe_trigger_stressor(state, rng)
    if new_stressor:
        effects = STRESSOR_EFFECTS[new_stressor]
        print(f"\n  ⚡ {effects['description']}")

    # Process patron costs
    for villager in state.villagers.values():
        if villager.is_alive and villager.dependents:
            calculate_patron_costs(villager, state, rng)


def update_villager_states(state: VillageState, rng: np.random.RandomState):
    """Update individual villager states with differentiated memory decay"""

    for villager in state.villagers.values():
        if not villager.is_alive:
            continue

        # Stress decay (modified by neuroticism and trauma)
        base_stress_decay = 0.95
        trauma_penalty = villager.trauma_score * 0.3  # Trauma makes stress stickier
        stress_decay = base_stress_decay - trauma_penalty * 0.1
        villager.stress = max(0.0, villager.stress * max(0.8, stress_decay))

        # Fear decay - faster if not accused, slower if traumatized
        if not villager.is_accused_currently:
            fear_decay = DECAY_RATES['suspicions']  # Fast
        else:
            fear_decay = 0.98  # Slow when accused
        
        # Trauma slows fear recovery
        fear_decay = fear_decay ** (1 + villager.trauma_score * 0.5)
        villager.emotional_state.fear = max(0.0, villager.emotional_state.fear * fear_decay)

        # Anger decays
        villager.emotional_state.anger *= DECAY_RATES['hatred']

        # Grief decays slowly
        villager.emotional_state.grief *= 0.99
        
        # Despair decays slowly, faster with hope
        despair_decay = 0.98 if villager.emotional_state.hope > 0.5 else 0.995
        villager.emotional_state.despair *= despair_decay

        # Trauma decays very slowly
        if villager.family_executions > 0:
            villager.trauma_score *= DECAY_RATES['family_trauma']
        else:
            villager.trauma_score *= DECAY_RATES['trauma']
        
        # Suspicions decay faster than other emotions
        for target_id in list(villager.emotional_state.suspicions.keys()):
            villager.emotional_state.suspicions[target_id] *= DECAY_RATES['suspicions']
            if villager.emotional_state.suspicions[target_id] < 0.01:
                del villager.emotional_state.suspicions[target_id]
        
        # Hatreds decay, but slowly
        for target_id in list(villager.emotional_state.hatreds.keys()):
            villager.emotional_state.hatreds[target_id] *= DECAY_RATES['hatred']
            if villager.emotional_state.hatreds[target_id] < 0.01:
                del villager.emotional_state.hatreds[target_id]
        
        # Loyalties decay even more slowly
        for target_id in list(villager.emotional_state.loyalties.keys()):
            villager.emotional_state.loyalties[target_id] *= DECAY_RATES['loyalty']
            if villager.emotional_state.loyalties[target_id] < 0.01:
                del villager.emotional_state.loyalties[target_id]

        # Recent accusations window
        if villager.days_since_last_accusation < 999:
            villager.days_since_last_accusation += 1
            if villager.days_since_last_accusation > 30:
                villager.times_accused_recent = max(0, villager.times_accused_recent - 1)

        # Panic and conformity pressure affect everyone
        villager.emotional_state.fear = min(1.0,
            villager.emotional_state.fear + state.panic_level * 0.03)
        
        # High conformity pressure increases stress for non-conformists
        if state.conformity_pressure > 0.6:
            nonconformity = 1 - villager.conformity_score
            villager.stress = min(1.0, villager.stress + nonconformity * state.conformity_pressure * 0.02)

        # Neuroticism amplifies stress
        villager.stress = min(1.0,
            villager.stress + villager.personality.neuroticism * 0.01)
        
        # Historical village violence keeps everyone on edge
        villager.emotional_state.fear = min(1.0,
            villager.emotional_state.fear + state.historical_violence * 0.02)
        
        # Alliance history decays slowly (people forget old favors)
        for target_id in list(villager.alliance_history.keys()):
            if rng.random() < 0.02:  # 2% chance per day to "forget" one interaction
                villager.alliance_history[target_id] = max(0, villager.alliance_history[target_id] - 1)
                if villager.alliance_history[target_id] <= 0:
                    del villager.alliance_history[target_id]
