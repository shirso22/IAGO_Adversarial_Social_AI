"""
Chain Accusations v2 - With Circuit Breakers

The cascade engine, now with historical rate limits:
1. Trial capacity - max trials per day
2. Confession fatigue - system exhaustion reduces pressure
3. Scheduling delays - chain-named aren't tried immediately  
4. Elite intervention - high-status accusations trigger scrutiny
5. Lower base rates - not everyone breaks, not everyone names many

Integration: Same as v1, but add trial scheduling logic to main.py
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from Villager import Villager, Gender, MaritalStatus, Occupation
from Village import VillageState
from Relationships import RelationshipType


# ============================================================================
# CONFIGURATION - Tune these for cascade intensity
# ============================================================================

CHAIN_CONFIG = {
    # Confession rates
    'base_confession_prob': 0.25,        # Base probability of naming others (was 0.4)
    'max_confession_prob': 0.85,         # Cap on confession probability
    'min_confession_prob': 0.05,         # Floor on confession probability
    
    # Number of names
    'base_names_poisson_lambda': 1.0,    # Poisson parameter for base names (was 2.0)
    'max_names_per_confession': 4,       # Hard cap on names (was 8)
    
    # Trial scheduling
    'min_trial_delay': 1,                # Minimum days before chain-accused is tried
    'max_trial_delay': 4,                # Maximum days before chain-accused is tried
    'max_trials_per_day': 3,             # Court capacity limit
    
    # System fatigue
    'fatigue_window': 5,                 # Days to look back for fatigue calc
    'fatigue_per_execution': 0.03,       # Confession prob reduction per recent execution
    'fatigue_per_trial': 0.02,           # Additional reduction per recent trial
    
    # Elite intervention
    'elite_status_threshold': 0.7,       # Status above which intervention possible
    'elite_intervention_prob': 0.4,      # Probability of intervention for elites
    
    # Recovery mechanics
    'acquittal_cascade_reduction': 0.3,  # Prob that chain-accused are released on acquittal
}


@dataclass
class ChainAccusationResult:
    """Result of pressuring an accused for names"""
    accused_id: int
    named_ids: List[int]
    confession_type: str  # 'refused', 'partial', 'full'
    reduced_sentence: bool
    intervention_occurred: bool = False  # Elite protection triggered


@dataclass 
class TrialSchedule:
    """Tracks scheduled trials with delays"""
    scheduled_trials: Dict[int, int] = field(default_factory=dict)  # villager_id -> scheduled_day
    
    def schedule_trial(self, villager_id: int, current_day: int, rng: np.random.RandomState):
        """Schedule a trial for a future day"""
        if villager_id not in self.scheduled_trials:
            delay = rng.randint(
                CHAIN_CONFIG['min_trial_delay'], 
                CHAIN_CONFIG['max_trial_delay'] + 1
            )
            self.scheduled_trials[villager_id] = current_day + delay
    
    def get_trials_for_day(self, day: int, trial_queue: List[int]) -> List[int]:
        """Get villagers whose trials are scheduled for today, respecting capacity"""
        # Immediate accusations (not from chain) get priority
        immediate = [vid for vid in trial_queue if vid not in self.scheduled_trials]
        
        # Chain-accused whose day has come
        scheduled_ready = [
            vid for vid in trial_queue 
            if vid in self.scheduled_trials and self.scheduled_trials[vid] <= day
        ]
        
        # Combine, respecting daily limit
        combined = immediate + scheduled_ready
        max_today = CHAIN_CONFIG['max_trials_per_day']
        
        return combined[:max_today]
    
    def clear_scheduled(self, villager_id: int):
        """Remove from schedule (after trial or release)"""
        if villager_id in self.scheduled_trials:
            del self.scheduled_trials[villager_id]


def calculate_system_fatigue(state: VillageState) -> float:
    """
    Calculate how exhausted the persecution system is.
    High fatigue = less aggressive interrogation = fewer confessions.
    """
    window = CHAIN_CONFIG['fatigue_window']
    
    # Count recent executions
    recent_executions = len([
        t for t in state.recent_execution_timestamps 
        if t > state.timestep - window
    ])
    
    # Count recent trials (approximated by recent accusations that went to trial)
    recent_trials = len([
        (t, _, _) for t, _, _ in state.recent_accusations 
        if t > state.timestep - window
    ])
    
    fatigue = (
        recent_executions * CHAIN_CONFIG['fatigue_per_execution'] +
        recent_trials * CHAIN_CONFIG['fatigue_per_trial']
    )
    
    return min(fatigue, 0.4)  # Cap fatigue effect at 40% reduction


def check_elite_intervention(
    accused: Villager, 
    state: VillageState, 
    rng: np.random.RandomState
) -> bool:
    """
    Check if elite status triggers intervention that limits chain accusations.
    
    Historically, when accusations reached nobility/clergy, sometimes
    higher authorities stepped in to slow things down.
    """
    if accused.social_status < CHAIN_CONFIG['elite_status_threshold']:
        return False
    
    # Higher status = higher intervention chance
    intervention_prob = CHAIN_CONFIG['elite_intervention_prob']
    intervention_prob += (accused.social_status - CHAIN_CONFIG['elite_status_threshold']) * 0.5
    
    # Trust in authority affects willingness to intervene
    intervention_prob *= state.trust_in_authority
    
    # Low panic = more likely to intervene (cooler heads)
    intervention_prob *= (1.5 - state.panic_level)
    
    return rng.random() < intervention_prob


def calculate_confession_probability(
    accused: Villager,
    state: VillageState,
    pressure_level: float = 0.5
) -> float:
    """
    How likely is this person to name others under pressure?
    Now includes system fatigue.
    """
    prob = CHAIN_CONFIG['base_confession_prob']
    
    # Pain tolerance is the main defense
    prob -= accused.pain_tolerance * 0.25
    
    # Psychological factors
    prob += accused.emotional_state.fear * 0.2
    prob += accused.emotional_state.despair * 0.15
    prob += (1 - accused.personality.conscientiousness) * 0.08
    
    # Agreeableness effects (reduced from v1)
    prob += accused.personality.agreeableness * 0.05  # Want to cooperate
    prob -= accused.personality.agreeableness * 0.1   # Don't want to harm
    
    # External pressure
    prob += pressure_level * 0.15
    prob += state.panic_level * 0.1
    
    # Prior trauma
    prob += accused.trauma_score * 0.1
    
    # Machiavellians name strategically
    prob += accused.personality.machiavellianism * 0.15
    
    # CIRCUIT BREAKER: System fatigue reduces pressure
    fatigue = calculate_system_fatigue(state)
    prob -= fatigue
    
    # CIRCUIT BREAKER: Very high historical violence can trigger backlash
    if state.historical_violence > 0.6:
        prob -= (state.historical_violence - 0.6) * 0.2
    
    return np.clip(
        prob, 
        CHAIN_CONFIG['min_confession_prob'], 
        CHAIN_CONFIG['max_confession_prob']
    )


def calculate_num_names(
    accused: Villager,
    state: VillageState,
    confession_prob: float,
    rng: np.random.RandomState
) -> int:
    """
    How many accomplices does the accused name?
    More conservative than v1.
    """
    if rng.random() > confession_prob:
        return 0
    
    # Lower base rate
    base = rng.poisson(CHAIN_CONFIG['base_names_poisson_lambda'])
    
    # Machiavellians name more (but less than before)
    base += int(accused.personality.machiavellianism * 2)
    
    # Desperation adds names (reduced)
    base += int(accused.emotional_state.despair * 1.5)
    
    # Enemies to name
    num_hated = len([h for h in accused.emotional_state.hatreds.values() if h > 0.4])
    base += min(num_hated, 2)
    
    # High panic increases names (reduced)
    base += int(state.panic_level * 1.5)
    
    # CIRCUIT BREAKER: Fatigue reduces names
    fatigue = calculate_system_fatigue(state)
    base = int(base * (1 - fatigue))
    
    # Cap
    alive_others = sum(
        1 for v in state.villagers.values() 
        if v.is_alive and v.id != accused.id and not v.is_accused_currently
    )
    
    return min(base, alive_others, CHAIN_CONFIG['max_names_per_confession'])


def select_accusation_targets(
    accused: Villager,
    num_targets: int,
    state: VillageState,
    rng: np.random.RandomState
) -> List[int]:
    """
    Who does the accused name? Same logic as v1 but uses passed rng.
    """
    if num_targets <= 0:
        return []
    
    candidates = []
    
    for v in state.villagers.values():
        if v.id == accused.id:
            continue
        if not v.is_alive:
            continue
        if v.is_accused_currently:
            continue
        
        weight = 0.1
        
        # Hatred - primary driver
        hatred = accused.emotional_state.hatreds.get(v.id, 0)
        weight += hatred * 2.0
        
        # Suspicion
        suspicion = accused.emotional_state.suspicions.get(v.id, 0)
        weight += suspicion * 1.5
        
        # Vulnerability
        from Utils import get_vulnerability
        vuln = get_vulnerability(v, state)
        weight += vuln * 0.6  # Reduced from 0.8
        
        # Rivalry
        if (accused.id, v.id) in state.relationships:
            rel = state.relationships[(accused.id, v.id)]
            if RelationshipType.RIVALRY in rel.relationship_types:
                weight += 0.4
        
        # Family protection - stronger than v1
        if (accused.id, v.id) in state.relationships:
            rel = state.relationships[(accused.id, v.id)]
            if RelationshipType.FAMILY in rel.relationship_types:
                family_penalty = 2.5 * (1 - accused.personality.psychopathy)
                family_penalty *= (1 - accused.emotional_state.despair * 0.4)
                weight -= family_penalty
        
        # Loyalty protection
        loyalty = accused.emotional_state.loyalties.get(v.id, 0)
        weight -= loyalty * 1.5
        
        # Patron protection
        if accused.patron_id == v.id:
            weight -= 1.5
        
        # High status penalty - stronger deterrent
        if v.social_status > 0.6:
            weight -= v.social_status * 0.8
        
        if weight > 0:
            candidates.append((v.id, weight))
    
    if not candidates:
        return []
    
    ids, weights = zip(*candidates)
    weights = np.array(weights)
    weights = np.maximum(weights, 0.01)
    probs = weights / weights.sum()
    
    num_to_select = min(num_targets, len(candidates))
    selected = rng.choice(
        ids, 
        size=num_to_select, 
        replace=False, 
        p=probs
    )
    
    return list(selected)


def apply_chain_accusation_effects(
    accused: Villager,
    named_ids: List[int],
    state: VillageState,
    trial_schedule: TrialSchedule,
    rng: np.random.RandomState,
    verbose: bool = True
):
    """
    Apply state changes from chain accusations.
    Now includes trial scheduling.
    """
    for named_id in named_ids:
        named = state.villagers.get(named_id)
        if not named:
            continue
        
        # The named person becomes accused
        named.is_accused_currently = True
        named.times_accused_total += 1
        named.times_accused_recent += 1
        named.accusations_received.append(accused.id)
        named.days_since_last_accusation = 0
        
        # Add to trial queue with scheduled delay
        if named_id not in state.trial_queue:
            state.trial_queue.append(named_id)
            trial_schedule.schedule_trial(named_id, state.timestep, rng)
        
        # Record accusation
        state.recent_accusations.append((state.timestep, accused.id, named_id))
        
        # Emotional effects (slightly reduced)
        named.emotional_state.fear = min(1.0, named.emotional_state.fear + 0.35)
        named.stress = min(1.0, named.stress + 0.25)
        named.reputation = max(0.0, named.reputation - 0.12)
        
        # Hatred toward accuser
        named.emotional_state.hatreds[accused.id] = min(1.0,
            named.emotional_state.hatreds.get(accused.id, 0) + 0.5)
        
        # Destroy loyalty
        if accused.id in named.emotional_state.loyalties:
            named.emotional_state.loyalties[accused.id] = max(0,
                named.emotional_state.loyalties[accused.id] - 0.4)
        
        if verbose:
            delay = trial_schedule.scheduled_trials.get(named_id, state.timestep) - state.timestep
            print(f"    👉 {accused.name} names {named.name} (trial in {delay} days)")
    
    # Village effects (reduced)
    if named_ids:
        state.panic_level = min(1.0, state.panic_level + len(named_ids) * 0.02)
        state.rumor_saturation = min(1.0, state.rumor_saturation + len(named_ids) * 0.04)
        state.social_cohesion = max(0.0, state.social_cohesion - len(named_ids) * 0.015)


def extract_chain_accusations(
    accused: Villager,
    state: VillageState,
    trial_schedule: TrialSchedule,
    rng: np.random.RandomState,
    pressure_level: float = 0.5,
    verbose: bool = True
) -> ChainAccusationResult:
    """
    Main entry point: pressure an accused person to name accomplices.
    
    Now includes:
    - Elite intervention check
    - System fatigue
    - Trial scheduling for named individuals
    """
    # Check for elite intervention
    intervention = check_elite_intervention(accused, state, rng)
    if intervention:
        if verbose:
            print(f"    ⚜️ {accused.name}'s status triggers scrutiny - no pressure for names")
        return ChainAccusationResult(
            accused_id=accused.id,
            named_ids=[],
            confession_type='refused',
            reduced_sentence=False,
            intervention_occurred=True
        )
    
    # Calculate confession probability
    confession_prob = calculate_confession_probability(accused, state, pressure_level)
    
    # How many names?
    num_names = calculate_num_names(accused, state, confession_prob, rng)
    
    # Confession type
    if num_names == 0:
        confession_type = 'refused'
    elif num_names <= 2:
        confession_type = 'partial'
    else:
        confession_type = 'full'
    
    # Select targets
    named_ids = select_accusation_targets(accused, num_names, state, rng)
    
    # Apply effects
    if named_ids:
        apply_chain_accusation_effects(
            accused, named_ids, state, trial_schedule, rng, verbose
        )
    
    # Reduced sentence for cooperation
    reduced_sentence = len(named_ids) >= 2 and confession_type == 'full'
    
    if verbose:
        if named_ids:
            print(f"    📜 Under interrogation, {accused.name} names {len(named_ids)} accomplices")
        else:
            print(f"    🤐 {accused.name} refuses to name accomplices")
    
    return ChainAccusationResult(
        accused_id=accused.id,
        named_ids=named_ids,
        confession_type=confession_type,
        reduced_sentence=reduced_sentence,
        intervention_occurred=False
    )


def handle_acquittal_cascade(
    acquitted_id: int,
    state: VillageState,
    trial_schedule: TrialSchedule,
    rng: np.random.RandomState,
    verbose: bool = True
) -> List[int]:
    """
    When someone is acquitted, people they named under pressure may be released.
    This is a circuit breaker that can stop cascades.
    """
    released = []
    
    # Find people this person named
    named_by_acquitted = [
        accused_id for (_, accuser_id, accused_id) in state.recent_accusations
        if accuser_id == acquitted_id
    ]
    
    for named_id in named_by_acquitted:
        named = state.villagers.get(named_id)
        if not named or not named.is_alive:
            continue
        if not named.is_accused_currently:
            continue
        
        # Chance of release
        if rng.random() < CHAIN_CONFIG['acquittal_cascade_reduction']:
            # Only release if this was their only accusation
            other_accusers = [
                acc_id for acc_id in named.accusations_received 
                if acc_id != acquitted_id
            ]
            
            if not other_accusers:
                named.is_accused_currently = False
                if named_id in state.trial_queue:
                    state.trial_queue.remove(named_id)
                trial_schedule.clear_scheduled(named_id)
                
                # Reputation recovery
                named.reputation = min(1.0, named.reputation + 0.08)
                named.emotional_state.fear = max(0, named.emotional_state.fear - 0.1)
                
                released.append(named_id)
                
                if verbose:
                    print(f"    ⚖️ {named.name} released - accusation discredited")
    
    return released


def get_chain_accusation_guilt_modifier(result: ChainAccusationResult) -> float:
    """How does confession affect verdict?"""
    if result.intervention_occurred:
        return -0.05  # Elite scrutiny slightly helps accused
    
    if result.confession_type == 'refused':
        return 0.03  # Slight suspicion
    elif result.confession_type == 'partial':
        return 0.08
    else:  # full
        return 0.12
    
    return 0.0


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def create_trial_schedule() -> TrialSchedule:
    """Create a new trial schedule tracker. Call once at simulation start."""
    return TrialSchedule()
