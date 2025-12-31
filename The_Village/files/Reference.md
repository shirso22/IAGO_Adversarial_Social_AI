

# Witch Trial Simulation: Technical Deep Dive

## Executive Summary

This document provides a comprehensive technical analysis of an agent-based simulation modeling the social dynamics of historical witch trials. The simulation implements a complex system of personality-driven decision-making, emotional contagion, social network effects, and institutional mechanics to explore how mass hysteria emerges from individual interactions.

The system models approximately 50 villagers (configurable) over 200 simulated days, tracking accusations, trials, executions, and the cascade effects that can spiral a community into crisis—or allow it to recover.

---

## 1. System Architecture

### 1.1 Module Overview

The simulation is organized into ten interconnected Python modules:

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.py                                │
│                    (Simulation Loop & Trials)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ ActionSelection │  │ ActionExecution │  │ChainAccusations │
│  (Utility Calc) │  │ (State Changes) │  │(Cascade Engine) │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Actions.py                              │
│              (Action Types, Metadata, Validation)               │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Villager.py   │  │   Village.py    │  │ Relationships.py│
│  (Agent Model)  │  │ (World State)   │  │ (Social Graph)  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌─────────────────┐
│  Personality_   │  │    Utils.py     │
│  and_emotions   │  │(Math & Helpers) │
└─────────────────┘  └─────────────────┘
```

### 1.2 Data Flow

Each simulation timestep follows this flow:

1. **Action Selection Phase**: Each living villager evaluates available actions, computes utility scores, and probabilistically selects an action.
2. **Action Execution Phase**: Selected actions modify agent and world state.
3. **Trial Phase**: Accused villagers are tried (with capacity limits), potentially triggering chain accusations.
4. **State Update Phase**: Decay functions run, stressors are processed, and emotional states evolve.

---

## 2. Agent Model (Villager)

### 2.1 Attribute Categories

Each villager is defined by approximately 40 attributes across five categories:

#### Immutable Attributes (set at initialization)
| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | int | Unique identifier |
| `name` | str | Generated medieval name with bynames for duplicates |
| `gender` | Gender | MALE or FEMALE |
| `age` | int | 15-85, beta distribution skewed young |
| `occupation` | Occupation | 11 types including HEALER, MIDWIFE, CLERGY |
| `beauty` | float | 0-1, physical attractiveness |

#### Personality (Big Five + Dark Triad + Seven Sins)
```python
@dataclass
class Personality:
    # Big Five (0-1 scales)
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    
    # Dark Triad (most people score low)
    machiavellianism: float = 0.2
    narcissism: float = 0.2
    psychopathy: float = 0.1
    
    # Seven Deadly Sins
    envy: float = 0.3
    greed: float = 0.3
    wrath: float = 0.3
    lust: float = 0.3
    pride: float = 0.3
    gluttony: float = 0.3
    sloth: float = 0.3
```

Personality traits are generated using beta distributions calibrated to produce realistic population distributions. The Dark Triad uses heavily right-skewed distributions (e.g., psychopathy: `beta(1, 8)`).

#### Emotional State (volatile, changes constantly)
```python
@dataclass
class EmotionalState:
    # Acute emotions (high volatility)
    fear: float = 0.3
    anger: float = 0.2
    joy: float = 0.4
    shame: float = 0.2
    grief: float = 0.1
    desire: float = 0.3
    
    # Sustained emotions (medium volatility)
    resentment: float = 0.2
    hope: float = 0.4
    despair: float = 0.2
    
    # Directed emotions (toward specific individuals)
    romantic_attractions: Dict[int, float]
    hatreds: Dict[int, float]
    envies: Dict[int, float]
    loyalties: Dict[int, float]
    suspicions: Dict[int, float]
```

#### Dynamic Attributes
| Attribute | Range | Description |
|-----------|-------|-------------|
| `social_status` | 0-1 | Pareto distributed, affects credibility |
| `wealth` | 0-1 | Correlated with status |
| `stress` | 0-1 | Accumulated long-term stress |
| `reputation` | 0-1 | Community standing |
| `trauma_score` | 0-1 | Accumulated psychological damage |
| `conformity_score` | 0-1 | How "normal" they appear |

#### State Flags
- `is_alive`, `is_accused_currently`, `is_on_trial`, `is_imprisoned`
- `witnessed_executions`, `family_executions`
- `patron_id`, `dependents[]`

### 2.2 Name Generation

The system generates unique medieval names with collision handling:

```python
# First occurrence
"John Smith"

# Second occurrence with same base name
"John Smith the Younger"  # or
"John Smith of the Hill"

# Subsequent collisions use random bynames
"John Smith the Bold"
```

This uses two pools of bynames (descriptive and locational) with fallback to ensure uniqueness.

---

## 3. Social Network (Relationships)

### 3.1 Multi-Edge Graph Model

Relationships are stored as a dictionary keyed by `(source_id, target_id)` tuples. Critically, each edge can contain **multiple relationship types**:

```python
@dataclass
class Relationship:
    source: int
    target: int
    relationship_types: Set[RelationshipType]  # Can contain multiple!
    overall_strength: float
    overall_trust: float
    type_strengths: Dict[RelationshipType, float]
    type_attributes: Dict[RelationshipType, dict]
```

### 3.2 Relationship Types

| Type | Trust Modifier | Description |
|------|---------------|-------------|
| FAMILY | +0.8 | Blood relations, assigned in clusters |
| MARRIAGE | +0.9 | Spousal bonds |
| FRIENDSHIP | +0.7 | Social bonds |
| ECONOMIC | +0.5 | Business relationships |
| RIVALRY | -0.3 | Competition/conflict |
| PATRONAGE | +0.6 | Power relationships (directed) |
| NEIGHBOR | +0.4 | Proximity |

### 3.3 Network Generation

The initial network is built in three phases:

1. **Family Clusters**: Villagers assigned to ~N/5 families with high-trust FAMILY edges.
2. **Economic Network**: Preferential attachment based on social status (rich get more connections).
3. **Rivalries**: 15% chance between similar-status villagers within 0.2 status and 0.3 wealth.

---

## 4. Action System

### 4.1 Action Categories

The 30+ action types are organized into categories:

| Category | Actions |
|----------|---------|
| AGGRESSIVE | ACCUSE_WITCHCRAFT, SPREAD_RUMOR, COUNTER_ACCUSE, INSULT_PUBLICLY |
| DEFENSIVE | COUNTER_ACCUSE, TESTIFY_FOR, CONFESS, HIDE_ACTIVITY, FLEE_VILLAGE |
| SOCIAL | FORM_ALLIANCE, BREAK_ALLIANCE, DEFEND_PERSON, SOCIALIZE, RECONCILE |
| ECONOMIC | SHARE_RESOURCES, REQUEST_HELP, WITHHOLD_RESOURCES |
| ROMANTIC | COURT_PERSON, PROPOSE_MARRIAGE |
| INSTITUTIONAL | ATTEND_CHURCH, APPEAL_TO_AUTHORITY, SEEK_PROTECTION, OFFER_PATRONAGE |
| INFORMATION | SPY_ON, GATHER_EVIDENCE, SHARE_GOSSIP, SEEK_ADVICE |
| TRIAL | TESTIFY_AGAINST, TESTIFY_FOR, CONFESS |

### 4.2 Action Metadata

Each action has associated metadata:

```python
ACTION_METADATA = {
    ActionType.ACCUSE_WITCHCRAFT: {
        "category": ActionCategory.AGGRESSIVE,
        "base_cost": 0.08,
        "requires_target": True,
        "risk": 0.15,
    },
    # ...
}
```

### 4.3 Action Selection Algorithm

Action selection uses a **utility-weighted softmax selection**:

```python
def select_action(actor, state, rng, top_k=5):
    # 1. Generate all (action, target) pairs
    scored_actions = []
    for action_type in get_available_actions(actor, state):
        for target in potential_targets:
            utility = compute_action_utility(actor, action_type, target.id, state)
            if utility > 0:
                scored_actions.append((action_type, target.id, utility))
    
    # 2. Take top k by utility
    top_actions = sorted(scored_actions, key=lambda x: x[2], reverse=True)[:top_k]
    
    # 3. Softmax selection with temperature
    utilities = [max(a[2], 0.01) for a in top_actions]
    temperature = 0.5
    utilities = utilities ** (1 / temperature)
    probs = utilities / sum(utilities)
    
    # 4. Probabilistic selection
    return rng.choice(top_actions, p=probs)
```

The temperature parameter controls exploration vs. exploitation:
- Lower temperature (0.2) → nearly deterministic, always picks highest utility
- Higher temperature (1.0) → more uniform random selection

---

## 5. Utility Computation

### 5.1 Core Formula

The utility of an action is computed as:

```
U(action, target) = base_utility 
                  - risk_penalty 
                  - cost_penalty 
                  + action_specific_factors
```

### 5.2 Risk Tolerance

Risk tolerance is personality-driven:

```python
risk_tolerance = (
    psychopathy * 0.3 +
    (1 - neuroticism) * 0.3 +
    machiavellianism * 0.2 +
    (1 - conscientiousness) * 0.2
)
risk_penalty = action_risk * (1 - risk_tolerance)
```

### 5.3 Accusation Utility (Key Action)

The `ACCUSE_WITCHCRAFT` utility calculation demonstrates the complexity:

```python
def compute_accusation_utility(actor, target, state):
    utility = 0.1  # base
    
    # Motivators
    utility += hatreds[target] * 0.35         # Hatred drives accusations
    utility += machiavellianism * 0.25        # Strategic accusers
    utility += fear * 0.18                    # Scapegoating
    utility += vulnerability(target) * 0.25   # Easy targets attractive
    
    # Rivalry bonus
    if RIVALRY in relationship:
        utility += 0.2
    
    # Patron targeting (high-value for Machiavellians)
    if target.dependents > 2:
        utility += machiavellianism * 0.2
    
    # Environmental factors
    utility += historical_violence * 0.12     # Normalization of violence
    utility += panic_level * 0.15             # Mob mentality
    
    # Inhibitors
    utility -= agreeableness * 0.2            # Nice people hesitate
    utility -= loyalties[target] * 0.5        # Loyalty protects
    utility -= serial_accuser_penalty * 0.12  # Diminishing returns
    
    # Family penalty
    if FAMILY in relationship:
        utility -= 0.4
    
    return utility
```

---

## 6. Vulnerability System

### 6.1 Vulnerability Calculation

Vulnerability determines how likely someone is to be accused:

```python
def get_vulnerability(villager, state):
    v = 0.0
    
    # Demographics
    v += 0.3 if gender == FEMALE else 0.0
    v += abs(age - 40) / 40 * 0.2  # U-shaped age curve
    v += 0.3 if WIDOWED else 0.0
    v += 0.2 if SINGLE and age > 30 else 0.0
    
    # Social position
    v += (1 - social_status) * 0.4      # Low status = vulnerable
    v += (1 - conformity_score) * 0.3   # Non-conformists targeted
    v += isolation_score * 0.3          # Network isolation
    
    # Suspicious markers
    v += 0.2 if owns_cat else 0.0
    v += 0.15 if lives_alone else 0.0
    v += 0.1 if has_suspicious_marks else 0.0
    
    # Occupation risk
    v += 0.25 if occupation in [HEALER, MIDWIFE] else 0.0
    
    # Feedback loops
    v += min(times_accused_recent * 0.2, 0.5)
    
    # Protection
    v -= patron_protection * 0.4
    
    return clip(v, 0, 1)
```

### 6.2 Historical Accuracy

The vulnerability coefficients are calibrated to historical witch trial demographics:
- ~80% of accused were women
- Widows and spinsters over-represented
- Healers/midwives at elevated risk
- Low-status individuals more vulnerable

---

## 7. Trial System

### 7.1 Trial Flow

```
┌─────────────────┐
│ Accusation Made │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Added to Queue │
└────────┬────────┘
         ▼
┌─────────────────┐    ┌─────────────────┐
│ Trial Scheduled │───▶│ Gather Testimony│
└────────┬────────┘    └────────┬────────┘
         │                      ▼
         │             ┌─────────────────┐
         │             │Chain Accusations│
         │             │  (under duress) │
         │             └────────┬────────┘
         ▼                      ▼
┌─────────────────────────────────────────┐
│            Compute Guilt Score          │
└────────────────────┬────────────────────┘
                     ▼
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│    CONVICTED    │     │    ACQUITTED    │
│   (Executed)    │     │   (Released)    │
└────────┬────────┘     └────────┬────────┘
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Trauma to all   │     │ Acquittal       │
│ Patron collapse │     │ Cascade Check   │
└─────────────────┘     └─────────────────┘
```

### 7.2 Guilt Score Calculation

```python
guilt_score = 0.3  # base

# Evidence
guilt_score += vulnerability * 0.25
guilt_score += min(num_accusers * 0.08, 0.25)

# Social standing
guilt_score -= social_status * 0.2

# Testimony
guilt_score += len(testimony_against) * 0.08
guilt_score -= len(testimony_for) * 0.06
guilt_score -= len(defenders) * 0.05

# Witness credibility
for witness in testimony_against:
    guilt_score += witness.social_status * 0.03

# Confession modifier
guilt_score += chain_accusation_guilt_modifier  # 0.03-0.12

# Environmental
guilt_score += panic_level * 0.15
guilt_score -= (trust_in_authority - 0.5) * 0.1

# Gathered evidence
guilt_score += min(total_evidence * 0.5, 0.2)

guilt_score = clip(guilt_score, 0.1, 0.9)
convicted = random() < guilt_score
```

---

## 8. Chain Accusation System

### 8.1 Overview

Chain accusations model the historical phenomenon where accused individuals, under duress, named "accomplices"—creating cascade effects that could spiral out of control.

### 8.2 Configuration Parameters

```python
CHAIN_CONFIG = {
    'base_confession_prob': 0.25,      # Base probability of naming others
    'max_confession_prob': 0.85,       # Cap
    'min_confession_prob': 0.05,       # Floor
    
    'base_names_poisson_lambda': 1.0,  # Expected number of names
    'max_names_per_confession': 4,     # Hard cap
    
    'min_trial_delay': 1,              # Days before chain-accused tried
    'max_trial_delay': 4,
    'max_trials_per_day': 3,           # Court capacity
    
    'fatigue_window': 5,               # Days for fatigue calculation
    'fatigue_per_execution': 0.03,     # Confession reduction per execution
    'fatigue_per_trial': 0.02,
    
    'elite_status_threshold': 0.7,     # Status for intervention check
    'elite_intervention_prob': 0.4,
    
    'acquittal_cascade_reduction': 0.3,  # Probability chain-accused released
}
```

### 8.3 Confession Probability

```python
def calculate_confession_probability(accused, state, pressure_level):
    prob = 0.25  # base
    
    # Defenses
    prob -= pain_tolerance * 0.25
    
    # Vulnerabilities
    prob += fear * 0.2
    prob += despair * 0.15
    prob += (1 - conscientiousness) * 0.08
    prob += trauma_score * 0.1
    
    # External pressure
    prob += pressure_level * 0.15
    prob += panic_level * 0.1
    
    # Strategic naming
    prob += machiavellianism * 0.15
    
    # Circuit breakers
    prob -= system_fatigue
    if historical_violence > 0.6:
        prob -= (historical_violence - 0.6) * 0.2
    
    return clip(prob, 0.05, 0.85)
```

### 8.4 Target Selection

When naming accomplices, the accused selects targets based on weighted probability:

```python
def select_accusation_targets(accused, num_targets, state, rng):
    weights = []
    for villager in living_villagers:
        weight = 0.5  # base
        
        # Positive factors
        weight += hatreds[villager] * 2.0       # Name enemies
        weight += suspicions[villager] * 1.5
        weight += rivalry_intensity * 1.0
        weight += (1 - villager.reputation) * 0.5
        
        # Negative factors (protections)
        weight -= loyalties[villager] * 1.5     # Protect friends
        weight -= 1.5 if villager is patron     # Protect patron
        weight -= villager.social_status * 0.8  # Hesitate to name elites
        
        if weight > 0:
            candidates.append((villager.id, weight))
    
    # Weighted random selection without replacement
    return rng.choice(candidates, size=num_targets, p=normalized_weights)
```

---

## 9. Circuit Breakers

The simulation includes multiple mechanisms to prevent runaway cascades:

### 9.1 Trial Capacity Limits

```python
max_trials_per_day = 3  # Court capacity
```

Trials are scheduled with delays (1-4 days), preventing immediate cascade execution.

### 9.2 System Fatigue

```python
def calculate_system_fatigue(state):
    recent_executions = count_executions_in_window(5_days)
    recent_trials = count_trials_in_window(5_days)
    
    fatigue = (
        recent_executions * 0.03 +
        recent_trials * 0.02
    )
    return min(fatigue, 0.4)  # Cap at 40% reduction
```

High fatigue reduces confession probability, modeling institutional exhaustion.

### 9.3 Elite Intervention

```python
def check_elite_intervention(accused, state, rng):
    if accused.social_status < 0.7:
        return False
    
    intervention_prob = 0.4
    intervention_prob += (status - 0.7) * 0.5
    intervention_prob *= trust_in_authority
    intervention_prob *= (1.5 - panic_level)  # Calm heads intervene more
    
    return random() < intervention_prob
```

When elites are accused, there's a chance authorities step in to slow proceedings.

### 9.4 Acquittal Cascade

```python
def handle_acquittal_cascade(acquitted_id, state, trial_schedule):
    # Find people named by the acquitted
    named_by_acquitted = [...]
    
    for named in named_by_acquitted:
        if random() < 0.3:  # 30% chance
            if no_other_accusers(named):
                release(named)  # Charges dropped
```

Acquittals can trigger release of chain-accused individuals, breaking cascades.

### 9.5 Violence Backlash

At very high historical violence (>0.6), confession probability decreases:

```python
if historical_violence > 0.6:
    confession_prob -= (historical_violence - 0.6) * 0.2
```

This models community backlash against excessive persecution.

---

## 10. External Stressors

### 10.1 Stressor Types

| Stressor | Duration | Effects |
|----------|----------|---------|
| PLAGUE | 20-40 days | +0.4 resource scarcity, healers suspected |
| FAMINE | 30-60 days | +0.5 resource scarcity, +0.3 panic |
| HARSH_WINTER | 40-80 days | +0.25 scarcity, +0.2 conformity pressure |
| INQUISITOR_VISIT | 10-20 days | +0.3 panic, +0.25 conformity pressure |
| POLITICAL_UPHEAVAL | 15-30 days | -0.2 trust in authority |
| RELIGIOUS_REVIVAL | 10-25 days | +0.2 conformity pressure |

### 10.2 Stressor Triggering

```python
def maybe_trigger_stressor(state, rng):
    if state.active_stressors:
        return None  # Only one at a time
    
    base_probability = 0.003  # ~0.3% per day
    
    # Modify by conditions
    if high_historical_violence:
        probability *= 1.5  # Trauma attracts more trauma
    if low_social_cohesion:
        probability *= 1.3
    
    if random() < probability:
        return weighted_random_stressor()
```

### 10.3 Stressor Effects on Vulnerability

```python
def get_stressor_suspicion_modifier(villager, state):
    modifier = 0.0
    
    if PLAGUE in active_stressors:
        if occupation in [HEALER, MIDWIFE]:
            modifier += 0.3 * 0.7  # Healers blamed
    
    if FAMINE in active_stressors:
        if wealth > 0.6:
            modifier += 0.15  # Wealthy resented
    
    if HARSH_WINTER in active_stressors:
        if lives_alone:
            modifier += 0.1
    
    return modifier
```

---

## 11. State Decay Functions

### 11.1 Decay Rates

```python
DECAY_RATES = {
    'panic': 0.92,              # Fast decay when no violence
    'rumors': 0.95,             # Rumors fade quickly
    'suspicions': 0.90,         # Suspicions are volatile
    'hatred': 0.98,             # Hatred is persistent
    'loyalty': 0.995,           # Loyalty is very sticky
    'trauma': 0.998,            # Trauma barely decays
    'family_trauma': 0.9995,    # Family trauma is near-permanent
    'historical_violence': 0.995,
}
```

### 11.2 Village State Update

```python
def update_village_state(state, rng):
    recent_violence = any_deaths_in_last_10_days()
    
    # Panic decay (slower during active crisis)
    if recent_violence:
        panic_decay = 0.97
    else:
        panic_decay = 0.92
    
    # Trauma slows recovery
    trauma_resistance = 1 + historical_violence * 0.3
    panic_decay = panic_decay ** (1 / trauma_resistance)
    
    state.panic_level *= panic_decay
    
    # Other decay
    state.rumor_saturation *= 0.95
    state.historical_violence *= 0.995
    state.conformity_pressure = max(0.5, conformity_pressure * 0.97)
    
    # Recovery (only without violence)
    if not recent_violence:
        state.social_cohesion = min(1.0, social_cohesion + 0.008)
    
    # Stressor processing
    apply_stressor_effects(state, rng)
    maybe_trigger_stressor(state, rng)
```

### 11.3 Individual State Update

```python
def update_villager_states(state, rng):
    for villager in living_villagers:
        # Stress decay (slower with trauma)
        stress_decay = 0.95 - trauma_score * 0.03
        villager.stress *= max(0.8, stress_decay)
        
        # Fear decay (slower when accused or traumatized)
        if is_accused:
            fear_decay = 0.98
        else:
            fear_decay = 0.90
        fear_decay = fear_decay ** (1 + trauma_score * 0.5)
        villager.fear *= fear_decay
        
        # Directed emotion decay
        for target in hatreds:
            hatreds[target] *= 0.98
        for target in loyalties:
            loyalties[target] *= 0.995
        for target in suspicions:
            suspicions[target] *= 0.90
        
        # Ambient fear from village state
        villager.fear += panic_level * 0.03
        villager.fear += historical_violence * 0.02
```

---

## 12. Patronage System

### 12.1 Protection Mechanics

High-status villagers can offer protection to lower-status individuals:

```python
def get_patron_protection_value(patron, state):
    if not patron or not patron.is_alive:
        return 0.0
    
    base = patron.social_status * 0.6
    
    # Reputation matters
    base *= (0.5 + patron.reputation * 0.5)
    
    # Accused patrons can't protect well
    if patron.is_accused_currently:
        base *= 0.3
    
    # Too many dependents dilutes protection
    if len(patron.dependents) > 3:
        base *= 3 / len(patron.dependents)
    
    return base
```

### 12.2 Patron Collapse

When a patron is executed, cascade effects hit dependents:

```python
def handle_patron_collapse(patron, state):
    for dependent in patron.dependents:
        # Loss of protection
        dependent.patron_id = None
        
        # Psychological impact
        dependent.fear += 0.3
        dependent.stress += 0.2
        dependent.despair += 0.15
        dependent.trauma_score += 0.2
        
        # Guilt by association
        dependent.reputation -= 0.1
```

### 12.3 Strategic Value

Patrons are high-value targets for Machiavellian accusers:

```python
# In accusation utility calculation
if target.dependents and len(target.dependents) > 2:
    utility += machiavellianism * 0.2
    utility += min(len(target.dependents) * 0.03, 0.15)
```

---

## 13. Simulation Termination

### 13.1 Termination Conditions

The simulation ends when any condition is met:

1. **Maximum Steps Reached**: `timestep >= max_steps` (default: 200)

2. **Village Decimated**: 
   ```python
   alive_count < village_size * min_survival_rate  # default: 30%
   ```

3. **Peace Restored**:
   ```python
   timestep > peace_threshold_days and        # default: 30
   panic_level < peace_panic_level and        # default: 0.05
   sum(accusations_last_10_days) == 0 and
   len(trial_queue) == 0
   ```

### 13.2 Output Statistics

The simulation tracks comprehensive statistics:

```python
stats = {
    'accusations_per_timestep': [],
    'deaths_per_timestep': [],
    'panic_over_time': [],
    'alive_count': [],
    'actions_taken': {},
    'historical_violence_over_time': [],
    'stressors_triggered': [],
    'chain_accusations_per_day': [],
    'trials_per_day': [],
    'acquittal_releases': 0,
}
```

---

## 14. Mathematical Foundations

### 14.1 Probability Distributions

| Use | Distribution | Parameters | Rationale |
|-----|--------------|------------|-----------|
| Personality traits | Beta(2, 2) | Symmetric | Most traits are average |
| Dark Triad | Beta(1.5, 5) | Right-skewed | Most people score low |
| Psychopathy | Beta(1, 8) | Heavily right-skewed | Very rare |
| Social status | Pareto(3) | Power law | Inequality |
| Age | Beta(2, 5) * 70 + 15 | Left-skewed | Young population |
| Names per confession | Poisson(1.0) | Count data | Discrete events |

### 14.2 Utility Softmax

Action selection uses a temperature-scaled softmax:

```
P(action_i) = exp(U_i / T) / Σ exp(U_j / T)
```

With T = 0.5, this gives a moderate exploration rate.

### 14.3 Decay Functions

Most decay follows exponential form:

```
x(t+1) = x(t) * r
```

With resistance modification for trauma:

```
effective_rate = base_rate ^ (1 / (1 + trauma * k))
```

---

## 15. Configuration Guide

### 15.1 Core Parameters

```python
VILLAGE_POPULATION = 50       # 10-200 reasonable
SIMULATION_LENGTH = 200       # Max days
RANDOM_SEED = 696             # For reproducibility
INITIAL_PANIC = 0.35          # 0.0-1.0
```

### 15.2 Tuning Cascade Intensity

To increase cascade severity:
- Increase `base_confession_prob` (0.25 → 0.40)
- Increase `max_names_per_confession` (4 → 8)
- Decrease `fatigue_per_execution` (0.03 → 0.01)
- Increase `max_trials_per_day` (3 → 5)

To decrease cascade severity:
- Decrease `base_confession_prob` (0.25 → 0.15)
- Increase trial delays
- Decrease `base_names_poisson_lambda`
- Increase `elite_intervention_prob`

### 15.3 Balancing Tips

1. **If too few accusations**: Increase `INITIAL_PANIC` or decrease agreeableness coefficients in accusation utility.

2. **If cascades spiral too fast**: Increase circuit breaker strengths (fatigue, elite intervention).

3. **If simulations always end in peace**: Decrease panic decay rate or increase stressor frequency.

4. **If simulations always end in decimation**: Increase acquittal cascade probability or trial delays.

---

## 16. Extension Points

### 16.1 Adding New Actions

1. Add enum value to `ActionType`
2. Add metadata to `ACTION_METADATA`
3. Add utility calculation case in `compute_action_utility()`
4. Add execution handler in `execute_action()` and implement `_execute_<action>()`
5. Add availability check in `get_available_actions()` if needed

### 16.2 Adding New Stressors

1. Add enum value to `StressorType`
2. Add effects dictionary to `STRESSOR_EFFECTS`
3. Add weight to stressor selection in `maybe_trigger_stressor()`
4. Add vulnerability modifier in `get_stressor_suspicion_modifier()` if needed

### 16.3 Adding New Personality Traits

1. Add field to `Personality` dataclass
2. Add generation in `generate_personality()`
3. Incorporate into relevant utility calculations
4. Consider adding to interaction vector calculations

---

## 17. Known Limitations

1. **Same-sex relationships**: Currently disabled per original design (line 201 of Utils.py).

2. **Static occupations**: Occupations don't change during simulation.

3. **No births/immigration**: Population can only decrease.

4. **Single stressor limit**: Only one external stressor active at a time.

5. **No economic simulation**: Wealth changes only through specific actions.

6. **Simplified legal system**: No appeals, lawyers, or complex proceedings.

---

## Appendix A: File Reference

| File | Lines | Purpose |
|------|-------|---------|
| main.py | ~350 | Simulation loop, trial logic, entry point |
| Villager.py | ~90 | Agent data model |
| Village.py | ~250 | World state, network generation |
| Relationships.py | ~80 | Multi-edge relationship model |
| Personality_and_emotions.py | ~60 | Personality and emotional state models |
| Actions.py | ~280 | Action types, metadata, validation |
| ActionSelection.py | ~640 | Utility computation, action selection |
| ActionExecution.py | ~840 | State change handlers |
| ChainAccusations.py | ~530 | Cascade engine, circuit breakers |
| Utils.py | ~940 | Vulnerability, decay, stressors, helpers |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Agent** | A simulated villager with personality, emotions, and behaviors |
| **Chain Accusation** | Secondary accusations extracted under duress during trials |
| **Circuit Breaker** | Mechanism that limits runaway cascade effects |
| **Conformity Score** | How "normal" a villager appears to the community |
| **Hysteria** | High panic + high accusations + low trust state |
| **Patron** | High-status villager offering protection to dependents |
| **Stressor** | External event that affects village conditions |
| **Utility** | Expected value of taking an action |
| **Vulnerability** | How likely someone is to be accused |
