"""The Village - where the simulation of historical awfulness unfolds"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from Villager import Villager, Gender, MaritalStatus, Occupation
from Relationships import Relationship, RelationshipType
from Personality_and_emotions import Personality, EmotionalState


# ============================================================================
# NAME GENERATOR (because "Node 47" is too dystopian even for us)
# ============================================================================

MALE_NAMES = [
    "Thomas", "John", "William", "Richard", "Robert", "Henry", "Edward", "George",
    "James", "Francis", "Nicholas", "Simon", "Walter", "Roger", "Ralph", "Hugh",
    "Gilbert", "Geoffrey", "Peter", "Michael", "Adam", "Martin", "Stephen", "David",
    "Andrew", "Philip", "Edmund", "Lawrence", "Gregory", "Bernard", "Roland", "Alan",
    "Reginald", "Godfrey", "Bartholomew", "Augustine", "Benedict", "Clement", "Oswald", "Harold"
]

FEMALE_NAMES = [
    "Agnes", "Alice", "Joan", "Margaret", "Mary", "Elizabeth", "Anne", "Catherine",
    "Eleanor", "Beatrice", "Maud", "Edith", "Emma", "Matilda", "Isabella", "Rose",
    "Sarah", "Dorothy", "Grace", "Martha", "Jane", "Frances", "Bridget", "Winifred",
    "Cecily", "Margery", "Constance", "Juliana", "Petronilla", "Sybil", "Lettice", "Thomasina",
    "Avice", "Rohesia", "Clarice", "Dionisia", "Gunhilda", "Helewise", "Idonea", "Aldith"
]

SURNAMES = [
    "Smith", "Fletcher", "Cooper", "Miller", "Baker", "Thatcher", "Fisher", "Shepherd",
    "Carter", "Taylor", "Turner", "Walker", "Mason", "Wright", "Webb", "Wood",
    "Green", "Brown", "White", "Black", "Young", "Old", "Long", "Short",
    "Hill", "Brook", "Field", "Stone", "Ford", "Wells", "Cross", "Lane",
    "Marsh", "Dale", "Holt", "Grove", "Church", "Bridge", "Tower", "Hall"
]

# Bynames for disambiguating duplicates
LOCATIONAL_BYNAMES = [
    "of the Hill", "of the Dale", "by the Brook", "at the Cross", "of the Green",
    "by the Mill", "at the Ford", "of the Marsh", "by the Church", "of the Wood",
    "at the Bridge", "by the Well", "of the Field", "at the Tower", "by the Gate"
]

DESCRIPTIVE_BYNAMES = [
    "the Elder", "the Younger", "the Red", "the Fair", "the Tall", "the Short",
    "the Stout", "the Wise", "the Bold", "the Meek", "the Swift", "the Strong",
    "the Quiet", "the Lame", "the One-Eyed"
]

# Track used names for deduplication
_used_names: Dict[str, int] = {}


def generate_unique_name(gender: Gender, rng: np.random.RandomState, 
                         used_names: Dict[str, int]) -> str:
    """
    Generate a unique villager name with medieval bynames for duplicates.
    
    First occurrence: "John Smith"
    Second occurrence: "John Smith the Younger" or "John Smith of the Hill"
    """
    first = rng.choice(MALE_NAMES if gender == Gender.MALE else FEMALE_NAMES)
    last = rng.choice(SURNAMES)
    base_name = f"{first} {last}"
    
    if base_name not in used_names:
        used_names[base_name] = 1
        return base_name
    
    # Name collision - add a byname
    count = used_names[base_name]
    used_names[base_name] = count + 1
    
    # First duplicate gets "the Younger" or "the Elder" treatment
    if count == 1:
        # Retroactively, the first one should have been "the Elder" but we can't rename them
        # So second one gets a byname
        byname = rng.choice(DESCRIPTIVE_BYNAMES + LOCATIONAL_BYNAMES)
    else:
        # Subsequent duplicates get random bynames
        all_bynames = DESCRIPTIVE_BYNAMES + LOCATIONAL_BYNAMES
        byname = rng.choice(all_bynames)
    
    unique_name = f"{first} {last} {byname}"
    
    # In rare cases of triple+ collision with same byname, add a number
    while unique_name in used_names:
        byname = rng.choice(DESCRIPTIVE_BYNAMES + LOCATIONAL_BYNAMES)
        unique_name = f"{first} {last} {byname}"
    
    used_names[unique_name] = 1
    return unique_name


# ============================================================================
# VILLAGE STATE
# ============================================================================

@dataclass
class VillageState:
    """The collective state of the village - tracks macro-level dynamics"""
    
    # Village-level emotional climate
    panic_level: float = 0.2  # 0-1, aggregate fear weighted by social influence
    trust_in_authority: float = 0.7  # 0-1, faith in governance structures
    social_cohesion: float = 0.6  # 0-1, strength of community bonds
    resource_scarcity: float = 0.3  # 0-1, how desperate are people?
    rumor_saturation: float = 0.2  # 0-1, how much unverified info is circulating

    # Historical trauma (for hysteresis - these decay very slowly)
    historical_violence: float = 0.0  # Accumulated violence memory
    total_executions: int = 0  # All-time execution count
    recent_execution_timestamps: List[int] = field(default_factory=list)  # For velocity calc
    
    # External stressors
    active_stressors: Dict[str, int] = field(default_factory=dict)  # stressor_type -> days_remaining
    base_resource_scarcity: float = 0.3  # Pre-stressor baseline
    
    # Conformity pressure (elevated during stressors)
    conformity_pressure: float = 0.5  # 0-1, social pressure to conform

    # Time tracking
    timestep: int = 0

    # Entity storage
    villagers: Dict[int, Villager] = field(default_factory=dict)
    relationships: Dict[Tuple[int, int], Relationship] = field(default_factory=dict)

    # Event tracking
    recent_accusations: List[Tuple[int, int, int]] = field(default_factory=list)  # (timestep, accuser, accused)
    recent_deaths: List[Tuple[int, int]] = field(default_factory=list)  # (timestep, villager_id)
    trial_queue: List[int] = field(default_factory=list)  # villagers awaiting trial


# ============================================================================
# VILLAGE INITIALIZATION
# ============================================================================

def generate_personality(rng: np.random.RandomState) -> Personality:
    """Generate a personality profile with realistic distributions"""
    return Personality(
        openness=rng.beta(2, 2),
        conscientiousness=rng.beta(2, 2),
        extraversion=rng.beta(2, 2),
        agreeableness=rng.beta(2, 2),
        neuroticism=rng.beta(2, 2),
        machiavellianism=rng.beta(1.5, 5),  # Most people are low
        narcissism=rng.beta(1.5, 5),
        psychopathy=rng.beta(1, 8),  # Very rare
        envy=rng.beta(2, 3),
        greed=rng.beta(2, 3),
        wrath=rng.beta(2, 3),
        lust=rng.beta(2, 2),
        pride=rng.beta(2, 3),
        gluttony=rng.beta(2, 3),
        sloth=rng.beta(2, 3),
    )


def generate_emotional_state(rng: np.random.RandomState) -> EmotionalState:
    """Generate initial emotional state"""
    return EmotionalState(
        fear=rng.beta(2, 5),
        anger=rng.beta(2, 5),
        joy=rng.beta(3, 2),
        shame=rng.beta(2, 5),
        grief=rng.beta(1.5, 5),
        desire=rng.beta(2, 3),
        resentment=rng.beta(2, 5),
        hope=rng.beta(3, 2),
        despair=rng.beta(2, 5),
    )


def assign_occupation(age: int, gender: Gender, status: float, rng: np.random.RandomState) -> Occupation:
    """Assign occupation based on demographics - historically accurate inequality included"""
    if age < 16:
        return Occupation.FARMER  # Child labor, yay feudalism

    if status > 0.8:
        return Occupation.NOBILITY if rng.random() < 0.7 else Occupation.CLERGY

    if gender == Gender.FEMALE:
        weights = [0.5, 0.15, 0.15, 0.1, 0.0, 0.0, 0.05, 0.0, 0.05, 0.0, 0.0]
    else:
        weights = [0.4, 0.05, 0.0, 0.15, 0.1, 0.02, 0.15, 0.05, 0.05, 0.01, 0.02]

    occupations = list(Occupation)
    return rng.choice(occupations, p=np.array(weights) / sum(weights))


def assign_marital_status(age: int, gender: Gender, rng: np.random.RandomState) -> MaritalStatus:
    """Assign marital status based on demographics"""
    if age < 16:
        return MaritalStatus.SINGLE
    elif age < 25:
        return MaritalStatus.MARRIED if rng.random() < 0.4 else MaritalStatus.SINGLE
    elif age < 50:
        probs = [0.15, 0.7, 0.1, 0.05]
        return rng.choice(list(MaritalStatus), p=probs)
    else:
        probs = [0.1, 0.5, 0.35, 0.05]
        return rng.choice(list(MaritalStatus), p=probs)


def generate_villager(node_id: int, rng: np.random.RandomState, 
                      used_names: Dict[str, int]) -> Villager:
    """Generate a single villager with all attributes"""

    # Age distribution (skewed young, medieval demographics)
    age = int(rng.beta(2, 5) * 70 + 15)  # 15-85, most are young

    # Gender (roughly 50/50, slight female bias)
    gender = Gender.FEMALE if rng.random() < 0.52 else Gender.MALE

    # Social status (power law distribution, most are peasants)
    social_status = min(rng.pareto(3) * 0.1, 1.0)

    # Wealth (correlated with status but noisier)
    wealth = np.clip(social_status * 0.7 + rng.beta(2, 5) * 0.3, 0, 1)

    # Demographics-based attributes
    occupation = assign_occupation(age, gender, social_status, rng)
    marital_status = assign_marital_status(age, gender, rng)

    # Personality and emotions
    personality = generate_personality(rng)
    emotional_state = generate_emotional_state(rng)

    return Villager(
        id=node_id,
        name=generate_unique_name(gender, rng, used_names),
        age=age,
        gender=gender,
        occupation=occupation,
        beauty=rng.beta(2, 2),
        personality=personality,
        emotional_state=emotional_state,
        social_status=social_status,
        wealth=wealth,
        marital_status=marital_status,
        conformity_score=rng.beta(5, 2),  # Most people are conformist
        has_suspicious_marks=rng.random() < 0.1,
        owns_cat=rng.random() < 0.15,
        lives_alone=marital_status == MaritalStatus.WIDOWED and rng.random() < 0.3,
        church_attendance=rng.beta(3, 2),
        rationality=rng.beta(3, 2),
        pain_tolerance=rng.beta(2, 2),
    )


def generate_relationships(state: VillageState, rng: np.random.RandomState):
    """Build the social network structure"""
    ids = list(state.villagers.keys())
    n = len(ids)

    # 1. Family clusters
    num_families = n // 5
    families = [[] for _ in range(num_families)]
    for vid in ids:
        families[rng.randint(0, num_families)].append(vid)

    for family in families:
        for i, v1 in enumerate(family):
            for v2 in family[i + 1:]:
                rel = Relationship(source=v1, target=v2)
                rel.add_relationship_type(
                    RelationshipType.FAMILY,
                    strength=rng.uniform(0.7, 1.0),
                    trust=rng.uniform(0.6, 1.0)
                )
                state.relationships[(v1, v2)] = rel
                state.relationships[(v2, v1)] = rel

    # 2. Economic relationships (preferential attachment to high-status)
    statuses = np.array([state.villagers[i].social_status for i in ids])
    for vid in ids:
        num_conn = max(1, int(rng.poisson(2)))
        probs = statuses + 0.1
        probs[vid] = 0
        probs = probs / probs.sum()
        targets = rng.choice(ids, size=min(num_conn, n - 1), replace=False, p=probs)
        for tid in targets:
            if (vid, tid) not in state.relationships:
                rel = Relationship(source=vid, target=tid)
                rel.add_relationship_type(
                    RelationshipType.ECONOMIC,
                    strength=rng.uniform(0.3, 0.7)
                )
                state.relationships[(vid, tid)] = rel

    # 3. Rivalries (between similar-status nodes)
    for vid in ids:
        if rng.random() < 0.15:
            v = state.villagers[vid]
            candidates = [
                i for i in ids if i != vid
                and abs(state.villagers[i].social_status - v.social_status) < 0.2
                and abs(state.villagers[i].wealth - v.wealth) < 0.3
            ]
            if candidates:
                rival = rng.choice(candidates)
                if (vid, rival) in state.relationships:
                    state.relationships[(vid, rival)].add_relationship_type(
                        RelationshipType.RIVALRY,
                        strength=rng.uniform(0.4, 0.8),
                        competition_intensity=rng.uniform(0.5, 1.0)
                    )
                else:
                    rel = Relationship(source=vid, target=rival)
                    rel.add_relationship_type(
                        RelationshipType.RIVALRY,
                        strength=rng.uniform(0.4, 0.8),
                        competition_intensity=rng.uniform(0.5, 1.0)
                    )
                    state.relationships[(vid, rival)] = rel


def initialize_village(size: int = 50, seed: int = 42) -> VillageState:
    """Create a village full of potential victims and perpetrators"""
    rng = np.random.RandomState(seed)
    state = VillageState()
    
    # Track used names to ensure uniqueness
    used_names: Dict[str, int] = {}

    # Generate villagers
    for i in range(size):
        villager = generate_villager(i, rng, used_names)
        state.villagers[i] = villager

    # Generate relationships
    generate_relationships(state, rng)

    return state
