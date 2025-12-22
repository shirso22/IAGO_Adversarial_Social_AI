"""Utility functions we will call frequently"""
    
    def get_node_vulnerability(self, node_id: int) -> float:
        """Calculate how vulnerable a node is to accusation"""

        node = self.G.nodes[node_id]

        # Base vulnerability factors
        v = 0.0

        # Gender (the depressing coefficient)
        v += 0.3 if node['gender'] == Gender.FEMALE else 0.0

        # Age effects (U-shaped)
        age_factor = abs(node['age'] - 40) / 40  # Higher at extremes
        v += age_factor * 0.2

        # Marital status
        if node['marital_status'] == MaritalStatus.WIDOWED:
            v += 0.3
        elif node['marital_status'] == MaritalStatus.SINGLE and node['age'] > 30:
            v += 0.2

        # Social status (inverse relationship)
        v += (1 - node['social_status']) * 0.4

        # Conformity (inverse)
        v += (1 - node['conformity_score']) * 0.3

        # Network isolation
        degree = self.G.degree(node_id)
        avg_degree = np.mean([d for n, d in self.G.degree()])
        isolation = max(0, (avg_degree - degree) / avg_degree)
        v += isolation * 0.3

        # Suspicious attributes
        if node['owns_cat']:
            v += 0.2
        if node['lives_alone']:
            v += 0.15
        if node['has_suspicious_marks']:
            v += 0.1

        # Previous accusations (feedback loop!)
        v += min(node['times_accused_recent'] * 0.2, 0.5)

        # Occupation risk
        if node['occupation'] in [Occupation.HEALER, Occupation.MIDWIFE]:
            v += 0.25

        return np.clip(v, 0, 1)

    def get_accusation_credibility(self, accuser_id: int, accused_id: int) -> float:
        """How believable is this accusation?"""

        accuser = self.G.nodes[accuser_id]
        accused = self.G.nodes[accused_id]

        credibility = 0.5  # Base

        # Accuser's social status helps
        credibility += accuser['social_status'] * 0.3

        # Accuser's gender (men more believed, sadly)
        if accuser['gender'] == Gender.MALE:
            credibility += 0.2

        # Accused's vulnerability makes accusations more believable
        credibility += self.get_node_vulnerability(accused_id) * 0.3

        # If there's a rivalry edge, more credible (motive is clear)
        if self.G.has_edge(accuser_id, accused_id):
            edge = self.G[accuser_id][accused_id]
            if edge.get('relationship_type') == RelationshipType.RIVALRY:
                credibility += 0.2

        # Diminishing returns on multiple accusations by same person
        num_previous = len(accuser['accusations_made'])
        credibility -= min(num_previous * 0.05, 0.3)

"""Interaction module or how villagers interact with each other and how interactions affect psychological state"""


#Baseline level for an individual villager
def get_base_interaction_vector(villager: Villager) -> np.ndarray

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


#Who you are interacting with affects the interaction
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

    # FLIRT: Suppressed by large status differential (unless narcissist/Machiavellian)
    if actor.gender != target.gender:  # Your "only opposite sex" constraint
        status_penalty = abs(status_diff) * (1 - actor.personality.narcissism)
        modifiers[3] *= max(0.1, 1 - status_penalty)
    else:
        modifiers[3] = 0  # Same-sex flirting disabled per your design

    # INSULT: Suppressed toward high-status unless angry/Machiavellian
    # Encouraged toward low-status if narcissistic/low-agreeableness
    if status_diff > 0:
        # Target is higher status - dangerous to insult
        safety_factor = (
            actor.personality.psychopathy * 0.3 +
            actor.emotional_state.anger * 0.4
        )
        modifiers[4] *= safety_factor
    else:
        # Target is lower status - safer to insult if you're an asshole
        cruelty_factor = (
            (1 - actor.personality.agreeableness) * 0.5 +
            actor.personality.narcissism * 0.3
        )
        modifiers[4] *= (1 + cruelty_factor)

    # RESPECT: High for high-status targets, low for low-status
    # Unless you're narcissistic (respect no one) or resentful
    base_respect = max(0, status_diff)
    narcissism_penalty = actor.personality.narcissism * 0.7
    resentment_penalty = actor.emotional_state.resentment * 0.5
    modifiers[5] *= (base_respect * (1 - narcissism_penalty - resentment_penalty) + 0.2)

    # Relationship quality affects everything
    if relationship_state > 0.5:
        # Positive relationship: more chill, less formal, less insult
        modifiers[0] *= 1.5
        modifiers[2] *= 0.5
        modifiers[4] *= 0.2
    elif relationship_state < -0.5:
        # Negative relationship: less chill, more formal, more insult
        modifiers[0] *= 0.3
        modifiers[2] *= 1.3
        modifiers[4] *= 2.0

    return modifiers

#Overall macro context of the village matters too
def compute_contextual_modifiers(
    village_state: Village,
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

    # Low institutional trust makes people more guarded (more formal, less respect)
    trust_factor = village_state.trust_in_authority
    modifiers[2] *= (1 + (1 - trust_factor) * 0.3)
    modifiers[5] *= trust_factor

    # If target is currently accused, interaction changes dramatically
    if target.is_accused_currently:
        modifiers[0] *= 0.2  # Avoid being friendly with accused (guilt by association)
        modifiers[2] *= 1.5  # More formal/distant
        modifiers[4] *= 1.8  # More acceptable to insult
        modifiers[5] *= 0.3  # Less respect

    # If actor is currently accused, they're desperate/defensive
    if actor.is_accused_currently:
        modifiers[0] *= 1.5  # Try to be friendly (seeking allies)
        modifiers[5] *= 1.8  # Show excessive respect (appeasement)
        modifiers[4] *= 0.1  # Avoid insults (can't afford more enemies)

    return modifiers


#Putting it all together in a final interaction vector
def compute_interaction_vector(
    actor: Villager,
    target: Villager,
    village_state: Village
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
    interaction_vector = interaction_vector / interaction_vector.sum()

    return interaction_vector

#State updates after interaction
def apply_interaction_effects(
    actor: Villager,
    target: Villager,
    interaction_vector: np.ndarray,
    village_state: Village
):
    """
    Update villager states based on interaction vector.
    vector = [chill, gossip, formal, flirt, insult, respect]
    """
    chill, gossip, formal, flirt, insult, respect = interaction_vector

    # === Target's emotional updates ===

    # CHILL increases joy, decreases stress slightly
    target.emotional_state.joy += chill * 0.1
    target.stress = max(0, target.stress - chill * 0.05)

    # GOSSIP increases knowledge but also potentially fear/anxiety
    # (depends on what's being gossiped about - for now, slight stress increase)
    target.emotional_state.fear += gossip * 0.05 * village_state.panic_level
    target.stress += gossip * 0.03

    # FORMAL doesn't affect emotions much (neutral interaction)
    # Maybe slight decrease in joy if person wanted warmth
    target.emotional_state.joy -= formal * 0.02

    # FLIRT affects romantic attraction, desire, joy
    if flirt > 0.1:
        # Update romantic attraction (bidirectional often, but depends on target's receptiveness)
        target_receptiveness = (
            target.personality.lust * 0.4 +
            target.emotional_state.desire * 0.3 +
            (1 - target.marital_status == MaritalStatus.MARRIED) * 0.3
        )
        if target_receptiveness > 0.4:
            target.emotional_state.romantic_attractions[actor.id] = \
                target.emotional_state.romantic_attractions.get(actor.id, 0) + flirt * 0.2
            target.emotional_state.joy += flirt * 0.1
            target.emotional_state.desire += flirt * 0.15
        else:
            # Unwanted flirting causes shame/discomfort
            target.emotional_state.shame += flirt * 0.1

    # INSULT increases anger, shame, decreases joy
    target.emotional_state.anger += insult * 0.3
    target.emotional_state.shame += insult * 0.2
    target.emotional_state.joy = max(0, target.emotional_state.joy - insult * 0.2)
    target.reputation -= insult * 0.02  # Public insults damage reputation

    # RESPECT increases confidence, decreases shame
    target.emotional_state.shame = max(0, target.emotional_state.shame - respect * 0.1)
    # Could model a "confidence" stat, or just reduce stress
    target.stress = max(0, target.stress - respect * 0.05)

    # === Relationship updates ===

    # Chill and respect increase loyalty
    loyalty_change = (chill * 0.1 + respect * 0.15)
    target.emotional_state.loyalties[actor.id] = \
        target.emotional_state.loyalties.get(actor.id, 0) + loyalty_change

    # Insult increases hatred, decreases loyalty
    hatred_change = insult * 0.2
    target.emotional_state.hatreds[actor.id] = \
        target.emotional_state.hatreds.get(actor.id, 0) + hatred_change
    target.emotional_state.loyalties[actor.id] = \
        max(0, target.emotional_state.loyalties.get(actor.id, 0) - insult * 0.15)

    # === Actor's updates (smaller, self-reflective) ===

    # Insulting someone might make high-agreeableness people feel guilt
    if insult > 0.3:
        guilt = insult * actor.personality.agreeableness
        actor.emotional_state.shame += guilt * 0.1

    # Successful flirting increases actor's joy/desire too
    if flirt > 0.3:
        actor.emotional_state.joy += flirt * 0.08
        actor.emotional_state.romantic_attractions[target.id] = \
            actor.emotional_state.romantic_attractions.get(target.id, 0) + flirt * 0.15

    # Showing respect to high-status individuals might make you feel secure
    if respect > 0.5 and target.social_status > actor.social_status:
        actor.stress = max(0, actor.stress - respect * 0.03)

#Group interactions
def compute_group_interaction(
    participants: List[Villager],
    village_state: Village
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
    village_state: Village
):
    """
    Apply all pairwise interactions, potentially with dampening for group settings.
    (In a group, individual interactions might be less intense than 1-on-1)
    """
    dampening_factor = 1.0 / np.sqrt(len(participants))  # Larger groups = less individual impact

    for (actor_id, target_id), vector in interactions.items():
        actor = village_state.villagers[actor_id]
        target = village_state.villagers[target_id]

        # Apply with dampening
        dampened_vector = vector * dampening_factor
        apply_interaction_effects(actor, target, dampened_vector, village_state)

        return np.clip(credibility, 0, 1)
