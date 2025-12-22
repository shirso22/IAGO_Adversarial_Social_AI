"""The Village"""

import networkx as nx
import numpy as np
from typing import List, Tuple

class VillageGraph:
    """Your simulation of historical awfulness"""

    #Variables that track village/society level dynamics
    panic_level: float  # 0-1, aggregate fear weighted by social influence
    trust_in_authority: float  # 0-1, faith in governance structures
    #judicial_legitimacy: float  # Faith in magistrates/courts
    #religious_authority: float  # Faith in clergy/church
    #police_loyalty: float       # Will enforcement apparatus obey orders?
    #economic_stability: float   # Faith in property rights/markets
    social_cohesion: float  # 0-1, strength of community bonds
    resource_scarcity: float  # 0-1, how desperate are people?
    rumor_saturation: float  # 0-1, how much unverified info is circulating
    #information_quality: float  # 0-1, accuracy of shared information
    #recent_execution_count: int  # Executions in last N timesteps
    #execution_velocity: float    # Change in execution rate


    def __init__(self,
                 village_size: int = 100,
                 random_seed: int = 42):

        self.G = nx.MultiDiGraph()  # or DiGraph if you want directional edges
        self.village_size = village_size
        self.rng = np.random.RandomState(random_seed)
        self.timestep = 0

        # Track global state
        self.collective_stress = 0.5
        self.recent_deaths = []
        self.recent_accusations = []

    def initialize_village(self):
        """Create your doomed population"""

        # Add nodes with attributes
        for i in range(self.village_size):
            villager = self._generate_villager(i)
            self.G.add_node(i, **villager.__dict__)

        # Generate social network structure
        self._generate_relationships()

        # Validate graph (optional but useful)
        self._validate_graph()

    def _generate_villager(self, node_id: int) -> Villager:
        """Generate a villager with realistic attribute distributions"""

        # Age distribution (skewed young, medieval demographics)
        age = int(self.rng.beta(2, 5) * 70 + 15)  # 15-85, most are young

        # Gender (roughly 50/50, slight male bias due to war deaths)
        gender = Gender.FEMALE if self.rng.random() < 0.52 else Gender.MALE

        # Social status (power law distribution, most are peasants)
        social_status = self.rng.pareto(3) * 0.1
        social_status = min(social_status, 1.0)

        # Wealth (correlated with status but noisier)
        wealth = social_status * 0.7 + self.rng.beta(2, 5) * 0.3
        wealth = np.clip(wealth, 0, 1)

        # Occupation (depends on age, gender, status)
        occupation = self._assign_occupation(age, gender, social_status)

        # Marital status (depends on age and gender)
        marital_status = self._assign_marital_status(age, gender)

        # Conformity (most people are pretty conformist)
        conformity = self.rng.beta(5, 2)  # Skewed toward high conformity

        # Suspicious attributes (rare)
        has_marks = self.rng.random() < 0.1
        owns_cat = self.rng.random() < 0.15
        lives_alone = marital_status == MaritalStatus.WIDOWED and self.rng.random() < 0.3

        return Villager(
            id=node_id,
            name=self._generate_name(gender),
            age=age,
            gender=gender,
            social_status=social_status,
            wealth=wealth,
            occupation=occupation,
            marital_status=marital_status,
            conformity_score=conformity,
            has_suspicious_marks=has_marks,
            owns_cat=owns_cat,
            lives_alone=lives_alone,
            church_attendance=self.rng.beta(3, 2),  # Most attend regularly
        )

    def _generate_relationships(self):
        """Build the social network structure"""

        # Strategy: Combine multiple network generation models

        # 1. Family ties (small-world, high clustering)
        self._add_family_networks()

        # 2. Economic relationships (preferential attachment to high-status)
        self._add_economic_networks()

        # 3. Geographic proximity (nearest neighbors)
        self._add_neighbor_networks()

        # 4. Random weak ties (for realism)
        self._add_random_ties()

        # 5. Rivalries (sparse, strategic)
        self._add_rivalries()

    def _add_family_networks(self):
        """Create family clusters with high internal connectivity"""

        # Group villagers into ~20 family units
        num_families = self.village_size // 5
        families = [[] for _ in range(num_families)]

        for node in self.G.nodes():
            family_id = self.rng.randint(0, num_families)
            families[family_id].append(node)

        # Connect family members
        for family in families:
            if len(family) < 2:
                continue

            # Create family bonds (high strength, high trust)
            for i, node1 in enumerate(family):
                for node2 in family[i+1:]:
                    self.G.add_edge(
                        node1, node2,
                        relationship_type=RelationshipType.FAMILY,
                        strength=self.rng.uniform(0.7, 1.0),
                        trust=self.rng.uniform(0.6, 1.0),
                        is_directed=False
                    )

    def _add_economic_networks(self):
        """High-status nodes attract more economic connections"""

        nodes = list(self.G.nodes())
        statuses = [self.G.nodes[n]['social_status'] for n in nodes]

        # Preferential attachment weighted by status
        for node in nodes:
            num_connections = max(1, int(self.rng.poisson(3)))

            # Higher status nodes attract connections
            probs = np.array(statuses) + 0.1  # Ensure non-zero
            probs[node] = 0  # Can't connect to self
            probs = probs / probs.sum()

            targets = self.rng.choice(nodes, size=num_connections,
                                     replace=False, p=probs)

            for target in targets:
                if not self.G.has_edge(node, target):
                    self.G.add_edge(
                        node, target,
                        relationship_type=RelationshipType.ECONOMIC,
                        strength=self.rng.uniform(0.3, 0.7),
                        trust=self.rng.uniform(0.4, 0.8)
                    )

    def _add_neighbor_networks(self):
        """Everyone knows their neighbors"""
        # Implementation left as exercise because you get the idea
        pass

    def _add_random_ties(self):
        """Weak ties that bridge communities"""
        pass

    def _add_rivalries(self):
        """Economic competition edges between similar-status nodes"""

        nodes = list(self.G.nodes())

        for node in nodes:
            # Small chance of having a rival
            if self.rng.random() < 0.2:
                status = self.G.nodes[node]['social_status']
                wealth = self.G.nodes[node]['wealth']

                # Find similar-status nodes (potential rivals)
                candidates = [
                    n for n in nodes
                    if n != node
                    and abs(self.G.nodes[n]['social_status'] - status) < 0.2
                    and abs(self.G.nodes[n]['wealth'] - wealth) < 0.3
                ]

                if candidates:
                    rival = self.rng.choice(candidates)

                    self.G.add_edge(
                        node, rival,
                        relationship_type=RelationshipType.RIVALRY,
                        strength=self.rng.uniform(0.4, 0.8),
                        trust=self.rng.uniform(0.1, 0.3),  # Low trust
                        competition_intensity=self.rng.uniform(0.5, 1.0)
                    )
