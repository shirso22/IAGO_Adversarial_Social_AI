#Defines relationships between villagers. The edges of the village graph essentially.

from enum import Enum

class RelationshipType(Enum):
    FAMILY = "family"  # Blood relations
    MARRIAGE = "marriage"  # Spousal
    FRIENDSHIP = "friendship"  # Social bonds
    ECONOMIC = "economic"  # Business relationships
    RIVALRY = "rivalry"  # Competition/conflict
    PATRONAGE = "patronage"  # Power relationships (directed)
    NEIGHBOR = "neighbor"  # Proximity

dataclass
class Relationship:
    """A single edge that can represent multiple relationship types"""

    source: int
    target: int

    # Instead of single type, store a SET of types
    relationship_types: Set[RelationshipType]  # <-- The fix

    # Aggregate properties across all relationship types
    overall_strength: float  # Combined strength of connection
    overall_trust: float     # Net trust level

    # Type-specific properties (use dicts keyed by type)
    type_strengths: dict[RelationshipType, float]  # Individual strengths
    type_attributes: dict[RelationshipType, dict]  # Type-specific data

    def add_relationship_type(self, rel_type: RelationshipType, strength: float, **kwargs):
        """Add another relationship type to this edge"""
        self.relationship_types.add(rel_type)
        self.type_strengths[rel_type] = strength
        self.type_attributes[rel_type] = kwargs

        # Recalculate aggregate properties
        self._update_aggregates()

    def _update_aggregates(self):
        """Recalculate overall strength/trust from component relationships"""

        if not self.type_strengths:
            self.overall_strength = 0.0
            self.overall_trust = 0.5
            return

        # Overall strength = max of individual strengths (strongest connection matters most)
        self.overall_strength = max(self.type_strengths.values())

        # Trust is trickier - rivalries reduce trust, family increases it
        trust_factors = []
        for rel_type in self.relationship_types:
            if rel_type == RelationshipType.FAMILY:
                trust_factors.append(0.8)
            elif rel_type == RelationshipType.FRIENDSHIP:
                trust_factors.append(0.7)
            elif rel_type == RelationshipType.RIVALRY:
                trust_factors.append(-0.3)  # Negative trust!
            elif rel_type == RelationshipType.ECONOMIC:
                trust_factors.append(0.5)
            else:
                trust_factors.append(0.5)

        # Average trust, clamped to [0, 1]
        self.overall_trust = np.clip(np.mean(trust_factors), 0, 1)

    def has_type(self, rel_type: RelationshipType) -> bool:
        """Check if this edge includes a specific relationship type"""
        return rel_type in self.relationship_types

    def get_competition_intensity(self) -> float:
        """Get rivalry intensity if this is a rivalry edge"""
        if RelationshipType.RIVALRY in self.relationship_types:
            return self.type_attributes[RelationshipType.RIVALRY].get('competition_intensity', 0.0)
        return 0.0
