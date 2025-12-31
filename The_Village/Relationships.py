"""Defines relationships between villagers. The edges of the village graph essentially."""

from dataclasses import dataclass, field
from typing import Set, Dict
from enum import Enum
import numpy as np


class RelationshipType(Enum):
    FAMILY = "family"  # Blood relations
    MARRIAGE = "marriage"  # Spousal
    FRIENDSHIP = "friendship"  # Social bonds
    ECONOMIC = "economic"  # Business relationships
    RIVALRY = "rivalry"  # Competition/conflict
    PATRONAGE = "patronage"  # Power relationships (directed)
    NEIGHBOR = "neighbor"  # Proximity


@dataclass  # <-- THE FIX: this decorator was missing
class Relationship:
    """A single edge that can represent multiple relationship types"""

    source: int
    target: int

    # Instead of single type, store a SET of types
    relationship_types: Set[RelationshipType] = field(default_factory=set)

    # Aggregate properties across all relationship types
    overall_strength: float = 0.5  # Combined strength of connection
    overall_trust: float = 0.5  # Net trust level

    # Type-specific properties (use dicts keyed by type)
    type_strengths: Dict[RelationshipType, float] = field(default_factory=dict)
    type_attributes: Dict[RelationshipType, dict] = field(default_factory=dict)

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
        trust_map = {
            RelationshipType.FAMILY: 0.8,
            RelationshipType.FRIENDSHIP: 0.7,
            RelationshipType.RIVALRY: -0.3,  # Negative trust!
            RelationshipType.ECONOMIC: 0.5,
            RelationshipType.MARRIAGE: 0.9,
            RelationshipType.NEIGHBOR: 0.4,
            RelationshipType.PATRONAGE: 0.6,
        }

        trust_factors = [trust_map.get(t, 0.5) for t in self.relationship_types]

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
