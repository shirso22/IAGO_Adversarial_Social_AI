"""Models personalities and emotions of individual villagers"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum

@dataclass
class Personality:
    # Big Five (0-1 scales, 0.5 = average)
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    # Dark Triad (0-1 scales, most people are LOW on these)
    machiavellianism: float = 0.2  # Typically low
    narcissism: float = 0.2
    psychopathy: float = 0.1  # Very low in most people

    # Seven Deadlies
    envy: float = 0.3
    greed: float = 0.3
    wrath: float = 0.3
    lust: float = 0.3
    pride: float = 0.3
    gluttony: float = 0.3
    sloth: float = 0.3

@dataclass
class EmotionalState:
    """Current emotional state - changes constantly"""

    # Acute emotions (high volatility, decay quickly)
    fear: float  # Spikes from threats, decays
    anger: float  # Spikes from slights, decays
    joy: float  # Positive events, decays
    shame: float  # Social failures, decays
    grief: float  # Losses, decays slowly
    desire: float # Romantic, financial, carnal, etc

    # Sustained emotions (medium volatility)
    resentment: float  # Accumulated anger, decays slowly
    hope: float  # Sustained positive outlook
    despair: float  # Sustained hopelessness

    # Directed emotions (toward specific individuals)
    romantic_attractions: Dict[int, float]  # node_id -> attraction strength
    hatreds: Dict[int, float]  # node_id -> hatred intensity
    envies: Dict[int, float]  # node_id -> envy level
    loyalties: Dict[int, float]  # node_id -> loyalty strength
    suspicions: Dict[int, float]  # node_id -> suspicion level

    # Meta-emotional state
    emotional_regulation: float  # Current capacity to control emotions
    emotional_exhaustion: float  # Depletion from sustained stress
