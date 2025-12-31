"""Models personalities and emotions of individual villagers"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Personality:
    """The unchanging core of human nature - for better or worse"""
    
    # Big Five (0-1 scales, 0.5 = average)
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    # Dark Triad (0-1 scales, most people are LOW on these)
    machiavellianism: float = 0.2
    narcissism: float = 0.2
    psychopathy: float = 0.1  # Very low in most people

    # Seven Deadlies (because medieval)
    envy: float = 0.3
    greed: float = 0.3
    wrath: float = 0.3
    lust: float = 0.3
    pride: float = 0.3
    gluttony: float = 0.3
    sloth: float = 0.3


@dataclass
class EmotionalState:
    """Current emotional state - changes constantly, unlike personality"""

    # Acute emotions (high volatility, decay quickly)
    fear: float = 0.3
    anger: float = 0.2
    joy: float = 0.4
    shame: float = 0.2
    grief: float = 0.1
    desire: float = 0.3  # Romantic, financial, carnal, etc.

    # Sustained emotions (medium volatility)
    resentment: float = 0.2  # Accumulated anger, decays slowly
    hope: float = 0.4
    despair: float = 0.2

    # Directed emotions (toward specific individuals)
    romantic_attractions: Dict[int, float] = field(default_factory=dict)
    hatreds: Dict[int, float] = field(default_factory=dict)
    envies: Dict[int, float] = field(default_factory=dict)
    loyalties: Dict[int, float] = field(default_factory=dict)
    suspicions: Dict[int, float] = field(default_factory=dict)

    # Meta-emotional state
    emotional_regulation: float = 0.5  # Current capacity to control emotions
    emotional_exhaustion: float = 0.3  # Depletion from sustained stress
