# IAGO: Adversarial Social Dynamics Simulation

> *"I am not what I am."* — Iago, Othello

An agent-based simulation of a medieval village during the witch hunt era, designed as a training environment for an adversarial reinforcement learning agent. The simulator models social dynamics, information cascades, and institutional breakdown to study how communities destabilize under pressure.

---

## The Simulator

This project models a medieval village as a social graph where nodes represent villagers and edges represent relationships.

### Agent Model

**Villagers** are modeled with:
- **Personality**: Big Five traits, Dark Triad tendencies, and Seven Deadly Sins
- **Emotional State**: Fear, anger, grief, desire, resentment, and directed emotions (hatreds, loyalties, suspicions) toward specific individuals
- **Social Position**: Status, wealth, reputation, and conformity score

**Relationships** form a multi-edge graph supporting:
- Family bonds, marriages, friendships
- Economic ties and rivalries  
- Patronage networks (protection in exchange for loyalty)

**Village Dynamics** capture macro-level phenomena:
- Panic propagation and decay
- Institutional trust
- Rumor saturation
- External stressors (plague, famine, inquisitor visits)

### Simulation Mechanics

Each timestep, villagers evaluate available actions through a utility function combining personality, emotional state, relationships, and environmental factors. Available actions include:

- Accusing someone of witchcraft
- Spreading rumors, forming alliances, seeking protection
- Testifying for or against the accused
- Courting romantic partners
- Attending church

Accusations trigger trials. Trials can trigger **chain accusations** where the accused names "accomplices" under duress, potentially creating cascade effects.

---

## Simulation Results

### Bimodal Outcomes

![Distribution of Final Survivor Fraction](The_Village/sim_outputs/Ensemble%20Survivor%20Fraction%20Histogram.png)

Ensemble simulations reveal a bimodal distribution: villages either survive mostly intact (~95%+ survival) or collapse catastrophically (~30% survival). Fewer than 5% of simulations land between 40-80% survival.

This pattern suggests a **phase transition** where small differences in initial conditions cascade into radically different outcomes.

### Early Divergence

![Average Panic Level Over Time by Outcome](The_Village/sim_outputs/Mean%20Panic%20Level.png)

Trajectory divergence occurs within the first 25 days:

- **High-survival runs** (green, n=55): Panic decays from 0.35 to near-zero by day 50. The system reaches equilibrium without sustained accusations.

- **Low-survival runs** (red, n=39): Panic rises to 0.9+ within two weeks and remains elevated. Circuit breakers (trial capacity limits, system fatigue, elite intervention) slow the cascade but cannot stop it.

The narrow confidence bands indicate that once a trajectory is established, outcomes become highly predictable.

---

## IAGO: The Adversarial Agent

*Under active development*

Named after Shakespeare's villain, IAGO is a reinforcement learning agent that operates as an ordinary villager with the objective of destabilizing the community while remaining undetected.

### Design Challenges

IAGO must:

1. **Infer hidden states**: Villager personalities and emotional states are only partially observable through actions
2. **Identify leverage points**: The bimodal outcome distribution suggests critical nodes and timing windows exist
3. **Exploit social mechanics**: Gossip, strategic accusations, alliance manipulation, testimony timing
4. **Maintain cover**: Avoid the credibility loss and suspicion that come from being too active or too connected to the executed

### Motivation

The mechanics being modeled—information cascades, trust network exploitation, strategic ambiguity—appear in many contexts beyond historical witch trials. IAGO provides a framework for understanding these dynamics through adversarial optimization.

---

## Technical Documentation

- **[Reference.md](The_Village/files/Reference.md)**: Complete system architecture, utility functions, trial mechanics, and configuration guide
- **[Simulation Runner README](The_Village/run/README.md)**: Usage guide for running simulations with presets and batch execution

### Quick Start

```bash
# Run with defaults
python simulation_runner.py

# Run a high-panic scenario
python simulation_runner.py --preset chaos

# Batch run for statistical analysis
python simulation_runner.py --batch 100 --quiet

# Parameter sweep
python simulation_runner.py --sweep
```

---

## Research Scope & Future Work

This simulator is an experimental environment for studying social dynamics and training adversarial agents.

### Planned Extensions

1. **Adversarial-Defensive Training**: Train competing agents—one attempting to destabilize, another attempting to stabilize the community.

2. **Intervention Strategies**: Evaluate minimum interventions required to shift trajectories, including early warning detection and network hub protection.

3. **Adaptive Villagers**: Implement learning mechanisms so villagers update beliefs about manipulation likelihood based on outcomes.

4. **LLM Integration**: Replace IAGO's learned policy with a language model that articulates manipulation strategies, enabling interpretability research.

5. **Real Network Topologies**: Initialize from empirical social network data to study vulnerability profiles of different community structures.

---

## Installation

```bash
# Clone repository
git clone https://github.com/[your-repo]/iago-simulator.git
cd iago-simulator

# Install dependencies
pip install numpy

# Run simulation
python main.py
```

---

## Citation

```bibtex
@software{iago_simulator,
  title={IAGO: Adversarial Social Dynamics Simulation},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]/iago-simulator}
}
```

---

*"And what's he then that says I play the villain?"*
