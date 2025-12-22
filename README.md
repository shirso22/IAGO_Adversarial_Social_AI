
# IAGO: An Adversarial AI agent for social dynamics

## Medieval Village during Witch Hunt Era Simulator

This models a medieval village during the Witch Hunt era as a graph. Nodes represent villagers, edges represent relationships between them (family, marriage, economic, etc). 

Villager personalities are modelled using Big 5, Dark Triad and Seven Deadly Sins framework. Their emotional states (anger, fear, stress, romantic attraction to others, etc) are also tracked. Personality, current emotional state and overall village dynamics
determine actions and interactions with others (accusing someone of witchcraft, publicly defending someone, going to church, gossip, etc) 

Village level dynamics, such as mass hysteria, institutional trust, collective stress, witch trial mechanics, etc are implemented.

The goal is to model a robust social environment, taking inspiration from real historical medieval village dynamics. The intention is to observe emergent behavior, such as cascade accusations that tear the community part, mob rule due to complete erosion of institutional trust, etc


## Adversarial RL based agent: IAGO (under active development)

Named after Shakespeare's famous villan, IAGO is an adversarial RL agent that will be just another villager, but learn to weaponize disinformation, gossip, accusations, etc to systematically destabilize the Village, for example through triggering cascade accusations 
by targeting the right villagers at the right time. 
While staying completely undetected themselves, like his literary counterpart. 

IAGO has to infer hidden psychological, emotional and personality states of villagers through observable actions, while successfully hiding their own true nature.

## Research Scope and Future Work

This simulator is intended as an experimental environment rather than a finished model. Extensions can include:

1) Training and evaluating adversarial and defensive agents. One agent tries to destabilize while the other tries to stabilize the community.

2) Systematic assessment of intervention and detection strategies.

3) Having the Villagers be adaptive and more resilient to manipualation.

4) Integration with language-model interfaces for explicit articulation of manipulation strategies.

