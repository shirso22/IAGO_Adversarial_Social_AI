import numpy as np
from typing import List, Dict
import sys 

# ============================================================================
# SIMULATION CONFIGURATION
# ============================================================================
# Adjust these values to customize your particular flavor of mass hysteria

VILLAGE_POPULATION = 50          # Number of villagers (10-200 reasonable range)
SIMULATION_LENGTH = 200          # Maximum days to simulate
RANDOM_SEED = 100                 # For reproducibility 
INITIAL_PANIC = 0.3             # Starting panic level (0.0-1.0)
VERBOSE_OUTPUT = True            # Print detailed event log
PRINT_TO_CONSOLE = True         # Set to True to print full simulation output to console

# Termination thresholds
MIN_SURVIVAL_RATE = 0.30         # Stop if population drops below this fraction
PEACE_THRESHOLD_DAYS = 30        # Days of low panic before declaring peace
PEACE_PANIC_LEVEL = 0.05         # Panic must be below this for peace

# ============================================================================

# Import from our modules
from Villager import Villager, Gender, MaritalStatus, Occupation
from Village import VillageState, initialize_village
from Relationships import Relationship, RelationshipType
from Actions import ActionType, Action
from ActionSelection import select_action, get_testimony_actions
from ActionExecution import execute_action
from Utils import (
    get_vulnerability,
    get_accusation_credibility,
    update_village_state,
    update_villager_states,
)


# ============================================================================
# TRIAL SYSTEM
# ============================================================================

def gather_testimony(accused_id: int, state: VillageState, rng: np.random.RandomState):
    """
    Before a trial, give villagers a chance to testify for or against.
    This happens separately from the main action loop.
    """
    accused = state.villagers.get(accused_id)
    if not accused or not accused.is_on_trial:
        return

    # Each villager gets a chance to testify
    for villager in state.villagers.values():
        if villager.id == accused_id or not villager.is_alive:
            continue

        testimony = get_testimony_actions(villager, accused_id, state, rng)
        if testimony:
            action_type, target_id = testimony
            execute_action(villager, action_type, target_id, state, rng, verbose=True)


def conduct_trial(accused_id: int, state: VillageState, rng: np.random.RandomState) -> bool:
    """Conduct a trial. Returns True if convicted."""

    accused = state.villagers[accused_id]
    accused.is_on_trial = True

    print(f"\n  ⚖️ TRIAL: {accused.name} stands accused of witchcraft")

    # Gather testimony
    gather_testimony(accused_id, state, rng)

    # Base conviction probability
    guilt_score = 0.3

    # Vulnerability increases conviction
    guilt_score += get_vulnerability(accused, state) * 0.25

    # Number of accusers matters
    num_accusers = len(set(accused.accusations_received))
    guilt_score += min(num_accusers * 0.08, 0.25)

    # Social status provides protection
    guilt_score -= accused.social_status * 0.2

    # Count testimony
    testimony_against = state.__dict__.get('testimony_against', {}).get(accused_id, [])
    testimony_for = state.__dict__.get('testimony_for', {}).get(accused_id, [])
    defenders = state.__dict__.get('defenders', {}).get(accused_id, [])

    # Testimony and defenders affect outcome
    guilt_score += len(testimony_against) * 0.08
    guilt_score -= len(testimony_for) * 0.06
    guilt_score -= len(defenders) * 0.05

    # Weigh testimony by social status
    for witness_id in testimony_against:
        witness = state.villagers.get(witness_id)
        if witness:
            guilt_score += witness.social_status * 0.03

    for witness_id in testimony_for:
        witness = state.villagers.get(witness_id)
        if witness:
            guilt_score -= witness.social_status * 0.03

    # Panic increases convictions
    guilt_score += state.panic_level * 0.15

    # Trust in authority affects fair trials (high trust = more fair = lower conviction)
    guilt_score -= (state.trust_in_authority - 0.5) * 0.1

    # Gathered evidence against accused
    total_evidence = sum(
        v.__dict__.get('gathered_evidence', {}).get(accused_id, 0)
        for v in state.villagers.values()
    )
    guilt_score += min(total_evidence * 0.5, 0.2)

    guilt_score = np.clip(guilt_score, 0.1, 0.9)

    convicted = rng.random() < guilt_score

    # Clean up trial state
    accused.is_on_trial = False
    accused.is_accused_currently = False

    # Clear testimony records for this trial
    if 'testimony_against' in state.__dict__ and accused_id in state.__dict__['testimony_against']:
        del state.__dict__['testimony_against'][accused_id]
    if 'testimony_for' in state.__dict__ and accused_id in state.__dict__['testimony_for']:
        del state.__dict__['testimony_for'][accused_id]
    if 'defenders' in state.__dict__ and accused_id in state.__dict__['defenders']:
        del state.__dict__['defenders'][accused_id]

    if convicted:
        print(f"  💀 {accused.name} is found GUILTY and executed!")
        accused.is_alive = False
        state.recent_deaths.append((state.timestep, accused_id))
        state.total_executions += 1
        state.recent_execution_timestamps.append(state.timestep)

        # HISTORICAL VIOLENCE - this execution leaves a scar on the village
        state.historical_violence = min(1.0, state.historical_violence + 0.03)  # Was 0.1

        # Aftermath
        state.panic_level = min(1.0, state.panic_level + 0.02)
        state.social_cohesion = max(0.0, state.social_cohesion - 0.05)

        # PATRON COLLAPSE - if executed was a patron, cascade effects
        if accused.dependents:
            from Utils import handle_patron_collapse
            print(f"    ⚠️ {accused.name}'s {len(accused.dependents)} dependents lose protection!")
            handle_patron_collapse(accused, state)

        # Clear accused from anyone's patron
        for v in state.villagers.values():
            if v.patron_id == accused_id:
                v.patron_id = None
                v.emotional_state.fear = min(1.0, v.emotional_state.fear + 0.2)

        # TRAUMA - everyone who witnessed is affected
        for v in state.villagers.values():
            if v.is_alive and v.id != accused_id:
                # Witnessing execution increases trauma
                v.witnessed_executions += 1
                v.trauma_score = min(1.0, v.trauma_score + 0.05)
                v.emotional_state.fear = min(1.0, v.emotional_state.fear + 0.05)

        # Traumatize family and allies more severely
        for (s, t), rel in state.relationships.items():
            if s == accused_id or t == accused_id:
                other_id = t if s == accused_id else s
                other = state.villagers.get(other_id)
                if other and other.is_alive:
                    if RelationshipType.FAMILY in rel.relationship_types:
                        other.emotional_state.grief = min(1.0, other.emotional_state.grief + 0.5)
                        other.emotional_state.fear = min(1.0, other.emotional_state.fear + 0.3)
                        other.stress = min(1.0, other.stress + 0.3)
                        # Family executions leave permanent scars
                        other.family_executions += 1
                        other.trauma_score = min(1.0, other.trauma_score + 0.3)
                    elif RelationshipType.FRIENDSHIP in rel.relationship_types:
                        other.emotional_state.grief = min(1.0, other.emotional_state.grief + 0.2)
                        other.emotional_state.fear = min(1.0, other.emotional_state.fear + 0.15)
                        other.trauma_score = min(1.0, other.trauma_score + 0.1)
    else:
        print(f"  ✨ {accused.name} is found NOT GUILTY and released!")
        accused.reputation = min(1.0, accused.reputation + 0.1)
        state.panic_level = max(0.0, state.panic_level - 0.03)

        # Those who testified against failed - slight reputation hit
        for witness_id in testimony_against:
            witness = state.villagers.get(witness_id)
            if witness:
                witness.reputation = max(0.0, witness.reputation - 0.02)

    return convicted


# ============================================================================
# MAIN SIMULATION LOOP
# ============================================================================

def run_simulation(
    village_size: int = None,
    num_timesteps: int = None,
    seed: int = None,
    initial_panic: float = None,
    verbose: bool = None,
    print_to_console: bool = None
):
    """Run the witch trial simulation"""

    # Use config values if not explicitly provided
    village_size = village_size if village_size is not None else VILLAGE_POPULATION
    num_timesteps = num_timesteps if num_timesteps is not None else SIMULATION_LENGTH
    seed = seed if seed is not None else RANDOM_SEED
    initial_panic = initial_panic if initial_panic is not None else INITIAL_PANIC
    verbose = verbose if verbose is not None else VERBOSE_OUTPUT
    print_to_console = print_to_console if print_to_console is not None else PRINT_TO_CONSOLE

    rng = np.random.RandomState(seed)

    if verbose:
        print("=" * 60)
        print("🏘️  WITCH TRIAL SIMULATION")
        print("=" * 60)
        print(f"Village size: {village_size}")
        print(f"Simulation length: {num_timesteps} days")
        print("=" * 60)

    # Initialize
    state = initialize_village(village_size, seed)
    state.panic_level = initial_panic

    # Statistics tracking
    stats = {
        'accusations_per_timestep': [],
        'deaths_per_timestep': [],
        'panic_over_time': [],
        'alive_count': [],
        'actions_taken': {},  # Track action frequency
        'stressors_triggered': [],  # Track external stressors
        'historical_violence_over_time': [],  # Track trauma accumulation
    }

    # Seed initial panic with an event
    if initial_panic > 0.2:
        print("\n📜 A mysterious illness has struck the village...")
        print("   Whispers of dark magic begin to spread.\n")

    # Main loop
    for t in range(num_timesteps):
        state.timestep = t
        accusations_this_step = 0
        deaths_this_step = 0

        alive_count = sum(1 for v in state.villagers.values() if v.is_alive)

        if verbose and t % 10 == 0:
            print(f"\n{'='*60}")
            print(f"--- Day {t} --- (Panic: {state.panic_level:.2f}, Alive: {alive_count}/{village_size})")
            print(f"{'='*60}")

        # Each villager takes an action
        alive_villagers = [v for v in state.villagers.values()
                         if v.is_alive and not v.is_imprisoned]
        rng.shuffle(alive_villagers)

        daily_actions = []

        for villager in alive_villagers:
            action_type, target_id = select_action(villager, state, rng)

            if action_type != ActionType.PASS:
                success = execute_action(villager, action_type, target_id, state, rng,
                                        verbose=(verbose and action_type in [
                                            ActionType.ACCUSE_WITCHCRAFT,
                                            ActionType.COUNTER_ACCUSE,
                                            ActionType.DEFEND_PERSON,
                                            ActionType.FORM_ALLIANCE,
                                            ActionType.BREAK_ALLIANCE,
                                            ActionType.SHARE_RESOURCES,
                                            ActionType.PROPOSE_MARRIAGE,
                                            ActionType.FLEE_VILLAGE,
                                            ActionType.SEEK_PROTECTION,
                                        ]))

                daily_actions.append((action_type.name, success))

                if action_type == ActionType.ACCUSE_WITCHCRAFT and success:
                    accusations_this_step += 1

                # Track action frequency
                action_name = action_type.name
                stats['actions_taken'][action_name] = stats['actions_taken'].get(action_name, 0) + 1

        # Process trials
        trials_today = list(state.trial_queue)  # Copy to avoid modification during iteration
        state.trial_queue = []

        for accused_id in trials_today:
            if state.villagers[accused_id].is_alive:
                convicted = conduct_trial(accused_id, state, rng)
                if convicted:
                    deaths_this_step += 1

        # Update states
        update_village_state(state, rng)
        update_villager_states(state, rng)

        # Track statistics
        stats['accusations_per_timestep'].append(accusations_this_step)
        stats['deaths_per_timestep'].append(deaths_this_step)
        stats['panic_over_time'].append(state.panic_level)
        stats['alive_count'].append(sum(1 for v in state.villagers.values() if v.is_alive))
        stats['historical_violence_over_time'].append(state.historical_violence)

        # Log any active stressors
        for stressor in state.active_stressors:
            if stressor not in stats['stressors_triggered']:
                stats['stressors_triggered'].append(stressor)

        # Check for simulation end conditions
        alive = sum(1 for v in state.villagers.values() if v.is_alive)
        if alive < village_size * MIN_SURVIVAL_RATE:
            if verbose:
                print(f"\n⚰️ The village has been decimated. Only {alive} remain.")
            break

        # If panic drops to near zero and no recent activity, simulation might stabilize
        if t > PEACE_THRESHOLD_DAYS and state.panic_level < PEACE_PANIC_LEVEL and sum(stats['accusations_per_timestep'][-10:]) == 0:
            if verbose:
                print(f"\n🕊️ Peace has returned to the village after {t} days.")
            break

    # Final report
    if verbose:
        print_final_report(state, stats, village_size)

    return state, stats


def print_final_report(state: VillageState, stats: Dict, village_size: int):
    """Print comprehensive simulation statistics"""
    print("\n" + "=" * 60)
    print("📊 SIMULATION COMPLETE")
    print("=" * 60)

    total_deaths = sum(stats['deaths_per_timestep'])
    total_accusations = sum(stats['accusations_per_timestep'])
    survivors = sum(1 for v in state.villagers.values() if v.is_alive)

    print(f"\nDuration: {state.timestep + 1} days")
    print(f"Total accusations: {total_accusations}")
    print(f"Total executions: {total_deaths}")
    print(f"Survivors: {survivors}/{village_size} ({100 * survivors / village_size:.1f}%)")
    print(f"Peak panic level: {max(stats['panic_over_time']):.2f}")
    print(f"Final panic level: {state.panic_level:.2f}")
    print(f"Historical violence (trauma): {state.historical_violence:.2f}")

    # Stressor history
    if stats.get('stressors_triggered'):
        print(f"\n⚡ External stressors that occurred:")
        for stressor in stats['stressors_triggered']:
            print(f"  - {stressor}")

    # Action breakdown
    print("\n📋 Actions taken:")
    sorted_actions = sorted(stats['actions_taken'].items(), key=lambda x: x[1], reverse=True)
    for action_name, count in sorted_actions[:15]:
        print(f"  {action_name}: {count}")

    # Most accused
    if total_accusations > 0:
        most_accused = max(state.villagers.values(), key=lambda v: v.times_accused_total)
        print(f"\nMost accused: {most_accused.name} ({most_accused.times_accused_total} accusations)")

    # Most prolific accuser
    most_accusations_made = max(state.villagers.values(), key=lambda v: len(v.accusations_made))
    if most_accusations_made.accusations_made:
        print(f"Most prolific accuser: {most_accusations_made.name} ({len(most_accusations_made.accusations_made)} accusations)")

    # Patron statistics
    patrons = [v for v in state.villagers.values() if v.dependents and v.is_alive]
    if patrons:
        biggest_patron = max(patrons, key=lambda v: len(v.dependents))
        print(f"Largest patron: {biggest_patron.name} ({len(biggest_patron.dependents)} dependents)")

    # Most traumatized survivor
    living = [v for v in state.villagers.values() if v.is_alive]
    if living:
        most_traumatized = max(living, key=lambda v: v.trauma_score)
        if most_traumatized.trauma_score > 0.1:
            print(f"Most traumatized survivor: {most_traumatized.name} (trauma: {most_traumatized.trauma_score:.2f})")

    # Strongest alliance (highest mutual loyalty)
    max_loyalty = 0
    best_pair = None
    for v1 in state.villagers.values():
        if not v1.is_alive:
            continue
        for v2_id, loyalty in v1.emotional_state.loyalties.items():
            v2 = state.villagers.get(v2_id)
            if v2 and v2.is_alive:
                mutual = loyalty + v2.emotional_state.loyalties.get(v1.id, 0)
                if mutual > max_loyalty:
                    max_loyalty = mutual
                    best_pair = (v1, v2)

    if best_pair:
        print(f"Strongest alliance: {best_pair[0].name} & {best_pair[1].name} (loyalty: {max_loyalty/2:.2f})")

    # Marriages formed
    marriages = sum(1 for v in state.villagers.values()
                   if v.marital_status == MaritalStatus.MARRIED) // 2
    print(f"Marriages: {marriages}")

    # Print the dead
    dead = [v for v in state.villagers.values() if not v.is_alive]
    if dead:
        print(f"\n💀 The dead ({len(dead)}):")
        # Show patrons first (their deaths are significant)
        patron_deaths = [v for v in dead if v.dependents]
        if patron_deaths:
            print("  Fallen patrons:")
            for v in patron_deaths[:5]:
                print(f"    ✝ {v.name} ({len(v.dependents)} dependents orphaned)")

        # Then others
        non_patron_dead = [v for v in dead if not v.dependents]
        for v in non_patron_dead[:15]:
            print(f"  ✝ {v.name}, {v.age}yo {v.occupation.value}")
        if len(dead) > 20:
            print(f"  ... and {len(dead) - 20} more")

    # Survivors by gender
    surviving_females = sum(1 for v in state.villagers.values()
                           if v.is_alive and v.gender == Gender.FEMALE)
    surviving_males = sum(1 for v in state.villagers.values()
                         if v.is_alive and v.gender == Gender.MALE)
    print(f"\nSurvivors by gender: {surviving_females} women, {surviving_males} men")

    # Calculate if women were disproportionately targeted
    total_females = sum(1 for v in state.villagers.values() if v.gender == Gender.FEMALE)
    total_males = sum(1 for v in state.villagers.values() if v.gender == Gender.MALE)
    if total_females > 0 and total_males > 0:
        female_death_rate = (total_females - surviving_females) / total_females
        male_death_rate = (total_males - surviving_males) / total_males
        if female_death_rate > male_death_rate * 1.2:
            print(f"  ⚠️ Women were disproportionately targeted ({female_death_rate*100:.1f}% vs {male_death_rate*100:.1f}%)")

    # Average trauma among survivors
    if living:
        avg_trauma = np.mean([v.trauma_score for v in living])
        if avg_trauma > 0.05:
            print(f"\n🩹 Average survivor trauma: {avg_trauma:.2f}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🔮" * 30 + "\n")

    # Save output to file first
    original_stdout = sys.stdout
    with open("simulation_output.txt", "w") as f:
        sys.stdout = f
        state, stats = run_simulation()
    sys.stdout = original_stdout # Restore original stdout

    if PRINT_TO_CONSOLE:
        # If printing to console, run again, but direct output to console
        print("\n" + "🔮" * 30 + "\n") # Separator for console output
        print("\n--- Console Output (also saved to simulation_output.txt) ---\n")
        # Re-run the simulation with console printing enabled for its duration
        # Or, ideally, load from the saved state/stats and just print report
        # For simplicity, we'll re-run, but for large simulations, loading is better
        state, stats = run_simulation(print_to_console=True) # Pass explicitly for console output

    print("Simulation output saved to simulation_output.txt")
    print("\n" + "🔮" * 30 + "\n")
