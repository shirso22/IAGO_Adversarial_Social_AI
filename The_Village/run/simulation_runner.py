#!/usr/bin/env python3
"""
Modular Simulation Runner
=========================
A flexible runner for the witch trial simulation with presets,
batch execution, and analysis tools.

Usage:
    python simulation_runner.py                    # Run with defaults
    python simulation_runner.py --preset chaos     # Run a preset scenario
    python simulation_runner.py --batch 10         # Run 10 simulations
    python simulation_runner.py --sweep            # Parameter sweep
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""
    
    # Core parameters
    village_size: int = 50
    max_steps: int = 200
    seed: int = 696
    initial_panic: float = 0.35
    
    # Output settings
    verbose: bool = True
    print_to_console: bool = True
    save_output: bool = True
    output_dir: str = "simulation_results"
    
    # Termination thresholds
    min_survival_rate: float = 0.30
    peace_threshold_days: int = 30
    peace_panic_level: float = 0.05
    
    # Metadata
    name: str = "default"
    description: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SimulationConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# PRESETS
# ============================================================================

PRESETS: Dict[str, SimulationConfig] = {
    "default": SimulationConfig(
        name="default",
        description="Standard simulation with moderate parameters"
    ),
    
    "small_village": SimulationConfig(
        name="small_village",
        village_size=20,
        max_steps=150,
        initial_panic=0.25,
        description="A small, tight-knit community"
    ),
    
    "large_town": SimulationConfig(
        name="large_town",
        village_size=150,
        max_steps=300,
        initial_panic=0.30,
        description="A larger settlement with complex social dynamics"
    ),
    
    "chaos": SimulationConfig(
        name="chaos",
        village_size=75,
        max_steps=200,
        initial_panic=0.65,
        description="High initial panic - things spiral quickly"
    ),
    
    "slow_burn": SimulationConfig(
        name="slow_burn",
        village_size=50,
        max_steps=400,
        initial_panic=0.15,
        peace_threshold_days=50,
        description="Low initial tension, longer simulation"
    ),
    
    "fragile_peace": SimulationConfig(
        name="fragile_peace",
        village_size=40,
        max_steps=200,
        initial_panic=0.20,
        min_survival_rate=0.50,
        description="Peace is possible but precarious"
    ),
    
    "quick_test": SimulationConfig(
        name="quick_test",
        village_size=15,
        max_steps=50,
        initial_panic=0.40,
        verbose=False,
        description="Fast test run for debugging"
    ),
}


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

class SimulationRunner:
    """Manages simulation execution and results."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results: List[Dict] = []
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Create output directory if needed."""
        if self.config.save_output:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def run_single(self, seed_override: Optional[int] = None) -> Tuple[any, Dict]:
        """Run a single simulation."""
        # Import here to avoid circular imports
        from main import run_simulation
        
        seed = seed_override if seed_override is not None else self.config.seed
        
        state, stats = run_simulation(
            village_size=self.config.village_size,
            max_steps=self.config.max_steps,
            seed=seed,
            initial_panic=self.config.initial_panic,
            verbose=self.config.verbose,
            print_to_console=self.config.print_to_console
        )
        
        # Extract summary metrics
        summary = self._extract_summary(state, stats)
        summary['seed'] = seed
        summary['config_name'] = self.config.name
        
        self.results.append(summary)
        
        if self.config.save_output:
            self._save_result(summary, seed)
        
        return state, stats
    
    def run_batch(self, num_runs: int, base_seed: int = 1000) -> List[Dict]:
        """Run multiple simulations with different seeds."""
        print(f"\n{'='*60}")
        print(f"🔮 BATCH RUN: {num_runs} simulations")
        print(f"{'='*60}\n")
        
        original_verbose = self.config.verbose
        self.config.verbose = False  # Suppress individual run output
        
        for i in range(num_runs):
            seed = base_seed + i
            print(f"  Running simulation {i+1}/{num_runs} (seed: {seed})...", end=" ")
            
            try:
                state, stats = self.run_single(seed_override=seed)
                survivors = self.results[-1]['survivors']
                deaths = self.results[-1]['total_deaths']
                print(f"✓ ({survivors} survived, {deaths} executed)")
            except Exception as e:
                print(f"✗ Error: {e}")
        
        self.config.verbose = original_verbose
        
        # Print aggregate statistics
        self._print_batch_summary()
        
        return self.results
    
    def parameter_sweep(
        self,
        param_name: str,
        values: List,
        runs_per_value: int = 3
    ) -> Dict[str, List[Dict]]:
        """Sweep over a parameter with multiple runs per value."""
        print(f"\n{'='*60}")
        print(f"📊 PARAMETER SWEEP: {param_name}")
        print(f"   Values: {values}")
        print(f"   Runs per value: {runs_per_value}")
        print(f"{'='*60}\n")
        
        sweep_results = {}
        original_value = getattr(self.config, param_name)
        original_verbose = self.config.verbose
        self.config.verbose = False
        
        for value in values:
            setattr(self.config, param_name, value)
            sweep_results[str(value)] = []
            
            print(f"\n  {param_name} = {value}:")
            
            for i in range(runs_per_value):
                seed = 1000 + i
                print(f"    Run {i+1}/{runs_per_value}...", end=" ")
                
                try:
                    self.run_single(seed_override=seed)
                    sweep_results[str(value)].append(self.results[-1])
                    print(f"✓")
                except Exception as e:
                    print(f"✗ {e}")
        
        # Restore original values
        setattr(self.config, param_name, original_value)
        self.config.verbose = original_verbose
        
        # Print sweep summary
        self._print_sweep_summary(param_name, sweep_results)
        
        return sweep_results
    
    def _extract_summary(self, state, stats: Dict) -> Dict:
        """Extract key metrics from simulation results."""
        return {
            'duration': state.timestep + 1,
            'total_accusations': sum(stats['accusations_per_timestep']),
            'total_deaths': sum(stats['deaths_per_timestep']),
            'survivors': sum(1 for v in state.villagers.values() if v.is_alive),
            'survival_rate': sum(1 for v in state.villagers.values() if v.is_alive) / len(state.villagers),
            'peak_panic': max(stats['panic_over_time']) if stats['panic_over_time'] else 0,
            'final_panic': state.panic_level,
            'historical_violence': state.historical_violence,
            'trials_conducted': sum(stats.get('trials_per_day', [])),
            'chain_accusations': sum(stats.get('chain_accusations_per_day', [])),
            'stressors': len(stats.get('stressors_triggered', [])),
        }
    
    def _save_result(self, summary: Dict, seed: int):
        """Save individual result to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.output_dir}/run_{self.config.name}_{seed}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'config': self.config.to_dict(),
                'summary': summary
            }, f, indent=2)
    
    def _print_batch_summary(self):
        """Print aggregate statistics for batch run."""
        if not self.results:
            return
        
        print(f"\n{'='*60}")
        print("📈 BATCH SUMMARY")
        print(f"{'='*60}")
        
        # Calculate statistics
        metrics = ['duration', 'total_deaths', 'survivors', 'peak_panic', 'final_panic']
        
        for metric in metrics:
            values = [r[metric] for r in self.results]
            print(f"\n  {metric}:")
            print(f"    Mean: {np.mean(values):.2f}")
            print(f"    Std:  {np.std(values):.2f}")
            print(f"    Min:  {np.min(values):.2f}")
            print(f"    Max:  {np.max(values):.2f}")
        
        # Survival rate
        survival_rates = [r['survival_rate'] for r in self.results]
        print(f"\n  Overall survival rate: {np.mean(survival_rates)*100:.1f}%")
    
    def _print_sweep_summary(self, param_name: str, sweep_results: Dict):
        """Print summary of parameter sweep."""
        print(f"\n{'='*60}")
        print(f"📊 SWEEP SUMMARY: {param_name}")
        print(f"{'='*60}")
        
        print(f"\n{'Value':<15} {'Deaths':<12} {'Survivors':<12} {'Peak Panic':<12}")
        print("-" * 51)
        
        for value, results in sweep_results.items():
            if results:
                avg_deaths = np.mean([r['total_deaths'] for r in results])
                avg_survivors = np.mean([r['survivors'] for r in results])
                avg_panic = np.mean([r['peak_panic'] for r in results])
                print(f"{value:<15} {avg_deaths:<12.1f} {avg_survivors:<12.1f} {avg_panic:<12.2f}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def list_presets():
    """Display available presets."""
    print("\n📋 Available Presets:")
    print("-" * 60)
    for name, config in PRESETS.items():
        print(f"\n  {name}:")
        print(f"    {config.description}")
        print(f"    Village: {config.village_size}, Steps: {config.max_steps}, Panic: {config.initial_panic}")


def create_custom_config(**kwargs) -> SimulationConfig:
    """Create a custom configuration."""
    return SimulationConfig(**kwargs)


def load_config(filepath: str) -> SimulationConfig:
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return SimulationConfig.from_dict(data)


def save_config(config: SimulationConfig, filepath: str):
    """Save configuration to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run witch trial simulations with various configurations"
    )
    
    # Mode selection
    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESETS.keys()),
        help="Use a preset configuration"
    )
    parser.add_argument(
        "--list-presets", "-l",
        action="store_true",
        help="List available presets"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        metavar="N",
        help="Run N simulations with different seeds"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter sweep on initial_panic"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Load configuration from JSON file"
    )
    
    # Override parameters
    parser.add_argument("--size", type=int, help="Village size")
    parser.add_argument("--steps", type=int, help="Max simulation steps")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--panic", type=float, help="Initial panic level")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # List presets and exit
    if args.list_presets:
        list_presets()
        return
    
    # Build configuration
    if args.config:
        config = load_config(args.config)
    elif args.preset:
        config = PRESETS[args.preset]
    else:
        config = SimulationConfig()
    
    # Apply overrides
    if args.size:
        config.village_size = args.size
    if args.steps:
        config.max_steps = args.steps
    if args.seed:
        config.seed = args.seed
    if args.panic:
        config.initial_panic = args.panic
    if args.quiet:
        config.verbose = False
        config.print_to_console = False
    if args.no_save:
        config.save_output = False
    
    # Create runner
    runner = SimulationRunner(config)
    
    # Execute based on mode
    if args.sweep:
        runner.parameter_sweep(
            param_name="initial_panic",
            values=[0.1, 0.25, 0.4, 0.55, 0.7],
            runs_per_value=3
        )
    elif args.batch:
        runner.run_batch(args.batch)
    else:
        print(f"\n🔮 Running simulation: {config.name}")
        print(f"   {config.description}\n")
        runner.run_single()
    
    print("\n✨ Done!\n")


# ============================================================================
# EXAMPLE USAGE (when imported as module)
# ============================================================================

def example_usage():
    """
    Examples of using the runner programmatically:
    
    # Basic run with preset
    runner = SimulationRunner(PRESETS['chaos'])
    state, stats = runner.run_single()
    
    # Custom configuration
    config = SimulationConfig(
        village_size=80,
        initial_panic=0.5,
        name="custom_test"
    )
    runner = SimulationRunner(config)
    runner.run_batch(5)
    
    # Parameter sweep
    runner = SimulationRunner(SimulationConfig())
    results = runner.parameter_sweep(
        param_name="village_size",
        values=[20, 50, 100],
        runs_per_value=3
    )
    """
    pass


if __name__ == "__main__":
    main()
