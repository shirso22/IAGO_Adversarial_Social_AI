# Witch Trial Simulation Runner

A modular runner for executing witch trial simulations with configurable parameters, batch execution, and analysis tools.

## Quick Start

```bash
# Run with default settings
python simulation_runner.py

# Run a chaotic scenario
python simulation_runner.py --preset chaos

# Run 10 simulations and get aggregate stats
python simulation_runner.py --batch 10 --quiet
```

## Requirements

- Python 3.7+
- NumPy
- All simulation module files (`Villager.py`, `Village.py`, `main.py`, etc.)

## Command Line Usage

### Basic Run

```bash
python simulation_runner.py
```

Runs a single simulation with default parameters and prints the full output.

### Using Presets

```bash
python simulation_runner.py --preset <preset_name>
```

Available presets:

| Preset | Village Size | Max Days | Initial Panic | Description |
|--------|--------------|----------|---------------|-------------|
| `default` | 50 | 200 | 0.35 | Standard balanced simulation |
| `small_village` | 20 | 150 | 0.25 | Tight-knit community |
| `large_town` | 150 | 300 | 0.30 | Complex social dynamics |
| `chaos` | 75 | 200 | 0.65 | High panic, rapid escalation |
| `slow_burn` | 50 | 400 | 0.15 | Gradual tension build-up |
| `fragile_peace` | 40 | 200 | 0.20 | Peace possible but precarious |
| `quick_test` | 15 | 50 | 0.40 | Fast run for testing |

List all presets:
```bash
python simulation_runner.py --list-presets
```

### Batch Runs

Run multiple simulations with different random seeds to get statistical insights:

```bash
python simulation_runner.py --batch 10
```

This runs 10 simulations and prints aggregate statistics (mean, std, min, max) for key metrics.

### Parameter Sweep

Test how a parameter affects outcomes:

```bash
python simulation_runner.py --sweep
```

By default, sweeps across initial panic levels (0.1, 0.25, 0.4, 0.55, 0.7) with 3 runs each.

### Override Parameters

Override any parameter directly:

```bash
python simulation_runner.py --size 100 --steps 300 --panic 0.5 --seed 42
```

| Flag | Description |
|------|-------------|
| `--size N` | Village population |
| `--steps N` | Maximum simulation days |
| `--seed N` | Random seed for reproducibility |
| `--panic X` | Initial panic level (0.0-1.0) |
| `--quiet` | Suppress detailed output |
| `--no-save` | Don't save results to files |

### Load Custom Config

```bash
python simulation_runner.py --config my_config.json
```

## Python API Usage

### Basic Run

```python
from simulation_runner import SimulationRunner, PRESETS

runner = SimulationRunner(PRESETS['default'])
state, stats = runner.run_single()
```

### Custom Configuration

```python
from simulation_runner import SimulationRunner, SimulationConfig

config = SimulationConfig(
    village_size=80,
    max_steps=250,
    initial_panic=0.45,
    seed=12345,
    verbose=True,
    name="my_experiment"
)

runner = SimulationRunner(config)
state, stats = runner.run_single()
```

### Batch Runs

```python
runner = SimulationRunner(PRESETS['chaos'])
results = runner.run_batch(num_runs=20, base_seed=1000)

# results is a list of summary dictionaries
for r in results:
    print(f"Seed {r['seed']}: {r['survivors']} survived, {r['total_deaths']} executed")
```

### Parameter Sweep

```python
runner = SimulationRunner(SimulationConfig())

# Sweep over village size
results = runner.parameter_sweep(
    param_name="village_size",
    values=[20, 50, 100, 150],
    runs_per_value=5
)

# Sweep over initial panic
results = runner.parameter_sweep(
    param_name="initial_panic",
    values=[0.1, 0.3, 0.5, 0.7],
    runs_per_value=3
)
```

### Save/Load Configurations

```python
from simulation_runner import save_config, load_config, SimulationConfig

# Save
config = SimulationConfig(village_size=100, name="saved_config")
save_config(config, "my_config.json")

# Load
config = load_config("my_config.json")
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `village_size` | int | 50 | Number of villagers (10-200 recommended) |
| `max_steps` | int | 200 | Maximum simulation days |
| `seed` | int | 696 | Random seed for reproducibility |
| `initial_panic` | float | 0.35 | Starting panic level (0.0-1.0) |
| `verbose` | bool | True | Print detailed event log |
| `print_to_console` | bool | True | Output to console |
| `save_output` | bool | True | Save results to JSON files |
| `output_dir` | str | "simulation_results" | Directory for saved results |
| `min_survival_rate` | float | 0.30 | Stop if population drops below this |
| `peace_threshold_days` | int | 30 | Days of low panic before peace declared |
| `peace_panic_level` | float | 0.05 | Panic threshold for peace |

## Output

### Console Output

Each simulation prints a day-by-day log (unless `--quiet`) followed by a summary:

```
📊 SIMULATION COMPLETE
============================================================
Duration: 127 days
Total accusations: 45
Total executions: 12
Survivors: 38/50 (76.0%)
Peak panic level: 0.72
...
```

### Saved Results

Results are saved as JSON files in `simulation_results/`:

```
simulation_results/
  run_chaos_1000_20250101_143022.json
  run_chaos_1001_20250101_143025.json
  ...
```

Each file contains:
```json
{
  "config": {
    "village_size": 75,
    "initial_panic": 0.65,
    ...
  },
  "summary": {
    "duration": 142,
    "total_deaths": 18,
    "survivors": 57,
    "peak_panic": 0.81,
    ...
  }
}
```

### Summary Metrics

| Metric | Description |
|--------|-------------|
| `duration` | Days until simulation ended |
| `total_accusations` | Number of accusations made |
| `total_deaths` | Number of executions |
| `survivors` | Villagers alive at end |
| `survival_rate` | Fraction of village surviving |
| `peak_panic` | Highest panic level reached |
| `final_panic` | Panic level at simulation end |
| `historical_violence` | Accumulated trauma score |
| `trials_conducted` | Number of trials held |
| `chain_accusations` | Accusations from confessions |

## Examples

### Compare Small vs Large Villages

```python
from simulation_runner import SimulationRunner, SimulationConfig

for size in [20, 50, 100]:
    config = SimulationConfig(village_size=size, verbose=False)
    runner = SimulationRunner(config)
    results = runner.run_batch(10)
```

### Find Tipping Point for Panic

```bash
python simulation_runner.py --sweep
```

### Reproducible Research Run

```bash
python simulation_runner.py --preset default --seed 42 --batch 100
```

### Quick Sanity Check

```bash
python simulation_runner.py --preset quick_test --quiet
```

## Termination Conditions

Simulations end when any of these occur:

1. **Maximum days reached** - Simulation hits `max_steps`
2. **Village decimated** - Population falls below `min_survival_rate`
3. **Peace restored** - Panic stays below `peace_panic_level` for `peace_threshold_days` with no accusations or pending trials

## Tips

- Use `--quiet` for batch runs to avoid overwhelming output
- Set `seed` for reproducible results when debugging
- Start with `quick_test` preset when developing
- Use parameter sweeps to understand how variables affect outcomes
- Batch runs with 20+ iterations give more statistically meaningful results
