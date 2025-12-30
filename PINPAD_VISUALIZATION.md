# PinPad Position Visit Tracking and Visualization

This feature adds position visit tracking and heatmap visualization for the PinPad and PinPadEasy environments.

## What's New

### Position Visit Tracking
Both `PinPad` and `PinPadEasy` environments now track:
- The number of times the agent visits each grid position
- Statistics about exploration coverage
- Visual heatmap representations

### Features Added

1. **Position Visit Counter** (`position_visit_counts`)
   - A 16x14 numpy array tracking visits to each position
   - Updated automatically whenever the agent moves to a valid (non-wall) position

2. **Heatmap Visualization** (`get_position_heatmap()`)
   - Generates a color-coded heatmap showing exploration patterns
   - Color gradient: Blue (rarely visited) → Cyan → Green → Yellow → Red (frequently visited)
   - Walls are shown in gray
   - Returns a 64x64x3 RGB image suitable for logging

3. **Position Statistics** (`get_position_stats()`)
   - `total_visits`: Total number of position visits
   - `unique_positions_visited`: Number of different positions explored
   - `total_valid_positions`: Total non-wall positions in the environment
   - `coverage_ratio`: Percentage of the environment explored (0.0 to 1.0)
   - `max_visits_single_position`: Highest visit count for any position
   - `mean_visits_per_visited_position`: Average visits across explored positions

## WandB Integration

When `wandb_logging` is enabled in the config, the following are automatically logged every `log_every` steps:

### Visualization
- `exploration/position_heatmap`: Visual heatmap image showing exploration patterns

### Metrics
- `exploration/total_visits`: Total position visits
- `exploration/unique_positions_visited`: Number of unique positions explored
- `exploration/coverage_ratio`: Fraction of environment explored
- `exploration/max_visits_single_position`: Peak visit count
- `exploration/mean_visits_per_visited_position`: Average visits per explored position

## Usage

### In Code
```python
from embodied.envs.pinpad import PinPad

# Create environment
env = PinPad(task="three", length=1000, seed=42)

# ... run agent ...

# Get visualization
heatmap = env.get_position_heatmap()  # Returns (64, 64, 3) RGB numpy array

# Get statistics
stats = env.get_position_stats()
print(f"Coverage: {stats['coverage_ratio']:.1%}")
print(f"Unique positions: {stats['unique_positions_visited']}/{stats['total_valid_positions']}")
```

### In Training
The visualization is automatically integrated into the training loop when using `embodied.run.train_eval`. No additional configuration is needed - it will automatically detect PinPad environments and log the heatmap.

### Viewing in WandB
1. Start a training run with `wandb_logging: true` in the config
2. Navigate to your WandB project
3. Look for the following in your run:
   - **Media** tab: `exploration/position_heatmap` images
   - **Charts** tab: `exploration/*` metrics showing coverage over time

## Example Interpretation

### Heatmap Colors
- **Blue regions**: Rarely or never visited - potential areas for more exploration
- **Green/Yellow regions**: Moderately visited - balanced exploration
- **Red regions**: Frequently visited - possibly over-explored or high-value areas

### Coverage Ratio
- **Low (< 0.3)**: Agent is under-exploring the environment
- **Medium (0.3-0.7)**: Healthy exploration pattern
- **High (> 0.7)**: Extensive exploration, possibly covering most of the environment

## Implementation Details

### Position Tracking
- Positions are tracked whenever the agent successfully moves (i.e., not blocked by walls)
- The counter is persistent across episode resets within the same environment instance
- This provides a cumulative view of exploration throughout training

### Heatmap Generation
- Uses a 4-step color gradient for smooth visualization
- Normalized by the maximum visit count for consistent scaling
- Scaled up 4x for better visibility (from 16x14 to 64x64)

### Performance
- Minimal overhead: counter update is O(1) per step
- Heatmap generation only happens during logging (every `log_every` steps)
- No impact on training performance

## Configuration

No additional configuration is required. The feature works with existing configs:

```yaml
wandb_logging: true  # Enable WandB logging
log_every: 1000      # How often to log (including heatmaps)
```

## Notes

- The position tracking is cumulative throughout the entire training run
- Each parallel environment tracks its own positions independently
- Only the first environment's heatmap is logged to avoid clutter
- Position counters reset if the environment is recreated (e.g., new training run)
