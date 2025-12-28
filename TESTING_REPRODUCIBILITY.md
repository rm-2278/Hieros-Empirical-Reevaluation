# Testing Reproducibility

This document explains how to test that the reproducibility fixes are working correctly.

## Quick Verification Test

To verify that metrics are identical across two runs with the same seed:

1. **Run the first training session:**
```bash
python hieros/train.py --configs seed --wandb_logging=false --tensorboard_logging=false --steps=1000 --seed=42 --logdir=logs/run1
```

2. **Run the second training session:**
```bash
python hieros/train.py --configs seed --wandb_logging=false --tensorboard_logging=false --steps=1000 --seed=42 --logdir=logs/run2
```

3. **Compare the metrics:**
```bash
# Compare metrics.jsonl files
diff logs/run1/metrics.jsonl logs/run2/metrics.jsonl

# If the runs are reproducible, diff should show no differences
# You should see no output from this command
```

## Expected Results

### What Should Be Identical

With the reproducibility fixes in place, the following should be identical across runs with the same seed:

- **All training metrics** (loss values, gradients, etc.)
  - `Subactor-0/wm_loss`
  - `Subactor-0/actor_loss`
  - `Subactor-0/critic_loss`
  - All other subactor metrics
- **Environment interactions** (rewards, actions, observations)
  - `train/reward`
  - `episode/score`
  - `episode/length`
- **Model parameters** (weights at each step)
- **Replay buffer sampling** (same sequences sampled at each step)

### What Might Still Differ

- **Wall-clock time measurements** (these depend on system load)
- **GPU memory usage** (varies based on other processes)

## Troubleshooting

If you find that metrics are not identical:

### 1. Check Configuration Settings

Ensure the following settings are correctly configured:

```yaml
seed: 42  # Or any consistent seed value
data_loaders: 0  # Must be 0 for perfect reproducibility
```

### 2. Verify Environment Setup

- Ensure you're using the same Python version
- Ensure you're using the same PyTorch version
- Ensure CUDA versions match if using GPU

### 3. Check for Warnings

Look for warnings like:
```
Warning: torch.use_deterministic_algorithms may impact performance
```

These warnings are expected and indicate that deterministic algorithms are active.

### 4. Verify Seeds Are Being Set

Check the console output at the start of training. You should see:
```
Random seed set to: 42
```

## Advanced Testing

### Testing Individual Components

You can test individual components for reproducibility:

#### Test Replay Buffer Selectors

```python
from embodied.replay import selectors

# Test Uniform selector
selector1 = selectors.Uniform(seed=42)
selector2 = selectors.Uniform(seed=42)

# Add keys and sample
for i in range(100):
    selector1[f"key_{i}"] = None
    selector2[f"key_{i}"] = None

samples1 = [selector1() for _ in range(50)]
samples2 = [selector2() for _ in range(50)]

assert samples1 == samples2, "Not reproducible!"
```

#### Test PyTorch Operations

```python
import torch

# Test with seeded generator
gen1 = torch.Generator()
gen1.manual_seed(42)
gen2 = torch.Generator()
gen2.manual_seed(42)

# Test any PyTorch operation
result1 = torch.randn(10, generator=gen1)
result2 = torch.randn(10, generator=gen2)

assert torch.equal(result1, result2), "Not reproducible!"
```

## Performance Considerations

Setting `data_loaders=0` disables multi-threaded data loading, which may impact training speed:

- **Without multi-threading (data_loaders=0):** Fully reproducible, but slower data loading
- **With multi-threading (data_loaders>0):** Faster data loading, but not fully reproducible

If you need both performance and reproducibility:
1. Use multi-threading during development/hyperparameter search
2. Switch to `data_loaders=0` for final reproducible runs
3. Consider using more powerful hardware to compensate for the single-threaded bottleneck

## Reporting Issues

If you encounter reproducibility issues after applying these fixes:

1. Verify all configuration settings match the recommendations above
2. Check that you're running identical code (same commit)
3. Ensure system dependencies are identical (Python, PyTorch, CUDA versions)
4. Document the specific metrics that differ
5. Provide configuration files and run logs
6. Report the issue with full details for investigation
