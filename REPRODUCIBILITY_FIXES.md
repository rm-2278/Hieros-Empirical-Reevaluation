# Reproducibility Fixes and Testing Guide

## Summary of Changes

This document describes the changes made to fix reproducibility issues in Hieros training and how to test them.
This does not make the result perfectly reproducible in all cases due to inherent non-determinism in some hardware and software operations, but significantly improves determinism across runs with the same seed.

## Issues Fixed

### 1. Replay Buffer Seeding
**Problem:** The replay buffer was not using the configured seed, causing non-deterministic sampling.

**Solution:** Modified `make_replay()` in `hieros/hieros.py` to pass the `seed` parameter to all replay buffer types:
- `Uniform`
- `TimeBalanced`
- `TimeBalancedNaive`
- `EfficientTimeBalanced`

**Code Changes:**
```python
seed = config.seed if hasattr(config, 'seed') else 0
replay = embodied.replay.Uniform(length, size, directory, seed=seed, **kw)
```

### 2. Environment Seeding
**Problem:** All parallel environments were receiving the same seed, causing identical behavior across environments.

**Solution:** Modified `make_env()` and `make_envs()` in `hieros/train.py` to give each environment a unique seed:
```python
# Each environment gets base_seed + index
kwargs['seed'] = config.seed + env_index
```

This ensures:
- Environment 0 gets seed 42
- Environment 1 gets seed 43
- Environment 2 gets seed 44
- etc.

### 3. Enhanced Determinism
**Problem:** Some PyTorch and Python operations could still be non-deterministic.

**Solution:** Enhanced `set_seed()` in `hieros/tools.py` with:
- `PYTHONHASHSEED` environment variable
- `torch.use_deterministic_algorithms(True)` for PyTorch operations

**Code Changes:**
```python
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # ... existing seed setting code ...
    torch.use_deterministic_algorithms(True)
```

## How to Test Reproducibility

### Quick Test (Recommended)
Run the same training command twice with identical parameters and compare outputs:

```bash
# Run 1
python hieros/train.py --configs seed --wandb_prefix=test_run1 --wandb_name=seed --seed=42

# Run 2
python hieros/train.py --configs seed --wandb_prefix=test_run2 --wandb_name=seed --seed=42
```

Then compare the metrics from both runs. Key metrics to check:
- `Subactor-0/model_loss` should be identical
- `train/reward` should be identical
- `episode/score` should be identical

### Detailed Testing Steps

1. **Clean Environment:**
   ```bash
   rm -rf logs/
   ```

2. **Run First Training:**
   ```bash
   python hieros/train.py --configs seed --wandb_prefix=repro_test --wandb_name=run1 --seed=42 --logdir=logs/run1
   ```

3. **Run Second Training:**
   ```bash
   python hieros/train.py --configs seed --wandb_prefix=repro_test --wandb_name=run2 --seed=42 --logdir=logs/run2
   ```

4. **Compare Logs:**
   ```bash
   # Compare metrics.jsonl files
   diff logs/run1/metrics.jsonl logs/run2/metrics.jsonl
   
   # If identical, you should see no output
   # Any differences indicate non-reproducibility
   ```

5. **Compare TensorBoard Logs:**
   ```bash
   tensorboard --logdir=logs/
   # Visually compare the curves - they should be identical
   ```

### Expected Behavior

With the fixes applied:
- ✓ Same metrics at each step
- ✓ Same loss values
- ✓ Same model predictions
- ✓ Same exploration behavior
- ✓ Identical training curves in TensorBoard

### Known Limitations

1. **Multi-threading:** If using `data_loaders > 0`, there may still be some non-determinism due to thread scheduling. The fixes minimize this, but perfect reproducibility may require setting `data_loaders=0`.

2. **GPU-specific operations:** Some GPU operations may vary slightly across different GPU models or drivers, even with deterministic algorithms enabled.

3. **Performance Impact:** `torch.use_deterministic_algorithms(True)` may reduce performance slightly, as it forces deterministic but potentially slower algorithms. The implementation uses `warn_only=True` to allow operations without deterministic implementations to still run (with a warning).

4. **Non-deterministic operations:** Some PyTorch operations don't have deterministic implementations. These will generate warnings but won't crash the program.

## Hyperparameter Improvements for pinpad-easy

### New Configuration: `pinpad-easy-director`

Added a new configuration inspired by the Director paper (Hafner et al., 2022), which successfully solved similar sequential decision-making tasks.

**Key Changes:**
- `batch_size`: 8 → 16 (larger batches for stable learning)
- `batch_length`: 16 → 32 (better temporal credit assignment)
- `train_ratio`: 8 → 128 (significantly more gradient steps per environment step)
- `imag_horizon`: 16 → 32 (longer planning horizon for multi-step tasks)
- `actor_entropy`: 3e-4 → 1e-3 (increased exploration for discrete actions)

**Usage:**
```bash
python hieros/train.py --configs pinpad-easy-director --task=pinpad-easy_three
```

**Rationale:**
1. **Higher train_ratio:** Director typically uses train_ratios between 128-512. This allows the model to learn more from each environment interaction, which is crucial for sample efficiency.

2. **Longer batch_length:** Sequential tasks like PinPad require understanding temporal dependencies. Longer sequences help the model learn better credit assignment.

3. **Longer imag_horizon:** For tasks requiring multi-step planning (like visiting pads in sequence), longer imagination horizons enable better long-term decision making.

4. **Higher entropy:** Discrete action spaces benefit from more exploration to discover the correct action sequences.

### Comparison Table

| Parameter | Original | Director-Inspired | Rationale |
|-----------|----------|-------------------|-----------|
| batch_size | 8 | 16 | Stable gradients |
| batch_length | 16 | 32 | Temporal credit |
| train_ratio | 8 | 128 | Sample efficiency |
| imag_horizon | 16 | 32 | Long-term planning |
| actor_entropy | 3e-4 | 1e-3 | Exploration |

## References

- Director: Hierarchical RL with Temporal Abstraction (Hafner et al., 2022)
- DreamerV3: Mastering Diverse Domains through World Models (Hafner et al., 2023)
