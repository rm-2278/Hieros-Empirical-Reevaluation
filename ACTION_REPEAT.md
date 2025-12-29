# Action Repeat in Hieros

## Overview

This document clarifies how action repetition (frame skipping) works in Hieros and why the `action_repeat` configuration parameter exists.

## TL;DR

**For hierarchical RL training**: Set `action_repeat: 1` in your config. Frame skipping is handled by environment-specific parameters (e.g., `env.atari.repeat: 4`).

**For non-hierarchical baselines** (e.g., Dreamer): Set `action_repeat` to match your environment's frame skip rate.

## How Action Repetition Works

### Environment-Level Frame Skipping

Each environment handles frame skipping independently:

1. **Atari**: Uses internal repetition in the `step()` method
   ```yaml
   env:
     atari: {repeat: 4}  # Action repeated 4 times per step
   ```

2. **DMC/BSuite**: Uses the `ActionRepeat` wrapper
   ```yaml
   env:
     dmc: {repeat: 2}  # Action repeated 2 times per step
   ```

3. **DMLab**: Uses internal `num_steps` parameter
   ```yaml
   env:
     dmlab: {repeat: 4}  # Action repeated 4 times per step
   ```

4. **PinPad**: No repetition (operates at single-step level)
   ```yaml
   env:
     pinpad: {}  # No frame skipping
   ```

### The `action_repeat` Configuration Parameter

The global `action_repeat` parameter in the config serves **only for step accounting**:

- It divides training steps: `config.steps //= config.action_repeat`
- It sets the logger multiplier to report metrics at the correct scale
- It does **NOT** control actual action repetition in the environment

### Why Does `action_repeat` Exist?

The parameter exists for **backward compatibility** with non-hierarchical RL algorithms (e.g., Dreamer) where:
- The agent makes decisions at a fixed frequency
- Action repeat determines the temporal scale of learning
- Consistent step counting across different action repeat values is important

## Hierarchical RL and Temporal Abstraction

In Hieros, **temporal abstraction is handled by the hierarchy itself**:

- Higher-level policies (managers) operate at lower frequencies
- Lower-level policies (workers) operate at higher frequencies
- This creates natural temporal abstraction without needing global action repeat

**Example**: With a 2-level hierarchy using k=4 temporal abstraction:
- Level 1 (worker): Acts every environment step
- Level 0 (manager): Acts every 4 environment steps

Adding a global `action_repeat=4` on top of this would be redundant and confusing.

## Recommendations

### For Hierarchical Training (Hieros)

Set `action_repeat: 1` to avoid confusion:

```yaml
hieros_config:
  action_repeat: 1  # No global step division
  env:
    atari: {repeat: 4}  # Environment handles frame skipping
```

### For Non-Hierarchical Baselines (Dreamer)

Match `action_repeat` to your environment's frame skip:

```yaml
dreamer_config:
  action_repeat: 4  # Match environment frame skip
  env:
    atari: {repeat: 4}
```

## Migration Guide

If you have existing configs with `action_repeat > 1` for hierarchical training:

1. **Option A - Quick Fix**: Leave as-is. The code will print a warning but work correctly.

2. **Option B - Clean Fix**: Update your config:
   ```diff
   - action_repeat: 4
   - steps: 400000
   + action_repeat: 1
   + steps: 100000  # Divide by old action_repeat value
   ```

## Related Issues

- Issue: "Action-repeat not being used?" - This document explains the design
- Inspiration: [Director](https://github.com/danijar/director) uses per-environment repeat, no global parameter

## References

- Environment wrappers: `embodied/core/wrappers.py`
- Atari environment: `embodied/envs/atari.py`
- DMC environment: `embodied/envs/dmc.py`
- Training script: `hieros/train.py`
