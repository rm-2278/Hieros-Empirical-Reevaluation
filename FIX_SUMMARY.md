# Fix for Tensor Dimension Mismatch in Debug Visualization

## Issue Summary

After PR #23 which added debug visualization for subgoal rewards, the following error was reported:

```
Exception in hieros policy: The expanded size of the tensor (128) must match the existing size (8) at non-singleton dimension 2.  Target sizes: [8, 1, 128].  Tensor sizes: [8, 8, 8]
```

## Dimension Clarification

### What is "8"?
The `8` in `[8, 8, 8]` refers to:
- **First dimension (8)**: The batch size (number of parallel environments, typically `config.envs["amount"]`)
- **Second and third dimensions (8, 8)**: The compressed subgoal shape `config.subgoal_shape = [8, 8]`

### What is "128"?
The `128` in `[8, 1, 128]` refers to:
- The **full feature dimension** of the deterministic state (`dyn_deter`), which can be 128, 256, or other values depending on the hierarchy level in the configuration

## Root Cause

The debug visualization code (lines 383-399 in `hieros.py`) was caching **compressed** subgoals with shape `[batch, 8, 8]` but then passing them directly to `_subgoal_reward` which expects **decoded** (full dimensional) subgoals with shape `[batch, time, features]`.

During normal training, subgoals are stored in the replay buffer in their **decoded** form (after being decompressed from `[8, 8]` to `[128]` or similar), but the debug visualization was caching the compressed form.

## The Fix

Two minimal changes were made:

### 1. Decode the Compressed Subgoal (Line 394)
```python
# Before (WRONG):
subgoal_with_time = cached_subgoal.unsqueeze(1)  # [batch, 1, 8, 8]

# After (CORRECT):
decoded_subgoal = subactor.decode_subgoal(cached_subgoal, isfirst=False)  # [batch, 128]
subgoal_with_time = decoded_subgoal.unsqueeze(1)  # [batch, 1, 128]
```

### 2. Replace expand with reshape (Line 1239)
```python
# Before (BROKEN):
reshaped_subgoal = subgoal.reshape(
    [subgoal.shape[0] * subgoal.shape[1]] + list(subgoal.shape[2:])
).expand(state_representation.shape)  # expand fails

# After (FIXED):
reshaped_subgoal = subgoal.reshape(
    [subgoal.shape[0] * subgoal.shape[1]] + list(subgoal.shape[2:])
).reshape(state_representation.shape)  # reshape works
```

The `expand` operation was failing because it cannot change non-singleton dimensions. Using `reshape` instead allows the tensor to be reshaped to match the target dimensions when the total number of elements matches.

## Validation

A comprehensive test (`test_subgoal_reward_fix.py`) was created that:
1. Reproduces the original bug
2. Validates that the fix resolves the issue
3. Explains the dimension semantics

All tests pass successfully.

## Files Changed

- `hieros/hieros.py` (2 changes):
  - Line 394: Added `decode_subgoal` call for debug visualization  
  - Line 1239: Replaced `.expand()` with `.reshape()` in `_subgoal_reward`
- `test_subgoal_reward_fix.py` (new file): Validation test

## Impact

- **Minimal**: Only 2 lines changed in production code
- **Scope**: Fixes both the debug visualization issue and a latent bug in `_subgoal_reward`
- **Risk**: Low - the reshape operation is mathematically equivalent when dimensions are compatible
