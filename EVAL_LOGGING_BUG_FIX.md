# Evaluation Logging Bug - Analysis and Fix

## Problem Statement

There is a fundamental bug in when evaluation scores are logged to jsonl files and W&B (Weights & Biases). The scores logged to both outputs are the same, but they are logged at the **wrong step** (a later step than when evaluation actually occurred).

## Bug Analysis

### Current (Buggy) Flow

1. **Step 30**: `should_eval(step)` triggers → evaluation runs
2. **Step 30**: `per_episode` callbacks execute, adding eval metrics to the `metrics` object
3. **Step 30**: Evaluation completes, but **NO `logger.write()` is called**
4. **Steps 30-50**: Training continues with `driver_train(policy_train, steps=100)`
5. **Step 50**: `should_log(step)` triggers in `train_step` function
6. **Step 50**: `logger.add(metrics.result())` includes the eval metrics from step 30
7. **Step 50**: `logger.write(fps=True)` writes everything to jsonl/W&B
8. **Result**: Eval metrics from step 30 are logged with step 50

### Code Location

The bug is in `embodied/run/train_eval.py`, lines 161-166:

```python
while step < args.steps:
    if should_eval(step):
        print("Starting evaluation at step", int(step))
        driver_eval.reset()
        driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps))
        # BUG: No logger.write() here!
    driver_train(policy_train, steps=100)
```

### Impact

- **JSONL files**: Eval scores appear at a later step than when evaluation occurred
- **W&B logs**: Eval scores appear at a later step than when evaluation occurred
- **The scores themselves are correct**, but the step/timestamp is wrong
- This makes it appear that evaluation happened later than it actually did
- Creates confusion when analyzing training curves and comparing different runs

## Answer to Original Questions

1. **Are scores logged to JSONL same as scores logged to W&B?**
   - Yes, the actual scores are the same in both JSONL and W&B
   - However, they are logged at the wrong step in both outputs

2. **What is the bug?**
   - Evaluation metrics are logged at a later step than when evaluation actually occurred
   - This happens because `logger.write()` is not called immediately after evaluation completes
   - The metrics sit in the `metrics` object until the next `should_log(step)` trigger

3. **How should I address it?**
   - **Minimal code change**: Add `logger.write()` immediately after evaluation
   - This ensures eval metrics are logged at the correct step

## Fix Applied

### Code Change

Added two lines immediately after evaluation in `embodied/run/train_eval.py` (lines 167-168):

```python
while step < args.steps:
    if should_eval(step):
        print("Starting evaluation at step", int(step))
        driver_eval.reset()
        driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps))
        # Log evaluation metrics immediately at the correct step
        logger.add(metrics.result())
        logger.write()
    driver_train(policy_train, steps=100)
```

### Why This Works

- `logger.add(metrics.result())`: Adds accumulated metrics (including eval metrics) to the logger at the **current step**
- `logger.write()`: Writes the metrics to all outputs (JSONL, W&B, TensorBoard) immediately
- Both jsonl and W&B use the same logger, so they will both log at the correct step

### Trade-offs

- **Pro**: Eval metrics are now logged at the correct step
- **Pro**: Minimal code change (2 lines)
- **Pro**: Works for both JSONL and W&B outputs
- **Potential Con**: Slightly more frequent writes to disk/network (once per eval instead of batched with training logs)
- **Note**: This does not change when training metrics are logged - they continue to be logged according to `should_log(step)`

## Testing

See `test_eval_timing_simple.py` for a demonstration of:
- The buggy behavior (eval metrics logged at wrong step)
- The fixed behavior (eval metrics logged at correct step)

Run with:
```bash
python test_eval_timing_simple.py
```

## Verification

To verify the fix works in a real training run:

1. Start a training run with evaluation enabled
2. Check the metrics.jsonl file (or W&B dashboard)
3. Verify that eval metrics (e.g., `eval_episode/avg_score`) appear at steps that match the `eval_every` configuration
4. Previously, there would be a delay of approximately `log_every - eval_every` steps

## Alternative Approaches Considered

1. **No code change** (document only): Would not fix the actual bug
2. **Separate logger instances**: More invasive, would require refactoring
3. **Immediate write with flag**: Could add complexity without clear benefit

The chosen approach (Option 2 from the original plan) provides the minimal, surgical fix that resolves the issue.
