# Summary: Evaluation Logging Bug Fix

## Questions Answered

### 1. Are the scores logged to jsonl files the same as scores logged to W&B?

**Answer: YES**

The actual evaluation scores (e.g., `avg_score`, `avg_length`) are identical in both JSONL files and W&B logs. They use the same `Logger` class which distributes metrics to all configured outputs.

However, **both** were affected by the timing bug - both showed eval scores at the wrong step.

### 2. What is the bug in the evaluation logic?

**The Bug:**

Evaluation metrics are logged at a **later step** than when evaluation actually occurred.

**Root Cause:**

In `embodied/run/train_eval.py`, the training loop structure is:

```python
while step < args.steps:
    if should_eval(step):
        # Evaluation runs here (e.g., at step 30)
        driver_eval(policy_eval, episodes=...)
        # BUG: No logger.write() called here!
    
    # Training continues, step counter advances
    driver_train(policy_train, steps=100)
```

When evaluation runs:
1. `per_episode` callbacks add eval metrics to the `metrics` object
2. But `logger.write()` is NOT called
3. Metrics sit in the `metrics` object
4. Training continues, step counter advances (e.g., from 30 to 50)
5. Later, when `should_log(step)` triggers in the training step handler, `logger.write()` is called
6. Eval metrics from step 30 get written with step 50

**Impact:**
- JSONL files show eval at wrong step (delayed)
- W&B shows eval at wrong step (delayed)
- Training curves appear incorrect
- Makes it hard to correlate eval performance with training progress

### 3. How should I address it?

**Solution: Add logger.write() immediately after evaluation**

The fix is minimal - just 2 lines added in `embodied/run/train_eval.py`:

```python
while step < args.steps:
    if should_eval(step):
        driver_eval(policy_eval, episodes=...)
        # FIX: Log immediately at the correct step
        logger.add(metrics.result())
        logger.write()
    driver_train(policy_train, steps=100)
```

**Why this works:**
- `logger.add(metrics.result())` - Adds eval metrics to logger at current step
- `logger.write()` - Writes to all outputs (JSONL, W&B, TensorBoard) immediately
- Eval metrics are now logged at the exact step when evaluation occurred

**Benefits:**
- ✅ Minimal change (2 lines)
- ✅ Fixes both JSONL and W&B simultaneously
- ✅ No change to training metric logging behavior
- ✅ No security vulnerabilities introduced
- ✅ Clear and maintainable

**Trade-offs:**
- Slightly more frequent I/O (eval metrics written immediately vs batched)
- This is acceptable since evaluation is relatively infrequent

## Files Changed

1. **embodied/run/train_eval.py** - The fix (2 lines added)
2. **EVAL_LOGGING_BUG_FIX.md** - Detailed documentation
3. **test_eval_timing_simple.py** - Test demonstrating the bug and fix

## Verification

Run the test to see the bug demonstrated:
```bash
python test_eval_timing_simple.py
```

The test shows:
- **Buggy behavior**: Eval at step 30 logged at step 50
- **Fixed behavior**: Eval at step 30 logged at step 30

## Conclusion

The bug has been identified, documented, and fixed with a minimal, surgical change. Both JSONL and W&B will now log evaluation metrics at the correct step, making training analysis much more accurate and reliable.
