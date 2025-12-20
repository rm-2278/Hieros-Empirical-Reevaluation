Tier 1: Smoke test
Purpose: Code logic & GPU memory
```
python hieros/train.py --configs atari100k small_model_size_old --task=atari_pong --steps=1000 --eval_every=500 --eval_eps=1 --log_every=1000 --batch_size=4 --batch_length=16
```

Tier 2: Pilot run
```
python hieros/train.py --configs atari100k --task=atari_pong --steps=10000 --eval_every=2000 --eval_eps=5 --batch_size=8 --batch_length=32
```
(Modify train_every to decrease number of updates per step if needed)


Tier 2.5: Longer pilot run
```
python hieros/train.py --configs atari100k --task=atari_pong --steps=40000 --eval_every=10000 --eval_eps=5 --batch_size=8 --batch_length=32
```
(2h 5m 53s on RTX 5070)


Tier 3: Full run
```
python hieros/train.py --configs atari100k --task=atari_pong --batch_size=8 --batch_length=32
```

### Notes
Default log_every: 1000 steps \

RTX 5070 (16GB):
- Tested up to batch_size=16, batch_length=64 in small_model_size_old config
- Tested up to batch_size=4, batch_length=64 in default config (batch_size=8 causes OOM)
- Tested up to batch_size=8, batch_length=32 in default config (batch_size=16 causes OOM)