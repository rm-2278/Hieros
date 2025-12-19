Tier 1: Smoke test
Purpose: Code logic & GPU memory
```
python hieros/train.py --configs atari100k --task=atari_pong
--steps=1000 --eval_every=500 --eval_eps=1
```

Tier 2: Pilot run
```
python hieros/train.py --configs atari100k --task=atari_alien
--steps=40000 --eval_every=100 --eval_eps=10
```
(Modify train_every to decrease number of updates per step if needed)

Tier 3: Full run
```
python hieros/train.py --configs atari100k --task=atari_alien
```