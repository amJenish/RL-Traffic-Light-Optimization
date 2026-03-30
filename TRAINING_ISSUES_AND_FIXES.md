# Training Issues & Fixes

Run analysed: `logs/2026-03-26_18-19-21_DoubleDQNPolicy_CompositeReward`

---

## What the data shows

| Metric | Value |
|---|---|
| Pre-train test mean reward | ~**-3.7** |
| Post-train test mean reward | ~**-9.6** |
| Train reward (episode 1) | **-363** |
| Train reward (episode 500) | **-190 to -220** |
| Epsilon at test | **0.05** (not 0) |
| Steps pre-train | **2155** (identical every day) |
| Steps post-train | **~2000** (varies) |

The pre-train test has **identical step counts (2155 every day)** — the untrained
policy at ε=1.0 in eval mode is still frozen at ε=1.0 (random walk). Random
switching on a lightly-loaded synthetic network averages out to near-zero reward.

The trained policy (ε=0.05) greedily follows a Q-network that overfit to the
5 train days. It makes deliberate but wrong decisions on test days — worse than
random.

---

## Root Causes

### 1. Q-network is overfitting 5 train days
5 train days is not enough variety. The model memorises train-day patterns and
fails to generalise to the 5 held-out test days.

### 2. Learning rate barely decays
`LR = 0.01 → 0.0001` via cosine, but after 100 epochs the LR is still
**~0.00867**. `TOTAL_UPDATES` is estimated at 360,000 but actual updates are
closer to 1,000,000 (≈2,000 decisions/episode × 500 episodes), so the cosine
scheduler thinks it has barely started.

### 3. Target network too infrequent
`TARGET_UPDATE = 200`. With 1,000+ gradient updates per episode, the online
network races far ahead of the target — Q-targets become stale and training
becomes unstable.

### 4. Replay buffer too large relative to training data
Buffer capacity 100,000 — early random-exploration transitions (meaningless
noise) stay in the buffer for hundreds of episodes, diluting useful samples.

### 5. Overshoot penalty dominates reward signal
Train rewards are −100 to −500 per episode while test rewards are −3 to −14.
The soft `OVERSHOOT_COEFF = 4.0` penalty is overwhelming the queue signal,
making the loss noisy and disconnected from actual traffic performance.

---

## Fixes (in priority order)

### Fix 1 — Increase training data
Increase `TRAIN_SIZE` to at least 7-8 days (requires more than 10 total days of
data) so the agent sees more traffic variety and can generalise.

### Fix 2 — Lower starting learning rate
```python
LEARNING_RATE = 0.001   # was 0.01
LR_MIN        = 0.00005 # was 0.0001
```

### Fix 3 — More frequent target network sync
```python
TARGET_UPDATE = 50   # was 200
```

### Fix 4 — Shrink replay buffer
```python
BUFFER_CAPACITY = 20_000   # was 100_000
```
Keeps recent, relevant transitions in the buffer and evicts stale random-phase
data faster.

### Fix 5 — Reduce overshoot penalty
```python
OVERSHOOT_COEFF = 1.0   # was 4.0
```
Or switch to hard enforcement (force a switch at `max_green`) to remove the
penalty from the reward signal entirely.

---

## Summary of recommended `main.py` changes

```python
LEARNING_RATE   = 0.001
LR_MIN          = 0.00005
TARGET_UPDATE   = 50
BUFFER_CAPACITY = 20_000
OVERSHOOT_COEFF = 1.0
TRAIN_SIZE      = 7      # if data allows
```
