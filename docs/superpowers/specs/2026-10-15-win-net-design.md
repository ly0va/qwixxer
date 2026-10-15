# Win net: a symmetric P(win) network trained TD-Gammon style

**Status:** design (branch `win-net`).

## Goal

Today's pair net predicts the final score difference as a Gaussian
(μ, σ); P(win) is derived as Φ((cdiff + μ) / σ). Try a net that predicts
P(win) directly, trained by TD(λ) self-play on win/loss outcomes like
TD-Gammon, with the game's symmetries built into its shape.

Success = a stronger Master: the new static bot beats today's Master
head-to-head, and Master+ built on it beats today's Master+.

## Scope

- 2-player games only. Output = P(I finish ahead of my opponent), a tie
  counting ½.
- Trained from scratch (random init), pure self-play.
- Not touched: `pair_model/`, `champion.txt`, the web app. Shipping the
  net (a hand-written forward pass for WASM, retiring `pair_model/`) is a
  separate decision after the results.
- Out of scope: 3+ players, an online (per-move) TD trainer, a warm start
  from the current net.

## Inputs

The same information as today's 45 pair features, regrouped. Per player,
from `board_features` (`src/dqn/pair.rs`):

- per row: progress, marks (/11), locked, probability weight
  (`f[i]`, `f[4+i]`, `f[8+i]`, `f[12+i]`);
- global: strikes, blanks, aggregate probability weight, lockable rows
  (`f[16..20]`).

Arranged as, from "me" against "opp":

- 4 **row tokens** of 8 numbers: `[me row features | opp row features]`,
  in the order red, yellow, green, blue;
- 9 **globals**: `[me globals | opp globals | cdiff / 100]` (cdiff
  clamped to ±1 as today).

Dropped from the 45: opponent count and the three all-opponent summaries
(constant or duplicates in 2-player).

The **swapped** input (opp against me) swaps the halves of every token
and of the globals and negates cdiff; it is derived from the stored
input, never recomputed from states.

## Net

```
Row encoder φ (shared by all 4 rows)       8 → W → W        ReLU
Pool inside pairs        a = φ(red) + φ(yellow),  d = φ(green) + φ(blue)
Pair encoder ψ (shared by both pairs)      W → W            ReLU
Pool across pairs        r = ψ(a) + ψ(d)
Head                     [r | 9 globals] → 2W → W → 1       ReLU, linear out
                         = g(me, opp)

z = g(me, opp) − g(opp, me)          P(me ahead) = sigmoid(z)
```

W = 32 by default, set by the capacity check (below).

Properties, exact for any weights:

- **Antisymmetry:** z(opp, me) = −z(me, opp), so P(me ahead) +
  P(opp ahead) = 1; an even position (identical boards, cdiff 0) gives ½.
- **Colour invariance:** z is unchanged by red ↔ yellow, green ↔ blue,
  and ascending ↔ descending (red ↔ green with yellow ↔ blue), and so by
  all 8 relabellings they generate. The row features are already
  direction-free, so each relabelling only permutes the row tokens.

Both orientations run as one batch of twice the size. No board-swap or
colour-permutation augmentation in training: the shape makes them
redundant.

## Capacity check

Before self-play training, check that the shape can represent what
today's net knows:

- Positions: post-move states from 2-player games of today's Master
  against itself; 300,000 for training, 50,000 held out (separate
  seeds).
- Target: today's net's P(win) (`PairStrategy::win_prob_multi`) for the
  position.
- Supervised fit (cross-entropy against the soft target, Adam 1e-3) at
  W = 16, 32, 64.
- Report held-out mean absolute error in P per width. Pick the smallest
  W with MAE ≤ 0.02; if none reaches it, report and stop for a decision.

The fitted nets are thrown away; training starts from random weights.

## Training

- **Games:** 2-player self-play, both seats the current net, ε-greedy
  exploration on today's schedule (0.2 · 0.95^iter, floor 0.07).
- **Stored per game:** two chains, one per player, of the input after
  each of that player's decisions (active and passive, as today's pair
  recorder), and the outcome (1, 0, ½ from that player's side).
- **Fresh targets:** before each training pass, the current net
  re-evaluates every stored input in the replay buffer and the
  λ-returns are rebuilt per chain:
  `G_last = outcome`, `G_t = (1 − λ)·P(x_{t+1}) + λ·G_{t+1}`, λ = 0.8.
- **Loss:** binary cross-entropy between sigmoid(z) and G_t. Adam,
  learning rate 4e-4.
- **Iterations:** 5,000 games each; replay buffer of the last 3
  iterations; 2 epochs per iteration over the buffer; 160 iterations
  (800,000 games, today's net's budget). All configurable.
- **Each iteration:** save a checkpoint and run a 400-game paired bench
  against Master (`pair`), logged as the learning curve.
- Deterministic from the base seed, like `pair-train`.

## Integration

- `src/dqn/win.rs`: `WinModel` (burn), input building, and `WinStrategy`
  implementing `Bot` and `WinProb`. Panics with more than one opponent.
- `src/dqn/win_train.rs`: the trainer and the capacity check.
- Model dir `win_model/` (`model.mpk`, `iter-N.mpk` checkpoints).
- CLI: `win-train` (training), `win-fit` (capacity check); bench bots
  `win` (static) and `win-search` (`SearchBot<WinStrategy>`, production
  `SearchConfig`).

**Value units.** The search assumes a bot's static value is a probit
z-score: the blend uses Φ(value) (`src/strategy/search.rs:608`, `:756`)
and the 0.15 gate is a gap in those units. `WinStrategy::evaluate`
therefore returns Φ⁻¹(sigmoid(z)), the probit equivalent of its P(win).
Move order is unchanged (monotone), and search, gate and blend apply as
they are. `WinProb::win_prob_multi` returns sigmoid(z). The static
pipeline (`src/strategy/bot_impl.rs`) is checked for any other threshold
in value units.

## Success criteria

1. Capacity check: some W ≤ 64 reaches held-out MAE ≤ 0.02.
2. Static: `win` vs `pair`, 20,000 paired 2-player games; success = the
   99% paired CI on `win`'s share lies above 50%. Both bots also benched
   against GA for context.
3. Search: `win-search` vs `pair-search`, a 10,000-game smoke, then
   100,000 games on the big machine if it leans positive.

Results go to `docs/EXPERIMENTS.md` as a new phase, win or lose.

## Testing

- Antisymmetry: random inputs; z(swapped) = −z within float tolerance,
  for a randomly initialised net.
- Colour invariance: each of the 8 relabellings of real positions leaves
  z unchanged (tolerance), for a random net.
- Even position: identical boards and cdiff 0 give P = ½.
- Input building: tokens and globals match `board_features` of both
  states; the swapped input equals the input built with the players
  exchanged.
- λ-returns: a hand-computed chain (outcome 1, ½, 0) matches.
- Fresh targets: after a weight change, rebuilt targets differ from the
  old ones and match a direct recomputation.
- Value units: `evaluate` = Φ⁻¹(`win_prob_multi`) within tolerance; order
  of candidates equals order of P.
- Panics with two opponents.
- Smoke: `win-train` with tiny settings (2 iterations, 20 games) runs end
  to end and writes a checkpoint.
