# Win net: gated policy iteration on rollout targets

**Status:** design (branch `win-net`).

## Goal

Replace the manual `win-fit --rollouts K --player <dir>` rounds with one
command that iterates: the current champion net plays, its rollouts label
the positions, a candidate static net is fitted to those labels, and the
candidate replaces the champion only if it beats it head-to-head.

Why: two manual rounds took the net from 50.56% to 51.0% against Master,
and a third fell back to 50.6%. The loop must not hand a regressed net to
the next round, and should keep finding the smaller gains of later rounds.

Success = the champion's share against Master rises over the rounds and
ends above the best manual round (pi2, 51.0% over 1M games, seed 18).

## Scope

- New CLI subcommand `win-pi`, and a new module `src/dqn/win_pi.rs`.
  Reuses `rollout_positions_by`, `train_epochs`, `bench_vs_pair` and the
  win model save/load code from `win_train.rs` / `win.rs`.
- 2-player only. Same `WinModel` architecture; the width comes from the
  initial net.
- Not touched: `win-train`, `win-fit` (it stays for one-off fits),
  `pair_model/`, `champion.txt`, `dqn_model/`, the web app.
- Out of scope: resuming a stopped run, an SPRT gate, changing the
  architecture or the rollout labeller.

## The loop

State: the **champion** (a win net; round 0 may start from Master), a
**window** of labelled positions from the last `R` rounds, a round counter.

Round `i` (1-based):

1. **Generate.** `rollout_positions_by(positions, seed_i, champion, …, k,
   1.0)`: turn-end positions from the champion's self-play games, each
   labelled with the champion's `k`-rollout P(win). Pure rollout labels
   (blend 1); Master's static term is not used.
2. **Window.** Append the round's positions; drop the oldest round's when
   more than `R` rounds are held. A round whose candidate is rejected
   still contributes its data (it is fresh data from the same champion).
3. **Fit.** The candidate starts from the champion's weights and trains
   `epochs` epochs on the whole window with `train_epochs`, at learning
   rate `lr`, with a round-dependent shuffle seed.
4. **Gate.** Sequential paired head-to-head, candidate vs champion
   (see "Gate"). Promote on pass: the candidate becomes the champion.
5. **Yardstick.** Bench the champion (after the gate) against Master with
   `bench_vs_pair` for `master_games` games (0 = skip). Logged only; never
   decides anything.
6. **Save and log** (see "Output").

### Starting champion

- `--init <dir>`: a win model dir (e.g. `win_fit_r_best/w64`); its config
  sets the width.
- Without `--init`: round 1's player is Master (`PairStrategy`) and the
  candidate is a fresh net of `--width` (default 64) fitted from random
  init. Round 1's gate compares the candidate against Master; it passes
  when z > `gate_z` as usual. If it fails, the run stops with a message:
  there is no win net to continue from.

### Seeds

All derived from one `--seed` (default fixed), so a run is reproducible:

- positions of round `i`: `seed ⊕ hash(i, "pos")`
- fit shuffle and init of round `i`: `seed ⊕ hash(i, "fit")`
- gate dice of round `i`: `seed ⊕ hash(i, "gate")`, so each round's gate
  plays fresh dice (no selection on one fixed dice set across rounds)
- the Master yardstick uses `bench_vs_pair`'s fixed `BENCH_SEED`, so
  rows are comparable across rounds and runs.

## Gate

`head_to_head(a, b, pairs, seed) -> Vec<f32>`: for each pair index `p`,
two games with shared dice streams (as in `bench_vs_pair`), `a` in seat 0
then seat 1; returns each pair's mean score for `a` (win 1, tie ½,
loss 0), in `[0, 1]`. Bots are built per rayon worker via factories, like
`rollout_positions_by`. Pair dice come from `seed` and the pair index
only, so blocks can be played incrementally and the result does not
depend on thread count or block size.

Sequential decision, blocks of `gate_block` games (default 100 000, i.e.
50 000 pairs), up to `gate_max` games (default 1 000 000):

- after each block, over all pairs so far: share `s` = mean of pair
  scores; `se` = sample std of pair scores / √pairs; `z = (s − 0.5) / se`.
- `z > gate_z` (default 2.0) → **promote**, stop.
- `z < 0` → **reject**, stop (behind with no reason to expect a
  catch-up; an equal net costs nothing to drop).
- after `gate_max` games without either → **reject**.
- `se == 0` (identical play in every pair) → reject.

Why per-pair std: paired dice make the two games of a pair correlated, so
the pair, not the game, is the independent unit; this measures the
pairing's actual variance reduction instead of assuming one.

Peeking after every block raises the false-promotion rate of an equal net
above the one-shot ~2.3%; with at most 10 looks this is accepted, since
promoting an equal net is harmless and a worse net (true share < 0.5) is
very unlikely to reach z > 2.

## Output

`--dir <dir>` (default `win_pi`); the command refuses to start if it
exists (no silent overwrite; no resume).

- `<dir>/round-<i>/{config.json, model.mpk}`: every round's candidate,
  promoted or not (benchable with `QWIXXER_WIN_MODEL`).
- `<dir>/champion/{config.json, model.mpk}`: the champion after each round
  (rewritten on promotion; written at round 1 when starting from
  `--init`, so it always exists after the first round).
- `<dir>/training.csv`, one row per round:
  `round,positions,loss,gate_games,share_vs_champion,z,promoted,share_vs_master,elapsed_s`
  — `share_vs_master` empty when `master_games` is 0. A round-0 row
  benches the initial champion against Master when `--init` is given:
  `0,,,,,,,share,elapsed`.
- Console: one line per round with the same numbers, shares to 2 decimals.

## CLI

```
qwixxer win-pi [--init <dir> | --width 64] [--rounds 10]
               [--positions 300000] [--rollouts 64] [--window 2]
               [--epochs 20] [--lr 3e-4]
               [--gate-z 2.0] [--gate-block 100000] [--gate-max 1000000]
               [--master-games 200000] [--seed <u64>] [--dir win_pi]
```

`--lr` defaults to 3e-4, below `win-fit`'s 1e-3, because the candidate is
warm-started from a net already near the targets; a from-scratch round 1
(no `--init`) uses 1e-3.

Validation before any work: `--rollouts ≥ 1`, `--window ≥ 1`,
`gate_block` even and ≥ 2, `gate_max ≥ gate_block`, `--dir` absent,
`--init` loadable.

## Cost

A round ≈ one manual `win-fit --rollouts 64 --player` round (labelling
dominates) plus 100k–1M gate games between two win nets plus the Master
bench. Window fitting costs `R`× a single round's fit.

## Testing

- `head_to_head`: deterministic for a seed; a bot against itself gives
  every pair score exactly ½ (a deterministic bot plays the same game
  twice; it scores x in seat 0 and 1 − x in seat 1); result independent of how the
  pairs are split into blocks.
- Gate decision as a pure function of pair scores: promote / reject on
  the z thresholds, reject on `z < 0`, reject at `gate_max`, reject on
  zero variance.
- Window: keeps at most `R` rounds, drops the oldest, keeps a rejected
  round's data.
- Seeds: round seeds differ by round and by purpose; same `--seed` → same
  positions.
- End-to-end smoke: two rounds with tiny settings (few positions, k = 2,
  small gate) from a tiny `--init` net write `round-1/`, `round-2/`,
  `champion/` and a 3-row `training.csv`; `--dir` existing → refuses.
