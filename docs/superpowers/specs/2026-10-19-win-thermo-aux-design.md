# Win net: thermometer inputs, auxiliary heads, held-out fitting

**Status:** design (branch `win-net`).

## Goal

Find out whether the win net is limited by how it is fitted, by how its
inputs are encoded, or by how little each rollout teaches it, and get a
stronger static net out of whichever it is.

The evidence so far: every `win-pi` fit warm-started from a champion was
rejected except two (turn-end rounds 2–7: z −1.20 … −2.30; candidate
rounds after the first promotion: z −0.13, −1.06, −4.10), and the worst
came from the longest fits on the cleanest data. Each fit restarts Adam at
a constant lr 3e-4 for 20 epochs and keeps the last weights, and no fit
has ever been measured on held-out data, so drift, overfitting and real
learning cannot be told apart. The champion's remaining error on
candidate differences is about 0.018 in P(win).

Success = an arm (below) beats the current champion
(`win_pi_cand/round-1`, 51.31% vs Master) head to head with z > 2, and
scores above 51.31% against Master. Either way, the held-out curves say
whether the earlier rejections were drift, overfitting or a capacity
limit.

## Scope

- New: a stored dataset with auxiliary labels (`win-data`), a fit command
  for one arm with held-out measurement (`win-arm`), and a head-to-head
  command between two win model dirs (`win-gate`).
- `WinModel` gains two config options, thermometer inputs and auxiliary
  heads, both off by default; `ManualWinNet` and `WinStrategy` play
  thermometer nets.
- 2-player only. Same trunk shape as today (W = 64).
- Not touched: `win-pi`, `win-search`, `pair-train`, model artifacts
  (`win_model/`, `pair_model/`, `champion.txt`, `dqn_model/`), the web app.
- Out of scope: SPRT, the decision benchmark, wiring the new fit into
  `win-pi`, shipping a thermometer net to WASM. Each is a follow-up once
  this experiment has a result.

## Data: `win-data`

One labelled dataset, generated once and reused by every arm.

**Games and samples.** The same as one `win-pi --labels cand` round:
turn-end positions from a player's self-play games (`--player DIR`, a
win model dir, required), plus the candidates of the decisions
win-search would search (top 3 by static value, production gate and
margin), built by
`phase1_entry` / `phase2_entry` / `passive_entry`. Default size 400,000
turn ends (two rounds' worth, about 1.7M candidate samples), `k = 64`
rollouts for both kinds (`--rollouts`, `--cand-rollouts`).

**Labels**, all means over the sample's `k` full-game CRN rollouts by the
player (an ended position gets its exact values):

| Label | Value |
|---|---|
| `win` | mean outcome for me (1 / ½ / 0) |
| `win_var` | sample variance of the `k` outcomes (0 when ended) |
| `marks[8]` | final row totals, me then opp, red/yellow/green/blue, / 12 |
| `strikes[2]` | final strikes, me then opp, / 4 |
| `diff` | final (my − opp) points / 100 |

They come from the final states `rollout_final_states` already returns;
the labellers today reduce those to the outcome only.

**Frame.** Every sample is stored in the turn-end frame (opponent to move
next), as now. A passive candidate (me to move next) is stored with the
players exchanged: states (opp, me), `win → 1 − win`, the two halves of
`marks` and of `strikes` exchanged, `diff → −diff`.

**Stored as states, not inputs.** Each sample keeps its two `State`s, so
any input encoding can be built at fit time. Candidate samples also keep
their decision (index and candidate order) and each decision its paired
outcome variances, as `CandGroup` does today, for the difference
diagnostic.

**Held-out split by game.** A game is held out when a hash of its index
and the seed puts it in the held-out share (`--holdout`, default 0.1); all
its turn ends and candidates go with it. No game contributes to both
sides.

**File.** One directory: a flat little-endian binary file with a version
header, plus `meta.json` (player, seed, sizes, k, holdout share). No new
dependency. Generation is deterministic from `--seed`.

## Model

`WinModelConfig` gains `thermo: bool` and `aux: bool` (both default
false). Existing model dirs, whose `config.json` lacks them, load
unchanged and play bit-for-bit as before.

### Thermometer inputs (`thermo`)

Per row and player, after today's 4 scalars:

- 10 progress bits: bit j = [cells passed ≥ j], j = 1…10, where cells
  passed = `row_progress` × 10 (a locked row has all 10 set);
- 12 mark bits: bit j = [row total ≥ j], j = 1…12.

A row token becomes 52 numbers: `[my 26 | opp 26]`. Globals gain 3 strike
bits per player, [strikes ≥ 1, ≥ 2, ≥ 3], after today's 9: 15 in all. The
input is 4 × 52 + 15 = 223. The swapped input exchanges the halves of
every token and of the per-player globals and negates cdiff, as today.

All features are measured from each row's own start, so the 8 colour
relabellings still only permute the tokens, and antisymmetry and colour
invariance remain exact.

### Net

```
row encoder φ (shared)    token → 64 → 64        ReLU    e_r
pool in colour pairs      a = e_red + e_yellow,  d = e_green + e_blue
pair encoder ψ (shared)   64 → 64                ReLU
pool across pairs         r = ψ(a) + ψ(d)
head                      [r | globals] → 128 → 64   ReLU    h(x)
win output                64 → 1                         g(x)
P(me ahead) = sigmoid(g(x) − g(swap x))     ← the only output used in play
```

Unchanged from today except the token and globals widths.

### Auxiliary heads (`aux`), training only

All linear, all reading the trunk on both orientations (the forward pass
already runs x and swap x as one batch):

- **row marks**, one map shared by the 4 rows: my final total in row r
  = `m([e_r(x) | h(x)])`, the opponent's = `m([e_r(swap x) | h(swap x)])`
  (128 → 1);
- **strikes**: mine = `s(h(x))`, the opponent's = `s(h(swap x))`
  (64 → 1);
- **score difference**: `a(h(x)) − a(h(swap x))` (64 → 1).

Swapping the players exchanges the marks and strikes predictions and
negates the difference; a colour relabelling permutes the row
predictions with the rows. `ManualWinNet` and play ignore the heads.

### Warm start

`--init <dir>` with a plain (non-thermometer) net and `--thermo`: copy
every weight; the new input columns (bits in the row encoder's first
layer, strike bits in the head's first layer) start at zero; aux heads
start from their default init. The new net's P(win) equals the init's at
step 0 (within float tolerance). Without `--init`, all weights are
random (from scratch).

## Fit: `win-arm`

Fits one arm on a `win-data` dataset and saves a loadable model dir.

```
win-arm --data DIR --out DIR [--init DIR] [--thermo] [--aux]
        [--epochs 10] [--lr 3e-4] [--wd 1e-4] [--ema 0.999]
        [--aux-weights 1.0,0.5,0.5] [--seed 42] [--legacy-fit]
```

- **`--legacy-fit`:** today's fit, unchanged: `train_epochs` (Adam,
  constant lr, no weight decay, no averaging), the last epoch's weights
  saved. The held-out measurements still run every epoch. Only arm A0
  uses it.

- **Optimiser:** AdamW, weight decay `--wd`; batch 1024, as today.
- **Schedule:** learning rate decays from `--lr` to 0 on a cosine over
  all steps (1% linear warmup).
- **Weight averaging:** an exponential moving average of the weights
  (`--ema` per step); all evaluation and saving use the averaged weights.
- **Loss:** BCE(win) + w₁ · MSE(marks, mean of 8) + w₂ · MSE(strikes,
  mean of 2) + w₃ · MSE(diff), the last three only with `--aux`.
- **Every epoch, on the held-out games:**
  - win excess loss: mean BCE − mean H(win label), against its noise
    floor, the mean of `win_var / (2k · w(1 − w))` over labels with
    0 < w < 1 (what a perfect predictor would score on these noisy
    labels);
  - each aux head's MSE (with `--aux`);
  - candidate difference error: `diff_stats` on the held-out decisions
    (difference RMSE, its noise, flips);
  - the same win loss on the training set, for the train/held-out gap.
- **Early stopping:** keep the epoch with the lowest held-out difference
  RMSE; save its averaged weights to `--out`.
- **Output:** `--out/config.json`, `--out/model.mpk`, and
  `--out/fit.csv` with one row per epoch:
  `epoch,train_win_loss,heldout_excess,heldout_floor,heldout_marks,heldout_strikes,heldout_diff,diff_rmse,diff_noise,flip_share,best,elapsed_s`
  (aux columns empty without `--aux`).

## Gate: `win-gate`

```
win-gate --a DIR --b DIR|master [--games 200000] [--seed S]
```

Paired head-to-head (`head_to_head`, shared dice per pair, both seats):
prints games, a's share and z. With `--b master` it plays Master
(`pair_model/`). 200,000 games resolve about ±0.14pp at z = 2.

## The experiment

1. **Data:** `win-data` by the champion, 400,000 turn ends + candidates,
   k = 64, 10% of games held out (about 2 h on the big machine).
2. **Arms**, all fitted on that dataset:

   | Arm | Init | thermo | aux | Fit |
   |---|---|---|---|---|
   | A0 | champion | – | – | `--legacy-fit --epochs 20` |
   | A | champion | – | – | new fit |
   | B | champion | ✓ | – | new fit |
   | C | champion | – | ✓ | new fit |
   | D | champion | ✓ | ✓ | new fit |
   | E | scratch | ✓ | ✓ | new fit, `--lr 1e-3`, `--epochs 20` |

   A0 against A isolates the fit fixes; A against B, C and D the input
   and target changes; E shows how far from scratch gets on this data.
3. **Read the held-out curves first.** Training loss falling while
   held-out excess rises = overfitting; flat held-out excess well above
   its floor = underfitting; A0's held-out difference RMSE rising over
   its 20 epochs = drift.
4. **Gate** the arms with the best held-out difference error against the
   champion with `win-gate` (200,000 games), and the best of those
   against Master.

## Testing

- **Thermometer encoding:** bits match hand-computed rows (fresh row,
  mid row, row with 5 marks, locked row); strike bits; the swapped input
  equals the input built with the players exchanged.
- **Symmetry with thermo and aux, random weights:** z(swap) = −z; each of
  the 8 colour relabellings leaves z unchanged and permutes the row-marks
  predictions; swapping exchanges the marks and strikes predictions and
  negates the difference prediction.
- **Warm start:** a plain net widened with `--thermo` gives the same
  logits as the original on real positions (tolerance).
- **Old dirs:** a model dir without the new config fields loads and gives
  the same logits as before (golden values).
- **ManualWinNet:** matches the burn forward for a thermometer net
  (tolerance).
- **Labels:** aux labels for a hand-built ended position are exact; a
  passive candidate's sample has the players exchanged with `1 − win`,
  exchanged marks and strikes, and negated diff.
- **Dataset:** write → read round-trips exactly; generation is
  deterministic for a seed; no game index appears on both sides of the
  split; held-out share is near `--holdout`.
- **Fit:** cosine schedule values at start, middle, end; EMA of a known
  weight sequence; the noise floor on hand values; the saved weights are
  the best epoch's, not the last (a scripted metric sequence);
  `--legacy-fit` saves the same weights as `train_epochs` on the same
  data and seed.
- **Gate:** `win-gate` of a dir against itself gives 50% exactly.
- **Smoke:** `win-data` at tiny size, then `win-arm` for 2 epochs with
  `--thermo --aux`, write a loadable dir with a 2-row `fit.csv`; `bench
  win` plays it.
