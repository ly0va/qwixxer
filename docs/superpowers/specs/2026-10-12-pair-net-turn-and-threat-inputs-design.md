# Pair Net: Turn-Position and Threat Inputs — Design

**Date:** 2026-10-12
**Status:** approved in conversation; spec under review
**Branch:** `pair-inputs` (from `leaf-calibration`)

## Goal

Give the pair value net inputs it provably lacks, so the static net (and, if
it transfers, pair-search) improves, without breaking 2–4 player play.

Evidence motivating each input:

- **Turn position.** Phase 23's `leaf_calibration` measured a ±2pp
  side-to-move bias (Φ(a·z + b) fits b ≈ +0.064 / −0.056): the net cannot know
  whose roll is next, so it averages over it. Leaf correction did not help
  search (comparisons cancel the offset), but training targets bootstrap across
  decision points with different sides to move, where the bias does not cancel.
- **Game-end pressure.** Phases 16/19: the net cannot tell good safe locks from
  bad (game-shortening while behind); 10× lock-sample weight did not help —
  representation-limited. Phase 23: the leaf is ~25% overconfident once a row is
  locked (a ≈ 0.75). In multiplayer the net sees per-row detail only for the
  paired (leading) opponent; other opponents appear as an aggregate.

Success is judged by head-to-head against the current checkpoint (Phase 22
rule), with mechanism probes that must move for the gain to be attributed to
these inputs.

## Scope

In: 7 new per-board inputs (45 → 59 total), turn-position plumbing through the
`Bot` evaluation API, a model migration tool, a head-to-head training bench,
tests, and the evaluation recipe.

Out (deferred): richer mark probabilities (per-row near-miss probabilities);
"locks remaining" (redundant with per-row locked flags, dropped); training from
scratch; any change to search, meta-rules, or the web crate (which does not use
the pair net). Player counts above 4 are unsupported for the new inputs
(clamped, debug-asserted).

## 1. Inputs

### Layout

Each board block grows from 20 to 27; the existing 20 keep their indices and
the new 7 are appended:

| block index | input | value |
|---|---|---|
| 0–19 | unchanged: per-row progress, marks, locked, weighted prob (4 each); strikes, blanks, aggregate wprob, lockable count | — |
| 20 | turns until this player rolls | `rolls_in / 3`, clamped to ≤ 1 |
| 21 | colour phase pending (this player is the active player, mid-turn) | 0 / 1 |
| 22 | P(forced strike) | exact, [0, 1] |
| 23–26 | P(row r locked by someone else within the next round), r = 0..3 | exact, [0, 1] |

Full vector: `[our block 27 | paired-opponent block 27 | pair-level 5]`. The
pair-level inputs (cdiff, opponent count, max opp progress, max opp strikes,
opp lockable sum) are unchanged in meaning and move from 40–44 to 54–58.
`BOARD_FEATURES = 27`, `PAIR_FEATURES = 59`.

### Definitions

**`rolls_in`**: number of rolls before this player next rolls, 0 = rolls next.
Divided by 3 (max for 4 players). With 2 players it is 0 or 1.

**`pending`**: 1 iff this player is the active player of the current turn and
its colour phase (phase 2) has not happened yet. True only for evaluations made
at passive decisions (mid-turn) and for the opponent's simulated passive
decision inside our active phase 1 (where we are the pending active player).

**P(forced strike)**: probability that a full roll `[W1, W2, R, Y, G, B]`
offers this board no legal mark at all — no white-sum mark on any row and no
white+colour mark on any row — under the real rules (`Row::can_mark`,
including the 5-mark requirement on terminal numbers; locked rows offer
nothing). Computed exactly: for each of the 36 ordered white pairs, the board
is strike-forced iff the white sum is unmarkable everywhere and, for each row
independently, the row's colour die gives no markable `Wj + C`; so

    P = Σ_{w1,w2} (1/36) · [white sum unmarkable] · Π_rows (#{c : W1+c, W2+c unmarkable} / 6)

Independent of opponents and of turn position.

**Lock threat, row r, board b**: probability that some player other than b
gets an opportunity to lock row r during the next round of n rolls (one roll
per player). A player k ≠ b "can lock r" on a roll by s if, with that roll's
dice:

- the white sum equals r's terminal number and r is lockable for k now
  (k has ≥ 5 marks in r, row unlocked) — available on every roll; or
- s = k (k's own roll) and k can lock r with its colour die, either directly
  (`Wj + C = terminal`, ≥ 5 marks) or by a white mark followed by a colour
  mark in the same turn (4 marks → white mark in r → colour terminal).

Per roll by s, P_s(r) = P(dice give some k ≠ b a lock opportunity on r),
computed with the **shared** white sum handled exactly (not as independent
players). Rolls are independent, and each player rolls once per round, so
`threat = 1 − Π_s (1 − P_s(r))` regardless of order — no turn input needed.
The per-roll pieces depend only on each row's `(total, free)` (12 × 11
states) and are precomputed into tables. Opportunity, not intent: a player is
assumed to lock when able (production forces safe locks; game-ending locks are
taken only when winning — a documented over-estimate). The pending colour
phase at a passive decision is excluded (the `pending` input covers it). If
row r is already locked, threat = 0.

### Augmentation

`permute_colors` gains a fifth per-row group at block offset 23 (lock threat),
permuted exactly like progress/marks/locked/wprob. Inputs 20–22 are
row-invariant and untouched.

### Multiplayer properties

Every new input is a function of (this board's state, this board's clock, the
set of other players' states). None depends on the order of the opponent list.
Board-swap sample doubling recomputes the inputs from the paired opponent's
side.

## 2. Turn-position plumbing

### Types (new module `src/strategy/clock.rs`)

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct BoardClock { pub rolls_in: u8, pub pending: bool }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvalClock {
    pub us: BoardClock,
    /// Aligned with the `opp_states` slice passed alongside (≤ 3 opponents).
    pub opps: [BoardClock; 3],
}
```

Constructors take seat geometry relative to the evaluating player, with
opponents identified by **seat offset** (not list position), so non-turn-ordered
lists stay correct:

- `EvalClock::turn_complete(n, next_roller_offset, opp_offsets)` —
  nothing pending.
- `EvalClock::mid_turn(n, active_offset, opp_offsets)` — next roller is the
  seat after the active one; the active player's board is `pending`.
- Convenience forms for turn-ordered lists (`opp_offsets = 1..n`).

`rolls_in` for a board at seat offset `o` = `(o − next_roller_offset) mod n`.
`n ≤ 4` is debug-asserted.

### API change

`Bot::evaluate`, `evaluate_batch`, `evaluate_batch_multi` take an `EvalClock`
(per group for `_multi`). `WinProb::win_z_multi` / `win_prob_multi` take one
per group. GA (`DNA`) and the old DQN accept and ignore it. The pair bot passes
`clock.us` to the candidate block and the paired opponent's clock to its block.
Every call site must supply a clock (compile-enforced); there is no default.

### Clock at each call site

| call site | moment | next roller | pending |
|---|---|---|---|
| passive decision candidates (`passive_phase1_choices` → eval) | mid-turn | seat after the active player | active player |
| active phase-1 plans / phase-2 candidates | turn complete | seat after us | none |
| opponent's simulated passive decision inside our active phase 1 (`simulate_opp_phase1`) | mid-turn, their frame | seat after us | us |
| simulator rollout decisions (`sim.rs`, `BatchedRollouts`) | same rules, from `sim.active` and the player's seat | | |
| search leaves (`search_pick`) | turn start | `sim.active` | none |
| distillation entries / rollout finals | turn complete | the entry's `first_active` | none |
| lock adjudication / search shortlisting | as the decision they belong to | | |

### Training snapshots

`Snapshot = (State, Vec<State>, EvalClock)`, recording the clock used at that
decision. `build_pair_samples` and distillation emission read stored clocks for
both the forward and the board-swapped sample; they never recompute clocks from
list order. `PairSample` serde handles the larger feature array.

## 3. Migration and training

### `pair-migrate --from <dir> --to <dir>` (CLI subcommand)

- `PairModelConfig` gains an `input` size field (default `PAIR_FEATURES`); the
  tool loads a 45-input model with `input = 45`.
- First-layer weight rows map old → new: 0–19 → 0–19 (our block), 20–39 →
  27–46 (opponent block), 40–44 → 54–58 (pair-level); the 14 new rows are
  zero. Biases and later layers are copied unchanged.
- Writes only to `--to`; never touches `pair_model/` unless pointed there.
- Loading a model whose input size differs from `PAIR_FEATURES` fails with an
  explicit message naming `pair-migrate`.
- A migrated net is **bit-identical** in output to the original: each new
  input contributes `0.0 · x = 0.0`, which leaves float sums unchanged.
- Old `iter-*.mpk` and `ckpt/` are incompatible; reference models for
  `pair-alt` must be migrated (they then play identically).

### `pair-train --bench-ref <dir>`

Per-iteration benchmark becomes a static head-to-head against the reference
pair model in `<dir>` (paired dice, fixed seed, reported as margin and z)
instead of vs GA. Without the flag, behaviour is unchanged.

### Training recipe

    B=./target/release/qwixxer
    $B pair-migrate --from pair_model --to pair_model_mig
    cp -r pair_model_mig pair_model_ref                  # frozen reference = current, migrated
    cp pair_model_mig/model.mpk pair_model/model.mpk     # warm start
    $B pair-train --distill --distill-passive --distill-m 2 --epsilon-lock 0.10 \
      -g 5000 -e 5 -c --start-iteration 90 -i 30 -b 500000 --bench-ref pair_model_ref

Warm start from the migrated production model; recipe otherwise identical to
Phase 22, so gains are attributable to the inputs.

## 4. Tests

Input math:
- P(forced strike) equals brute force over all 6⁶ rolls with real move
  generation (cases: terminal 5-mark rule, locked rows, sparse and dense
  boards).
- Lock threat equals an independent reference: per roll, enumerate 6⁶ dice
  and ask the move generator whether any other player can lock row r
  (including the white-then-colour 4-mark case); multiply across rolls. For 2,
  3, 4 players, including a board whose own player could lock (excluded).
- Colour symmetry: inputs of a colour-swapped game (via `State::from_parts`;
  asc↔desc mirrored) equal the permuted inputs of the original, for all three
  swaps.

Turn plumbing:
- Constructor table: n = 2, 3, 4 × complete/mid-turn × seat offsets.
- Order invariance: shuffling `opp_states` together with clock entries leaves
  every board's inputs unchanged.
- Ground truth vs `Game::play`: a recording strategy logs every clock it
  receives over full 2-, 3-, 4-player games; each is checked against the true
  seat of the next roller and the pending player. The simulator equivalence
  test (`sim.rs`) is extended to compare clocks.
- Board-swapped samples equal `pair_features` computed fresh from the paired
  opponent's perspective with its true clock.

Migration:
- Migrated net output is bit-identical to the original on random positions and
  arbitrary clocks; index map tested directly; load guard message tested.
- End-to-end: `pair` and `pair-search` with the migrated production model
  produce byte-identical bench output to the current build — 1v1 seed 3,
  3-player seed 5, 4-player — proving the plumbing changed nothing else.
- Full `cargo test --lib --release`; all examples build.

## 5. Evaluation (pre-registered)

- **Primary (adoption):** static head-to-head, new vs migrated current
  (`pair` vs `pair-alt`), 200k games, seed 42: margin > 0 at z ≥ 2; and no
  regression in the Phase 22 suite (vs GA, DQN, Opportunist, Conservative; 3p
  and 4p mixes).
- **Headline, reported not required:** pair-search head-to-head (`pair-search`
  vs `pair-search-alt`), 50k games.
- **Mechanism probes (predictions):** `leaf_calibration` side-to-move
  |b| shrinks from ≈ 0.06 to < 0.03; the Phase 17 lock-ab suppression edge
  grows above +0.04pp. If static improves but neither moves, the write-up
  states the gain is not attributable to the intended mechanism.
- Checkpoint selection: best on the `--bench-ref` head-to-head curve; confirm
  the top 2–3 with the 200k primary bench.
- Results recorded as an EXPERIMENTS.md phase, including failures.

## Risks

- **Hot-path cost.** Lock threat and forced strike add per-evaluation work in
  search rollouts. Mitigation: table lookups; measure games/s before/after;
  forced strike can be cached per state if needed.
- **Plumbing errors that do not crash.** A wrong clock yields wrong inputs
  silently. Mitigation: compile-enforced clock parameter, ground-truth tests
  against `Game::play`, byte-identical migrated benches.
- **No transfer to search.** Phase 22 showed static gains may not reach
  pair-search; the headline metric is reported, not required.
