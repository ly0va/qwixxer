# Win net: tempo (who moves next)

**Status:** design (branch `win-tempo`, from `win-net`).

## Goal

Let the win net represent the value of moving next, and make every place
that evaluates it say who moves next.

The evidence (`docs/EXPERIMENTS.md`, Phase 28): the net's logit is
antisymmetric, z(a, b) = g(a, b) − g(b, a), so V(a, b) + V(b, a) = 1 and the
net cannot tell who moves next. Every label is taken at a turn end with the
opponent to move, and the opponent's active roll is worth something: from
the empty board the first player to roll scores 0.531 ± 0.0025 in
self-play, where the net must say 0.5. Both the old champion and E2 (fitted
from scratch on these labels) overrate the player who has just moved by
+0.022 on held-out turn ends. Doubling the data (r1 → r2) barely moved the
nets' own error on candidate differences, and W48 plays as well as W64,
which points at systematic error such as this rather than label noise or
capacity.

Success = a tempo arm T (below) has turn-end bias ≈ 0 on `win-eval`, lower
own error than the matched control on the benchmark and on r2's held-out
set, and beats the matched control and E2 head to head with z > 2.

## Scope

- `WinModelConfig` gains one option, a tempo head (off by default) saved in
  `tempo.mpk`; `ManualWinNet` and `WinStrategy` play tempo nets; `win-arm --tempo`
  fits them.
- The `Bot` and `WinProb` traits gain a who-moves-next flag whose default
  ignores it; the passive decisions and the search leaves pass it.
- 2-player only, as the win net is.
- **Unchanged, bit for bit:** every bot without a tempo head (the pair net
  / Master, every existing win net, GA, the heuristic bots), the web app
  (it uses only the pair net), the data format and `win-data`, model
  artifacts (`win_model/`, `pair_model/`).
- Out of scope: playing out the opponent's remaining colour phase before a
  passive evaluation (option (b) in the design discussion; a later arm if
  tempo pays), a tempo part for the auxiliary difference head (aux is not
  used in play), new inputs.

## The frame

The net answers one question: **P(first player wins | the first player has
just finished a turn, the second player moves next)**. This is the frame of
every stored sample (turn ends, and candidates, which `win-data` already
stores from the side of whoever has just moved). Both kinds of decision
ask it:

- Opponent moves next (every active-player decision): V(me, opp).
- I move next (passive phase-1 decisions): 1 − V(opp, me). The opponent is
  the one who has just moved; in data terms this is the opponent's turn
  end. Approximation: at a passive decision the opponent's colour phase is
  still to come, so its board is at most one colour mark short of the turn
  ends the net is trained on.

So no new data, frame inputs or labels are needed. Any function of (a, b)
splits into a part that flips sign when the players swap and a part that
does not, so with both parts the net is unconstrained; the data contains
positions and their mirrors, each as "first player has just moved", and
their labels sum to less than 1 by the value of moving next. That shortfall
is what the symmetric part learns.

## Model

`WinModelConfig` gains `tempo: Option<bool>` (`None` in older dirs, as
`thermo` and `aux`; `is_tempo()` = `Some(true)`).

The tempo head is one `Linear` (W → 1, with bias) on the trunk's last
hidden layer h (the input of `out`). Like the auxiliary heads it lives
beside the net, in `tempo.mpk`, so `WinModel` and `model.mpk` are unchanged
and every existing dir loads exactly as today. Its weight and bias start at
**zero**, so a fresh tempo net starts as exactly the antisymmetric net it
extends.

Logit, for input x and its player swap x′:

    z = g(x) − g(x′) + t(x) + t(x′),   t(x) = tempo(h(x))

z is the logit of P(first player wins) in the frame above;
½ (z(a, b) + z(b, a)) = t(x) + t(x′) is minus the value of moving next, in
logit units. The trunk already runs on both orientations, so the cost is
one W-wide dot product per orientation.

Implemented in `ArmModel::forward` (training; `ArmModel` gains
`tempo: Option<Linear>`) and `ManualWinNet` (`logit`, `logit_states`,
`logit_batch`; `with_tempo` copies the head). `save_tempo` writes
`tempo.mpk`; `WinStrategy::load` loads it when the config says
`"tempo": true`.

## Who moves next: the traits

    pub enum Next { Opp, Me }

`Bot` (in `src/strategy/mod.rs`) gains:

    fn evaluate_batch_next(&self, candidates: &[State], opp_states: &[State], next: Next) -> Vec<f32>
    fn evaluate_batch_multi_next(&self, groups: &[(&[State], &[State], Next)]) -> Vec<Vec<f32>>

whose defaults drop `next` and call `evaluate_batch` /
`evaluate_batch_multi`. `WinProb` gains `win_prob_multi_next(groups: &[(&State, &[State], Next)])`,
default likewise.

`WinStrategy` overrides all three. **Without a tempo head it hands every
call to today's methods unchanged**, so its results are bit-identical
whatever `next` says (no reliance on floating-point identities such as
probit(−z) = −probit(z)). With a tempo head:

- `Next::Opp`: as today, from z(cand, opp).
- `Next::Me`: from −z(opp, cand), i.e. P = 1 − P(opp, cand), in the same
  units as today (probit for `evaluate*`, P for `win_prob_multi*`).
- Groups of both kinds in one `*_multi_next` call still run as one batch.

`bot_impl` gains `eval_decision_next(bot, decision, eval_opps, next)`;
`eval_decision` stays as the `Next::Opp` case.

### Call sites that pass `Next::Me`

| Site | What |
|---|---|
| `bot_impl::passive_phase1_impl` | every bot's passive decision (production bots, `game.rs`, the `TurnEnds` recorders that play `win-data` games) |
| `sim::play_sim_turn` | the passive decision (both fidelities) |
| `sim::BatchedRollouts::step_turn` | passive pending choices: `PendingChoice` carries its `Next`, `resolve` calls `evaluate_batch_multi_next` |
| `search.rs` passive path | the shortlist's static values and the unsearched fallback in `explain_passive_phase1` |
| `win_distill::capture_passive` | the static values that rank and gate the candidates |
| search leaves (`win_prob_multi` on the survivors at the horizon) | `Next::Me` when the survivors' next mover is seat 0 (the entry's `first_active` is 0 after whole rounds), else `Next::Opp` |

Everything else stays `Next::Opp` by keeping today's calls: every
active-player decision, `passive_entry`'s internals (they evaluate the
active player's own decisions), `win-eval`, the probes, `pair_train`
(pair net only).

## Fit: `win-arm --tempo`

- From scratch: `WinModelConfig::with_tempo(Some(true))`; the config written
  to the arm dir records `"tempo": true`.
- With `--init`: a net without a tempo head gains a zero one (as
  `widen_to_thermo` widens a plain net), so the warm start begins as the
  init net; a tempo init loads its `tempo.mpk` and needs `--tempo`.
- No other change: loss, schedule, EMA, best-epoch selection and the
  held-out measurements are as today; the tempo head is in the optimizer's
  parameters and the weight average with the rest of `ArmModel`.

## The experiment

On `win_data_r2` (E's play, 800k turn ends, 64 rollouts):

| Arm | Flags |
|---|---|
| E2′ (control) | `--thermo --aux --lr 1e-3 --epochs 10` |
| **T** | the same + `--tempo` |

Both from scratch with the same seed, so they differ only in the head.
`--epochs 10` because E2 (20 epochs) peaked at epoch 7 and then drifted.

Measured:

- `win-eval` on `bench_e2` and on `win_data_r2`: turn-end bias (T ≈ 0,
  E2′ ≈ +0.022), own error on candidate differences, pick agreement,
  regret.
- `win-gate` T vs E2′ and T vs E2, 1M games each; T vs Master; T static
  and win-search vs GA (`bench`).
- The 1-ply probe on T, to see whether the concentration of error
  changes.

Decision: T is promoted if it beats E2 with z > 2. If T and E2′ tie, tempo
does not pay in play at option (a); option (b) is the next arm only if T
shows a clear own-error gain.

## Testing

Model:

- A tempo net with a zero head gives the same logits as the same weights
  without the head (burn and `ManualWinNet`).
- With a non-zero head, z(a, b) + z(b, a) = 2·(t(x) + t(x′)) and
  z(a, b) − z(b, a) = 2·(g(x) − g(x′)).
- `ManualWinNet` matches the burn forward (`ArmModel::forward`, and the
  trunk plus head by hand) for a tempo net, plain and thermometer inputs.
- Save and load round-trip a tempo net (`tempo.mpk`); a model dir without
  `tempo` in its config loads as a non-tempo net.

Traits and call sites:

- `WinStrategy` with a tempo head: `Next::Me` equals 1 − P(opp, cand)
  (`win_prob_multi_next`) and −probit-of-z(opp, cand) (`evaluate*_next`);
  mixed `Next` groups in one multi call equal the separate calls.
- Without a tempo head (`WinStrategy`) and for the pair net, `*_next` with
  `Next::Me` returns exactly the `Next::Opp` result on random positions,
  and whole games and searches are unchanged (existing determinism tests
  stay green).
- A recording `Bot` (logs the `Next` of every call) shows: passive
  decisions in `passive_phase1_impl`, `play_sim_turn`, `BatchedRollouts`,
  the search passive path and `capture_passive` pass `Me`; active
  decisions pass `Opp`.
- Search leaves: a recording `WinProb` sees `Me` for entries with
  `first_active` 0 and `Opp` otherwise.

Fit:

- `win-arm --tempo` writes `"tempo": true` and a model that loads and
  plays.
- On a small synthetic dataset whose labels give the player to move a
  fixed advantage (mirror pairs labelled p and 1 − p − δ), a tempo arm
  learns ½(z(a, b) + z(b, a)) < 0 and fits better than the same arm
  without the head.
