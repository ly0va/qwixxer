# Win net: candidate distillation in `win-pi`

**Status:** design (branch `win-net`).

## Goal

Close part of the gap between the static win net and its search. `win`
(pi2) loses to `pair-search` 48.4–51.6, and `win-search` plays about as
well as `pair-search`, so search is worth about 1.6pp of share over the
static net. Seven `win-pi` rounds on turn-end rollout labels moved
nothing (all rejected; the loss sat at the label-noise floor).

Success = at least one promoted round, and the champion's share against
Master ends above pi2's 51.06% on the same yardstick. Expectation from
the pair net's distillation legs (Phases 18–22: +0.5, +0.25, +0.25pp
static, each closing a fifth or less of the static–search gap): about
+0.3pp from the first rounds, shrinking after that.

## Why candidate targets

The turn-end labels only ever label the move the net *chose*. The
alternatives at a decision get no label, so nothing in the data tells the
net it ranked two close candidates wrongly; and the move order is the
only thing the net's values are used for (Phases 24/25: a correction that
is the same for every candidate changes no pick).

Search's edge comes from exactly those close decisions: it rolls out the
top candidates on shared dice and picks by the result. Candidate
distillation labels every top candidate at those decisions with its own
full-game rollout value, on dice shared across the candidates (CRN), and
fits the net to those values alongside the turn-end labels.

The targets are full-game rollouts of the static champion (its policy's
value, V^π), as in `pair-train --distill`, not search's blended scores:
those mix in the static net itself and are cut at a one-round horizon,
so they would mostly teach the net its own values back.

## Scope

- `win-pi` gains candidate capture and labelling; the loop, gate, window,
  Master yardstick, seeds and output layout stay as they are.
- Code goes in `src/dqn/win_pi.rs` (or a new `src/dqn/win_distill.rs` if
  `win_pi.rs` passes ~1000 lines), reusing `rollout_final_states`,
  `phase1_entry`, `phase2_entry`, `passive_entry`, `passive_first_active`,
  `context_seed` and `gates_with` from `src/strategy/search.rs`, and the
  `*_choices` functions from `bot_impl.rs`.
- 2-player only, same `WinModel`, same BCE loss and `train_epochs`.
- Not touched: `win-search` itself, `pair-train`, `win-fit`, `win-gap`,
  model artifacts, the web app.
- Out of scope: lock firings (Phases 18–19 showed lock targets do not fix
  the lock blind spot by data alone; `win-search` keeps adjudicating
  them), a within-decision ranking loss, playing the games with search.

## Capture

The round's games are the same champion self-play games that produce the
turn-end positions (`rollout_positions_by`'s games, seeded `seed + 1000·g`).
A recorder wraps the champion in both seats; besides the turn ends, at
each of its decisions it rebuilds the candidates the static pipeline
ranks and keeps the decision when **win-search would search it**:

- **Active phase 1** (`active_phase1_choices`, plans collapsed to distinct
  phase-1 marks by best value), **active phase 2**
  (`active_phase2_choices`) and **passive phase 1**
  (`passive_phase1_choices`): a `Decision::Choices` with ≥ 2 candidates,
  values from the champion's `evaluate_batch`, and `gates_with(…,
  GATE_MARGIN)` close or endgame — the production gate, with the
  production margin.
- Kept: the top `SEARCH_CANDS` (3) candidates by static value, or all of
  them if fewer.
- Forced decisions (`Decision::Forced`, including safe-lock firings) are
  not captured.

Capture never changes a move: the recorder plays what the plain static
champion plays, so the turn-end positions are unchanged byte for byte.

## Labels

After the game, per captured decision, each kept candidate's entry is
built as win-search builds it (`phase1_entry` / `phase2_entry` /
`passive_entry`), and `rollout_final_states(champion, entries,
context_seed(state, opps, dice), k_c, first_active, false)` runs `k_c`
full-game rollouts per candidate on shared dice. The label is the mean
outcome for me (win 1, tie ½, loss 0); an entry whose game already ended
gets its exact result.

Training sample: `(win_input(entry[0], entry[1]), label)` on the
lock-propagated entry boards, as for turn ends. Frame: active entries
have the opponent to move next, the turn-end frame. Passive entries have
**me** to move next (`passive_first_active`); they are stored as
`(swap_input(x), 1 − label)` — the opponent's view, with the other player
(me) to move — so every sample is in the turn-end frame. Under the
current antisymmetric net the two forms give the same loss; the
conversion keeps the data correct for a later net that is not.

Each kept candidate is emitted `cand_m` times (default 1), the analogue
of `pair-train`'s `m_gated`, to set the candidate share of the data.

## The round

1. **Generate** turn-end positions as now, plus the candidate samples of
   the same games. `--positions` still counts turn ends; the candidate
   count follows from the games.
2. **Diagnose** (before fitting, on the new round's candidates): the
   champion's error on candidate *differences* — per decision, each
   non-top candidate's predicted P minus the top one's, against the same
   difference of labels. Reported as the RMSE of that error and an
   estimate of the label noise in the differences (CRN sample variance
   of the paired outcome differences / `k_c`), plus the share of decisions
   where the labels' best candidate differs from the champion's pick.
   This is the headroom: if the difference RMSE is at the noise, the
   round cannot help.
3. **Window, fit, gate, yardstick, save** as now. The window holds each
   round's turn ends and candidates together; the fit trains on the union.

## CLI

New `win-pi` options:

```
--labels turn|cand        turn = today's behaviour (default)
--cand-rollouts 64        k_c, full-game rollouts per candidate
--cand-m 1                copies of each candidate sample
```

`--labels turn` must reproduce today's runs exactly (same positions,
same fits, same CSV apart from the new empty columns).

Validation: `--cand-rollouts ≥ 2` (the noise estimate needs a variance),
`--cand-m ≥ 1`.

## Output

`training.csv` gains, after `positions`:
`cand_decisions,cand_samples,diff_rmse,diff_noise,flip_share` (empty for
`--labels turn` and for the round-0 row). The console line gains the
candidate count, the difference RMSE, the noise and the flip share.

## Cost and the smoke run

Per game, turn-end labelling costs ~25 positions × 64 rollouts. Candidate
labelling adds (captured decisions) × ≤ 3 × `k_c` rollouts, each from a
mid-game position; if ~10 decisions per player are gated that is about
2–3× the turn-end cost. A smoke run (`--rounds 1`, small `--positions`,
no gate) on the user's machine measures decisions per game, the candidate
share of samples and seconds per round before the full run; `--positions`
and `--cand-m` are set from it.

## Testing

- Capture leaves play unchanged: same seed, recorder with and without
  capture → identical turn-end positions.
- Capture keeps only gated decisions with ≥ 2 candidates, at most 3
  candidates each, sorted by static value; forced decisions are not
  captured. (Force the gate with a huge margin / a zero margin in tests
  via an injectable margin.)
- Labels: deterministic for a seed; an ended entry gets its exact
  result; candidates of one decision share dice (two identical candidates
  get identical labels).
- Passive frame: a passive sample is stored as the swapped input with
  `1 − label`.
- Difference diagnostic as a pure function: zero error when predictions
  equal labels; flip share counts argmax disagreements; noise from paired
  variances.
- `--labels turn` reproduces the current smoke output and CSV rows.
- End-to-end smoke with `--labels cand`, tiny settings: rounds run,
  candidate columns filled, `cand_samples = cand_m ×` kept candidates.
