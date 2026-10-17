# Win Candidate Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `win-pi --labels cand`: label the top candidates of the decisions win-search would search with their own full-game rollout values, and fit the win net to them alongside the turn-end labels.

**Architecture:** A new module `src/dqn/win_distill.rs` holds the pure parts: capturing a decision's shortlist (mirroring `SearchBot`'s candidate building and gate), labelling it with CRN full-game rollouts, and the difference diagnostic. `win_train.rs`'s turn-end recorder gains an optional capture, and a new `rollout_round_by` returns turn ends plus labelled decisions from the same games. `win_pi.rs` mixes the candidate samples into the round's data, logs five new CSV columns, and `main.rs` gets three flags.

**Tech Stack:** Rust 2021, burn (ndarray backend), rayon, clap (main.rs only).

**Spec:** `docs/superpowers/specs/2026-10-17-win-candidate-distillation-design.md`

## Global Constraints

- 2-player only; same `WinModel`, same BCE loss and `train_epochs`.
- Capture never changes a move: turn-end positions must be identical with and without capture, and `--labels turn` must reproduce today's runs.
- Candidate shortlist = top `SEARCH_CANDS` (3) by static value, gate = `gates_with(…, GATE_MARGIN)` close or endgame, ≥ 2 candidates; `Decision::Forced` (incl. safe-lock firings) never captured.
- Labels: full-game rollouts by the champion, `rollout_final_states(…, context_seed(state, opps, dice), k_c, first_active, false)`; mean outcome for me; ended entry = exact result.
- Entries with me to move next (`first_active == 0`) are stored as `(swap_input(x), 1 − label)`.
- New flags: `--labels turn|cand` (default turn), `--cand-rollouts 64`, `--cand-m 1`; validation `--cand-rollouts ≥ 2`, `--cand-m ≥ 1`.
- CSV header becomes `round,positions,cand_decisions,cand_samples,diff_rmse,diff_noise,flip_share,loss,gate_games,share_vs_champion,z,promoted,share_vs_master,elapsed_s`; the five new columns are empty for `--labels turn` and the round-0 row.
- Formatting: `rustfmt --edition 2021 <touched files>` only. Never `cargo fmt`; never rustfmt `src/dqn/pair.rs`.
- Never commit `.superpowers/`, `AGENTS.md`, `web-ui.txt`, `win_training.csv`, `win_model/`, `win_pi/`; never edit `AGENTS.md`; don't touch `champion.txt`, `pair_model/`, `dqn_model/`. If `web/Cargo.lock` changes, `git checkout web/Cargo.lock`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Check test output before every commit; never chain a commit after a test command.
- Tests run with `cargo test --lib --release <filter>`; the known pre-existing warning is the unused `double` in `src/state.rs`.

## Review Focus

1. A fresh run (no `--init`, Master plays round 1) with `--labels cand`: capture runs with Master's values, there is no win-net champion to diagnose, and the diagnostic columns must be empty, not a crash — pinned by `a_fresh_candidate_run_has_no_champion_to_diagnose` (Task 4).
2. A candidate whose entry already ended the game: its label must be the exact result, broadcast over all rollouts so the paired variance is still defined — pinned by `an_ended_candidate_gets_its_exact_result_and_crn_holds` (Task 2).
3. Decisions with no opponent (solo): capture must return `None` before evaluating anything — pinned in `a_shut_gate_or_no_opponent_captures_nothing` (Task 1).
4. The window total in the CSV `positions` column now includes candidate samples × `cand_m`: pinned in `candidate_rounds_log_their_decisions_and_headroom` (Task 4).
5. The refactor of `rollout_positions_by` into `rollout_round_by` must not change its output by a bit — pinned by the golden `rollout_positions_by_output_is_pinned`, recorded before the refactor (Task 3).

---

### Task 1: Capturing the decisions win-search would search

**Files:**
- Create: `src/dqn/win_distill.rs`
- Modify: `src/dqn/mod.rs` (next to `pub mod win_pi;`)

**Interfaces:**
- Consumes: `active_phase1_choices`, `active_phase2_choices`, `passive_phase1_choices`, `Decision` (`crate::strategy::bot_impl`); `context_seed`, `gates_with`, `passive_entry`, `passive_first_active`, `phase1_entry`, `phase2_entry`, `SEARCH_CANDS` (`crate::strategy::search`); `Bot::evaluate_batch`.
- Produces:
  ```rust
  pub struct CandDecision { pub entries: Vec<(Vec<State>, bool)>, pub first_active: usize, pub seed: u64 }
  pub fn capture_active1(bot: &impl Bot, state: &State, opps: &[State], dice: [u8; 6], margin: f32) -> Option<CandDecision>
  pub fn capture_active2(bot: &impl Bot, state: &State, opps: &[State], dice: [u8; 6], has_marked: bool, margin: f32) -> Option<CandDecision>
  pub fn capture_passive(bot: &impl Bot, state: &State, opps: &[State], dice: [u8; 6], active_player: usize, margin: f32) -> Option<CandDecision>
  ```

- [ ] **Step 1: Write the module skeleton and the failing tests**

Add to `src/dqn/mod.rs`, right after the `win_pi` line:

```rust
#[cfg(feature = "dqn")]
pub mod win_distill;
```

Create `src/dqn/win_distill.rs`:

```rust
//! Candidate distillation for the win net: the decisions win-search would
//! search, their top candidates' full-game rollout values, and the static
//! net's error on the differences between candidates.
//!
//! Design: docs/superpowers/specs/2026-10-17-win-candidate-distillation-design.md

use crate::state::{Mark, State};
use crate::strategy::bot_impl::{active_phase1_choices, active_phase2_choices, passive_phase1_choices, Decision};
use crate::strategy::search::{
    context_seed, gates_with, passive_entry, passive_first_active, phase1_entry, phase2_entry, SEARCH_CANDS,
};
use crate::strategy::Bot;

/// A decision win-search would search, ready to label: one entry per kept
/// candidate (best static value first), built as win-search builds it.
#[derive(Clone, Debug)]
pub struct CandDecision {
    /// (all-player states, game ended) per kept candidate.
    pub entries: Vec<(Vec<State>, bool)>,
    /// Entry-frame seat that takes the first rollout turn (0 = me).
    pub first_active: usize,
    /// Rollout seed shared by the candidates (CRN).
    pub seed: u64,
}

pub fn capture_active1(
    _bot: &impl Bot,
    _state: &State,
    _opps: &[State],
    _dice: [u8; 6],
    _margin: f32,
) -> Option<CandDecision> {
    todo!()
}

pub fn capture_active2(
    _bot: &impl Bot,
    _state: &State,
    _opps: &[State],
    _dice: [u8; 6],
    _has_marked: bool,
    _margin: f32,
) -> Option<CandDecision> {
    todo!()
}

pub fn capture_passive(
    _bot: &impl Bot,
    _state: &State,
    _opps: &[State],
    _dice: [u8; 6],
    _active_player: usize,
    _margin: f32,
) -> Option<CandDecision> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::pair::PairStrategy;
    use crate::dqn::win::win_input;
    use crate::dqn::win_train::turn_ends_by;

    fn pair() -> PairStrategy {
        PairStrategy::load("pair_model")
    }

    /// (me, opp, dice) contexts from Master's turn ends, a few dice each.
    fn contexts() -> Vec<(State, State, [u8; 6])> {
        let p = pair();
        let dice = [[1, 2, 3, 4, 5, 6], [3, 4, 6, 2, 5, 1], [6, 6, 1, 1, 2, 3], [2, 5, 4, 4, 3, 6]];
        turn_ends_by(24, 5, &|| p.clone())
            .into_iter()
            .zip(dice.iter().cycle())
            .map(|((me, opp), &d)| (me, opp, d))
            .collect()
    }

    fn board(e: &(Vec<State>, bool)) -> [f32; crate::dqn::win::WIN_INPUTS] {
        win_input(&e.0[0], &e.0[1])
    }

    #[test]
    fn an_open_gate_captures_every_choice_with_its_top_three() {
        let bot = pair();
        let (mut passive, mut active1, mut active2) = (0, 0, 0);
        for (me, opp, dice) in contexts() {
            let opps = [opp];
            let seed = context_seed(&me, &opps, dice);

            let n = match passive_phase1_choices(&me, &opps, dice) {
                Decision::Choices(c) => c.len(),
                Decision::Forced(_) => 0,
            };
            let d = capture_passive(&bot, &me, &opps, dice, 0, f32::INFINITY);
            assert_eq!(d.is_some(), n >= 2);
            if let Some(d) = d {
                passive += 1;
                assert_eq!(d.entries.len(), n.min(SEARCH_CANDS));
                assert_eq!(d.first_active, 0, "2-player passive: me next");
                assert_eq!(d.seed, seed);
            }

            let choices = match active_phase2_choices(&me, &opps, dice, false) {
                Decision::Choices(c) => c,
                Decision::Forced(_) => Vec::new(),
            };
            let d = capture_active2(&bot, &me, &opps, dice, false, f32::INFINITY);
            assert_eq!(d.is_some(), choices.len() >= 2);
            if let Some(d) = d {
                active2 += 1;
                assert_eq!(d.entries.len(), choices.len().min(SEARCH_CANDS));
                assert_eq!(d.first_active, 1);
                assert_eq!(d.seed, seed);
                // The first entry is the best static candidate's.
                let posts: Vec<State> = choices.iter().map(|c| c.1).collect();
                let values = bot.evaluate_batch(&posts, &opps);
                let best = (0..values.len()).fold(0, |b, i| if values[i] > values[b] { i } else { b });
                assert_eq!(board(&d.entries[0]), board(&phase2_entry(posts[best], &opps)));
            }

            let distinct = match active_phase1_choices(&bot, &me, &opps, dice).0 {
                Decision::Choices(plans) => {
                    let mut marks: Vec<Option<Mark>> = Vec::new();
                    for (m, _) in plans {
                        if !marks.contains(&m) {
                            marks.push(m);
                        }
                    }
                    marks.len()
                }
                Decision::Forced(_) => 0,
            };
            let d = capture_active1(&bot, &me, &opps, dice, f32::INFINITY);
            assert_eq!(d.is_some(), distinct >= 2);
            if let Some(d) = d {
                active1 += 1;
                assert_eq!(d.entries.len(), distinct.min(SEARCH_CANDS));
                assert_eq!(d.first_active, 1);
                assert_eq!(d.seed, seed);
            }
        }
        assert!(passive > 0 && active1 > 0 && active2 > 0, "{passive} {active1} {active2}");
    }

    #[test]
    fn a_shut_gate_or_no_opponent_captures_nothing() {
        // An empty board: no locks, no strikes, no candidate can lock, so the
        // endgame gate is shut; a negative margin shuts the close gate.
        let bot = pair();
        let (me, opp) = (State::default(), State::default());
        for dice in [[1, 2, 3, 4, 5, 6], [3, 4, 6, 2, 5, 1]] {
            assert!(capture_passive(&bot, &me, &[opp], dice, 0, -1.0).is_none());
            assert!(capture_active2(&bot, &me, &[opp], dice, false, -1.0).is_none());
            assert!(capture_active1(&bot, &me, &[opp], dice, -1.0).is_none());
            assert!(capture_passive(&bot, &me, &[], dice, 0, f32::INFINITY).is_none());
            assert!(capture_active2(&bot, &me, &[], dice, false, f32::INFINITY).is_none());
            assert!(capture_active1(&bot, &me, &[], dice, f32::INFINITY).is_none());
        }
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib --release win_distill 2>&1 | tail -20`
Expected: both tests FAIL, panicking at `not yet implemented`.

- [ ] **Step 3: Implement capture**

Replace the three `todo!()` stubs in `src/dqn/win_distill.rs` with:

```rust
/// The search shortlist when the gate fires: candidates sorted best static
/// value first, gated as win-search gates (close or endgame at `margin`),
/// cut to `SEARCH_CANDS`.
fn gated(
    mut cands: Vec<(Option<Mark>, State, f32)>,
    state: &State,
    opps: &[State],
    margin: f32,
) -> Option<Vec<(Option<Mark>, State)>> {
    if cands.len() < 2 {
        return None;
    }
    cands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let value_posts: Vec<(f32, State)> = cands.iter().map(|c| (c.2, c.1)).collect();
    let (close, endgame) = gates_with(&value_posts, state, opps, margin);
    (close || endgame).then(|| cands.into_iter().take(SEARCH_CANDS).map(|c| (c.0, c.1)).collect())
}

/// Each choice with its static value against `eval_opps`.
fn valued(bot: &impl Bot, choices: &[(Option<Mark>, State)], eval_opps: &[State]) -> Vec<(Option<Mark>, State, f32)> {
    let posts: Vec<State> = choices.iter().map(|c| c.1).collect();
    let values = bot.evaluate_batch(&posts, eval_opps);
    choices.iter().zip(values).map(|(&(m, s), v)| (m, s, v)).collect()
}

/// Active phase 1, as `SearchBot::explain_active_phase1` shortlists it:
/// plans collapsed to distinct phase-1 marks (each keeps its best plan),
/// valued against the simulated post-phase-1 opponents.
pub fn capture_active1(
    bot: &impl Bot,
    state: &State,
    opps: &[State],
    dice: [u8; 6],
    margin: f32,
) -> Option<CandDecision> {
    if opps.is_empty() {
        return None;
    }
    let (Decision::Choices(plans), sim_opp) = active_phase1_choices(bot, state, opps, dice) else {
        return None;
    };
    let mut cands: Vec<(Option<Mark>, State, f32)> = Vec::new();
    for (m, s, v) in valued(bot, &plans, &sim_opp) {
        match cands.iter_mut().find(|c| c.0 == m) {
            Some(c) if v > c.2 => {
                c.1 = s;
                c.2 = v;
            }
            Some(_) => {}
            None => cands.push((m, s, v)),
        }
    }
    let kept = gated(cands, state, opps, margin)?;
    Some(CandDecision {
        entries: kept
            .iter()
            .map(|&(m, _)| phase1_entry(bot, state, &sim_opp, dice, m))
            .collect(),
        first_active: 1 % (opps.len() + 1),
        seed: context_seed(state, opps, dice),
    })
}

/// Active phase 2, as `SearchBot::explain_active_phase2` shortlists it.
pub fn capture_active2(
    bot: &impl Bot,
    state: &State,
    opps: &[State],
    dice: [u8; 6],
    has_marked: bool,
    margin: f32,
) -> Option<CandDecision> {
    if opps.is_empty() {
        return None;
    }
    let Decision::Choices(choices) = active_phase2_choices(state, opps, dice, has_marked) else {
        return None;
    };
    let kept = gated(valued(bot, &choices, opps), state, opps, margin)?;
    Some(CandDecision {
        entries: kept.iter().map(|&(_, post)| phase2_entry(post, opps)).collect(),
        first_active: 1 % (opps.len() + 1),
        seed: context_seed(state, opps, dice),
    })
}

/// Passive phase 1, as `SearchBot::explain_passive_phase1` shortlists it:
/// each entry completes the rest of the turn, and the rollout starts at
/// the seat after the active player (me, in 2-player).
pub fn capture_passive(
    bot: &impl Bot,
    state: &State,
    opps: &[State],
    dice: [u8; 6],
    active_player: usize,
    margin: f32,
) -> Option<CandDecision> {
    if opps.is_empty() {
        return None;
    }
    let Decision::Choices(choices) = passive_phase1_choices(state, opps, dice) else {
        return None;
    };
    let kept = gated(valued(bot, &choices, opps), state, opps, margin)?;
    Some(CandDecision {
        entries: kept
            .iter()
            .map(|&(m, _)| passive_entry(bot, state, opps, dice, active_player, m))
            .collect(),
        first_active: passive_first_active(active_player, opps.len() + 1),
        seed: context_seed(state, opps, dice),
    })
}
```

If the compiler reports a `pub(crate)` item (e.g. `gates_with`) as private outside its module, it is still visible within the crate; if it reports `turn_ends_by` or a `bot_impl` item as not found, check the path against `src/strategy/mod.rs` (`pub mod bot_impl;`) and `src/dqn/win_train.rs`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib --release win_distill 2>&1 | tail -20`
Expected: `test result: ok. 2 passed`.

- [ ] **Step 5: Format and commit**

```bash
rustfmt --edition 2021 src/dqn/win_distill.rs src/dqn/mod.rs
git checkout web/Cargo.lock 2>/dev/null; git status --short
git add src/dqn/win_distill.rs src/dqn/mod.rs
git commit -m "feat(win): capture the decisions win-search would search, with their shortlists

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Candidate labels and the difference diagnostic

**Files:**
- Modify: `src/dqn/win_distill.rs`

**Interfaces:**
- Consumes: `CandDecision` (Task 1); `rollout_final_states` (`crate::strategy::search`, `pub(crate)`); `SimGame::outcome` (`crate::strategy::sim`); `win_input`, `swap_input`, `WIN_INPUTS` (`crate::dqn::win`).
- Produces:
  ```rust
  pub struct CandGroup { pub samples: Vec<([f32; WIN_INPUTS], f32)>, pub pair_var: Vec<f32>, pub swapped: bool }
  pub fn label_decision(bot: &impl Bot, d: &CandDecision, k: usize) -> CandGroup
  pub struct DiffReport { pub decisions: usize, pub diff_rmse: f32, pub diff_noise: f32, pub flip_share: f32 }
  pub fn diff_stats(preds: &[Vec<f32>], groups: &[CandGroup], k: usize) -> Option<DiffReport>
  ```

- [ ] **Step 1: Write the failing tests**

Add to `src/dqn/win_distill.rs`, above `#[cfg(test)]`:

```rust
/// One labelled decision: a training sample per kept candidate (best static
/// value first) and the CRN spread of each runner-up's difference to the
/// first candidate.
#[derive(Clone, Debug)]
pub struct CandGroup {
    /// (input, P(win)) in the turn-end frame (the other player to move).
    pub samples: Vec<([f32; WIN_INPUTS], f32)>,
    /// `pair_var[i − 1]`: sample variance over the rollouts of
    /// outcome(i) − outcome(0), for candidates i ≥ 1.
    pub pair_var: Vec<f32>,
    /// The entries had me to move next, so the samples are stored from the
    /// opponent's side (`swap_input`, `1 − label`).
    pub swapped: bool,
}

pub fn label_decision(_bot: &impl Bot, _d: &CandDecision, _k: usize) -> CandGroup {
    todo!()
}

/// The champion's error on the differences between candidates.
#[derive(Clone, Debug, PartialEq)]
pub struct DiffReport {
    pub decisions: usize,
    /// RMS over runner-ups of (p_i − p_0) − (y_i − y_0).
    pub diff_rmse: f32,
    /// The part of `diff_rmse` the labels' own noise explains: √(mean pair_var / k).
    pub diff_noise: f32,
    /// Share of decisions whose best label (for me) is not the first
    /// candidate, the static pick.
    pub flip_share: f32,
}

pub fn diff_stats(_preds: &[Vec<f32>], _groups: &[CandGroup], _k: usize) -> Option<DiffReport> {
    todo!()
}
```

Change the `use` lines at the top of the file to:

```rust
use crate::dqn::win::{swap_input, win_input, WIN_INPUTS};
use crate::state::{Mark, State};
use crate::strategy::bot_impl::{active_phase1_choices, active_phase2_choices, passive_phase1_choices, Decision};
use crate::strategy::search::{
    context_seed, gates_with, passive_entry, passive_first_active, phase1_entry, phase2_entry, rollout_final_states,
    SEARCH_CANDS,
};
use crate::strategy::sim::SimGame;
use crate::strategy::Bot;
```

and in the test module replace `use crate::dqn::win::win_input;` with nothing (it now comes through `super::*`), and replace `fn board(e: &(Vec<State>, bool)) -> [f32; crate::dqn::win::WIN_INPUTS]` with `fn board(e: &(Vec<State>, bool)) -> [f32; WIN_INPUTS]`.

Add these tests to the test module:

```rust
    #[test]
    fn an_ended_candidate_gets_its_exact_result_and_crn_holds() {
        let bot = pair();
        let opp = State::default();
        let mut done = State::default();
        for _ in 0..4 {
            done.apply_strike();
        }
        let live = phase2_entry(State::default(), &[opp]);
        let ended = phase2_entry(done, &[opp]);
        assert!(ended.1 && !live.1);
        let d = CandDecision {
            entries: vec![live.clone(), ended.clone(), live.clone()],
            first_active: 1,
            seed: 9,
        };
        let g = label_decision(&bot, &d, 8);
        assert!(!g.swapped);
        assert_eq!(g.samples.len(), 3);
        assert_eq!(g.samples[0].0, board(&live));
        // Four strikes: −20 against 0, an exact loss.
        assert_eq!(g.samples[1].1, 0.0);
        // Identical candidates share every rollout's dice.
        assert_eq!(g.samples[0].1, g.samples[2].1);
        assert_eq!(g.pair_var.len(), 2);
        assert_eq!(g.pair_var[1], 0.0);
        // Deterministic for a seed.
        assert_eq!(label_decision(&bot, &d, 8).samples, g.samples);
    }

    #[test]
    fn me_next_entries_are_stored_from_the_opponents_side() {
        let bot = pair();
        let opp = State::default();
        let mut marked = State::default();
        marked.apply_mark(Mark { row: 0, number: 2 });
        let entries = vec![phase2_entry(marked, &[opp]), phase2_entry(State::default(), &[opp])];
        let d = CandDecision {
            entries: entries.clone(),
            first_active: 0,
            seed: 9,
        };
        let g = label_decision(&bot, &d, 8);
        assert!(g.swapped);
        let finals = rollout_final_states(&bot, &entries, 9, 8, 0, false);
        let mean = finals[0].iter().map(|f| SimGame::outcome(f)).sum::<f32>() / 8.0;
        assert_eq!(g.samples[0].0, swap_input(&board(&entries[0])));
        assert!((g.samples[0].1 - (1.0 - mean)).abs() < 1e-6);
    }

    fn group(labels: &[f32], pair_var: &[f32], swapped: bool) -> CandGroup {
        CandGroup {
            samples: labels.iter().map(|&y| ([0.0; WIN_INPUTS], y)).collect(),
            pair_var: pair_var.to_vec(),
            swapped,
        }
    }

    #[test]
    fn a_constant_shift_has_no_difference_error() {
        let gs = vec![group(&[0.6, 0.4, 0.5], &[0.04, 0.08], false), group(&[0.3, 0.7], &[0.02], false)];
        let preds: Vec<Vec<f32>> = gs
            .iter()
            .map(|g| g.samples.iter().map(|s| s.1 + 0.1).collect())
            .collect();
        let r = diff_stats(&preds, &gs, 4).unwrap();
        assert_eq!(r.decisions, 2);
        assert!(r.diff_rmse < 1e-6, "{r:?}");
        assert!((r.diff_noise - (0.14f32 / 3.0 / 4.0).sqrt()).abs() < 1e-6, "{r:?}");
        // The second decision's best label is its runner-up.
        assert!((r.flip_share - 0.5).abs() < 1e-6, "{r:?}");
    }

    #[test]
    fn difference_error_is_rms_over_runner_ups() {
        let gs = vec![group(&[0.5, 0.5, 0.5], &[0.0, 0.0], false)];
        let r = diff_stats(&[vec![0.5, 0.8, 0.1]], &gs, 4).unwrap();
        // Errors 0.3 and −0.4.
        assert!((r.diff_rmse - (0.125f32).sqrt()).abs() < 1e-6, "{r:?}");
        assert_eq!(r.diff_noise, 0.0);
        // Ties keep the first candidate.
        assert_eq!(r.flip_share, 0.0);
    }

    #[test]
    fn flips_are_judged_for_me_in_swapped_groups() {
        // Stored [0.4, 0.6] from the opponent's side is [0.6, 0.4] for me.
        let swapped = diff_stats(&[vec![0.4, 0.6]], &[group(&[0.4, 0.6], &[0.0], true)], 4).unwrap();
        assert_eq!(swapped.flip_share, 0.0);
        let plain = diff_stats(&[vec![0.4, 0.6]], &[group(&[0.4, 0.6], &[0.0], false)], 4).unwrap();
        assert_eq!(plain.flip_share, 1.0);
    }

    #[test]
    fn no_decisions_no_report() {
        assert!(diff_stats(&[], &[], 4).is_none());
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib --release win_distill 2>&1 | tail -20`
Expected: the 6 new tests FAIL at `not yet implemented`; the 2 Task 1 tests pass.

- [ ] **Step 3: Implement labels and the diagnostic**

Replace the two stubs with:

```rust
/// Sample variance (n − 1); 0 below two values.
fn sample_var(xs: &[f32]) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (n - 1.0)
}

/// Label every kept candidate with `k` full-game CRN rollouts by `bot`:
/// mean outcome for me (win 1, tie ½); an ended entry's exact result stands
/// for every rollout.
pub fn label_decision(bot: &impl Bot, d: &CandDecision, k: usize) -> CandGroup {
    let finals = rollout_final_states(bot, &d.entries, d.seed, k, d.first_active, false);
    let outcomes: Vec<Vec<f32>> = finals
        .iter()
        .map(|f| {
            let o: Vec<f32> = f.iter().map(|states| SimGame::outcome(states)).collect();
            if o.len() == 1 {
                vec![o[0]; k]
            } else {
                o
            }
        })
        .collect();
    let swapped = d.first_active == 0;
    let samples = d
        .entries
        .iter()
        .zip(&outcomes)
        .map(|((e, _), o)| {
            assert_eq!(e.len(), 2, "the win net is 2-player only");
            let x = win_input(&e[0], &e[1]);
            let y = o.iter().sum::<f32>() / k as f32;
            if swapped {
                (swap_input(&x), 1.0 - y)
            } else {
                (x, y)
            }
        })
        .collect();
    let pair_var = outcomes[1..]
        .iter()
        .map(|o| {
            let diffs: Vec<f32> = o.iter().zip(&outcomes[0]).map(|(a, b)| a - b).collect();
            sample_var(&diffs)
        })
        .collect();
    CandGroup {
        samples,
        pair_var,
        swapped,
    }
}

/// `preds[g][i]` = the champion's P for `groups[g].samples[i]` (the
/// samples' frame). `None` when no decision has a runner-up.
pub fn diff_stats(preds: &[Vec<f32>], groups: &[CandGroup], k: usize) -> Option<DiffReport> {
    let (mut sq, mut var, mut pairs, mut flips) = (0.0f64, 0.0f64, 0usize, 0usize);
    for (p, g) in preds.iter().zip(groups) {
        let y: Vec<f32> = g.samples.iter().map(|s| s.1).collect();
        for i in 1..y.len() {
            let e = (p[i] - p[0]) - (y[i] - y[0]);
            sq += (e * e) as f64;
            var += g.pair_var[i - 1] as f64;
            pairs += 1;
        }
        // Best label for me; ties keep the first (static) candidate.
        let mine: Vec<f32> = y.iter().map(|&v| if g.swapped { 1.0 - v } else { v }).collect();
        let best = (1..mine.len()).fold(0, |b, i| if mine[i] > mine[b] { i } else { b });
        flips += (best != 0) as usize;
    }
    (pairs > 0).then(|| DiffReport {
        decisions: groups.len(),
        diff_rmse: (sq / pairs as f64).sqrt() as f32,
        diff_noise: (var / pairs as f64 / k as f64).sqrt() as f32,
        flip_share: flips as f32 / groups.len() as f32,
    })
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib --release win_distill 2>&1 | tail -20`
Expected: `test result: ok. 8 passed`.

- [ ] **Step 5: Format and commit**

```bash
rustfmt --edition 2021 src/dqn/win_distill.rs
git status --short
git add src/dqn/win_distill.rs
git commit -m "feat(win): label candidates by CRN full-game rollouts; difference diagnostic

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Capture in the round's games (`rollout_round_by`)

**Files:**
- Modify: `src/dqn/win_train.rs` (`TurnEnds` ~line 496, `rollout_positions_by` ~line 580, `turn_ends_by` ~line 651, tests module ~line 883)

**Interfaces:**
- Consumes: `CandDecision`, `CandGroup`, `capture_active1/2`, `capture_passive`, `label_decision` (Tasks 1–2).
- Produces:
  ```rust
  pub struct CandCfg { pub k: usize, pub margin: f32 }
  pub struct RoundData { pub positions: Vec<([f32; WIN_INPUTS], f32)>, pub groups: Vec<CandGroup> }
  pub fn rollout_round_by<P: Bot + Clone + 'static>(count: usize, seed: u64, player: &(dyn Fn() -> P + Sync), pair_dir: &str, k: usize, blend: f32, cand: Option<CandCfg>) -> RoundData
  ```
  `rollout_positions_by` keeps its signature and output.

- [ ] **Step 1: Pin the current output (characterization test, before any refactor)**

Add to the `win_train.rs` tests module:

```rust
    /// FNV-1a over the bits of every input and label.
    fn fingerprint(data: &[([f32; WIN_INPUTS], f32)]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for (x, y) in data {
            for v in x.iter().chain(std::iter::once(y)) {
                h = (h ^ v.to_bits() as u64).wrapping_mul(0x0100_0000_01b3);
            }
        }
        h
    }

    #[test]
    fn rollout_positions_by_output_is_pinned() {
        let data = rollout_positions_by(40, 11, &|| PairStrategy::load("pair_model"), "pair_model", 2, 1.0);
        assert_eq!(data.len(), 40);
        assert_eq!(fingerprint(&data), 0);
    }
```

- [ ] **Step 2: Record the golden value**

Run: `cargo test --lib --release rollout_positions_by_output_is_pinned 2>&1 | grep -A3 "left"`
Expected: FAIL with `left: <N>` / `right: 0`. Replace the `0` in the assertion with that `<N>` (as a `u64` literal), rerun, and expect PASS. This pins today's output; the test must still pass after Step 5.

- [ ] **Step 3: Write the failing capture test**

Add to the tests module:

```rust
    #[test]
    fn capture_leaves_the_positions_unchanged_and_labels_decisions() {
        use crate::strategy::search::{GATE_MARGIN, SEARCH_CANDS};
        let pair = || PairStrategy::load("pair_model");
        let plain = rollout_round_by(40, 11, &pair, "pair_model", 2, 1.0, None);
        assert!(plain.groups.is_empty());
        let cand = Some(CandCfg {
            k: 4,
            margin: GATE_MARGIN,
        });
        let with = rollout_round_by(40, 11, &pair, "pair_model", 2, 1.0, cand);
        assert_eq!(plain.positions, with.positions);
        assert!(!with.groups.is_empty());
        for g in &with.groups {
            assert!((2..=SEARCH_CANDS).contains(&g.samples.len()), "{}", g.samples.len());
            assert_eq!(g.pair_var.len(), g.samples.len() - 1);
            assert!(g.samples.iter().all(|s| (0.0..=1.0).contains(&s.1)));
        }
    }
```

Run: `cargo test --lib --release capture_leaves_the_positions 2>&1 | tail -5`
Expected: compile error — `rollout_round_by` / `CandCfg` not found.

- [ ] **Step 4: Add capture to `TurnEnds`**

Add to the imports at the top of `win_train.rs`:

```rust
use crate::dqn::win_distill::{capture_active1, capture_active2, capture_passive, label_decision, CandDecision, CandGroup};
```

Replace the `TurnEnds` struct and its `Strategy` impl with:

```rust
/// Plays a bot greedily and records (my state, opponent) at the end of each
/// of my active turns: after phase 2, the boundary a rollout restarts from.
/// With `capture`, also keeps the decisions win-search would search; that
/// never changes a move.
struct TurnEnds<B: Bot> {
    bot: B,
    recorded: Rc<RefCell<Vec<(State, State)>>>,
    /// The close-gate margin and the captured decisions.
    capture: Option<(f32, Rc<RefCell<Vec<CandDecision>>>)>,
}

impl<B: Bot> TurnEnds<B> {
    fn keep(&self, capture: impl FnOnce(&B, f32) -> Option<CandDecision>) {
        if let Some((margin, buf)) = &self.capture {
            if let Some(d) = capture(&self.bot, *margin) {
                buf.borrow_mut().push(d);
            }
        }
    }
}

impl<B: Bot> std::fmt::Debug for TurnEnds<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "TurnEnds")
    }
}

impl<B: Bot> Strategy for TurnEnds<B> {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        self.keep(|b, m| capture_active1(b, state, opp_states, dice, m));
        active_phase1_impl(&self.bot, state, opp_states, dice)
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        self.keep(|b, m| capture_active2(b, state, opp_states, dice, has_marked, m));
        let mark = active_phase2_impl(&self.bot, state, opp_states, dice, has_marked);
        let mut no_mark = *state;
        if !has_marked {
            no_mark.apply_strike();
        }
        self.recorded
            .borrow_mut()
            .push((after(state, mark, no_mark), opp_states[0]));
        mark
    }

    fn passive_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], active: usize) -> Option<Mark> {
        self.keep(|b, m| capture_passive(b, state, opp_states, dice, active, m));
        passive_phase1_impl(&self.bot, state, opp_states, dice)
    }
}
```

In `turn_ends_by`, add `capture: None,` to the `TurnEnds { … }` literal.

- [ ] **Step 5: Add `rollout_round_by` and make `rollout_positions_by` delegate**

Replace the whole `rollout_positions_by` function (doc comment included) with:

```rust
/// [`rollout_positions`] with the bot `player()` builds (once per worker
/// thread) in both seats of the games and the rollouts: one step of policy
/// iteration when it is the net fitted to the previous round's labels. The
/// blend's static term stays Master's.
pub fn rollout_positions_by<P: Bot + Clone + 'static>(
    count: usize,
    seed: u64,
    player: &(dyn Fn() -> P + Sync),
    pair_dir: &str,
    k: usize,
    blend: f32,
) -> Vec<([f32; WIN_INPUTS], f32)> {
    rollout_round_by(count, seed, player, pair_dir, k, blend, None).positions
}

/// Candidate capture for [`rollout_round_by`].
#[derive(Clone, Copy, Debug)]
pub struct CandCfg {
    /// Full-game rollouts per candidate.
    pub k: usize,
    /// Close-gate margin (production: `GATE_MARGIN`).
    pub margin: f32,
}

/// One round's data from [`rollout_round_by`].
pub struct RoundData {
    pub positions: Vec<([f32; WIN_INPUTS], f32)>,
    /// The labelled decisions of every game played. The positions are cut
    /// to `count`; the last games' decisions are kept whole.
    pub groups: Vec<CandGroup>,
}

/// [`rollout_positions_by`] that, with `cand`, also keeps the decisions
/// win-search would search in the same games and labels their top
/// candidates with `cand.k` rollouts by the player
/// (docs/superpowers/specs/2026-10-17-win-candidate-distillation-design.md).
/// Capture never changes a move, so the positions are the same either way.
pub fn rollout_round_by<P: Bot + Clone + 'static>(
    count: usize,
    seed: u64,
    player: &(dyn Fn() -> P + Sync),
    pair_dir: &str,
    k: usize,
    blend: f32,
    cand: Option<CandCfg>,
) -> RoundData {
    // Games are labelled one per task, and a game's cost grows with the
    // square of its length: size each batch from the positions still needed
    // (~25 turn ends per game) so all cores stay busy past the slowest
    // games. Seeds depend only on the game index, so batching never changes
    // the output.
    let mut positions = Vec::with_capacity(count);
    let mut groups = Vec::new();
    let mut games = 0u64;
    while positions.len() < count {
        let per_game = if games == 0 {
            25.0
        } else {
            positions.len() as f64 / games as f64
        };
        let batch = (((count - positions.len()) as f64 / per_game).ceil() as u64).max(64);
        let chunk: Vec<(Vec<([f32; WIN_INPUTS], f32)>, Vec<CandGroup>)> = (games..games + batch)
            .into_par_iter()
            .map_init(
                || (PairStrategy::load(pair_dir), player()),
                |(pair, player), g| {
                    let game_seed = seed.wrapping_add(g.wrapping_mul(1000));
                    let bufs: [Rc<RefCell<Vec<(State, State)>>>; 2] = Default::default();
                    let caps: [Rc<RefCell<Vec<CandDecision>>>; 2] = Default::default();
                    let players: Vec<Player> = (0..2)
                        .map(|i| {
                            let rec = TurnEnds {
                                bot: player.clone(),
                                recorded: bufs[i].clone(),
                                capture: cand.map(|c| (c.margin, caps[i].clone())),
                            };
                            Player::new(
                                Box::new(rec),
                                Box::new(SmallRng::seed_from_u64(game_seed.wrapping_add(i as u64))),
                            )
                        })
                        .collect();
                    Game::new(players).play();
                    let ends: Vec<(State, State)> = bufs.iter().flat_map(|b| b.take()).collect();
                    // Inputs and Master's estimate on the lock-propagated boards.
                    let boards: Vec<Vec<State>> = ends.iter().map(|(me, opp)| phase2_entry(*me, &[*opp]).0).collect();
                    let groups: Vec<(&State, &[State])> = boards.iter().map(|b| (&b[0], &b[1..])).collect();
                    let stat = pair.win_prob_multi(&groups);
                    let roll = if blend == 0.0 {
                        vec![0.0; ends.len()]
                    } else {
                        rollout_labels(&*player, &ends, k, game_seed ^ 0x5011_0075)
                    };
                    let pos = boards
                        .iter()
                        .zip(stat.into_iter().zip(roll))
                        .map(|(b, (s, r))| (win_input(&b[0], &b[1]), blend * r + (1.0 - blend) * s))
                        .collect();
                    let labelled = match cand {
                        Some(c) => caps
                            .iter()
                            .flat_map(|b| b.take())
                            .map(|d| label_decision(&*player, &d, c.k))
                            .collect(),
                        None => Vec::new(),
                    };
                    (pos, labelled)
                },
            )
            .collect();
        for (p, g) in chunk {
            positions.extend(p);
            groups.extend(g);
        }
        games += batch;
    }
    positions.truncate(count);
    RoundData { positions, groups }
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test --lib --release -- rollout_positions_by_output_is_pinned capture_leaves_the_positions 2>&1 | tail -8`
Expected: `test result: ok. 2 passed`. The pinned fingerprint must be unchanged; if it changed, the refactor altered the output — fix the refactor, never the golden.

Then the whole win_train suite: `cargo test --lib --release win_train 2>&1 | tail -5`
Expected: all pass.

- [ ] **Step 7: Format and commit**

```bash
rustfmt --edition 2021 src/dqn/win_train.rs
git status --short
git add src/dqn/win_train.rs
git commit -m "feat(win): rollout_round_by: candidate capture and labels in the round's own games

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `win-pi --labels cand`

**Files:**
- Modify: `src/dqn/win_pi.rs` (imports, `WinPiCfg` ~line 156, `validate` ~line 183, `policy_iteration` ~line 242, tests)
- Modify: `src/main.rs` (`WinPi` args ~line 421, dispatch ~line 1018)

**Interfaces:**
- Consumes: `rollout_round_by`, `CandCfg`, `RoundData` (Task 3); `diff_stats`, `DiffReport` (Task 2); `ManualWinNet::logits`, `sigmoid` (`crate::dqn::win`); `GATE_MARGIN` (`crate::strategy::search`).
- Produces: `pub enum Labels { Turn, Cand }`; `WinPiCfg` fields `labels: Labels`, `cand_rollouts: usize`, `cand_m: usize`; CLI flags `--labels`, `--cand-rollouts`, `--cand-m`.

- [ ] **Step 1: Update the existing tests and write the new ones (failing)**

In the `win_pi.rs` tests module, add to `tiny_cfg`'s literal:

```rust
            labels: Labels::Turn,
            cand_rollouts: 2,
            cand_m: 1,
```

In `validate_refuses_bad_settings_before_any_work`, before `std::fs::create_dir_all(&ok.dir)`, add:

```rust
        assert!(err(WinPiCfg {
            cand_rollouts: 1,
            ..tiny_cfg(ok.dir.clone(), None)
        })
        .contains("--cand-rollouts"));
        assert!(err(WinPiCfg {
            cand_m: 0,
            ..tiny_cfg(ok.dir.clone(), None)
        })
        .contains("--cand-m"));
```

In `two_rounds_from_an_init_net_write_the_run`, replace everything from `assert_eq!(\n            lines[0],` through the end of the `for (i, row)` loop with:

```rust
        assert_eq!(lines[0], HEADER);
        assert_eq!(lines.len(), 4, "{csv}");
        assert!(lines[1].starts_with("0,,,,,,,,,,,,"), "{}", lines[1]);
        assert_eq!(lines[1].split(',').count(), 14, "{}", lines[1]);
        for (i, row) in lines[2..].iter().enumerate() {
            let f: Vec<&str> = row.split(',').collect();
            assert_eq!(f.len(), 14, "{row}");
            assert_eq!(f[0], (i + 1).to_string());
            // --labels turn: no candidate columns.
            assert!(f[2..7].iter().all(|c| c.is_empty()), "{row}");
            assert!(f[11] == "0" || f[11] == "1", "{row}");
            assert!(!f[12].is_empty(), "{row}");
        }
```

and add above `fn tmp`:

```rust
    const HEADER: &str = "round,positions,cand_decisions,cand_samples,diff_rmse,diff_noise,flip_share,\
                          loss,gate_games,share_vs_champion,z,promoted,share_vs_master,elapsed_s";
```

In `a_fresh_start_that_loses_to_master_stops_cleanly`, change `rows[0].split(',').nth(6)` to `rows[0].split(',').nth(11)`.

Add two tests:

```rust
    fn init_net(base: &str) -> String {
        let init = format!("{base}/init");
        let device = Default::default();
        <MyBackend as Backend>::seed(&device, 3);
        let mcfg = WinModelConfig::new().with_width(4);
        let model: WinModel<MyBackend> = mcfg.init(&device);
        save_win(&init, &mcfg, &model);
        init
    }

    #[test]
    fn candidate_rounds_log_their_decisions_and_headroom() {
        let base = tmp("cand");
        let cfg = WinPiCfg {
            labels: Labels::Cand,
            cand_m: 2,
            rounds: 1,
            ..tiny_cfg(format!("{base}/run"), Some(init_net(&base)))
        };
        assert!(policy_iteration(&cfg).is_some());
        let csv = std::fs::read_to_string(format!("{}/training.csv", cfg.dir)).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3, "{csv}");
        let f: Vec<&str> = lines[2].split(',').collect();
        assert_eq!(f.len(), 14, "{}", lines[2]);
        let decisions: usize = f[2].parse().unwrap();
        let samples: usize = f[3].parse().unwrap();
        assert!(decisions > 0, "{}", lines[2]);
        // 2–3 candidates per decision, each emitted cand_m = 2 times.
        assert!(samples >= 4 * decisions && samples <= 6 * decisions, "{}", lines[2]);
        assert_eq!(samples % 2, 0);
        assert!(f[4..7].iter().all(|c| !c.is_empty()), "{}", lines[2]);
        // The window total counts the turn ends and the candidate samples.
        assert_eq!(f[1].parse::<usize>().unwrap(), cfg.positions + samples);
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn a_fresh_candidate_run_has_no_champion_to_diagnose() {
        let base = tmp("fresh-cand");
        let cfg = WinPiCfg {
            labels: Labels::Cand,
            master_games: 0,
            ..tiny_cfg(format!("{base}/run"), None)
        };
        assert!(policy_iteration(&cfg).is_none());
        let csv = std::fs::read_to_string(format!("{}/training.csv", cfg.dir)).unwrap();
        let row = csv.lines().nth(1).unwrap();
        let f: Vec<&str> = row.split(',').collect();
        assert!(!f[2].is_empty() && !f[3].is_empty(), "{row}");
        assert!(f[4..7].iter().all(|c| c.is_empty()), "{row}");
        std::fs::remove_dir_all(&base).unwrap();
    }
```

and in `two_rounds_from_an_init_net_write_the_run`, replace the six lines that build `init` (from `let init = format!("{base}/init");` through `save_win(&init, &mcfg, &model);`) with `let init = init_net(&base);`.

Run: `cargo test --lib --release win_pi 2>&1 | tail -5`
Expected: compile errors — `Labels`, `labels`, `cand_rollouts`, `cand_m` not found.

- [ ] **Step 2: Add the config, validation and `Labels`**

In `win_pi.rs` replace the import lines

```rust
use crate::dqn::win::{WinModel, WinModelConfig, WinStrategy, WIN_INPUTS};
use crate::dqn::win_train::{bench_vs_pair, outcome, rollout_positions_by, train_epochs, Ad};
```

with

```rust
use crate::dqn::win::{sigmoid, ManualWinNet, WinModel, WinModelConfig, WinStrategy, WIN_INPUTS};
use crate::dqn::win_distill::diff_stats;
use crate::dqn::win_train::{bench_vs_pair, outcome, rollout_round_by, train_epochs, Ad, CandCfg};
use crate::strategy::search::GATE_MARGIN;
```

(If `rollout_positions_by` is still used elsewhere in the file, keep it in the import.)

Above `pub struct WinPiCfg`, add:

```rust
/// What labels a round's data: turn ends only, or also the top candidates
/// of the decisions win-search would search
/// (docs/superpowers/specs/2026-10-17-win-candidate-distillation-design.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Labels {
    Turn,
    Cand,
}
```

Add to `WinPiCfg`, after `pub pair_dir: String,`:

```rust
    pub labels: Labels,
    /// Full-game rollouts per candidate (`Labels::Cand`).
    pub cand_rollouts: usize,
    /// Copies of each candidate sample (`Labels::Cand`).
    pub cand_m: usize,
```

In `validate`, after the `gate_max` check, add:

```rust
        if self.cand_rollouts < 2 {
            return Err("--cand-rollouts must be at least 2".into());
        }
        if self.cand_m == 0 {
            return Err("--cand-m must be at least 1".into());
        }
```

- [ ] **Step 3: Generate, diagnose and log candidates in `policy_iteration`**

Replace the header string literal

```rust
        "round,positions,loss,gate_games,share_vs_champion,z,promoted,share_vs_master,elapsed_s\n",
```

with

```rust
        "round,positions,cand_decisions,cand_samples,diff_rmse,diff_noise,flip_share,\
         loss,gate_games,share_vs_champion,z,promoted,share_vs_master,elapsed_s\n",
```

Replace the round-0 log line `log(format!("0,,,,,,,{},{}", csv(s), start.elapsed().as_secs()));` with:

```rust
            log(format!("0{},{},{}", ",".repeat(11), csv(s), start.elapsed().as_secs()));
```

Replace the generate block — from `// 1. Generate: the champion's own turn ends, labelled by its rollouts.` through `push_window(&mut window, data, cfg.window);` — with:

```rust
        // 1. Generate: the champion's own turn ends, labelled by its rollouts;
        // with `--labels cand`, also the top candidates of the decisions
        // win-search would search in the same games.
        let pos_seed = round_seed(cfg.seed, round, POS);
        let cand = (cfg.labels == Labels::Cand).then_some(CandCfg {
            k: cfg.cand_rollouts,
            margin: GATE_MARGIN,
        });
        let round_data = match &champion {
            None => rollout_round_by(
                cfg.positions,
                pos_seed,
                &|| PairStrategy::load(&cfg.pair_dir),
                &cfg.pair_dir,
                cfg.rollouts,
                1.0,
                cand,
            ),
            Some(m) => {
                let win = WinStrategy::from_model(&m.valid());
                rollout_round_by(
                    cfg.positions,
                    pos_seed,
                    &|| win.clone(),
                    &cfg.pair_dir,
                    cfg.rollouts,
                    1.0,
                    cand,
                )
            }
        };

        // 2. Diagnose: the champion's error on the new candidates'
        // differences (none while Master is the champion).
        let diag = champion.as_ref().and_then(|m| {
            let net = ManualWinNet::from_model(&m.valid());
            let preds: Vec<Vec<f32>> = round_data
                .groups
                .iter()
                .map(|g| {
                    let xs: Vec<[f32; WIN_INPUTS]> = g.samples.iter().map(|s| s.0).collect();
                    net.logits(&xs).into_iter().map(sigmoid).collect()
                })
                .collect();
            diff_stats(&preds, &round_data.groups, cfg.cand_rollouts)
        });
        let cand_decisions = round_data.groups.len();
        let mut data = round_data.positions;
        let turn_ends = data.len();
        for g in &round_data.groups {
            for _ in 0..cfg.cand_m {
                data.extend_from_slice(&g.samples);
            }
        }
        let cand_samples = data.len() - turn_ends;
        let diag_csv = diag
            .as_ref()
            .map(|d| format!("{:.4},{:.4},{:.4}", d.diff_rmse, d.diff_noise, d.flip_share))
            .unwrap_or_else(|| ",,".into());
        let cand_csv = match cfg.labels {
            Labels::Turn => ",,,,".to_string(),
            Labels::Cand => format!("{cand_decisions},{cand_samples},{diag_csv}"),
        };
        let cand_note = match cfg.labels {
            Labels::Turn => String::new(),
            Labels::Cand => format!(
                ", {cand_decisions} decisions ({cand_samples} samples){}",
                diag.as_ref()
                    .map(|d| format!(
                        ", diff RMSE {:.4} (noise {:.4}), flips {:.1}%",
                        d.diff_rmse,
                        d.diff_noise,
                        d.flip_share * 100.0
                    ))
                    .unwrap_or_default()
            ),
        };

        // 3. Window: every round's data enters, whatever its gate decides.
        push_window(&mut window, data, cfg.window);
```

Renumber the following step comments: `// 3. Fit` → `// 4. Fit`, `// 4. Gate` → `// 5. Gate`, `// 5. Yardstick (logged only) and 6. log.` → `// 6. Yardstick (logged only) and 7. log.`

In the `row` closure, change the format string from

```rust
                "{round},{},{loss:.5},{},{:.4},{:.2},{},{},{}",
```

to

```rust
                "{round},{},{cand_csv},{loss:.5},{},{:.4},{:.2},{},{},{}",
```

In the per-round `println!`, change `"Round {round:>2}: {} positions, loss {loss:.4}, …"` to start with `"Round {round:>2}: {} positions{cand_note}, loss {loss:.4}, …"` (the rest of the string and its arguments unchanged).

Update the module doc comment's `Design:` line to list both specs:

```rust
//! Design: docs/superpowers/specs/2026-10-17-win-policy-iteration-design.md,
//! docs/superpowers/specs/2026-10-17-win-candidate-distillation-design.md
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib --release win_pi 2>&1 | tail -8`
Expected: `test result: ok. 15 passed` (13 existing + 2 new).

- [ ] **Step 5: Wire the CLI**

In `src/main.rs`, inside `WinPi { … }`, after the `seed` field add:

```rust
        /// turn = turn-end rollout labels; cand = also the top candidates of the decisions win-search would search
        #[arg(long, default_value = "turn", value_parser = ["turn", "cand"])]
        labels: String,
        /// Full-game rollouts per candidate (--labels cand)
        #[arg(long, default_value = "64")]
        cand_rollouts: usize,
        /// Copies of each candidate sample (--labels cand)
        #[arg(long, default_value = "1")]
        cand_m: usize,
```

In the dispatch, add `labels, cand_rollouts, cand_m,` to the `Some(Commands::WinPi { … })` pattern after `seed,`, and to the `WinPiCfg { … }` literal after `pair_dir: "pair_model".into(),`:

```rust
                labels: if labels == "cand" {
                    dqn::win_pi::Labels::Cand
                } else {
                    dqn::win_pi::Labels::Turn
                },
                cand_rollouts,
                cand_m,
```

Check: `cargo build --release 2>&1 | tail -3` → builds. `./target/release/qwixxer win-pi --cand-rollouts 1 --dir /tmp/claude-1000/wpi-x; echo $?` → prints `win-pi: --cand-rollouts must be at least 2` and `2`. `./target/release/qwixxer win-pi --help | grep -A1 labels` → shows the flag with `[possible values: turn, cand]`.

- [ ] **Step 6: Full suite**

Run: `cargo test --lib --release 2>&1 | tail -5`
Expected: all pass (151 before this plan + 2 + 6 + 2 + 2 = 163).

- [ ] **Step 7: Format and commit**

```bash
rustfmt --edition 2021 src/dqn/win_pi.rs src/main.rs
git checkout web/Cargo.lock 2>/dev/null; git status --short
git add src/dqn/win_pi.rs src/main.rs
git commit -m "feat(win): win-pi --labels cand: candidate samples, difference diagnostic, CSV columns

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

## After the plan (user's machine)

The smoke run the spec asks for, to size `--positions` and `--cand-m`:

```bash
./target/release/qwixxer win-pi --init win_model --labels cand --rounds 1 \
  --positions 20000 --gate-max 100000 --master-games 0 --dir win_pi_cand_smoke
```

Read from it: decisions per game (`cand_decisions` / (20000 / 25)), the candidate share (`cand_samples` / `positions`), seconds per round, and the headroom (`diff_rmse` vs `diff_noise`, `flip_share`).
