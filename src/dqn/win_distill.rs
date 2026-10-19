//! Candidate distillation for the win net: the decisions win-search would
//! search, their top candidates' full-game rollout values, and the static
//! net's error on the differences between candidates.
//!
//! Design: docs/superpowers/specs/2026-10-17-win-candidate-distillation-design.md

use crate::dqn::win::{swap_input, win_input, WIN_INPUTS};
use crate::state::{Mark, State};
use crate::strategy::bot_impl::{active_phase1_choices, active_phase2_choices, passive_phase1_choices, Decision};
use crate::strategy::search::{
    context_seed, gates_with, passive_entry, passive_first_active, phase1_entry, phase2_entry, rollout_final_states,
    SEARCH_CANDS,
};
use crate::strategy::sim::SimGame;
use crate::strategy::{Bot, Next};

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
fn valued(
    bot: &impl Bot,
    choices: &[(Option<Mark>, State)],
    eval_opps: &[State],
    next: Next,
) -> Vec<(Option<Mark>, State, f32)> {
    let posts: Vec<State> = choices.iter().map(|c| c.1).collect();
    let values = bot.evaluate_batch_next(&posts, eval_opps, next);
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
    for (m, s, v) in valued(bot, &plans, &sim_opp, Next::Opp) {
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
    let kept = gated(valued(bot, &choices, opps, Next::Opp), state, opps, margin)?;
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
    let kept = gated(valued(bot, &choices, opps, Next::Me), state, opps, margin)?;
    Some(CandDecision {
        entries: kept
            .iter()
            .map(|&(m, _)| passive_entry(bot, state, opps, dice, active_player, m))
            .collect(),
        first_active: passive_first_active(active_player, opps.len() + 1),
        seed: context_seed(state, opps, dice),
    })
}

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

/// Sample variance (n − 1); 0 below two values.
pub(crate) fn sample_var(xs: &[f32]) -> f32 {
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

/// What the difference diagnostic reads from one decision: its labels in
/// the samples' frame, the runner-ups' paired variances, and the frame.
pub struct GroupView<'a> {
    pub labels: Vec<f32>,
    pub pair_var: &'a [f32],
    pub swapped: bool,
}

/// `preds[g][i]` = the champion's P for `groups[g].samples[i]` (the
/// samples' frame). `None` when no decision has a runner-up.
pub fn diff_stats(preds: &[Vec<f32>], groups: &[CandGroup], k: usize) -> Option<DiffReport> {
    let views: Vec<GroupView> = groups
        .iter()
        .map(|g| GroupView {
            labels: g.samples.iter().map(|s| s.1).collect(),
            pair_var: &g.pair_var,
            swapped: g.swapped,
        })
        .collect();
    diff_stats_views(preds, &views, k)
}

/// [`diff_stats`] over [`GroupView`]s.
pub fn diff_stats_views(preds: &[Vec<f32>], groups: &[GroupView], k: usize) -> Option<DiffReport> {
    let (mut sq, mut var, mut pairs, mut flips) = (0.0f64, 0.0f64, 0usize, 0usize);
    for (p, g) in preds.iter().zip(groups) {
        let y = &g.labels;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::pair::PairStrategy;
    use crate::dqn::win_train::turn_ends_by;

    fn pair() -> PairStrategy {
        PairStrategy::load("pair_model")
    }

    /// (me, opp, dice) contexts from Master's turn ends, a few dice each.
    fn contexts() -> Vec<(State, State, [u8; 6])> {
        let dice = [
            [1, 2, 3, 4, 5, 6],
            [3, 4, 6, 2, 5, 1],
            [6, 6, 1, 1, 2, 3],
            [2, 5, 4, 4, 3, 6],
        ];
        turn_ends_by(24, 5, &pair)
            .into_iter()
            .zip(dice.iter().cycle())
            .map(|((me, opp), &d)| (me, opp, d))
            .collect()
    }

    fn board(e: &(Vec<State>, bool)) -> [f32; WIN_INPUTS] {
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
        assert!(
            passive > 0 && active1 > 0 && active2 > 0,
            "{passive} {active1} {active2}"
        );
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
        let gs = vec![
            group(&[0.6, 0.4, 0.5], &[0.04, 0.08], false),
            group(&[0.3, 0.7], &[0.02], false),
        ];
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

    #[test]
    fn passive_captures_are_valued_with_me_moving_next() {
        use crate::strategy::recorder::Recorder;
        use crate::strategy::Next;
        let bot = Recorder::default();
        let (me, opps) = (State::default(), [State::default()]);
        let dice = [3, 4, 2, 3, 5, 1];
        assert!(capture_passive(&bot, &me, &opps, dice, 0, f32::INFINITY).is_some());
        assert_eq!(bot.take()[0].next, Next::Me);
        assert!(capture_active2(&bot, &me, &opps, dice, false, f32::INFINITY).is_some());
        assert_eq!(bot.take()[0].next, Next::Opp);
    }
}
