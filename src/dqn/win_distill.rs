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
}
