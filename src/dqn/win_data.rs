//! Stored datasets for the win net: turn ends and the candidates of the
//! decisions win-search would search, labelled with full-game rollouts
//! (win and auxiliary labels), split into training and held-out games.
//!
//! Design: docs/superpowers/specs/2026-10-19-win-thermo-aux-design.md

use crate::dqn::win_distill::{sample_var, CandDecision};
use crate::state::State;
use crate::strategy::search::rollout_final_states;
use crate::strategy::sim::SimGame;
use crate::strategy::Bot;

/// One labelled position in the turn-end frame (the other player to move
/// next). Every label is a mean over the position's rollouts.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub me: State,
    pub opp: State,
    /// Mean outcome for me (win 1, tie ½, loss 0).
    pub win: f32,
    /// Sample variance of the outcomes (0 for an ended position).
    pub win_var: f32,
    /// Final row totals / 12: mine red, yellow, green, blue, then the opponent's.
    pub marks: [f32; 8],
    /// Final strikes / 4: mine, the opponent's.
    pub strikes: [f32; 2],
    /// Final (my − opponent's) points / 100.
    pub diff: f32,
}

/// Exact equality: the boards byte for byte (`State`'s own `==` is a
/// dominance test) and every label.
impl PartialEq for Sample {
    fn eq(&self, o: &Self) -> bool {
        self.me.to_bytes() == o.me.to_bytes() && self.opp.to_bytes() == o.opp.to_bytes() && self.floats() == o.floats()
    }
}

impl Sample {
    /// Labels from the rollouts' final boards (`finals`: one (me, opp) per
    /// rollout; a single element for an ended position, its exact result).
    pub fn from_finals(me: State, opp: State, finals: &[Vec<State>]) -> Sample {
        let n = finals.len() as f32;
        let outcomes: Vec<f32> = finals.iter().map(|f| SimGame::outcome(f)).collect();
        let (mut marks, mut strikes, mut diff) = ([0.0f32; 8], [0.0f32; 2], 0.0f32);
        for f in finals {
            for side in 0..2 {
                for (r, &t) in f[side].row_totals().iter().enumerate() {
                    marks[4 * side + r] += t as f32 / 12.0 / n;
                }
                strikes[side] += f[side].strikes as f32 / 4.0 / n;
            }
            diff += (f[0].count_points() - f[1].count_points()) as f32 / 100.0 / n;
        }
        Sample {
            me,
            opp,
            win: outcomes.iter().sum::<f32>() / n,
            win_var: sample_var(&outcomes),
            marks,
            strikes,
            diff,
        }
    }

    /// The same sample from the other player's side.
    pub fn swapped(&self) -> Sample {
        let mut marks = [0.0f32; 8];
        marks[..4].copy_from_slice(&self.marks[4..]);
        marks[4..].copy_from_slice(&self.marks[..4]);
        Sample {
            me: self.opp,
            opp: self.me,
            win: 1.0 - self.win,
            win_var: self.win_var,
            marks,
            strikes: [self.strikes[1], self.strikes[0]],
            diff: -self.diff,
        }
    }

    /// The labels as stored: win, win_var, marks, strikes, diff.
    pub fn floats(&self) -> [f32; 13] {
        let mut f = [0.0f32; 13];
        f[0] = self.win;
        f[1] = self.win_var;
        f[2..10].copy_from_slice(&self.marks);
        f[10..12].copy_from_slice(&self.strikes);
        f[12] = self.diff;
        f
    }

    /// The inverse of [`Sample::floats`].
    pub fn from_floats(me: State, opp: State, f: [f32; 13]) -> Sample {
        let mut marks = [0.0f32; 8];
        marks.copy_from_slice(&f[2..10]);
        Sample {
            me,
            opp,
            win: f[0],
            win_var: f[1],
            marks,
            strikes: [f[10], f[11]],
            diff: f[12],
        }
    }
}

/// One labelled decision: a sample per kept candidate (best static value
/// first) and the CRN spread of each runner-up's difference to the first,
/// as [`crate::dqn::win_distill::CandGroup`] holds them.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionData {
    pub samples: Vec<Sample>,
    /// `pair_var[i − 1]`: sample variance of outcome(i) − outcome(0).
    pub pair_var: Vec<f32>,
    /// The entries had me to move next, so the samples are stored from the
    /// opponent's side ([`Sample::swapped`]).
    pub swapped: bool,
}

/// [`crate::dqn::win_distill::label_decision`] with every label: `k`
/// full-game CRN rollouts by `bot` per kept candidate.
pub fn label_decision_full(bot: &impl Bot, d: &CandDecision, k: usize) -> DecisionData {
    let finals = rollout_final_states(bot, &d.entries, d.seed, k, d.first_active, false);
    let swapped = d.first_active == 0;
    let samples = d
        .entries
        .iter()
        .zip(&finals)
        .map(|((e, _), f)| {
            assert_eq!(e.len(), 2, "the win net is 2-player only");
            let s = Sample::from_finals(e[0], e[1], f);
            if swapped {
                s.swapped()
            } else {
                s
            }
        })
        .collect();
    // An ended entry's exact result stands for every rollout.
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
    let pair_var = outcomes[1..]
        .iter()
        .map(|o| {
            let diffs: Vec<f32> = o.iter().zip(&outcomes[0]).map(|(a, b)| a - b).collect();
            sample_var(&diffs)
        })
        .collect();
    DecisionData {
        samples,
        pair_var,
        swapped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::pair::PairStrategy;
    use crate::dqn::win::win_input;
    use crate::dqn::win_distill::label_decision;
    use crate::state::Mark;
    use crate::strategy::search::{phase2_entry, rollout_final_states};

    #[test]
    fn labels_of_an_ended_position_are_exact() {
        let mut done = State::default();
        for _ in 0..4 {
            done.apply_strike();
        }
        let mut opp = State::default();
        opp.apply_mark(Mark { row: 0, number: 2 });
        let s = Sample::from_finals(done, opp, &[vec![done, opp]]);
        assert_eq!((s.win, s.win_var), (0.0, 0.0));
        assert_eq!(s.marks, [0.0, 0.0, 0.0, 0.0, 1.0 / 12.0, 0.0, 0.0, 0.0]);
        assert_eq!(s.strikes, [1.0, 0.0]);
        assert_eq!(s.diff, -21.0 / 100.0);
    }

    #[test]
    fn labels_average_the_rollouts() {
        let mut a = State::default();
        a.apply_mark(Mark { row: 0, number: 2 });
        a.apply_mark(Mark { row: 0, number: 3 });
        let mut b = State::default();
        b.apply_strike();
        // a: 3 points, b: −5; once each way round.
        let s = Sample::from_finals(State::default(), State::default(), &[vec![a, b], vec![b, a]]);
        assert_eq!((s.win, s.win_var), (0.5, 0.5));
        assert!((s.marks[0] - 1.0 / 12.0).abs() < 1e-7 && (s.marks[4] - 1.0 / 12.0).abs() < 1e-7);
        assert!((s.strikes[0] - 0.125).abs() < 1e-7 && (s.strikes[1] - 0.125).abs() < 1e-7);
        assert!(s.diff.abs() < 1e-7);
    }

    #[test]
    fn a_swapped_sample_is_the_other_players_view() {
        let mut a = State::default();
        a.apply_mark(Mark { row: 1, number: 4 });
        let b = State::default();
        let s = Sample {
            me: a,
            opp: b,
            win: 0.75,
            win_var: 0.2,
            marks: [1., 2., 3., 4., 5., 6., 7., 8.],
            strikes: [0.25, 0.5],
            diff: 0.1,
        };
        let t = s.swapped();
        assert_eq!((t.me.to_bytes(), t.opp.to_bytes()), (b.to_bytes(), a.to_bytes()));
        assert_eq!((t.win, t.win_var, t.diff), (0.25, 0.2, -0.1));
        assert_eq!(t.marks, [5., 6., 7., 8., 1., 2., 3., 4.]);
        assert_eq!(t.strikes, [0.5, 0.25]);
        assert_eq!(t.swapped(), s);
        assert_eq!(Sample::from_floats(s.me, s.opp, s.floats()), s);
    }

    #[test]
    fn full_labels_match_the_win_only_labels() {
        let bot = PairStrategy::load("pair_model");
        let opp = State::default();
        let mut marked = State::default();
        marked.apply_mark(Mark { row: 0, number: 2 });
        let mut done = State::default();
        for _ in 0..4 {
            done.apply_strike();
        }
        let live = vec![
            phase2_entry(State::default(), &[opp]),
            phase2_entry(done, &[opp]),
            phase2_entry(marked, &[opp]),
        ];
        let me_next = vec![phase2_entry(marked, &[opp]), phase2_entry(State::default(), &[opp])];
        for (entries, first_active) in [(live, 1), (me_next, 0)] {
            let d = CandDecision {
                entries,
                first_active,
                seed: 9,
            };
            let (g, full) = (label_decision(&bot, &d, 8), label_decision_full(&bot, &d, 8));
            assert_eq!(full.swapped, g.swapped);
            assert_eq!(full.pair_var, g.pair_var);
            assert_eq!(full.samples.len(), g.samples.len());
            for (s, (x, y)) in full.samples.iter().zip(&g.samples) {
                assert_eq!(win_input(&s.me, &s.opp), *x);
                assert_eq!(s.win, *y);
            }
            // Every auxiliary label swaps with the players, not only win.
            let finals = rollout_final_states(&bot, &d.entries, 9, 8, first_active, false);
            let e = &d.entries[0].0;
            let plain = Sample::from_finals(e[0], e[1], &finals[0]);
            let want = if first_active == 0 { plain.swapped() } else { plain };
            assert_eq!(full.samples[0], want);
        }
    }
}
