//! Test helpers: a bot that logs the frame (`Next`) of every evaluation,
//! and one that never wants a lock.

use super::search::WinProb;
use super::{Bot, Next};
use crate::state::State;
use std::cell::RefCell;

/// One logged evaluation: its frame, whether it was a search leaf, and the
/// strikes of the group's first candidate (or of the leaf's own board).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Call {
    pub next: Next,
    pub leaf: bool,
    pub strikes: u8,
}

#[derive(Debug, Default)]
pub struct Recorder {
    calls: RefCell<Vec<Call>>,
}

impl Recorder {
    /// The calls since the last `take`.
    pub fn take(&self) -> Vec<Call> {
        self.calls.take()
    }

    fn value(s: &State, opps: &[State]) -> f32 {
        let best = opps.iter().map(|o| o.count_points()).max().unwrap_or(0);
        (s.count_points() - best) as f32 / 10.0
    }

    fn log(&self, next: Next, leaf: bool, s: Option<&State>) {
        if let Some(s) = s {
            self.calls.borrow_mut().push(Call {
                next,
                leaf,
                strikes: s.strikes,
            });
        }
    }
}

impl Bot for Recorder {
    fn evaluate(&self, our_state: &State, opp_states: &[State]) -> f32 {
        self.evaluate_batch(std::slice::from_ref(our_state), opp_states)[0]
    }

    fn evaluate_batch(&self, candidates: &[State], opp_states: &[State]) -> Vec<f32> {
        self.evaluate_batch_next(candidates, opp_states, Next::Opp)
    }

    fn evaluate_batch_multi(&self, groups: &[(&[State], &[State])]) -> Vec<Vec<f32>> {
        groups
            .iter()
            .map(|(c, o)| self.evaluate_batch_next(c, o, Next::Opp))
            .collect()
    }

    fn evaluate_batch_next(&self, candidates: &[State], opp_states: &[State], next: Next) -> Vec<f32> {
        self.log(next, false, candidates.first());
        candidates.iter().map(|s| Self::value(s, opp_states)).collect()
    }

    fn evaluate_batch_multi_next(&self, groups: &[(&[State], &[State], Next)]) -> Vec<Vec<f32>> {
        groups
            .iter()
            .map(|(c, o, n)| self.evaluate_batch_next(c, o, *n))
            .collect()
    }
}

impl WinProb for Recorder {
    fn win_prob_multi(&self, groups: &[(&State, &[State])]) -> Vec<f32> {
        let framed: Vec<(&State, &[State], Next)> = groups.iter().map(|(s, o)| (*s, *o, Next::Opp)).collect();
        self.win_prob_multi_next(&framed)
    }

    fn win_prob_multi_next(&self, groups: &[(&State, &[State], Next)]) -> Vec<f32> {
        groups
            .iter()
            .map(|(s, o, n)| {
                self.log(*n, true, Some(s));
                1.0 / (1.0 + (-Self::value(s, o)).exp())
            })
            .collect()
    }
}

/// Values a board by minus its points, so it never takes a lock unless the
/// rule forces it (and strikes rather than mark).
#[derive(Debug, Clone, Copy, Default)]
pub struct ShunsPoints;

impl Bot for ShunsPoints {
    fn evaluate(&self, s: &State, _: &[State]) -> f32 {
        -(s.count_points() as f32)
    }
}

impl WinProb for ShunsPoints {
    fn win_prob_multi(&self, groups: &[(&State, &[State])]) -> Vec<f32> {
        groups
            .iter()
            .map(|(s, _)| 1.0 / (1.0 + (s.count_points() as f32 / 10.0).exp()))
            .collect()
    }
}

/// Red 2..6 marked: a white or red 12 is a safe lock.
pub fn lockable() -> State {
    let mut s = State::default();
    for n in 2..=6 {
        s.apply_mark(crate::state::Mark { row: 0, number: n });
    }
    s
}

/// Values a board by its lead in points (tenths), and a finished game by
/// its result: ±50 for a win or a loss, 0 for a tie.
#[derive(Debug, Clone, Copy, Default)]
pub struct Points;

impl Bot for Points {
    fn evaluate(&self, s: &State, opps: &[State]) -> f32 {
        let best = opps.iter().map(|o| o.count_points()).max().unwrap_or(0);
        (s.count_points() - best) as f32 / 10.0
    }

    fn terminal_value(&self, states: &[State]) -> f32 {
        (super::sim::SimGame::outcome(states) - 0.5) * 100.0
    }
}

impl WinProb for Points {
    fn win_prob_multi(&self, groups: &[(&State, &[State])]) -> Vec<f32> {
        groups
            .iter()
            .map(|(s, o)| 1.0 / (1.0 + (-self.evaluate(s, o)).exp()))
            .collect()
    }
}

/// Values every board by the frame alone: 1 when I move next, −1 when an
/// opponent does.
#[derive(Debug, Clone, Copy, Default)]
pub struct Framed;

impl Bot for Framed {
    fn evaluate(&self, _: &State, _: &[State]) -> f32 {
        -1.0
    }

    fn evaluate_batch_next(&self, candidates: &[State], _: &[State], next: Next) -> Vec<f32> {
        vec![if next == Next::Me { 1.0 } else { -1.0 }; candidates.len()]
    }

    fn evaluate_batch_multi_next(&self, groups: &[(&[State], &[State], Next)]) -> Vec<Vec<f32>> {
        groups
            .iter()
            .map(|(c, o, n)| self.evaluate_batch_next(c, o, *n))
            .collect()
    }
}

/// Row `row` with its five lowest numbers marked: a white or colour 12 (an
/// ascending row) is a lock.
pub fn lockable_row(row: usize) -> State {
    let mut s = State::default();
    let numbers: Vec<u8> = if row < 2 {
        (2..=6).collect()
    } else {
        (8..=12).rev().collect()
    };
    for n in numbers {
        s.apply_mark(crate::state::Mark { row, number: n });
    }
    s
}
