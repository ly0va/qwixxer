//! Test helper: a bot that logs the frame (`Next`) of every evaluation.

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
