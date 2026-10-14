pub mod bot_impl;
pub mod search;
pub mod sim;
pub(crate) use bot_impl::{active_phase1_impl, active_phase2_impl, passive_phase1_impl};

use crate::state::{format_dice, Mark, State};
use rand::Rng;

const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

pub trait Strategy: std::fmt::Debug {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark>;
    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark>;
    fn passive_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        active_player: usize,
    ) -> Option<Mark>;
    fn is_interactive(&self) -> bool {
        false
    }
}

pub trait Bot: std::fmt::Debug {
    fn evaluate(&self, our_state: &State, opp_states: &[State]) -> f32;
    fn evaluate_batch(&self, candidates: &[State], opp_states: &[State]) -> Vec<f32> {
        candidates.iter().map(|s| self.evaluate(s, opp_states)).collect()
    }
    /// Evaluate several independent (candidates, opp_states) groups. The
    /// default loops `evaluate_batch`; bots with batched inference override
    /// this to run all groups in a single forward pass.
    fn evaluate_batch_multi(&self, groups: &[(&[State], &[State])]) -> Vec<Vec<f32>> {
        groups.iter().map(|(c, o)| self.evaluate_batch(c, o)).collect()
    }
}

// ---- Interactive ----

#[derive(Debug, Default, Clone)]
pub struct Interactive;

impl Interactive {
    fn read_line() -> String {
        use std::io::{self, Write};
        print!("  {BOLD}> {RESET}");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        input
    }

    fn show_marks(marks: &[Mark]) {
        let w = if marks.len() >= 10 { 2 } else { 1 };
        for (i, mark) in marks.iter().enumerate() {
            print!(" {DIM}{:>w$}){RESET} {mark}", i + 1);
        }
        println!();
    }

    fn pick_mark(marks: &[Mark], allow_skip: bool) -> Option<Mark> {
        if allow_skip {
            println!("\n  {DIM}Enter number or Enter to skip:{RESET}");
        } else {
            println!("\n  {DIM}Enter number:{RESET}");
        }
        loop {
            let input = Self::read_line();
            let input = input.trim();
            if input.is_empty() && allow_skip {
                println!("  {DIM}Skipped.{RESET}");
                return None;
            }
            if let Ok(n) = input.parse::<usize>() {
                if n >= 1 && n <= marks.len() {
                    return Some(marks[n - 1]);
                }
            }
            if allow_skip {
                println!("  \x1b[91mInvalid. Enter 1-{} or Enter to skip.{RESET}", marks.len());
            } else {
                println!("  \x1b[91mInvalid. Enter 1-{}.{RESET}", marks.len());
            }
        }
    }
}

impl Strategy for Interactive {
    fn active_phase1(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        println!("\n  {BOLD}=== YOUR TURN (Phase 1: White Dice) ==={RESET}\n");
        println!("{state}");
        println!("{}", format_dice(dice));
        println!();

        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        if marks.is_empty() {
            println!("  {DIM}No white-sum marks available.{RESET}");
            return None;
        }
        print!("  {DIM}White sum {white_sum}:{RESET}");
        Self::show_marks(&marks);
        Self::pick_mark(&marks, true)
    }

    fn active_phase2(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        println!("\n  {BOLD}=== YOUR TURN (Phase 2: Color Dice) ==={RESET}\n");
        println!("{state}");
        if !has_marked {
            println!("  {DIM}(No mark in Phase 1 -- you must mark here or take a strike){RESET}");
        }

        let marks = state.generate_color_moves(dice);
        if marks.is_empty() {
            if !has_marked {
                println!("  {DIM}No color marks available -- strike.{RESET}");
            } else {
                println!("  {DIM}No color marks available.{RESET}");
            }
            return None;
        }
        print!("  {DIM}Color marks:{RESET}");
        Self::show_marks(&marks);
        Self::pick_mark(&marks, has_marked)
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        _opp_states: &[State],
        dice: [u8; 6],
        _active_player: usize,
    ) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        println!("\n  {BOLD}=== OPPONENT'S TURN (White Dice) ==={RESET}\n");
        println!("{state}");
        println!("  White sum: {BOLD}{white_sum}{RESET}");
        println!();

        let marks = state.generate_white_moves(white_sum);
        if marks.is_empty() {
            println!("  {DIM}No marks available.{RESET}");
            return None;
        }
        print!("  {DIM}Mark:{RESET}");
        Self::show_marks(&marks);
        Self::pick_mark(&marks, true)
    }

    fn is_interactive(&self) -> bool {
        true
    }
}

// ---- Conservative ----

/// Conservative strategy: only marks numbers that don't skip too many.
/// Picks the move creating the fewest new blanks, rejecting any that
/// would add more than `max_new_blanks`. Among moves with equal blanks,
/// picks the one with the highest resulting score.
#[derive(Debug, Clone)]
pub struct Conservative {
    max_new_blanks: u8,
}

impl Default for Conservative {
    fn default() -> Self {
        Self { max_new_blanks: 3 }
    }
}

impl Conservative {
    fn best_mark(state: &State, marks: &[Mark], max_new_blanks: u8) -> Option<Mark> {
        let current_blanks = state.blanks();
        marks
            .iter()
            .filter_map(|&mark| {
                let mut new_state = *state;
                new_state.apply_mark(mark);
                let new_blanks = new_state.blanks().saturating_sub(current_blanks);
                if new_blanks <= max_new_blanks {
                    Some((mark, new_blanks, new_state.count_points()))
                } else {
                    None
                }
            })
            // Minimize blanks first, then maximize resulting score
            .min_by_key(|(_, blanks, score)| (*blanks, -*score))
            .map(|(mark, _, _)| mark)
    }
}

impl Strategy for Conservative {
    fn active_phase1(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        Self::best_mark(state, &marks, self.max_new_blanks)
    }

    fn active_phase2(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        let cap = if has_marked {
            self.max_new_blanks
        } else {
            self.max_new_blanks
        };
        Self::best_mark(state, &marks, cap)
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        _opp_states: &[State],
        dice: [u8; 6],
        _active_player: usize,
    ) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        Self::best_mark(state, &marks, 0)
    }
}

// ---- Opportunist ----

/// Opportunist strategy: optimizes for probability (keeping free pointers
/// on common 2d6 sums like 6/7/8) to maximize marks on opponent turns.
#[derive(Debug, Default, Clone)]
pub struct Opportunist;

impl Opportunist {
    /// Filter marks to those within a blank budget, then pick by best probability.
    fn best_mark(state: &State, marks: &[Mark], max_new_blanks: u8) -> Option<Mark> {
        let current_blanks = state.blanks();
        marks
            .iter()
            .copied()
            .filter_map(|mark| {
                let mut new_state = *state;
                new_state.apply_mark(mark);
                let new_blanks = new_state.blanks().saturating_sub(current_blanks);
                if new_blanks <= max_new_blanks {
                    Some((mark, new_state.probability()))
                } else {
                    None
                }
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(mark, _)| mark)
    }
}

impl Strategy for Opportunist {
    fn active_phase1(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        Self::best_mark(state, &marks, 2)
    }

    fn active_phase2(
        &mut self,
        state: &State,
        _opp_states: &[State],
        dice: [u8; 6],
        _has_marked: bool,
    ) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        Self::best_mark(state, &marks, 2)
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        _opp_states: &[State],
        dice: [u8; 6],
        _active_player: usize,
    ) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        Self::best_mark(state, &marks, 1)
    }
}

/// The Opportunist with the shared pipeline's pruning: it never picks a mark
/// whose post-state another available mark dominates (e.g. yellow 7 when
/// yellow 5 is on offer). The web's Newbie; the CLI's `opportunist` (and
/// its recorded benchmarks) stay unpruned.
#[derive(Debug, Default, Clone)]
pub struct PrunedOpportunist;

impl PrunedOpportunist {
    fn best_mark(state: &State, marks: &[Mark], max_new_blanks: u8) -> Option<Mark> {
        let mut cands: Vec<(Mark, State)> = marks
            .iter()
            .map(|&m| {
                let mut s = *state;
                s.apply_mark(m);
                (m, s)
            })
            .collect();
        bot_impl::prune_dominated(&mut cands, |(_, s)| s);
        let kept: Vec<Mark> = cands.into_iter().map(|(m, _)| m).collect();
        Opportunist::best_mark(state, &kept, max_new_blanks)
    }
}

impl Strategy for PrunedOpportunist {
    fn active_phase1(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let marks = state.generate_white_moves(dice[0] + dice[1]);
        Self::best_mark(state, &marks, 2)
    }

    fn active_phase2(
        &mut self,
        state: &State,
        _opp_states: &[State],
        dice: [u8; 6],
        _has_marked: bool,
    ) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        Self::best_mark(state, &marks, 2)
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        _opp_states: &[State],
        dice: [u8; 6],
        _active_player: usize,
    ) -> Option<Mark> {
        let marks = state.generate_white_moves(dice[0] + dice[1]);
        Self::best_mark(state, &marks, 1)
    }
}

// ---- Random ----

#[derive(Debug, Default, Clone)]
pub struct Random;

impl Strategy for Random {
    fn active_phase1(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        if !marks.is_empty() {
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.5) {
                Some(marks[rng.gen_range(0..marks.len())])
            } else {
                None
            }
        } else {
            None
        }
    }

    fn active_phase2(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        if !marks.is_empty() {
            let mut rng = rand::thread_rng();
            if has_marked {
                // 50% mark random, 50% skip
                if rng.gen_bool(0.5) {
                    Some(marks[rng.gen_range(0..marks.len())])
                } else {
                    None
                }
            } else {
                // Must mark to avoid strike
                Some(marks[rng.gen_range(0..marks.len())])
            }
        } else {
            None
        }
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        _opp_states: &[State],
        dice: [u8; 6],
        _active_player: usize,
    ) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        if !marks.is_empty() {
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.5) {
                Some(marks[rng.gen_range(0..marks.len())])
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    /// Is `pick` dominated by another of `marks` (its post-state strictly worse)?
    fn dominated(state: &State, marks: &[Mark], pick: Mark) -> bool {
        let post = |m: Mark| {
            let mut s = *state;
            s.apply_mark(m);
            s
        };
        marks
            .iter()
            .any(|&m| post(pick).partial_cmp(&post(m)) == Some(std::cmp::Ordering::Less))
    }

    #[test]
    fn pruned_opportunist_never_picks_a_dominated_mark() {
        let mut rng = SmallRng::seed_from_u64(3);
        let (mut plain_dominated, mut colour_decisions) = (0, 0);
        for _ in 0..3000 {
            let mut marks = [[false; 11]; 4];
            for row in marks.iter_mut() {
                for m in row.iter_mut().take(9) {
                    *m = rng.gen_bool(0.2);
                }
            }
            let state = State::from_marks(&marks, 0);
            let dice: [u8; 6] = core::array::from_fn(|_| rng.gen_range(1..=6));
            let colour = state.generate_color_moves(dice);
            if colour.len() < 2 {
                continue;
            }
            colour_decisions += 1;
            if let Some(m) = PrunedOpportunist.active_phase2(&state, &[], dice, false) {
                assert!(!dominated(&state, &colour, m), "{m:?} from {colour:?}");
            }
            if let Some(m) = Opportunist.active_phase2(&state, &[], dice, false) {
                plain_dominated += dominated(&state, &colour, m) as usize;
            }
            let white = state.generate_white_moves(dice[0] + dice[1]);
            if let Some(m) = PrunedOpportunist.active_phase1(&state, &[], dice) {
                assert!(!dominated(&state, &white, m));
            }
        }
        // The plain Opportunist does pick dominated marks, so the test bites.
        assert!(
            colour_decisions > 500 && plain_dominated > 0,
            "{colour_decisions} {plain_dominated}"
        );
    }
}
