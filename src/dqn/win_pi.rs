//! Gated policy iteration for the win net: the champion's rollouts label its
//! own games, a warm-started candidate is fitted to a window of recent
//! rounds, and a sequential paired head-to-head decides promotion.
//!
//! Design: docs/superpowers/specs/2026-10-17-win-policy-iteration-design.md

use std::collections::VecDeque;

/// Seed purposes for [`round_seed`].
pub const POS: u64 = 1;
pub const FIT: u64 = 2;
pub const GATE: u64 = 3;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The seed for one purpose (positions, fit, gate) of one round, all
/// derived from the run's `--seed`.
pub fn round_seed(seed: u64, round: usize, purpose: u64) -> u64 {
    splitmix64(seed ^ splitmix64(((round as u64) << 8) | purpose))
}

/// Append a round's data and keep only the last `keep` rounds.
pub fn push_window<T>(window: &mut VecDeque<Vec<T>>, round: Vec<T>, keep: usize) {
    window.push_back(round);
    while window.len() > keep {
        window.pop_front();
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gate {
    Promote,
    Reject,
    Continue,
}

/// The sequential gate's decision over the pair scores so far (each in
/// [0, 1], the candidate's mean over a pair's two seats): returns the
/// decision, the share and z = (share − ½) / (pair-score std / √pairs).
/// Promote when z > `z_min`; reject when z < 0, at the cap (`at_max`), or
/// when the pair scores have no spread (z is then reported as 0).
pub fn gate_decision(scores: &[f32], z_min: f64, at_max: bool) -> (Gate, f64, f64) {
    let n = scores.len() as f64;
    let share = scores.iter().map(|&s| s as f64).sum::<f64>() / n.max(1.0);
    if scores.len() < 2 {
        return (Gate::Reject, share, 0.0);
    }
    let var = scores.iter().map(|&s| (s as f64 - share).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (var / n).sqrt();
    if se == 0.0 {
        return (Gate::Reject, share, 0.0);
    }
    let z = (share - 0.5) / se;
    let g = if z > z_min {
        Gate::Promote
    } else if z < 0.0 || at_max {
        Gate::Reject
    } else {
        Gate::Continue
    };
    (g, share, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_seeds_are_deterministic_and_distinct() {
        assert_eq!(round_seed(42, 3, POS), round_seed(42, 3, POS));
        let mut seen = std::collections::HashSet::new();
        for round in 0..50 {
            for purpose in [POS, FIT, GATE] {
                assert!(
                    seen.insert(round_seed(42, round, purpose)),
                    "collision at {round}/{purpose}"
                );
            }
        }
        assert_ne!(round_seed(42, 1, POS), round_seed(43, 1, POS));
    }

    #[test]
    fn window_keeps_the_last_rounds() {
        let mut w: VecDeque<Vec<u32>> = VecDeque::new();
        push_window(&mut w, vec![1], 2);
        push_window(&mut w, vec![2, 2], 2);
        assert_eq!(w.iter().flatten().count(), 3);
        // A third round drops the oldest; every round's data enters,
        // whatever its gate decided.
        push_window(&mut w, vec![3], 2);
        assert_eq!(w, VecDeque::from(vec![vec![2, 2], vec![3]]));
    }

    #[test]
    fn gate_promotes_a_clear_winner() {
        let scores: Vec<f32> = (0..200).map(|i| if i % 2 == 0 { 1.0 } else { 0.5 }).collect();
        let (g, share, z) = gate_decision(&scores, 2.0, false);
        assert_eq!(g, Gate::Promote);
        assert!((share - 0.75).abs() < 1e-9);
        assert!(z > 2.0);
    }

    #[test]
    fn gate_rejects_a_net_behind() {
        let scores: Vec<f32> = (0..200).map(|i| if i % 2 == 0 { 0.0 } else { 0.5 }).collect();
        assert_eq!(gate_decision(&scores, 2.0, false).0, Gate::Reject);
    }

    #[test]
    fn gate_continues_when_undecided_and_rejects_at_the_cap() {
        // Share just above ½, z ≈ 0.05: keep playing, or give up at the cap.
        let mut scores: Vec<f32> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        scores.push(0.75);
        let (g, _, z) = gate_decision(&scores, 2.0, false);
        assert!(z > 0.0 && z < 2.0, "z = {z}");
        assert_eq!(g, Gate::Continue);
        assert_eq!(gate_decision(&scores, 2.0, true).0, Gate::Reject);
    }

    #[test]
    fn gate_rejects_zero_variance() {
        assert_eq!(gate_decision(&[0.5; 10], 2.0, false).0, Gate::Reject);
        assert_eq!(gate_decision(&[0.5], 2.0, false).0, Gate::Reject);
    }
}
