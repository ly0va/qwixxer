//! Gated policy iteration for the win net: the champion's rollouts label its
//! own games, a warm-started candidate is fitted to a window of recent
//! rounds, and a sequential paired head-to-head decides promotion.
//!
//! Design: docs/superpowers/specs/2026-10-17-win-policy-iteration-design.md

use crate::dqn::win_train::outcome;
use crate::game::{Game, Player};
use crate::strategy::{Bot, Strategy};
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;
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

/// Paired games between `a` and `b`, one score per pair index: both games
/// of a pair share the dice streams (`seed`, pair index, seat), `a` in seat
/// 0 then seat 1, and the score is `a`'s mean outcome (win 1, tie ½). The
/// dice depend only on `seed` and the pair index, so splitting `pairs`
/// into blocks or changing the thread count never changes a score. Bots
/// are built once per worker thread.
pub fn head_to_head<A: Bot + Clone + 'static, B: Bot + Clone + 'static>(
    a: &(dyn Fn() -> A + Sync),
    b: &(dyn Fn() -> B + Sync),
    pairs: std::ops::Range<u64>,
    seed: u64,
) -> Vec<f32> {
    pairs
        .into_par_iter()
        .map_init(
            || (a(), b()),
            |(a, b), p| {
                let dice = |seat: u64| Box::new(SmallRng::seed_from_u64(seed.wrapping_add(p * 2 + seat)));
                let mut total = 0.0;
                for rot in 0..2 {
                    let x: Box<dyn Strategy> = Box::new(a.clone());
                    let y: Box<dyn Strategy> = Box::new(b.clone());
                    let players = if rot == 0 {
                        vec![Player::new(x, dice(0)), Player::new(y, dice(1))]
                    } else {
                        vec![Player::new(y, dice(0)), Player::new(x, dice(1))]
                    };
                    let mut game = Game::new(players);
                    game.play();
                    let s: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
                    total += outcome(s[rot], s[1 - rot]);
                }
                total / 2.0
            },
        )
        .collect()
}

pub struct GateResult {
    pub promoted: bool,
    pub games: usize,
    pub share: f64,
    pub z: f64,
}

/// The spec's sequential gate: play `block` games at a time (the last
/// block cut so the total never exceeds `max`) and stop as soon as
/// [`gate_decision`] promotes or rejects.
pub fn run_gate<A: Bot + Clone + 'static, B: Bot + Clone + 'static>(
    cand: &(dyn Fn() -> A + Sync),
    champ: &(dyn Fn() -> B + Sync),
    seed: u64,
    z_min: f64,
    block: usize,
    max: usize,
) -> GateResult {
    let (block_pairs, max_pairs) = ((block / 2) as u64, (max / 2) as u64);
    let mut scores: Vec<f32> = Vec::new();
    loop {
        let start = scores.len() as u64;
        let end = (start + block_pairs).min(max_pairs);
        scores.extend(head_to_head(cand, champ, start..end, seed));
        let (g, share, z) = gate_decision(&scores, z_min, end >= max_pairs);
        if g != Gate::Continue {
            return GateResult {
                promoted: g == Gate::Promote,
                games: scores.len() * 2,
                share,
                z,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::win::{WinModelConfig, WinStrategy};
    use crate::dqn::MyBackend;
    use burn::prelude::*;

    /// A random W=4 net. Burn's seed is global, so seeding and init are
    /// serialized: parallel tests must not interleave between the two.
    fn tiny_net(seed: u64) -> WinStrategy {
        static SEEDING: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = SEEDING.lock().unwrap();
        let device = Default::default();
        <MyBackend as Backend>::seed(&device, seed);
        WinStrategy::from_model(&WinModelConfig::new().with_width(4).init::<MyBackend>(&device))
    }

    #[test]
    fn a_bot_against_itself_scores_exactly_half_per_pair() {
        let a = tiny_net(1);
        let scores = head_to_head(&|| a.clone(), &|| a.clone(), 0..8, 7);
        assert_eq!(scores, vec![0.5; 8]);
    }

    #[test]
    fn head_to_head_is_deterministic_and_split_independent() {
        let (a, b) = (tiny_net(1), tiny_net(2));
        let whole = head_to_head(&|| a.clone(), &|| b.clone(), 0..12, 7);
        assert_eq!(whole, head_to_head(&|| a.clone(), &|| b.clone(), 0..12, 7));
        let mut split = head_to_head(&|| a.clone(), &|| b.clone(), 0..5, 7);
        split.extend(head_to_head(&|| a.clone(), &|| b.clone(), 5..12, 7));
        assert_eq!(whole, split);
        assert!(
            whole.iter().any(|&s| s != 0.5),
            "different nets should differ somewhere"
        );
        // Swapping the bots mirrors every pair score.
        let mirrored = head_to_head(&|| b.clone(), &|| a.clone(), 0..12, 7);
        for (x, y) in whole.iter().zip(&mirrored) {
            assert!((x + y - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn gate_on_equal_bots_rejects_after_one_block() {
        let a = tiny_net(1);
        let r = run_gate(&|| a.clone(), &|| a.clone(), 7, 2.0, 10, 40);
        assert!(!r.promoted);
        assert_eq!(r.games, 10);
    }

    #[test]
    fn gate_never_plays_past_its_cap() {
        let (a, b) = (tiny_net(1), tiny_net(2));
        // z_min unreachable and the cap not a multiple of the block: the
        // gate must stop at exactly `max` games unless it rejects earlier.
        let r = run_gate(&|| a.clone(), &|| b.clone(), 7, f64::INFINITY, 8, 14);
        assert!(!r.promoted);
        assert!(r.games <= 14 && r.games % 2 == 0, "games = {}", r.games);
        // Stopping before the cap is only allowed on a reject (z < 0, or
        // z reported as 0 for zero spread).
        assert!(r.games == 14 || r.z <= 0.0, "games = {}, z = {}", r.games, r.z);
    }

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
