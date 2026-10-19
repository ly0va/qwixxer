//! A win model measured on a benchmark: a win-data dataset whose held-out
//! labels come from many rollouts, so the model's own error on the
//! differences between candidates is measured rather than buried in the
//! label noise.

use crate::dqn::win::WinStrategy;
use crate::dqn::win_arm::{bce, entropy, noise_floor};
use crate::dqn::win_data::{DecisionData, Sample, Split};
use crate::state::State;
use crate::strategy::search::WinProb;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Measurements of one model on one split.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchReport {
    pub turns: usize,
    pub decisions: usize,
    /// Candidate differences (runner-up minus first) over all decisions.
    pub pairs: usize,
    /// Win excess (BCE − label entropy) and its noise floor, over the turn
    /// ends and every candidate.
    pub excess: f64,
    pub floor: f64,
    /// Mean prediction minus mean label over the turn ends: how much the
    /// model favours the player who has just moved.
    pub bias: f64,
    /// RMSE of the predicted candidate differences against the labels',
    /// and the labels' own noise on them.
    pub diff_rmse: f64,
    pub diff_noise: f64,
    /// √(diff RMSE² − noise²), floored at 0, and its bootstrap standard
    /// error over decisions.
    pub own: f64,
    pub own_se: f64,
    /// Share of decisions where the model picks the candidate with the best
    /// label, and the mean label value lost where it does not (both for
    /// the player deciding; the labels' noise inflates the regret a little).
    pub agree: f64,
    pub regret: f64,
}

/// Bootstrap resamples behind [`BenchReport::own_se`].
const RESAMPLES: usize = 200;

fn own_error(sq: f64, var: f64, pairs: f64) -> f64 {
    ((sq - var) / pairs).max(0.0).sqrt()
}

/// The report for predictions `turn_p` (one per turn end) and `dec_p` (one
/// per candidate of each decision), with labels from `k_turn` / `k_cand`
/// rollouts.
pub fn bench_stats(
    turns: &[Sample],
    turn_p: &[f32],
    decs: &[DecisionData],
    dec_p: &[Vec<f32>],
    k_turn: usize,
    k_cand: usize,
) -> BenchReport {
    let (mut excess, mut floor, mut n) = (0.0f64, 0.0f64, 0usize);
    let mut add = |s: &Sample, p: f32, k: usize| {
        excess += bce(s.win, p) - entropy(s.win);
        floor += noise_floor(s.win, s.win_var, k);
        n += 1;
    };
    for (s, &p) in turns.iter().zip(turn_p) {
        add(s, p, k_turn);
    }
    // Per decision: squared difference errors, their label noise, pairs.
    let mut per: Vec<(f64, f64, f64)> = Vec::with_capacity(decs.len());
    let (mut agree, mut regret) = (0usize, 0.0f64);
    for (d, p) in decs.iter().zip(dec_p) {
        for (s, &q) in d.samples.iter().zip(p) {
            add(s, q, k_cand);
        }
        let y: Vec<f32> = d.samples.iter().map(|s| s.win).collect();
        let (mut sq, mut var) = (0.0f64, 0.0f64);
        for i in 1..y.len() {
            let e = (p[i] - p[0]) - (y[i] - y[0]);
            sq += (e * e) as f64;
            var += d.pair_var[i - 1] as f64 / k_cand as f64;
        }
        per.push((sq, var, (y.len() - 1) as f64));
        let mine = |v: &[f32], i: usize| if d.swapped { 1.0 - v[i] } else { v[i] };
        let best = |v: &[f32]| (1..v.len()).fold(0, |b, i| if mine(v, i) > mine(v, b) { i } else { b });
        let (want, got) = (best(&y), best(p));
        agree += (want == got) as usize;
        regret += (mine(&y, want) - mine(&y, got)) as f64;
    }
    let sum = |rows: &mut dyn Iterator<Item = &(f64, f64, f64)>| {
        rows.fold((0.0, 0.0, 0.0), |a, r| (a.0 + r.0, a.1 + r.1, a.2 + r.2))
    };
    let (sq, var, pairs) = sum(&mut per.iter());
    let own = own_error(sq, var, pairs);
    let mut rng = SmallRng::seed_from_u64(0xBE7C);
    let boot: Vec<f64> = (0..RESAMPLES)
        .map(|_| {
            let (sq, var, pairs) = sum(&mut (0..per.len()).map(|_| &per[rng.gen_range(0..per.len())]));
            own_error(sq, var, pairs)
        })
        .collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let m = mean(&boot);
    let own_se = if per.len() > 1 {
        (boot.iter().map(|b| (b - m) * (b - m)).sum::<f64>() / (RESAMPLES - 1) as f64).sqrt()
    } else {
        0.0
    };
    let nd = decs.len().max(1) as f64;
    let np = pairs.max(1.0);
    BenchReport {
        turns: turns.len(),
        decisions: decs.len(),
        pairs: pairs as usize,
        excess: excess / n.max(1) as f64,
        floor: floor / n.max(1) as f64,
        bias: turns.iter().zip(turn_p).map(|(s, &p)| (p - s.win) as f64).sum::<f64>() / turns.len().max(1) as f64,
        diff_rmse: (sq / np).sqrt(),
        diff_noise: (var / np).sqrt(),
        own,
        own_se,
        agree: agree as f64 / nd,
        regret: regret / nd,
    }
}

/// P(win) for each sample's `me`, `opp` to move next.
fn predict(net: &WinStrategy, samples: &[&Sample]) -> Vec<f32> {
    samples
        .chunks(4096)
        .flat_map(|c| {
            let groups: Vec<(&State, &[State])> = c.iter().map(|s| (&s.me, std::slice::from_ref(&s.opp))).collect();
            net.win_prob_multi(&groups)
        })
        .collect()
}

/// `net` measured on `split`, whose labels come from `k_turn` / `k_cand`
/// rollouts.
pub fn bench(net: &WinStrategy, split: &Split, k_turn: usize, k_cand: usize) -> BenchReport {
    let turns: Vec<&Sample> = split.turns.iter().collect();
    let turn_p = predict(net, &turns);
    let cands: Vec<&Sample> = split.decisions.iter().flat_map(|d| &d.samples).collect();
    let mut flat = predict(net, &cands).into_iter();
    let dec_p: Vec<Vec<f32>> = split
        .decisions
        .iter()
        .map(|d| flat.by_ref().take(d.samples.len()).collect())
        .collect();
    bench_stats(&split.turns, &turn_p, &split.decisions, &dec_p, k_turn, k_cand)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::win_distill::{diff_stats_views, GroupView};

    fn s(win: f32, win_var: f32) -> Sample {
        Sample {
            me: State::default(),
            opp: State::default(),
            win,
            win_var,
            marks: [0.0; 8],
            strikes: [0.0; 2],
            diff: 0.0,
        }
    }

    fn dec(wins: &[f32], pair_var: &[f32], swapped: bool) -> DecisionData {
        DecisionData {
            samples: wins.iter().map(|&w| s(w, 0.2)).collect(),
            pair_var: pair_var.to_vec(),
            swapped,
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn a_hand_worked_benchmark() {
        // A: picks the runner-up (label 0.5) over the best (0.6): error
        // (0.53 − 0.50) − (0.5 − 0.6) = 0.13, regret 0.1. B (the other
        // player deciding): labels 0.6 / 0.7 for the mover, picks the best;
        // error (0.36 − 0.42) − (0.3 − 0.4) = 0.04.
        let decs = [dec(&[0.6, 0.5], &[0.016], false), dec(&[0.4, 0.3], &[0.032], true)];
        let dec_p = [vec![0.50, 0.53], vec![0.42, 0.36]];
        let turns = [s(0.25, 0.1), s(0.75, 0.1)];
        let r = bench_stats(&turns, &[0.3, 0.8], &decs, &dec_p, 4, 8);
        assert_eq!((r.turns, r.decisions, r.pairs), (2, 2, 2));
        // RMSE² (0.13² + 0.04²) / 2 = 0.00925; noise² (0.016 + 0.032) / 8 / 2 = 0.003.
        assert!(close(r.diff_rmse, 0.00925f64.sqrt()), "{}", r.diff_rmse);
        assert!(close(r.diff_noise, 0.003f64.sqrt()), "{}", r.diff_noise);
        assert!(close(r.own, 0.00625f64.sqrt()), "{}", r.own);
        assert!(r.own_se > 0.0);
        assert!(close(r.agree, 0.5) && close(r.regret, 0.05), "{} {}", r.agree, r.regret);
        assert!(close(r.bias, 0.05), "{}", r.bias);
    }

    #[test]
    fn differences_match_the_fit_diagnostic() {
        let decs = [
            dec(&[0.6, 0.5, 0.55], &[0.016, 0.02], false),
            dec(&[0.4, 0.3], &[0.032], true),
        ];
        let dec_p = [vec![0.50, 0.53, 0.6], vec![0.42, 0.36]];
        let r = bench_stats(&[], &[], &decs, &dec_p, 64, 64);
        let views: Vec<GroupView> = decs
            .iter()
            .map(|d| GroupView {
                labels: d.samples.iter().map(|s| s.win).collect(),
                pair_var: &d.pair_var,
                swapped: d.swapped,
            })
            .collect();
        let want = diff_stats_views(&dec_p, &views, 64).unwrap();
        assert!((r.diff_rmse - want.diff_rmse as f64).abs() < 1e-6);
        assert!((r.diff_noise - want.diff_noise as f64).abs() < 1e-6);
    }

    #[test]
    fn a_perfect_predictor_has_no_own_error() {
        let decs = [
            dec(&[0.6, 0.5], &[0.2], false),
            dec(&[0.4, 0.3, 0.45], &[0.1, 0.3], true),
        ];
        let dec_p: Vec<Vec<f32>> = decs.iter().map(|d| d.samples.iter().map(|s| s.win).collect()).collect();
        let turns = [s(0.25, 0.1), s(0.75, 0.1)];
        let r = bench_stats(&turns, &[0.25, 0.75], &decs, &dec_p, 64, 64);
        assert_eq!((r.diff_rmse, r.own, r.own_se), (0.0, 0.0, 0.0));
        assert_eq!((r.agree, r.regret, r.bias), (1.0, 0.0, 0.0));
        assert!(r.excess.abs() < 1e-9 && r.floor > 0.0);
    }

    #[test]
    fn the_net_is_measured_on_every_sample_in_order() {
        use crate::dqn::pair::PairStrategy;
        use crate::dqn::win_data::{generate, DataCfg};
        let cfg = DataCfg {
            player: "pair_model".into(),
            out: String::new(),
            positions: 60,
            rollouts: 2,
            cand_rollouts: 2,
            holdout: 1.0,
            keep: 1.0,
            seed: 3,
            resume: false,
        };
        let ds = generate(&cfg, &|| PairStrategy::load("pair_model"));
        let net = WinStrategy::load("win_model");
        let r = bench(&net, &ds.heldout, 2, 2);
        let one = |s: &Sample| net.win_prob_multi(&[(&s.me, std::slice::from_ref(&s.opp))])[0];
        let turn_p: Vec<f32> = ds.heldout.turns.iter().map(one).collect();
        let dec_p: Vec<Vec<f32>> = ds
            .heldout
            .decisions
            .iter()
            .map(|d| d.samples.iter().map(one).collect())
            .collect();
        assert_eq!(
            r,
            bench_stats(&ds.heldout.turns, &turn_p, &ds.heldout.decisions, &dec_p, 2, 2)
        );
        assert!(r.decisions > 0 && r.turns >= 60);
    }
}
