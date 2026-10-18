//! One arm of the thermometer/auxiliary experiment: the auxiliary heads,
//! the held-out metrics, the learning-rate schedule, weight averaging and
//! the fit (`win-arm`).
//!
//! Design: docs/superpowers/specs/2026-10-19-win-thermo-aux-design.md

use crate::dqn::win::{
    flat_tensor, sigmoid, swap_input, swap_thermo, thermo_input, win_input, AuxHeads, WinModel, WIN_INPUTS,
    WIN_INPUTS_T,
};
use crate::dqn::win_data::{DecisionData, Sample, Split};
use crate::dqn::win_distill::{diff_stats_views, DiffReport, GroupView};
use crate::dqn::win_train::bce_logits;
use crate::dqn::MyBackend;
use crate::state::State;
use burn::{module::Param, nn::Linear, prelude::*, tensor::TensorData};

/// The net being fitted, with its auxiliary heads when the arm has them.
/// The net goes to `model.mpk`, the heads to `aux.mpk`.
#[derive(Module, Debug)]
pub struct ArmModel<B: Backend> {
    pub net: WinModel<B>,
    pub aux: Option<AuxHeads<B>>,
}

/// Outputs for n input rows: the antisymmetric logit `[n, 1]`, and with
/// heads the row marks `[n, 8]` (mine red…blue, then the opponent's),
/// strikes `[n, 2]` (mine, the opponent's) and difference `[n, 1]`.
pub struct ArmOut<B: Backend> {
    pub z: Tensor<B, 2>,
    pub marks: Option<Tensor<B, 2>>,
    pub strikes: Option<Tensor<B, 2>>,
    pub diff: Option<Tensor<B, 2>>,
}

/// Targets in [`ArmOut`]'s shapes.
pub struct Targets<B: Backend> {
    pub win: Tensor<B, 2>,
    pub marks: Tensor<B, 2>,
    pub strikes: Tensor<B, 2>,
    pub diff: Tensor<B, 2>,
}

impl<B: Backend> ArmModel<B> {
    /// Both orientations run as one batch, as in [`WinModel::forward`]; the
    /// opponent's predictions are the heads on the swapped half.
    pub fn forward(&self, x: Tensor<B, 2>, swapped: Tensor<B, 2>) -> ArmOut<B> {
        let n = x.dims()[0];
        let (e, h) = self.net.trunk(Tensor::cat(vec![x, swapped], 0));
        let g = self.net.out.forward(h.clone());
        let z = g.clone().narrow(0, 0, n) - g.narrow(0, n, n);
        let Some(aux) = &self.aux else {
            return ArmOut {
                z,
                marks: None,
                strikes: None,
                diff: None,
            };
        };
        let w = h.dims()[1];
        let hr = h.clone().reshape([2 * n, 1, w]).repeat_dim(1, 4);
        let m = aux.marks.forward(Tensor::cat(vec![e, hr], 2)).reshape([2 * n, 4]);
        let marks = Tensor::cat(vec![m.clone().narrow(0, 0, n), m.narrow(0, n, n)], 1);
        let s = aux.strikes.forward(h.clone());
        let strikes = Tensor::cat(vec![s.clone().narrow(0, 0, n), s.narrow(0, n, n)], 1);
        let a = aux.diff.forward(h);
        let diff = a.clone().narrow(0, 0, n) - a.narrow(0, n, n);
        ArmOut {
            z,
            marks: Some(marks),
            strikes: Some(strikes),
            diff: Some(diff),
        }
    }
}

/// Inputs and swapped inputs of (me, opp) boards, flat, and their width.
pub fn encode(boards: &[(&State, &State)], thermo: bool) -> (Vec<f32>, Vec<f32>, usize) {
    let (mut x, mut s) = (Vec::new(), Vec::new());
    for &(me, opp) in boards {
        if thermo {
            let v = thermo_input(me, opp);
            x.extend_from_slice(&v);
            s.extend_from_slice(&swap_thermo(&v));
        } else {
            let v = win_input(me, opp);
            x.extend_from_slice(&v);
            s.extend_from_slice(&swap_input(&v));
        }
    }
    (x, s, if thermo { WIN_INPUTS_T } else { WIN_INPUTS })
}

/// Inputs, swapped inputs and targets of `samples` as tensors.
pub fn tensors<B: Backend>(
    samples: &[&Sample],
    thermo: bool,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>, Targets<B>) {
    let boards: Vec<(&State, &State)> = samples.iter().map(|s| (&s.me, &s.opp)).collect();
    let (x, s, d) = encode(&boards, thermo);
    let n = samples.len();
    let col = |f: &dyn Fn(&Sample) -> Vec<f32>, w: usize| {
        let v: Vec<f32> = samples.iter().flat_map(|s| f(s)).collect();
        Tensor::<B, 1>::from_floats(v.as_slice(), device).reshape([n, w])
    };
    let t = Targets {
        win: col(&|s| vec![s.win], 1),
        marks: col(&|s| s.marks.to_vec(), 8),
        strikes: col(&|s| s.strikes.to_vec(), 2),
        diff: col(&|s| vec![s.diff], 1),
    };
    (flat_tensor(&x, d, device), flat_tensor(&s, d, device), t)
}

/// (total loss, win BCE): BCE(win) + w[0]·MSE(marks) + w[1]·MSE(strikes) +
/// w[2]·MSE(difference), the last three only with heads.
pub fn arm_loss<B: Backend>(out: ArmOut<B>, t: &Targets<B>, w: &[f32]) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let win = bce_logits(out.z, t.win.clone());
    let mse = |p: Tensor<B, 2>, y: Tensor<B, 2>| {
        let d = p - y;
        (d.clone() * d).mean()
    };
    let mut total = win.clone();
    if let (Some(m), Some(s), Some(d)) = (out.marks, out.strikes, out.diff) {
        total = total
            + mse(m, t.marks.clone()).mul_scalar(w[0])
            + mse(s, t.strikes.clone()).mul_scalar(w[1])
            + mse(d, t.diff.clone()).mul_scalar(w[2]);
    }
    (total, win)
}

/// Binary entropy in nats; 0·ln 0 = 0.
pub fn entropy(w: f32) -> f64 {
    let h = |q: f64| if q > 0.0 { -q * q.ln() } else { 0.0 };
    h(w as f64) + h(1.0 - w as f64)
}

/// Binary cross-entropy of prediction `p` against label `w`, in nats
/// (`p` clamped away from 0 and 1).
pub fn bce(w: f32, p: f32) -> f64 {
    let (w, p) = (w as f64, (p as f64).clamp(1e-7, 1.0 - 1e-7));
    -(w * p.ln() + (1.0 - w) * (1.0 - p).ln())
}

/// The win excess (BCE − entropy) a perfect predictor scores on a label
/// that is the mean of `k` outcomes with sample variance `v`:
/// v / (2k·w(1 − w)); 0 for a label of exactly 0 or 1.
pub fn noise_floor(w: f32, v: f32, k: usize) -> f64 {
    let w = w as f64;
    if w <= 0.0 || w >= 1.0 {
        return 0.0;
    }
    v as f64 / (2.0 * k as f64 * w * (1.0 - w))
}

/// Learning rate at `step` of `total`: linear warmup over the first 1% of
/// steps (at least one), then a cosine from `lr` down to 0.
pub fn cosine_lr(lr: f64, step: usize, total: usize) -> f64 {
    let warm = (total / 100).max(1);
    if step < warm {
        return lr * (step + 1) as f64 / warm as f64;
    }
    let t = (step - warm) as f64 / (total - warm).max(1) as f64;
    lr * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
}

/// e ← decay·e + (1 − decay)·v.
pub fn ema_update(e: &mut [f32], v: &[f32], decay: f32) {
    for (a, &b) in e.iter_mut().zip(v) {
        *a = decay * *a + (1.0 - decay) * b;
    }
}

fn linears<B: Backend>(m: &ArmModel<B>) -> Vec<&Linear<B>> {
    let n = &m.net;
    let mut v = vec![&n.row1, &n.row2, &n.pair, &n.head1, &n.head2, &n.out];
    if let Some(a) = &m.aux {
        v.extend([&a.marks, &a.strikes, &a.diff]);
    }
    v
}

fn linears_mut<B: Backend>(m: &mut ArmModel<B>) -> Vec<&mut Linear<B>> {
    let n = &mut m.net;
    let mut v = vec![
        &mut n.row1,
        &mut n.row2,
        &mut n.pair,
        &mut n.head1,
        &mut n.head2,
        &mut n.out,
    ];
    if let Some(a) = &mut m.aux {
        v.extend([&mut a.marks, &mut a.strikes, &mut a.diff]);
    }
    v
}

/// Every layer's weights then bias, in [`linears`] order.
fn weights<B: Backend>(m: &ArmModel<B>) -> Vec<Vec<f32>> {
    linears(m)
        .into_iter()
        .flat_map(|l| {
            [
                l.weight.val().into_data().to_vec::<f32>().unwrap(),
                l.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().unwrap(),
            ]
        })
        .collect()
}

fn set_linear(l: &mut Linear<MyBackend>, w: &[f32], b: &[f32], device: &<MyBackend as Backend>::Device) {
    let (d_in, d_out) = (l.weight.shape().dims[0], l.weight.shape().dims[1]);
    l.weight = Param::from_tensor(Tensor::from_data(TensorData::new(w.to_vec(), [d_in, d_out]), device));
    l.bias = Some(Param::from_tensor(Tensor::from_data(
        TensorData::new(b.to_vec(), [d_out]),
        device,
    )));
}

/// An exponential moving average of a model's weights, updated after
/// every step; decay 0 keeps the live weights.
pub struct Ema {
    decay: f32,
    values: Vec<Vec<f32>>,
}

impl Ema {
    pub fn new<B: Backend>(m: &ArmModel<B>, decay: f32) -> Self {
        Ema {
            decay,
            values: weights(m),
        }
    }

    pub fn update<B: Backend>(&mut self, m: &ArmModel<B>) {
        let now = weights(m);
        if self.decay == 0.0 {
            self.values = now;
            return;
        }
        for (e, v) in self.values.iter_mut().zip(&now) {
            ema_update(e, v, self.decay);
        }
    }

    /// `m` with the averaged weights.
    pub fn apply(&self, mut m: ArmModel<MyBackend>) -> ArmModel<MyBackend> {
        let device = Default::default();
        for (i, l) in linears_mut(&mut m).into_iter().enumerate() {
            set_linear(l, &self.values[2 * i], &self.values[2 * i + 1], &device);
        }
        m
    }
}

/// The lowest metric offered so far.
#[derive(Default)]
pub struct Best {
    pub metric: Option<f64>,
}

impl Best {
    /// Whether `m` is a new lowest (the first offer always is).
    pub fn offer(&mut self, m: f64) -> bool {
        if self.metric.is_none_or(|b| m < b) {
            self.metric = Some(m);
            true
        } else {
            false
        }
    }
}

/// Predictions for samples, in order; the aux vectors are empty without heads.
#[derive(Default)]
pub struct Preds {
    pub p: Vec<f32>,
    pub marks: Vec<[f32; 8]>,
    pub strikes: Vec<[f32; 2]>,
    pub diff: Vec<f32>,
}

pub fn predict(model: &ArmModel<MyBackend>, samples: &[&Sample]) -> Preds {
    let device = Default::default();
    let thermo = model.net.is_thermo();
    let mut out = Preds::default();
    let v = |t: Tensor<MyBackend, 2>| t.into_data().to_vec::<f32>().unwrap();
    for chunk in samples.chunks(4096) {
        let boards: Vec<(&State, &State)> = chunk.iter().map(|s| (&s.me, &s.opp)).collect();
        let (x, s, d) = encode(&boards, thermo);
        let o = model.forward(flat_tensor(&x, d, &device), flat_tensor(&s, d, &device));
        out.p.extend(v(o.z).into_iter().map(sigmoid));
        if let (Some(m), Some(st), Some(df)) = (o.marks, o.strikes, o.diff) {
            out.marks
                .extend(v(m).chunks(8).map(|c| <[f32; 8]>::try_from(c).unwrap()));
            out.strikes
                .extend(v(st).chunks(2).map(|c| <[f32; 2]>::try_from(c).unwrap()));
            out.diff.extend(v(df));
        }
    }
    out
}

/// One split's measurements: win excess (BCE − label entropy) and its
/// noise floor over every sample, each head's MSE, and the difference
/// diagnostic over the decisions.
#[derive(Clone, Debug)]
pub struct EvalReport {
    pub samples: usize,
    pub excess: f64,
    pub floor: f64,
    pub marks: Option<f64>,
    pub strikes: Option<f64>,
    pub diff: Option<f64>,
    pub diffs: Option<DiffReport>,
}

/// `k_turn` / `k_cand`: rollouts behind a turn-end / candidate label.
pub fn evaluate(model: &ArmModel<MyBackend>, split: &Split, k_turn: usize, k_cand: usize) -> EvalReport {
    let cand: Vec<&Sample> = split.decisions.iter().flat_map(|d| &d.samples).collect();
    let all: Vec<&Sample> = split.turns.iter().chain(cand.iter().copied()).collect();
    let ks: Vec<usize> = split
        .turns
        .iter()
        .map(|_| k_turn)
        .chain(cand.iter().map(|_| k_cand))
        .collect();
    let pr = predict(model, &all);
    let n = all.len().max(1) as f64;
    let mean = |f: &dyn Fn(usize) -> f64| (0..all.len()).map(f).sum::<f64>() / n;
    let sq = |a: f32, b: f32| ((a - b) as f64).powi(2);
    let excess = mean(&|i| bce(all[i].win, pr.p[i]) - entropy(all[i].win));
    let floor = mean(&|i| noise_floor(all[i].win, all[i].win_var, ks[i]));
    let has_aux = !pr.diff.is_empty();
    let marks = has_aux.then(|| mean(&|i| (0..8).map(|j| sq(all[i].marks[j], pr.marks[i][j])).sum::<f64>() / 8.0));
    let strikes =
        has_aux.then(|| mean(&|i| (0..2).map(|j| sq(all[i].strikes[j], pr.strikes[i][j])).sum::<f64>() / 2.0));
    let diff = has_aux.then(|| mean(&|i| sq(all[i].diff, pr.diff[i])));
    // The decisions' predictions follow the turn ends', in order.
    let mut at = split.turns.len();
    let mut preds: Vec<Vec<f32>> = Vec::new();
    let views: Vec<GroupView> = split
        .decisions
        .iter()
        .map(|d| {
            preds.push(pr.p[at..at + d.samples.len()].to_vec());
            at += d.samples.len();
            GroupView {
                labels: d.samples.iter().map(|s| s.win).collect(),
                pair_var: &d.pair_var,
                swapped: d.swapped,
            }
        })
        .collect();
    EvalReport {
        samples: all.len(),
        excess,
        floor,
        marks,
        strikes,
        diff,
        diffs: diff_stats_views(&preds, &views, k_cand),
    }
}

/// The first `max` turn ends, and the first decisions up to `max` samples.
pub fn subset(split: &Split, max: usize) -> Split {
    let mut decisions: Vec<DecisionData> = Vec::new();
    let mut n = 0;
    for d in &split.decisions {
        if n + d.samples.len() > max {
            break;
        }
        n += d.samples.len();
        decisions.push(d.clone());
    }
    Split {
        turns: split.turns.iter().take(max).copied().collect(),
        decisions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::win::{WinModelConfig, TOKEN, T_TOKEN};
    use crate::dqn::win_distill::sample_var;
    use crate::state::Mark;
    use burn::tensor::ElementConversion;
    use rand::{Rng, SeedableRng};

    static SEEDING: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn random_arm(seed: u64, thermo: bool, aux: bool) -> ArmModel<MyBackend> {
        let _guard = SEEDING.lock().unwrap();
        let device = Default::default();
        <MyBackend as Backend>::seed(&device, seed);
        let net = WinModelConfig::new()
            .with_width(8)
            .with_thermo(thermo.then_some(true))
            .init(&device);
        ArmModel {
            net,
            aux: aux.then(|| AuxHeads::new(8, &device)),
        }
    }

    fn boards() -> (State, State) {
        let mut a = State::default();
        for m in [
            Mark { row: 0, number: 2 },
            Mark { row: 0, number: 5 },
            Mark { row: 2, number: 12 },
        ] {
            a.apply_mark(m);
        }
        a.apply_strike();
        let mut b = State::default();
        for m in [
            Mark { row: 1, number: 4 },
            Mark { row: 3, number: 9 },
            Mark { row: 3, number: 7 },
        ] {
            b.apply_mark(m);
        }
        (a, b)
    }

    fn values(t: Option<Tensor<MyBackend, 2>>) -> Vec<f32> {
        t.unwrap().into_data().to_vec::<f32>().unwrap()
    }

    /// Permute the 4 row tokens of every flat input row.
    fn relabel(x: &[f32], d: usize, token: usize, order: [usize; 4]) -> Vec<f32> {
        let mut y = x.to_vec();
        for row in 0..x.len() / d {
            for (slot, &src) in order.iter().enumerate() {
                let (to, from) = (row * d + token * slot, row * d + token * src);
                y[to..to + token].copy_from_slice(&x[from..from + token]);
            }
        }
        y
    }

    #[test]
    fn aux_predictions_follow_the_players_and_the_rows() {
        let device = Default::default();
        let (a, b) = boards();
        for thermo in [false, true] {
            let m = random_arm(31, thermo, true);
            let run =
                |x: &[f32], s: &[f32], d: usize| m.forward(flat_tensor(x, d, &device), flat_tensor(s, d, &device));
            let (x, s, d) = encode(&[(&a, &b)], thermo);
            let (xr, sr, _) = encode(&[(&b, &a)], thermo);
            let (o, r) = (run(&x, &s, d), run(&xr, &sr, d));
            let (zo, zr) = (
                o.z.into_data().to_vec::<f32>().unwrap(),
                r.z.into_data().to_vec::<f32>().unwrap(),
            );
            assert!((zo[0] + zr[0]).abs() < 1e-5);
            let (mo, mr) = (values(o.marks), values(r.marks));
            for i in 0..4 {
                assert!(
                    (mo[i] - mr[4 + i]).abs() < 1e-5 && (mo[4 + i] - mr[i]).abs() < 1e-5,
                    "thermo {thermo} row {i}"
                );
            }
            let (so, sr) = (values(o.strikes), values(r.strikes));
            assert!((so[0] - sr[1]).abs() < 1e-5 && (so[1] - sr[0]).abs() < 1e-5);
            let (dd, dr) = (values(o.diff), values(r.diff));
            assert!((dd[0] + dr[0]).abs() < 1e-5);
            // A colour relabelling permutes the row predictions with the rows.
            let token = if thermo { T_TOKEN } else { TOKEN };
            let order = [1usize, 0, 3, 2];
            let base = values(run(&x, &s, d).marks);
            let perm = values(run(&relabel(&x, d, token, order), &relabel(&s, d, token, order), d).marks);
            for side in 0..2 {
                for (slot, &src) in order.iter().enumerate() {
                    assert!((perm[4 * side + slot] - base[4 * side + src]).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn the_loss_adds_each_head_with_its_weight() {
        let device = Default::default();
        let t2 = |v: Vec<f32>, w: usize| Tensor::<MyBackend, 1>::from_floats(v.as_slice(), &device).reshape([1, w]);
        let out = || ArmOut {
            z: t2(vec![0.0], 1),
            marks: Some(t2(vec![0.0; 8], 8)),
            strikes: Some(t2(vec![0.0; 2], 2)),
            diff: Some(t2(vec![0.0], 1)),
        };
        let t = Targets {
            win: t2(vec![1.0], 1),
            marks: t2(vec![0.5; 8], 8),
            strikes: t2(vec![1.0; 2], 2),
            diff: t2(vec![0.2], 1),
        };
        let scalar = |x: Tensor<MyBackend, 1>| x.into_scalar().elem::<f32>();
        let (total, win) = arm_loss(out(), &t, &[1.0, 0.5, 0.5]);
        let ln2 = std::f32::consts::LN_2;
        assert!((scalar(win) - ln2).abs() < 1e-6);
        assert!((scalar(total) - (ln2 + 0.25 + 0.5 + 0.02)).abs() < 1e-6);
        let plain = ArmOut {
            marks: None,
            strikes: None,
            diff: None,
            ..out()
        };
        assert!((scalar(arm_loss(plain, &t, &[1.0, 0.5, 0.5]).0) - ln2).abs() < 1e-6);
    }

    #[test]
    fn metrics_are_finite_at_certain_labels() {
        assert_eq!((entropy(0.0), entropy(1.0)), (0.0, 0.0));
        assert!((entropy(0.5) - std::f64::consts::LN_2).abs() < 1e-12);
        assert!((bce(0.5, 0.5) - std::f64::consts::LN_2).abs() < 1e-7);
        for (w, p) in [(1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0)] {
            assert!(bce(w, p).is_finite(), "bce({w}, {p})");
        }
        assert_eq!((noise_floor(0.0, 0.0, 64), noise_floor(1.0, 0.0, 64)), (0.0, 0.0));
        assert!((noise_floor(0.5, 0.25, 64) - 1.0 / 128.0).abs() < 1e-12);
    }

    #[test]
    fn the_noise_floor_is_what_a_perfect_predictor_scores() {
        let (p, k, n) = (0.3f32, 64usize, 20_000);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(5);
        let (mut excess, mut floor) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let o: Vec<f32> = (0..k).map(|_| (rng.gen::<f32>() < p) as u8 as f32).collect();
            let w = o.iter().sum::<f32>() / k as f32;
            excess += bce(w, p) - entropy(w);
            floor += noise_floor(w, sample_var(&o), k);
        }
        let ratio = excess / floor;
        assert!(
            (ratio - 1.0).abs() < 0.1,
            "excess {} vs floor {}",
            excess / n as f64,
            floor / n as f64
        );
    }

    #[test]
    fn cosine_schedule_warms_up_then_decays_to_zero() {
        let lr = 1e-3;
        assert!((cosine_lr(lr, 0, 1000) - lr * 0.1).abs() < 1e-15);
        assert!((cosine_lr(lr, 9, 1000) - lr).abs() < 1e-15);
        assert!((cosine_lr(lr, 10, 1000) - lr).abs() < 1e-15);
        assert!((cosine_lr(lr, 505, 1000) - lr * 0.5).abs() < 1e-12);
        assert!(cosine_lr(lr, 999, 1000) < lr * 1e-4);
        assert!(cosine_lr(lr, 0, 1) > 0.0, "a one-step fit still steps");
    }

    #[test]
    fn averaging_and_best_tracking() {
        let mut e = [0.0f32, 2.0];
        ema_update(&mut e, &[1.0, 2.0], 0.5);
        ema_update(&mut e, &[1.0, 2.0], 0.5);
        assert_eq!(e, [0.75, 2.0]);
        let mut best = Best::default();
        let offered: Vec<bool> = [3.0, 1.0, 2.0, 1.0].iter().map(|&m| best.offer(m)).collect();
        assert_eq!(offered, [true, true, false, false]);
        assert_eq!(best.metric, Some(1.0));
    }

    #[test]
    fn averaged_weights_with_no_decay_are_the_live_weights() {
        let m = random_arm(32, true, true);
        let mut ema = Ema::new(&m, 0.0);
        ema.update(&m);
        let back = ema.apply(m.clone());
        let (a, b) = boards();
        let (x, s, d) = encode(&[(&a, &b)], true);
        let device = Default::default();
        let z = |m: &ArmModel<MyBackend>| {
            m.forward(flat_tensor(&x, d, &device), flat_tensor(&s, d, &device))
                .z
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        };
        assert_eq!(z(&back), z(&m));
    }

    #[test]
    fn labels_equal_to_the_predictions_score_no_error() {
        let m = random_arm(33, false, true);
        let (a, b) = boards();
        let raw: Vec<Sample> = [(a, b), (b, a), (a, a), (b, b)]
            .iter()
            .map(|&(me, opp)| Sample {
                me,
                opp,
                win: 0.0,
                win_var: 0.0,
                marks: [0.0; 8],
                strikes: [0.0; 2],
                diff: 0.0,
            })
            .collect();
        let refs: Vec<&Sample> = raw.iter().collect();
        let pr = predict(&m, &refs);
        let labelled: Vec<Sample> = raw
            .iter()
            .enumerate()
            .map(|(i, s)| Sample {
                win: pr.p[i],
                marks: pr.marks[i],
                strikes: pr.strikes[i],
                diff: pr.diff[i],
                ..*s
            })
            .collect();
        let split = Split {
            turns: labelled[..2].to_vec(),
            decisions: vec![DecisionData {
                samples: labelled[2..].to_vec(),
                pair_var: vec![0.0],
                swapped: false,
            }],
        };
        let r = evaluate(&m, &split, 64, 64);
        assert_eq!(r.samples, 4);
        assert!(r.excess.abs() < 1e-6 && r.floor == 0.0, "{r:?}");
        assert!(
            r.marks.unwrap() < 1e-12 && r.strikes.unwrap() < 1e-12 && r.diff.unwrap() < 1e-12,
            "{r:?}"
        );
        assert!(r.diffs.unwrap().diff_rmse < 1e-5);
        let plain = random_arm(34, false, false);
        let r = evaluate(&plain, &split, 64, 64);
        assert!(r.marks.is_none() && r.strikes.is_none() && r.diff.is_none());
        let sub = subset(&split, 1);
        assert_eq!((sub.turns.len(), sub.decisions.len()), (1, 0));
    }
}
