//! One arm of the thermometer/auxiliary experiment: the auxiliary heads,
//! the held-out metrics, the learning-rate schedule, weight averaging and
//! the fit (`win-arm`).
//!
//! Design: docs/superpowers/specs/2026-10-19-win-thermo-aux-design.md

use crate::dqn::win::{
    flat_tensor, sigmoid, swap_input, swap_thermo, thermo_input, win_input, AuxHeads, WinModel, WIN_INPUTS,
    WIN_INPUTS_T,
};
use crate::dqn::win::{load_tempo, save_aux, save_tempo, widen_to_thermo, zero_tempo, WinModelConfig};
use crate::dqn::win_data::{self, Dataset};
use crate::dqn::win_data::{DecisionData, Sample, Split};
use crate::dqn::win_distill::{diff_stats_views, DiffReport, GroupView};
use crate::dqn::win_pi::{load_win_ad, save_win};
use crate::dqn::win_train::bce_logits;
use crate::dqn::win_train::Ad;
use crate::dqn::MyBackend;
use crate::state::State;
use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, AdamWConfig, GradientsParams, Optimizer},
};
use burn::{module::Param, nn::Linear, prelude::*, tensor::TensorData};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use std::path::Path;

/// The net being fitted, with its auxiliary heads and tempo head when the
/// arm has them. The net goes to `model.mpk`, the heads to `aux.mpk` and
/// `tempo.mpk`.
#[derive(Module, Debug)]
pub struct ArmModel<B: Backend> {
    pub net: WinModel<B>,
    pub aux: Option<AuxHeads<B>>,
    /// The tempo head when the arm has one; saved to `tempo.mpk`.
    pub tempo: Option<Linear<B>>,
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
        let z = match &self.tempo {
            Some(head) => {
                let t = head.forward(h.clone());
                z + t.clone().narrow(0, 0, n) + t.narrow(0, n, n)
            }
            None => z,
        };
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
    if let Some(t) = &m.tempo {
        v.push(t);
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
    if let Some(t) = &mut m.tempo {
        v.push(t);
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

/// Training batch size, as `train_epochs`.
pub const BATCH: usize = 1024;

/// Training turn ends and training samples in the training-excess subset.
const TRAIN_SUBSET: usize = 50_000;

pub const FIT_HEADER: &str = "epoch,train_excess,heldout_excess,heldout_floor,heldout_marks,heldout_strikes,\
heldout_diff,diff_rmse,diff_noise,flip_share,best,elapsed_s";

/// `win-arm` settings.
pub struct ArmCfg {
    /// A `win-data` dir.
    pub data: String,
    /// Output model dir (must not exist).
    pub out: String,
    /// Warm start from this win model dir; `None` = from scratch.
    pub init: Option<String>,
    /// Thermometer inputs (a plain `init` is widened).
    pub thermo: bool,
    /// Auxiliary heads.
    pub aux: bool,
    /// Tempo head (the value of moving next).
    pub tempo: bool,
    pub epochs: usize,
    /// Peak learning rate (the constant one with `legacy`).
    pub lr: f64,
    /// AdamW weight decay.
    pub wd: f32,
    /// Weight-average decay per step (0 = the live weights).
    pub ema: f32,
    /// Aux loss weights: marks, strikes, difference.
    pub aux_weights: Vec<f32>,
    /// Width of a from-scratch net.
    pub width: usize,
    pub seed: u64,
    /// Today's fit: `train_epochs`' Adam at a constant lr, no weight decay,
    /// no averaging, the last epoch saved.
    pub legacy: bool,
}

impl ArmCfg {
    /// Every check that can fail before the dataset is read.
    pub fn validate(&self) -> Result<(), String> {
        if self.epochs == 0 {
            return Err("--epochs must be at least 1".into());
        }
        if !(self.lr > 0.0) {
            return Err("--lr must be positive".into());
        }
        if !(0.0..1.0).contains(&self.ema) {
            return Err("--ema must be in [0, 1)".into());
        }
        if !(self.wd >= 0.0) {
            return Err("--wd must be at least 0".into());
        }
        if self.aux_weights.len() != 3 || self.aux_weights.iter().any(|&w| !(w >= 0.0)) {
            return Err("--aux-weights takes 3 non-negative weights: marks,strikes,difference".into());
        }
        if self.legacy && (self.thermo || self.aux || self.tempo) {
            return Err("--legacy-fit refits the plain net without aux heads: drop --thermo, --aux and --tempo".into());
        }
        for f in ["meta.json", "data.bin"] {
            if !Path::new(&format!("{}/{f}", self.data)).exists() {
                return Err(format!("--data {}: {f} not found", self.data));
            }
        }
        match &self.init {
            Some(init) => {
                for f in ["config.json", "model.mpk"] {
                    if !Path::new(&format!("{init}/{f}")).exists() {
                        return Err(format!("--init {init}: {f} not found"));
                    }
                }
                let c = WinModelConfig::load(format!("{init}/config.json"))
                    .map_err(|e| format!("--init {init}: unreadable config.json ({e:?})"))?;
                if c.is_thermo() && !self.thermo {
                    return Err(format!("--init {init} is a thermometer net: add --thermo"));
                }
                if c.is_tempo() && !self.tempo {
                    return Err(format!("--init {init} has a tempo head: add --tempo"));
                }
            }
            None if self.width == 0 => return Err("--width must be at least 1".into()),
            None => {}
        }
        if Path::new(&self.out).exists() {
            return Err(format!(
                "{} already exists: move it aside or pick another --out",
                self.out
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct FitSummary {
    /// 1-based epoch whose weights were saved.
    pub best_epoch: usize,
    pub epochs: usize,
}

/// Fit one arm (docs/superpowers/specs/2026-10-19-win-thermo-aux-design.md,
/// "Fit"): writes `out/config.json`, `out/model.mpk` (the best epoch's
/// averaged net, without heads), `out/aux.mpk` and `out/tempo.mpk` when the
/// arm has them, and `out/fit.csv`.
pub fn fit(cfg: &ArmCfg) -> Result<FitSummary, String> {
    cfg.validate()?;
    let ds = win_data::load(&cfg.data).map_err(|e| format!("--data {}: {e}", cfg.data))?;
    if ds.heldout.turns.is_empty() && ds.heldout.decisions.is_empty() {
        return Err(format!(
            "--data {}: no held-out games (generate it with --holdout above 0)",
            cfg.data
        ));
    }
    if ds.train.turns.is_empty() && ds.train.decisions.is_empty() {
        return Err(format!("--data {}: no training games", cfg.data));
    }
    let device = Default::default();
    <Ad as Backend>::seed(&device, cfg.seed);
    let net: WinModel<Ad> = match &cfg.init {
        Some(dir) => {
            let (_, m) = load_win_ad(dir);
            if cfg.thermo && !m.is_thermo() {
                widen_to_thermo(&m, &device)
            } else {
                m
            }
        }
        None => WinModelConfig::new()
            .with_width(cfg.width)
            .with_thermo(cfg.thermo.then_some(true))
            .init(&device),
    };
    let width = net.row1.weight.shape().dims[1];
    let init_tempo = cfg
        .init
        .as_ref()
        .filter(|dir| WinModelConfig::load(format!("{dir}/config.json")).is_ok_and(|c| c.is_tempo()));
    let tempo = cfg.tempo.then(|| match init_tempo {
        Some(dir) => load_tempo::<Ad>(dir, width, &device),
        None => zero_tempo::<Ad>(width, &device),
    });
    let mcfg = WinModelConfig::new()
        .with_width(width)
        .with_thermo(net.is_thermo().then_some(true))
        .with_aux(cfg.aux.then_some(true))
        .with_tempo(cfg.tempo.then_some(true));
    let model = ArmModel {
        net,
        aux: cfg.aux.then(|| AuxHeads::new(width, &device)),
        tempo,
    };
    std::fs::create_dir_all(&cfg.out).map_err(|e| format!("{}: {e}", cfg.out))?;
    if cfg.legacy {
        run(cfg, &ds, &mcfg, model, AdamConfig::new().init())
    } else {
        run(
            cfg,
            &ds,
            &mcfg,
            model,
            AdamWConfig::new().with_weight_decay(cfg.wd).init(),
        )
    }
}

fn run<O: Optimizer<ArmModel<Ad>, Ad>>(
    cfg: &ArmCfg,
    ds: &Dataset,
    mcfg: &WinModelConfig,
    mut model: ArmModel<Ad>,
    mut optim: O,
) -> Result<FitSummary, String> {
    let device = Default::default();
    let start = std::time::Instant::now();
    // Turn ends then candidates, as win-pi orders a round's data.
    let samples: Vec<&Sample> = ds
        .train
        .turns
        .iter()
        .chain(ds.train.decisions.iter().flat_map(|d| &d.samples))
        .collect();
    let thermo = model.net.is_thermo();
    let total = cfg.epochs * samples.len().div_ceil(BATCH);
    let mut ema = Ema::new(&model, if cfg.legacy { 0.0 } else { cfg.ema });
    let train_sub = subset(&ds.train, TRAIN_SUBSET);
    let (kt, kc) = (ds.meta.rollouts, ds.meta.cand_rollouts);
    let log_path = format!("{}/fit.csv", cfg.out);
    std::fs::write(&log_path, format!("{FIT_HEADER}\n")).map_err(|e| format!("{log_path}: {e}"))?;
    let mut order: Vec<usize> = (0..samples.len()).collect();
    let (mut best, mut best_model, mut best_epoch, mut step) = (Best::default(), None, 0, 0);
    for epoch in 0..cfg.epochs {
        order.shuffle(&mut SmallRng::seed_from_u64(cfg.seed.wrapping_add(epoch as u64)));
        for chunk in order.chunks(BATCH) {
            let batch: Vec<&Sample> = chunk.iter().map(|&i| samples[i]).collect();
            let (x, s, t) = tensors::<Ad>(&batch, thermo, &device);
            let (loss, _) = arm_loss(model.forward(x, s), &t, &cfg.aux_weights);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            let lr = if cfg.legacy {
                cfg.lr
            } else {
                cosine_lr(cfg.lr, step, total)
            };
            model = optim.step(lr, model, grads);
            ema.update(&model);
            step += 1;
        }
        let avg = ema.apply(model.valid());
        let held = evaluate(&avg, &ds.heldout, kt, kc);
        let train_excess = evaluate(&avg, &train_sub, kt, kc).excess;
        let metric = held.diffs.as_ref().map_or(held.excess, |d| d.diff_rmse as f64);
        let is_best = if cfg.legacy {
            epoch + 1 == cfg.epochs
        } else {
            best.offer(metric)
        };
        if is_best {
            best_model = Some(avg.clone());
            best_epoch = epoch + 1;
        }
        let opt = |v: Option<f64>| v.map(|v| format!("{v:.6}")).unwrap_or_default();
        let d = held.diffs.as_ref();
        let row = format!(
            "{},{train_excess:.6},{:.6},{:.6},{},{},{},{},{},{},{},{}",
            epoch + 1,
            held.excess,
            held.floor,
            opt(held.marks),
            opt(held.strikes),
            opt(held.diff),
            opt(d.map(|d| d.diff_rmse as f64)),
            opt(d.map(|d| d.diff_noise as f64)),
            opt(d.map(|d| d.flip_share as f64)),
            is_best as u8,
            start.elapsed().as_secs()
        );
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&log_path)
                .map_err(|e| format!("{log_path}: {e}"))?;
            writeln!(f, "{row}").map_err(|e| format!("{log_path}: {e}"))?;
        }
        println!(
            "Epoch {:>2}: train excess {train_excess:.5}, held-out excess {:.5} (floor {:.5}){}{}, {}s",
            epoch + 1,
            held.excess,
            held.floor,
            d.map(|d| format!(", diff RMSE {:.4} (noise {:.4})", d.diff_rmse, d.diff_noise))
                .unwrap_or_default(),
            if is_best { " → best" } else { "" },
            start.elapsed().as_secs()
        );
    }
    let best_model: ArmModel<MyBackend> = best_model.expect("at least one epoch");
    save_win(&cfg.out, mcfg, &best_model.net);
    if let Some(heads) = &best_model.aux {
        save_aux(&cfg.out, heads);
    }
    if let Some(head) = &best_model.tempo {
        save_tempo(&cfg.out, head);
    }
    Ok(FitSummary {
        best_epoch,
        epochs: cfg.epochs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::pair::PairStrategy;
    use crate::dqn::win::{save_tempo, zero_tempo, WinModelConfig, TOKEN, T_TOKEN};
    use crate::dqn::win::{AuxPred, ManualWinNet, WinStrategy};
    use crate::dqn::win_data::{generate, load, save, DataCfg};
    use crate::dqn::win_distill::sample_var;
    use crate::dqn::win_pi::{load_win_ad, save_win};
    use crate::dqn::win_train::train_epochs;
    use crate::state::Mark;
    use crate::strategy::search::WinProb;
    use burn::module::AutodiffModule;
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
            tempo: None,
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

    fn tmp(name: &str) -> String {
        let d = std::env::temp_dir().join(format!("qwixxer-win-arm-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d.to_string_lossy().into_owned()
    }

    /// A tiny dataset by Master with decisions on both sides, and a W=4
    /// plain net to start from.
    fn tiny(base: &str) -> (String, String) {
        let data = format!("{base}/data");
        let cfg = DataCfg {
            player: "pair_model".into(),
            out: data.clone(),
            positions: 120,
            rollouts: 2,
            cand_rollouts: 2,
            holdout: 0.5,
            keep: 1.0,
            seed: 3,
            resume: false,
        };
        let ds = generate(&cfg, &|| PairStrategy::load("pair_model"));
        assert!(
            !ds.train.decisions.is_empty() && !ds.heldout.decisions.is_empty(),
            "pick another seed: one side has no decisions"
        );
        save(&data, &ds).unwrap();
        let init = format!("{base}/init");
        let device = Default::default();
        let mcfg = WinModelConfig::new().with_width(4);
        let mut model = mcfg.init::<MyBackend>(&device);
        // Positive biases: a random W=4 net otherwise often starts with a
        // whole layer dead (every ReLU zero on every input), and then no
        // gradient reaches the row encoder at all.
        for l in [
            &mut model.row1,
            &mut model.row2,
            &mut model.pair,
            &mut model.head1,
            &mut model.head2,
        ] {
            let d = l.weight.shape().dims[1];
            l.bias = Some(Param::from_tensor(Tensor::from_data(
                TensorData::new(vec![0.1f32; d], [d]),
                &device,
            )));
        }
        save_win(&init, &mcfg, &model);
        (data, init)
    }

    fn arm(data: &str, out: String, init: Option<String>) -> ArmCfg {
        ArmCfg {
            data: data.into(),
            out,
            init,
            thermo: false,
            aux: false,
            tempo: false,
            epochs: 2,
            lr: 3e-4,
            wd: 1e-4,
            ema: 0.999,
            aux_weights: vec![1.0, 0.5, 0.5],
            width: 4,
            seed: 42,
            legacy: false,
        }
    }

    #[test]
    fn the_legacy_fit_is_todays_fit() {
        let base = tmp("legacy");
        let (data, init) = tiny(&base);
        let out = format!("{base}/legacy");
        fit(&ArmCfg {
            legacy: true,
            ..arm(&data, out.clone(), Some(init.clone()))
        })
        .unwrap();
        let ds = load(&data).unwrap();
        let samples: Vec<([f32; WIN_INPUTS], f32)> = ds
            .train
            .turns
            .iter()
            .chain(ds.train.decisions.iter().flat_map(|d| &d.samples))
            .map(|s| (win_input(&s.me, &s.opp), s.win))
            .collect();
        let (mcfg, start) = load_win_ad(&init);
        let (want, _) = train_epochs(start, &samples, 2, 3e-4, 42);
        let want_dir = format!("{base}/want");
        save_win(&want_dir, &mcfg, &want.valid());
        let (got, want) = (WinStrategy::load(&out), WinStrategy::load(&want_dir));
        let xs: Vec<_> = samples.iter().map(|s| s.0).collect();
        assert_eq!(got.net.logits(&xs), want.net.logits(&xs));
        let log = std::fs::read_to_string(format!("{out}/fit.csv")).unwrap();
        let rows: Vec<&str> = log.lines().collect();
        assert_eq!(rows[0], FIT_HEADER);
        assert_eq!(rows.len(), 3);
        // The legacy fit saves its last epoch, and no heads.
        assert_eq!(rows[2].split(',').nth(10), Some("1"));
        assert!(!Path::new(&format!("{out}/aux.mpk")).exists());
        assert!(!WinModelConfig::load(format!("{out}/config.json")).unwrap().has_aux());
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn a_thermometer_arm_with_heads_fits_and_saves_a_playable_net() {
        let base = tmp("smoke");
        let (data, init) = tiny(&base);
        let out = format!("{base}/arm");
        let s = fit(&ArmCfg {
            thermo: true,
            aux: true,
            // The live weights, so two steps visibly move the new columns
            // (averaging has its own tests).
            ema: 0.0,
            ..arm(&data, out.clone(), Some(init))
        })
        .unwrap();
        assert!((1..=2).contains(&s.best_epoch) && s.epochs == 2);
        let log = std::fs::read_to_string(format!("{out}/fit.csv")).unwrap();
        let rows: Vec<Vec<&str>> = log.lines().skip(1).map(|l| l.split(',').collect()).collect();
        assert_eq!(rows.len(), 2);
        for r in &rows {
            assert_eq!(r.len(), 12);
            assert!(
                r[4..10].iter().all(|v| !v.is_empty()),
                "aux and diff columns filled: {r:?}"
            );
        }
        let bot = WinStrategy::load(&out);
        assert!(bot.net.is_thermo());
        // The heads are saved beside the net.
        assert!(Path::new(&format!("{out}/aux.mpk")).exists());
        assert!(WinModelConfig::load(format!("{out}/config.json")).unwrap().has_aux());
        // The widened bit columns trained: some weight of the 22 bit inputs
        // of each half (inputs 4..26 and 30..52 of the row encoder) moved
        // off zero.
        let (_, m) = load_win_ad(&out);
        let w = m.row1.weight.val().into_data().to_vec::<f32>().unwrap();
        let width = m.row1.weight.shape().dims[1];
        let bits = |from: usize, to: usize| w[from * width..to * width].iter().any(|&v| v != 0.0);
        assert!(bits(4, 26) && bits(30, 52));
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn a_from_scratch_arm_fits() {
        let base = tmp("scratch");
        let (data, _) = tiny(&base);
        let out = format!("{base}/arm");
        fit(&ArmCfg {
            thermo: true,
            aux: true,
            epochs: 1,
            ..arm(&data, out.clone(), None)
        })
        .unwrap();
        assert!(WinStrategy::load(&out).net.is_thermo());
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn held_out_games_without_decisions_stop_on_the_win_excess() {
        let base = tmp("no-decisions");
        let (data, init) = tiny(&base);
        let mut ds = load(&data).unwrap();
        ds.heldout.decisions.clear();
        let bare = format!("{base}/bare");
        save(&bare, &ds).unwrap();
        let out = format!("{base}/arm");
        fit(&arm(&bare, out.clone(), Some(init))).unwrap();
        let log = std::fs::read_to_string(format!("{out}/fit.csv")).unwrap();
        for row in log.lines().skip(1) {
            let r: Vec<&str> = row.split(',').collect();
            assert!(r[7].is_empty() && r[8].is_empty() && r[9].is_empty(), "{row}");
        }
        assert!(Path::new(&format!("{out}/model.mpk")).exists());
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn validate_refuses_bad_settings() {
        let base = tmp("validate");
        let (data, init) = tiny(&base);
        let ok = arm(&data, format!("{base}/out"), Some(init.clone()));
        assert_eq!(ok.validate(), Ok(()));
        let err = |c: ArmCfg| c.validate().unwrap_err();
        let base_cfg = || arm(&data, ok.out.clone(), Some(init.clone()));
        assert!(err(ArmCfg {
            epochs: 0,
            ..base_cfg()
        })
        .contains("--epochs"));
        assert!(err(ArmCfg { ema: 1.0, ..base_cfg() }).contains("--ema"));
        assert!(err(ArmCfg { lr: 0.0, ..base_cfg() }).contains("--lr"));
        assert!(err(ArmCfg {
            aux_weights: vec![1.0],
            ..base_cfg()
        })
        .contains("--aux-weights"));
        assert!(err(ArmCfg {
            legacy: true,
            aux: true,
            ..base_cfg()
        })
        .contains("--legacy-fit"));
        assert!(err(ArmCfg {
            data: format!("{base}/nope"),
            ..base_cfg()
        })
        .contains("--data"));
        assert!(err(ArmCfg {
            init: Some(format!("{base}/nope")),
            ..base_cfg()
        })
        .contains("--init"));
        let thermo_init = format!("{base}/thermo");
        let device = Default::default();
        let mcfg = WinModelConfig::new().with_width(4).with_thermo(Some(true));
        save_win(&thermo_init, &mcfg, &mcfg.init::<MyBackend>(&device));
        assert!(err(ArmCfg {
            init: Some(thermo_init),
            ..base_cfg()
        })
        .contains("add --thermo"));
        // A dataset without held-out games is refused when the fit loads it.
        let mut ds = load(&data).unwrap();
        ds.heldout = Split::default();
        let none = format!("{base}/none");
        save(&none, &ds).unwrap();
        assert!(fit(&arm(&none, format!("{base}/out2"), Some(init.clone())))
            .unwrap_err()
            .contains("no held-out"));
        std::fs::create_dir_all(&ok.out).unwrap();
        assert!(err(base_cfg()).contains("already exists"));
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn manual_heads_match_the_burn_heads() {
        let device = Default::default();
        let (a, b) = boards();
        for thermo in [false, true] {
            let m = random_arm(35, thermo, true);
            let net = ManualWinNet::from_model(&m.net).with_aux(m.aux.as_ref().unwrap());
            let (x, s, d) = encode(&[(&a, &b)], thermo);
            let o = m.forward(flat_tensor(&x, d, &device), flat_tensor(&s, d, &device));
            let (marks, strikes, diff) = (values(o.marks), values(o.strikes), values(o.diff));
            let got: AuxPred = net.aux_states(&a, &b).unwrap();
            for i in 0..8 {
                assert!(
                    (got.marks[i] - 12.0 * marks[i]).abs() < 1e-3,
                    "thermo {thermo} mark {i}"
                );
            }
            for i in 0..2 {
                assert!((got.strikes[i] - 4.0 * strikes[i]).abs() < 1e-3);
            }
            assert!((got.diff - 100.0 * diff[0]).abs() < 1e-2);
            assert!(ManualWinNet::from_model(&m.net).aux_states(&a, &b).is_none());
        }
    }

    #[test]
    fn a_fitted_arm_with_heads_loads_them() {
        let base = tmp("heads");
        let (data, init) = tiny(&base);
        let (with, without) = (format!("{base}/with"), format!("{base}/without"));
        fit(&ArmCfg {
            aux: true,
            epochs: 1,
            ..arm(&data, with.clone(), Some(init.clone()))
        })
        .unwrap();
        fit(&ArmCfg {
            epochs: 1,
            ..arm(&data, without.clone(), Some(init))
        })
        .unwrap();
        let (a, b) = boards();
        let p = WinStrategy::load(&with).net.aux_states(&a, &b).unwrap();
        assert!(p.marks.iter().chain(&p.strikes).all(|v| v.is_finite()) && p.diff.is_finite());
        assert!(WinStrategy::load(&without).net.aux_states(&a, &b).is_none());
        std::fs::remove_dir_all(&base).unwrap();
    }

    fn with_random_tempo(mut arm: ArmModel<MyBackend>, seed: u64) -> ArmModel<MyBackend> {
        let _guard = SEEDING.lock().unwrap();
        let device = Default::default();
        <MyBackend as Backend>::seed(&device, seed);
        arm.tempo = Some(burn::nn::LinearConfig::new(8, 1).with_bias(true).init(&device));
        arm
    }

    fn some_samples() -> Vec<Sample> {
        let (a, b) = boards();
        [(a, b), (b, a), (a, a), (State::default(), b)]
            .into_iter()
            .map(|(me, opp)| Sample {
                me,
                opp,
                win: 0.5,
                win_var: 0.25,
                marks: [0.0; 8],
                strikes: [0.0; 2],
                diff: 0.0,
            })
            .collect()
    }

    #[test]
    fn a_zero_tempo_head_leaves_the_arm_forward_unchanged() {
        for aux in [false, true] {
            let plain = random_arm(70, true, aux);
            let tempo = ArmModel {
                tempo: Some(zero_tempo(8, &Default::default())),
                ..plain.clone()
            };
            let s = some_samples();
            let refs: Vec<&Sample> = s.iter().collect();
            assert_eq!(predict(&plain, &refs).p, predict(&tempo, &refs).p, "aux {aux}");
        }
    }

    #[test]
    fn arm_forward_matches_the_manual_net_with_a_tempo_head() {
        for aux in [false, true] {
            let arm = with_random_tempo(random_arm(71, true, aux), 72);
            let net = ManualWinNet::from_model(&arm.net).with_tempo(arm.tempo.as_ref().unwrap());
            let s = some_samples();
            let refs: Vec<&Sample> = s.iter().collect();
            for (x, p) in s.iter().zip(predict(&arm, &refs).p) {
                let q = sigmoid(net.logit_states(&x.me, &x.opp));
                assert!((p - q).abs() < 1e-5, "aux {aux}: burn {p} vs manual {q}");
            }
        }
    }

    #[test]
    fn the_weight_average_covers_the_tempo_head() {
        let arm = with_random_tempo(random_arm(73, false, false), 74);
        let n_plain = weights(&random_arm(73, false, false)).len();
        let ema = Ema::new(&arm, 0.5);
        assert_eq!(ema.values.len(), n_plain + 2, "tempo weight and bias");
        let zeroed = ema.apply(ArmModel {
            tempo: Some(zero_tempo(8, &Default::default())),
            ..arm.clone()
        });
        assert_eq!(
            weights(&zeroed),
            weights(&arm),
            "apply restores the averaged tempo head"
        );
    }

    #[test]
    fn validate_refuses_a_tempo_init_without_tempo() {
        let base = tmp("tempo-init");
        let (data, init) = tiny(&base);
        // Make the init a tempo dir.
        let mut cfg = WinModelConfig::load(format!("{init}/config.json")).unwrap();
        cfg.tempo = Some(true);
        cfg.save(format!("{init}/config.json")).unwrap();
        save_tempo(&init, &zero_tempo(4, &Default::default()));
        let err = arm(&data, format!("{base}/out"), Some(init.clone()))
            .validate()
            .unwrap_err();
        assert!(err.contains("--tempo"), "{err}");
        assert!(ArmCfg {
            tempo: true,
            ..arm(&data, format!("{base}/out"), Some(init))
        }
        .validate()
        .is_ok());
        assert!(ArmCfg {
            tempo: true,
            legacy: true,
            ..arm(&data, format!("{base}/out2"), None)
        }
        .validate()
        .unwrap_err()
        .contains("--legacy-fit"));
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn a_tempo_arm_saves_its_head_and_plays() {
        let base = tmp("tempo-fit");
        let (data, init) = tiny(&base);
        for (out, init) in [(format!("{base}/scratch"), None), (format!("{base}/warm"), Some(init))] {
            fit(&ArmCfg {
                tempo: true,
                ..arm(&data, out.clone(), init)
            })
            .unwrap();
            let cfg = WinModelConfig::load(format!("{out}/config.json")).unwrap();
            assert!(cfg.is_tempo());
            assert!(Path::new(&format!("{out}/tempo.mpk")).exists());
            let bot = WinStrategy::load(&out);
            assert!(bot.net.has_tempo());
        }
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// Mirror pairs, each side "just moved" with label 0.45: the player to
    /// move next has the edge. Only a tempo net can fit both at 0.45.
    fn mirror_data(dir: &str) {
        let mut rng = SmallRng::seed_from_u64(9);
        let mut board = || {
            let mut s = State::default();
            for _ in 0..rng.gen_range(0..12) {
                let (row, number) = (rng.gen_range(0..4), rng.gen_range(2..=12));
                if s.can_mark(row, number) {
                    s.apply_mark(Mark { row, number });
                }
            }
            s
        };
        let sample = |me: State, opp: State| Sample {
            me,
            opp,
            win: 0.45,
            win_var: 0.25,
            marks: [0.0; 8],
            strikes: [0.0; 2],
            diff: 0.0,
        };
        let mut split = || {
            let mut turns = Vec::new();
            for _ in 0..200 {
                let (a, b) = (board(), board());
                turns.push(sample(a, b));
                turns.push(sample(b, a));
            }
            Split {
                turns,
                decisions: Vec::new(),
            }
        };
        let (train, heldout) = (split(), split());
        let meta = win_data::DataMeta {
            player: "synthetic".into(),
            seed: 0,
            games: 2,
            rollouts: 64,
            cand_rollouts: 64,
            holdout: 0.5,
            keep: None,
            train_turns: 0,
            train_decisions: 0,
            heldout_turns: 0,
            heldout_decisions: 0,
            bytes: 0,
        };
        save(dir, &Dataset { meta, train, heldout }).unwrap();
    }

    #[test]
    fn a_tempo_arm_learns_the_value_of_moving_next() {
        let base = tmp("tempo-learns");
        let data = format!("{base}/data");
        mirror_data(&data);
        let ds = load(&data).unwrap();
        let mean_p = |out: &str| {
            let bot = WinStrategy::load(out);
            let leaves: Vec<(&State, &[State])> = ds
                .heldout
                .turns
                .iter()
                .map(|s| (&s.me, std::slice::from_ref(&s.opp)))
                .collect();
            let p = bot.win_prob_multi(&leaves);
            p.iter().sum::<f32>() / p.len() as f32
        };
        let run = |out: String, tempo: bool| {
            fit(&ArmCfg {
                tempo,
                epochs: 40,
                lr: 1e-2,
                ema: 0.0,
                width: 8,
                ..arm(&data, out.clone(), None)
            })
            .unwrap();
            mean_p(&out)
        };
        let plain = run(format!("{base}/plain"), false);
        let tempo = run(format!("{base}/tempo"), true);
        assert!((plain - 0.5).abs() < 1e-3, "antisymmetric over mirror pairs: {plain}");
        assert!(tempo < 0.47, "the tempo net learns the mover's edge: {tempo}");
        std::fs::remove_dir_all(&base).unwrap();
    }
}
