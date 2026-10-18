//! Win net: a 2-player P(win) network, antisymmetric in the players and
//! invariant to the 8 colour relabellings by construction, trained by TD(λ)
//! self-play on win/loss outcomes (TD-Gammon style).
//!
//! Design: docs/superpowers/specs/2026-10-15-win-net-design.md

use super::pair::board_features;
use super::MyBackend;
use crate::state::State;
use crate::strategy::search::WinProb;
use crate::strategy::Bot;
use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    record::CompactRecorder,
};
use std::sync::Arc;

/// Row-token width: a row's 4 features for me, then the same 4 for the opponent.
pub const TOKEN: usize = 8;
/// Global inputs: 4 per player (strikes, blanks, aggregate weight, lockable
/// rows), then the current difference / 100.
pub const GLOBALS: usize = 9;
/// Full input: 4 row tokens (red, yellow, green, blue) + globals.
pub const WIN_INPUTS: usize = 4 * TOKEN + GLOBALS;

/// Input for "me" against "opp": today's two 20-feature board blocks
/// regrouped by row. The row features are direction-free, so every colour
/// relabelling only permutes the 4 tokens.
pub fn win_input(me: &State, opp: &State) -> [f32; WIN_INPUTS] {
    let (a, b) = (board_features(me), board_features(opp));
    let mut x = [0.0f32; WIN_INPUTS];
    for r in 0..4 {
        for k in 0..4 {
            x[TOKEN * r + k] = a[4 * k + r];
            x[TOKEN * r + 4 + k] = b[4 * k + r];
        }
    }
    x[32..36].copy_from_slice(&a[16..20]);
    x[36..40].copy_from_slice(&b[16..20]);
    let cdiff = (me.count_points() - opp.count_points()) as f32;
    x[40] = (cdiff / 100.0).clamp(-1.0, 1.0);
    x
}

/// The same position from the opponent's side: swap the halves of every
/// token and of the globals, negate the difference.
pub fn swap_input(x: &[f32; WIN_INPUTS]) -> [f32; WIN_INPUTS] {
    let mut s = *x;
    for r in 0..4 {
        for k in 0..4 {
            s[TOKEN * r + k] = x[TOKEN * r + 4 + k];
            s[TOKEN * r + 4 + k] = x[TOKEN * r + k];
        }
    }
    for k in 0..4 {
        s[32 + k] = x[36 + k];
        s[36 + k] = x[32 + k];
    }
    s[40] = -x[40];
    s
}

/// Thermometer row half: a player's 4 row scalars, 10 progress bits
/// (cells passed ≥ j), 12 mark bits (row total ≥ j).
pub const T_HALF: usize = 26;
/// Thermometer row token: my half, then the opponent's.
pub const T_TOKEN: usize = 2 * T_HALF;
/// Thermometer globals: today's 9, then 3 strike bits per player (≥ 1, 2, 3).
pub const T_GLOBALS: usize = GLOBALS + 6;
/// Full thermometer input.
pub const WIN_INPUTS_T: usize = 4 * T_TOKEN + T_GLOBALS;

/// One player's row bits: `progress` is the row's `row_progress` (cells
/// passed / 10; a locked row is 1), `total` its mark count.
fn row_bits(out: &mut [f32], progress: f32, total: u8) {
    let cells = (progress * 10.0).round() as usize;
    for j in 1..=10 {
        out[j - 1] = (cells >= j) as u8 as f32;
    }
    for j in 1..=12 {
        out[9 + j] = (total as usize >= j) as u8 as f32;
    }
}

/// [`win_input`] with each row's progress and mark count also as
/// thermometer bits, and each player's strikes as 3 bits
/// (docs/superpowers/specs/2026-10-19-win-thermo-aux-design.md). The bits
/// are direction-free like the scalars, so colour relabellings still only
/// permute the tokens.
pub fn thermo_input(me: &State, opp: &State) -> [f32; WIN_INPUTS_T] {
    let plain = win_input(me, opp);
    let totals = [me.row_totals(), opp.row_totals()];
    let mut x = [0.0f32; WIN_INPUTS_T];
    for r in 0..4 {
        for side in 0..2 {
            let half = &mut x[T_TOKEN * r + T_HALF * side..][..T_HALF];
            half[..4].copy_from_slice(&plain[TOKEN * r + 4 * side..][..4]);
            row_bits(&mut half[4..], plain[TOKEN * r + 4 * side], totals[side][r]);
        }
    }
    let g = 4 * T_TOKEN;
    x[g..g + GLOBALS].copy_from_slice(&plain[4 * TOKEN..]);
    for (side, strikes) in [me.strikes, opp.strikes].into_iter().enumerate() {
        for j in 1..=3 {
            x[g + GLOBALS + 3 * side + j - 1] = (strikes as usize >= j) as u8 as f32;
        }
    }
    x
}

/// [`swap_input`] for [`thermo_input`]: swap the halves of every token, the
/// per-player globals and the strike bits; negate the difference.
pub fn swap_thermo(x: &[f32; WIN_INPUTS_T]) -> [f32; WIN_INPUTS_T] {
    let mut s = *x;
    for r in 0..4 {
        let t = T_TOKEN * r;
        s[t..t + T_HALF].copy_from_slice(&x[t + T_HALF..t + T_TOKEN]);
        s[t + T_HALF..t + T_TOKEN].copy_from_slice(&x[t..t + T_HALF]);
    }
    let g = 4 * T_TOKEN;
    s[g..g + 4].copy_from_slice(&x[g + 4..g + 8]);
    s[g + 4..g + 8].copy_from_slice(&x[g..g + 4]);
    s[g + 8] = -x[g + 8];
    s[g + 9..g + 12].copy_from_slice(&x[g + 12..g + 15]);
    s[g + 12..g + 15].copy_from_slice(&x[g + 9..g + 12]);
    s
}

// ---- Model ----

/// g(me, opp): shared row encoder φ (8 → W → W), sums within the colour
/// pairs (red + yellow, green + blue), shared pair encoder ψ (W → W) summed
/// over both pairs, then a head over [pooled rows | 9 globals]
/// (W + 9 → 2W → W → 1). ReLU throughout, linear output.
#[derive(Module, Debug)]
pub struct WinModel<B: Backend> {
    pub row1: Linear<B>,
    pub row2: Linear<B>,
    pub pair: Linear<B>,
    pub head1: Linear<B>,
    pub head2: Linear<B>,
    pub out: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct WinModelConfig {
    /// Encoder width W (the head is 2W → W).
    #[config(default = 32)]
    pub width: usize,
}

impl WinModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WinModel<B> {
        let w = self.width;
        let lin = |d_in: usize, d_out: usize| LinearConfig::new(d_in, d_out).with_bias(true).init(device);
        WinModel {
            row1: lin(TOKEN, w),
            row2: lin(w, w),
            pair: lin(w, w),
            head1: lin(w + GLOBALS, 2 * w),
            head2: lin(2 * w, w),
            out: lin(w, 1),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> WinModel<B> {
    /// g per input row: `[n, WIN_INPUTS]` → `[n, 1]`.
    pub fn g(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = x.dims()[0];
        let tokens = x.clone().narrow(1, 0, 4 * TOKEN).reshape([n, 4, TOKEN]);
        let e = self
            .activation
            .forward(self.row2.forward(self.activation.forward(self.row1.forward(tokens))));
        let w = e.dims()[2];
        let asc = e.clone().narrow(1, 0, 2).sum_dim(1);
        let desc = e.narrow(1, 2, 2).sum_dim(1);
        let rows = self.activation.forward(self.pair.forward(asc)) + self.activation.forward(self.pair.forward(desc));
        let h = Tensor::cat(vec![rows.reshape([n, w]), x.narrow(1, 4 * TOKEN, GLOBALS)], 1);
        let h = self.activation.forward(self.head1.forward(h));
        let h = self.activation.forward(self.head2.forward(h));
        self.out.forward(h)
    }

    /// Antisymmetric logit z = g(x) − g(swapped), `[n, 1]`; both
    /// orientations run as one batch.
    pub fn forward(&self, x: Tensor<B, 2>, swapped: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = x.dims()[0];
        let both = self.g(Tensor::cat(vec![x, swapped], 0));
        both.clone().narrow(0, 0, n) - both.narrow(0, n, n)
    }
}

/// Rows of inputs as a `[n, WIN_INPUTS]` tensor.
pub fn input_tensor<B: Backend>(rows: &[[f32; WIN_INPUTS]], device: &B::Device) -> Tensor<B, 2> {
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([rows.len(), WIN_INPUTS])
}

/// Logits through the burn model (training-side reference; bots use the
/// manual inference path).
pub fn model_logits(model: &WinModel<MyBackend>, rows: &[[f32; WIN_INPUTS]]) -> Vec<f32> {
    let device = Default::default();
    let swapped: Vec<_> = rows.iter().map(swap_input).collect();
    let z = model.forward(input_tensor(rows, &device), input_tensor(&swapped, &device));
    z.into_data().to_vec::<f32>().unwrap()
}

// ---- Manual inference path ----

/// One dense layer, weights input-major as burn stores them:
/// `w[i * d_out..][..d_out]` are input i's weights to every output.
struct Dense {
    w: Vec<f32>,
    b: Vec<f32>,
    d_out: usize,
}

impl Dense {
    fn from_linear(layer: &Linear<MyBackend>) -> Self {
        let d_out = layer.weight.shape().dims[1];
        Dense {
            w: layer.weight.val().into_data().to_vec::<f32>().unwrap(),
            b: layer.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().unwrap(),
            d_out,
        }
    }

    /// `out = b + x · w`, then ReLU when `relu`. Zero inputs are skipped.
    fn apply(&self, x: &[f32], out: &mut Vec<f32>, relu: bool) {
        out.clear();
        out.extend_from_slice(&self.b);
        for (i, &xi) in x.iter().enumerate() {
            if xi != 0.0 {
                for (o, &wv) in out.iter_mut().zip(&self.w[i * self.d_out..(i + 1) * self.d_out]) {
                    *o += xi * wv;
                }
            }
        }
        if relu {
            for o in out.iter_mut() {
                *o = o.max(0.0);
            }
        }
    }
}

/// Plain-array copy of a [`WinModel`] for inference (burn's per-call
/// overhead dominates at these sizes, as for `ManualPairNet`). Numerics
/// match burn to a tolerance, not bit for bit.
pub struct ManualWinNet {
    width: usize,
    row1: Dense,
    row2: Dense,
    pair: Dense,
    head1: Dense,
    head2: Dense,
    out: Dense,
}

impl ManualWinNet {
    pub fn from_model(model: &WinModel<MyBackend>) -> Self {
        ManualWinNet {
            width: model.row1.weight.shape().dims[1],
            row1: Dense::from_linear(&model.row1),
            row2: Dense::from_linear(&model.row2),
            pair: Dense::from_linear(&model.pair),
            head1: Dense::from_linear(&model.head1),
            head2: Dense::from_linear(&model.head2),
            out: Dense::from_linear(&model.out),
        }
    }

    fn g(&self, x: &[f32; WIN_INPUTS]) -> f32 {
        let w = self.width;
        let (mut h, mut e) = (Vec::with_capacity(2 * w), Vec::with_capacity(w));
        let mut pooled = [vec![0.0f32; w], vec![0.0f32; w]];
        for r in 0..4 {
            self.row1.apply(&x[TOKEN * r..TOKEN * (r + 1)], &mut h, true);
            self.row2.apply(&h, &mut e, true);
            for (p, v) in pooled[r / 2].iter_mut().zip(&e) {
                *p += v;
            }
        }
        let mut cat = vec![0.0f32; w];
        for p in &pooled {
            self.pair.apply(p, &mut e, true);
            for (c, v) in cat.iter_mut().zip(&e) {
                *c += v;
            }
        }
        cat.extend_from_slice(&x[4 * TOKEN..]);
        self.head1.apply(&cat, &mut h, true);
        self.head2.apply(&h, &mut e, true);
        self.out.apply(&e, &mut h, false);
        h[0]
    }

    /// Antisymmetric logit z = g(x) − g(swap x).
    pub fn logit(&self, x: &[f32; WIN_INPUTS]) -> f32 {
        self.g(x) - self.g(&swap_input(x))
    }

    pub fn logits(&self, xs: &[[f32; WIN_INPUTS]]) -> Vec<f32> {
        xs.iter().map(|x| self.logit(x)).collect()
    }
}

pub fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Φ⁻¹(sigmoid(z)): the probit z-score of the logit's probability, the
/// value unit the search's gate and blend assume. Computed from the smaller
/// tail in f64 so it stays finite and strictly monotone for large |z|.
pub fn probit_of_logit(z: f32) -> f32 {
    let tail = 1.0 / (1.0 + (z.abs() as f64).exp()); // sigmoid(-|z|) ≤ ½
    let x = probit_lower(tail.max(f64::MIN_POSITIVE)); // ≤ 0
    (if z >= 0.0 { -x } else { x }) as f32
}

/// Φ⁻¹(p) for 0 < p ≤ ½ (Acklam's rational approximation, relative error
/// < 1.2e-9).
fn probit_lower(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

// ---- Strategy ----

#[derive(Clone)]
pub struct WinStrategy {
    pub net: Arc<ManualWinNet>,
}

impl std::fmt::Debug for WinStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "WinStrategy")
    }
}

impl WinStrategy {
    pub fn from_model(model: &WinModel<MyBackend>) -> Self {
        WinStrategy {
            net: Arc::new(ManualWinNet::from_model(model)),
        }
    }

    /// Load `dir/config.json` (the width) and `dir/model.mpk`.
    pub fn load(dir: &str) -> Self {
        let device = Default::default();
        let cfg = WinModelConfig::load(format!("{dir}/config.json")).expect("Failed to load win model config");
        let model = cfg
            .init::<MyBackend>(&device)
            .load_file(format!("{dir}/model"), &CompactRecorder::new(), &device)
            .expect("Failed to load win model");
        Self::from_model(&model)
    }

    /// The single opponent; an empty board when there is none (solo is
    /// unsupported, as for the pair bot).
    fn opponent(opps: &[State]) -> State {
        assert!(
            opps.len() <= 1,
            "the win net is 2-player only (got {} opponents)",
            opps.len()
        );
        opps.first().copied().unwrap_or_default()
    }
}

impl Bot for WinStrategy {
    fn evaluate(&self, our_state: &State, opp_states: &[State]) -> f32 {
        self.evaluate_batch(&[*our_state], opp_states)[0]
    }

    fn evaluate_batch(&self, candidates: &[State], opp_states: &[State]) -> Vec<f32> {
        self.evaluate_batch_multi(&[(candidates, opp_states)]).pop().unwrap()
    }

    /// Probit-unit values (see [`probit_of_logit`]); move order equals the
    /// order of P(win).
    fn evaluate_batch_multi(&self, groups: &[(&[State], &[State])]) -> Vec<Vec<f32>> {
        groups
            .iter()
            .map(|(candidates, opps)| {
                let opp = Self::opponent(opps);
                candidates
                    .iter()
                    .map(|c| probit_of_logit(self.net.logit(&win_input(c, &opp))))
                    .collect()
            })
            .collect()
    }
}

impl WinProb for WinStrategy {
    fn win_prob_multi(&self, groups: &[(&State, &[State])]) -> Vec<f32> {
        groups
            .iter()
            .map(|(our, opps)| sigmoid(self.net.logit(&win_input(our, &Self::opponent(opps)))))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Mark;

    /// Two different mid-game boards.
    pub(super) fn boards() -> (State, State) {
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

    #[test]
    fn input_regroups_board_features_into_row_tokens_and_globals() {
        let (a, b) = boards();
        let (fa, fb) = (board_features(&a), board_features(&b));
        let x = win_input(&a, &b);
        for r in 0..4 {
            for k in 0..4 {
                assert_eq!(x[TOKEN * r + k], fa[4 * k + r], "me row {r} feature {k}");
                assert_eq!(x[TOKEN * r + 4 + k], fb[4 * k + r], "opp row {r} feature {k}");
            }
        }
        assert_eq!(&x[32..36], &fa[16..20]);
        assert_eq!(&x[36..40], &fb[16..20]);
        let cdiff = (a.count_points() - b.count_points()) as f32;
        assert_eq!(x[40], (cdiff / 100.0).clamp(-1.0, 1.0));
    }

    #[test]
    fn swapped_input_equals_input_with_players_exchanged() {
        let (a, b) = boards();
        assert_eq!(swap_input(&win_input(&a, &b)), win_input(&b, &a));
        assert_eq!(swap_input(&swap_input(&win_input(&a, &b))), win_input(&a, &b));
    }

    #[test]
    fn thermometer_input_keeps_the_scalars_and_adds_level_bits() {
        let (a, b) = boards();
        let (p, x) = (win_input(&a, &b), thermo_input(&a, &b));
        for r in 0..4 {
            for side in 0..2 {
                let half = &x[T_TOKEN * r + T_HALF * side..][..T_HALF];
                assert_eq!(
                    &half[..4],
                    &p[TOKEN * r + 4 * side..][..4],
                    "row {r} side {side} scalars"
                );
            }
        }
        assert_eq!(&x[4 * T_TOKEN..4 * T_TOKEN + GLOBALS], &p[4 * TOKEN..]);
        // Me, red: marks 2 and 5, next free 6 → 4 cells passed, 2 marks.
        let red = &x[4..T_HALF];
        assert_eq!(&red[..10], &[1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]);
        assert_eq!(&red[10..], &[1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]);
        // Me, green (descending): mark 12, next free 11 → 1 cell, 1 mark.
        let green = &x[2 * T_TOKEN + 4..2 * T_TOKEN + T_HALF];
        assert_eq!(green[..10].iter().sum::<f32>(), 1.0);
        assert_eq!(green[10..].iter().sum::<f32>(), 1.0);
        // Opp, blue (descending): marks 9 and 7, next free 6 → 6 cells, 2 marks.
        let blue = &x[3 * T_TOKEN + T_HALF + 4..4 * T_TOKEN];
        assert_eq!(blue[..10].iter().sum::<f32>(), 6.0);
        assert_eq!(blue[10..].iter().sum::<f32>(), 2.0);
        // Strike bits: me one strike, opp none.
        assert_eq!(&x[4 * T_TOKEN + GLOBALS..], &[1., 0., 0., 0., 0., 0.]);
    }

    #[test]
    fn a_locked_row_sets_every_progress_bit() {
        let mut a = State::default();
        for n in [2, 3, 4, 5, 6, 12] {
            a.apply_mark(Mark { row: 0, number: n });
        }
        for _ in 0..3 {
            a.apply_strike();
        }
        let x = thermo_input(&a, &State::default());
        assert_eq!(&x[4..14], &[1.0; 10]);
        assert_eq!(x[14..T_HALF].iter().sum::<f32>(), 7.0, "6 marks and the lock bonus");
        assert_eq!(&x[4 * T_TOKEN + GLOBALS..], &[1., 1., 1., 0., 0., 0.]);
        // Fresh boards: no bits at all.
        let f = thermo_input(&State::default(), &State::default());
        for r in 0..4 {
            for side in 0..2 {
                assert!(f[T_TOKEN * r + T_HALF * side + 4..][..22].iter().all(|&v| v == 0.0));
            }
        }
    }

    #[test]
    fn swapped_thermometer_input_equals_input_with_players_exchanged() {
        let (a, b) = boards();
        assert_eq!(swap_thermo(&thermo_input(&a, &b)), thermo_input(&b, &a));
        assert_eq!(swap_thermo(&swap_thermo(&thermo_input(&a, &b))), thermo_input(&a, &b));
    }

    use crate::dqn::MyBackend;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    fn random_model(seed: u64, width: usize) -> WinModel<MyBackend> {
        let device = Default::default();
        <MyBackend as burn::tensor::backend::Backend>::seed(&device, seed);
        WinModelConfig::new().with_width(width).init::<MyBackend>(&device)
    }

    /// Random inputs in the features' ranges (cdiff in [-1, 1]).
    pub(super) fn random_inputs(n: usize, seed: u64) -> Vec<[f32; WIN_INPUTS]> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let mut x = [0.0f32; WIN_INPUTS];
                for v in x.iter_mut() {
                    *v = rng.gen::<f32>();
                }
                x[40] = rng.gen_range(-1.0..1.0);
                x
            })
            .collect()
    }

    /// Permute the row tokens: red↔yellow, green↔blue, ascending↔descending pairs.
    fn relabel(x: &[f32; WIN_INPUTS], ry: bool, gb: bool, pairs: bool) -> [f32; WIN_INPUTS] {
        let mut order = [0usize, 1, 2, 3];
        if ry {
            order.swap(0, 1);
        }
        if gb {
            order.swap(2, 3);
        }
        if pairs {
            order = [order[2], order[3], order[0], order[1]];
        }
        let mut y = *x;
        for (slot, &src) in order.iter().enumerate() {
            y[TOKEN * slot..TOKEN * (slot + 1)].copy_from_slice(&x[TOKEN * src..TOKEN * (src + 1)]);
        }
        y
    }

    #[test]
    fn logit_is_antisymmetric_for_a_random_net() {
        let model = random_model(7, 32);
        let xs = random_inputs(64, 1);
        let swapped: Vec<_> = xs.iter().map(swap_input).collect();
        for (z, zs) in model_logits(&model, &xs)
            .into_iter()
            .zip(model_logits(&model, &swapped))
        {
            assert!((z + zs).abs() < 1e-5, "z {z} vs swapped {zs}");
        }
    }

    #[test]
    fn logit_is_invariant_to_all_8_colour_relabellings() {
        let model = random_model(8, 16);
        let (a, b) = boards();
        let xs = [win_input(&a, &b), win_input(&b, &a)];
        let base = model_logits(&model, &xs);
        for code in 0..8u8 {
            let ys: Vec<_> = xs
                .iter()
                .map(|x| relabel(x, code & 1 != 0, code & 2 != 0, code & 4 != 0))
                .collect();
            for (z, zr) in base.iter().zip(model_logits(&model, &ys)) {
                assert!((z - zr).abs() < 1e-5, "relabelling {code}: {z} vs {zr}");
            }
        }
    }

    #[test]
    fn identical_boards_give_an_even_logit() {
        let model = random_model(9, 32);
        let (a, _) = boards();
        let z = model_logits(&model, &[win_input(&a, &a)])[0];
        assert!(z.abs() < 1e-6, "z = {z}");
    }

    use crate::strategy::search::{phi, WinProb};
    use crate::strategy::Bot;

    #[test]
    fn manual_net_matches_the_burn_model() {
        for width in [16, 32] {
            let model = random_model(11, width);
            let net = ManualWinNet::from_model(&model);
            let xs = random_inputs(32, 2);
            for (a, b) in model_logits(&model, &xs).into_iter().zip(net.logits(&xs)) {
                assert!((a - b).abs() < 1e-4, "width {width}: burn {a} vs manual {b}");
            }
        }
    }

    #[test]
    fn probit_of_logit_matches_phi_inverse_of_sigmoid() {
        for i in -80..=80 {
            let z = i as f32 / 10.0;
            let v = probit_of_logit(z);
            assert!((phi(v) - sigmoid(z)).abs() < 1e-4, "z {z}: phi({v}) vs {}", sigmoid(z));
        }
        assert_eq!(probit_of_logit(0.0), 0.0);
    }

    #[test]
    fn probit_of_logit_is_finite_and_monotone_at_extremes() {
        let zs = [-60.0f32, -40.0, -25.0, -20.0, -10.0, 0.0, 10.0, 20.0, 25.0, 40.0, 60.0];
        let vs: Vec<f32> = zs.iter().map(|&z| probit_of_logit(z)).collect();
        assert!(vs.iter().all(|v| v.is_finite()), "{vs:?}");
        assert!(vs.windows(2).all(|w| w[0] < w[1]), "not strictly increasing: {vs:?}");
    }

    #[test]
    fn evaluate_is_the_probit_of_win_prob() {
        let bot = WinStrategy::from_model(&random_model(12, 32));
        let (a, b) = boards();
        let mut c = a;
        c.apply_mark(Mark { row: 1, number: 3 });
        let candidates = [a, b, c];
        let values = bot.evaluate_batch(&candidates, &[b]);
        let groups: Vec<(&State, &[State])> = candidates.iter().map(|s| (s, std::slice::from_ref(&b))).collect();
        let probs = bot.win_prob_multi(&groups);
        for (v, p) in values.iter().zip(&probs) {
            assert!((phi(*v) - p).abs() < 1e-4, "phi({v}) vs {p}");
        }
        // Same order by value as by probability.
        let mut by_v: Vec<usize> = (0..3).collect();
        by_v.sort_by(|&i, &j| values[i].partial_cmp(&values[j]).unwrap());
        let mut by_p: Vec<usize> = (0..3).collect();
        by_p.sort_by(|&i, &j| probs[i].partial_cmp(&probs[j]).unwrap());
        assert_eq!(by_v, by_p);
    }

    #[test]
    fn win_prob_of_identical_boards_is_one_half() {
        let bot = WinStrategy::from_model(&random_model(13, 32));
        let (a, _) = boards();
        let p = bot.win_prob_multi(&[(&a, std::slice::from_ref(&a))])[0];
        assert!((p - 0.5).abs() < 1e-6, "p = {p}");
    }

    #[test]
    fn no_opponent_falls_back_to_empty_board() {
        let bot = WinStrategy::from_model(&random_model(14, 32));
        let (a, _) = boards();
        let solo = bot.evaluate(&a, &[]);
        let vs_empty = bot.evaluate(&a, &[State::default()]);
        assert_eq!(solo, vs_empty);
    }

    #[test]
    #[should_panic(expected = "2-player only")]
    fn two_opponents_panic() {
        let bot = WinStrategy::from_model(&random_model(15, 32));
        let (a, b) = boards();
        bot.evaluate(&a, &[b, b]);
    }
}
