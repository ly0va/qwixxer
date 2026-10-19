//! Win net: a 2-player P(win) network, antisymmetric in the players and
//! invariant to the 8 colour relabellings by construction, trained by TD(λ)
//! self-play on win/loss outcomes (TD-Gammon style).
//!
//! Design: docs/superpowers/specs/2026-10-15-win-net-design.md

use super::pair::board_features;
use super::MyBackend;
use crate::state::State;
use crate::strategy::search::WinProb;
use crate::strategy::{Bot, Next};
use burn::{
    module::Param,
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    record::CompactRecorder,
    tensor::TensorData,
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
    /// Thermometer inputs ([`thermo_input`]). `None` in model dirs written
    /// before it existed: an `Option` because burn's config loader rejects
    /// a missing plain field.
    pub thermo: Option<bool>,
    /// The dir also holds trained [`AuxHeads`] in `aux.mpk`. `None` in
    /// older dirs (an `Option` for the same reason as `thermo`).
    pub aux: Option<bool>,
    /// The dir also holds a tempo head in `tempo.mpk` (the value of moving
    /// next; docs/superpowers/specs/2026-10-20-win-tempo-design.md). `None`
    /// in older dirs (an `Option` for the same reason as `thermo`).
    pub tempo: Option<bool>,
}

impl WinModelConfig {
    pub fn is_thermo(&self) -> bool {
        self.thermo == Some(true)
    }

    pub fn has_aux(&self) -> bool {
        self.aux == Some(true)
    }

    pub fn is_tempo(&self) -> bool {
        self.tempo == Some(true)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> WinModel<B> {
        let w = self.width;
        let (token, globals) = if self.is_thermo() {
            (T_TOKEN, T_GLOBALS)
        } else {
            (TOKEN, GLOBALS)
        };
        let lin = |d_in: usize, d_out: usize| LinearConfig::new(d_in, d_out).with_bias(true).init(device);
        WinModel {
            row1: lin(token, w),
            row2: lin(w, w),
            pair: lin(w, w),
            head1: lin(w + globals, 2 * w),
            head2: lin(2 * w, w),
            out: lin(w, 1),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> WinModel<B> {
    /// Whether the net takes thermometer inputs (its row encoder's width).
    pub fn is_thermo(&self) -> bool {
        self.row1.weight.shape().dims[0] == T_TOKEN
    }

    /// The trunk per input row: every row's encoding e_r `[n, 4, W]` and
    /// the head's last hidden layer h `[n, W]`.
    pub fn trunk(&self, x: Tensor<B, 2>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [n, d] = x.dims();
        let token = self.row1.weight.shape().dims[0];
        let tokens = x.clone().narrow(1, 0, 4 * token).reshape([n, 4, token]);
        let e = self
            .activation
            .forward(self.row2.forward(self.activation.forward(self.row1.forward(tokens))));
        let w = e.dims()[2];
        let asc = e.clone().narrow(1, 0, 2).sum_dim(1);
        let desc = e.clone().narrow(1, 2, 2).sum_dim(1);
        let rows = self.activation.forward(self.pair.forward(asc)) + self.activation.forward(self.pair.forward(desc));
        let h = Tensor::cat(vec![rows.reshape([n, w]), x.narrow(1, 4 * token, d - 4 * token)], 1);
        let h = self.activation.forward(self.head1.forward(h));
        (e, self.activation.forward(self.head2.forward(h)))
    }

    /// g per input row: `[n, inputs]` → `[n, 1]`.
    pub fn g(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.out.forward(self.trunk(x).1)
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

/// Rows of `d` inputs each, flat, as a `[rows, d]` tensor.
pub fn flat_tensor<B: Backend>(flat: &[f32], d: usize, device: &B::Device) -> Tensor<B, 2> {
    Tensor::<B, 1>::from_floats(flat, device).reshape([flat.len() / d, d])
}

/// [`model_logits`] for thermometer inputs.
pub fn model_logits_t(model: &WinModel<MyBackend>, rows: &[[f32; WIN_INPUTS_T]]) -> Vec<f32> {
    let device = Default::default();
    let x: Vec<f32> = rows.iter().flatten().copied().collect();
    let s: Vec<f32> = rows.iter().flat_map(swap_thermo).collect();
    let z = model.forward(
        flat_tensor(&x, WIN_INPUTS_T, &device),
        flat_tensor(&s, WIN_INPUTS_T, &device),
    );
    z.into_data().to_vec::<f32>().unwrap()
}

/// A thermometer net with the same function as the plain `model`: every
/// weight copied, the new input weights (the row bits in the row encoder's
/// first layer, the strike bits in the head's first layer) zero.
pub fn widen_to_thermo<B: Backend>(model: &WinModel<B>, device: &B::Device) -> WinModel<B> {
    assert!(!model.is_thermo(), "already a thermometer net");
    let w = model.row1.weight.shape().dims[1];
    let mut out = model.clone();
    // Weights are input-major: input i's weights are `[i * d_out..][..d_out]`.
    let old = model.row1.weight.val().into_data().to_vec::<f32>().unwrap();
    let mut new = vec![0.0f32; T_TOKEN * w];
    for side in 0..2 {
        for k in 0..4 {
            let (src, dst) = (4 * side + k, T_HALF * side + k);
            new[dst * w..(dst + 1) * w].copy_from_slice(&old[src * w..(src + 1) * w]);
        }
    }
    out.row1.weight = Param::from_tensor(Tensor::from_data(TensorData::new(new, [T_TOKEN, w]), device));
    let d_out = model.head1.weight.shape().dims[1];
    let old = model.head1.weight.val().into_data().to_vec::<f32>().unwrap();
    let mut new = vec![0.0f32; (w + T_GLOBALS) * d_out];
    new[..old.len()].copy_from_slice(&old);
    out.head1.weight = Param::from_tensor(Tensor::from_data(TensorData::new(new, [w + T_GLOBALS, d_out]), device));
    out
}

/// Heads on the trunk that play never uses: each row's final marks (one map
/// shared by the rows), final strikes, and the final score difference.
/// Trained by `win-arm --aux`, saved to `aux.mpk` beside the net.
#[derive(Module, Debug)]
pub struct AuxHeads<B: Backend> {
    /// [e_r | h] (2W) → that row's final total / 12.
    pub marks: Linear<B>,
    /// h (W) → final strikes / 4.
    pub strikes: Linear<B>,
    /// h (W) → a; the difference / 100 is a(x) − a(swap x).
    pub diff: Linear<B>,
}

impl<B: Backend> AuxHeads<B> {
    pub fn new(width: usize, device: &B::Device) -> Self {
        let lin = |d_in: usize| LinearConfig::new(d_in, 1).with_bias(true).init(device);
        AuxHeads {
            marks: lin(2 * width),
            strikes: lin(width),
            diff: lin(width),
        }
    }
}

/// Write `dir/aux.mpk`; the dir's `config.json` says `aux: true`.
pub fn save_aux(dir: &str, heads: &AuxHeads<MyBackend>) {
    heads
        .clone()
        .save_file(format!("{dir}/aux"), &CompactRecorder::new())
        .expect("Failed to save the auxiliary heads");
}

/// A tempo head (W → 1, with bias) that adds nothing: a net given it plays
/// exactly as the antisymmetric net it extends.
pub fn zero_tempo<B: Backend>(width: usize, device: &B::Device) -> Linear<B> {
    let mut head = LinearConfig::new(width, 1).with_bias(true).init(device);
    head.weight = Param::from_tensor(Tensor::zeros([width, 1], device));
    head.bias = Some(Param::from_tensor(Tensor::zeros([1], device)));
    head
}

/// Write `dir/tempo.mpk`; the dir's `config.json` says `tempo: true`.
pub fn save_tempo(dir: &str, head: &Linear<MyBackend>) {
    head.clone()
        .save_file(format!("{dir}/tempo"), &CompactRecorder::new())
        .expect("Failed to save the tempo head");
}

/// Read `dir/tempo.mpk` for a net of `width`.
pub fn load_tempo<B: Backend>(dir: &str, width: usize, device: &B::Device) -> Linear<B> {
    LinearConfig::new(width, 1)
        .with_bias(true)
        .init(device)
        .load_file(format!("{dir}/tempo"), &CompactRecorder::new(), device)
        .expect("Failed to load the tempo head")
}

/// What the auxiliary heads predict for a position, in game units: final
/// row totals (mine red, yellow, green, blue, then the opponent's), final
/// strikes (mine, the opponent's), and the final (my − opponent's) points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuxPred {
    pub marks: [f32; 8],
    pub strikes: [f32; 2],
    pub diff: f32,
}

/// Plain-array copy of [`AuxHeads`].
struct ManualAux {
    marks: Dense,
    strikes: Dense,
    diff: Dense,
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

    /// [`Dense::apply`] for `n` inputs, `n` a multiple of [`LANES`]: input
    /// s is `x[s * stride..][..d_in]`, its output `out[s * d_out..][..d_out]`.
    /// Each weight tile is loaded once for [`LANES`] inputs, and each
    /// input's sums run in `apply`'s order, so the results are bit-identical.
    /// Zero inputs are not skipped (a branch per input costs more than it
    /// saves here); adding 0·w leaves a sum unchanged.
    fn apply_batch(&self, x: &[f32], stride: usize, n: usize, out: &mut Vec<f32>, relu: bool) {
        debug_assert_eq!(n % LANES, 0);
        out.clear();
        out.resize(n * self.d_out, 0.0);
        for s0 in (0..n).step_by(LANES) {
            let mut t0 = 0;
            while t0 + TILE <= self.d_out {
                self.block::<TILE>(x, stride, s0, t0, out, relu);
                t0 += TILE;
            }
            for t in t0..self.d_out {
                self.block::<1>(x, stride, s0, t, out, relu);
            }
        }
    }

    /// Outputs `t0..t0 + T` of inputs `s0..s0 + LANES`, the accumulators
    /// held in registers across the input loop.
    fn block<const T: usize>(&self, x: &[f32], stride: usize, s0: usize, t0: usize, out: &mut [f32], relu: bool) {
        let d_in = self.w.len() / self.d_out;
        let bias: &[f32; T] = self.b[t0..t0 + T].try_into().unwrap();
        let mut acc = [*bias; LANES];
        for i in 0..d_in {
            let w: &[f32; T] = self.w[i * self.d_out + t0..][..T].try_into().unwrap();
            for (l, a) in acc.iter_mut().enumerate() {
                let xi = x[(s0 + l) * stride + i];
                for k in 0..T {
                    a[k] += xi * w[k];
                }
            }
        }
        for (l, a) in acc.iter().enumerate() {
            let o = &mut out[(s0 + l) * self.d_out + t0..][..T];
            for k in 0..T {
                o[k] = if relu { a[k].max(0.0) } else { a[k] };
            }
        }
    }
}

/// Output-tile width of [`Dense::apply_batch`]: two AVX2 vectors.
const TILE: usize = 16;
/// Inputs per pass of [`Dense::apply_batch`]: each weight tile loaded is
/// used this many times.
const LANES: usize = 4;
/// Positions per pass of [`ManualWinNet::logit_batch`] (fastest of 4–256
/// on an i7-1355U).
const CHUNK: usize = 16;

/// Plain-array copy of a [`WinModel`] for inference (burn's per-call
/// overhead dominates at these sizes, as for `ManualPairNet`). Numerics
/// match burn to a tolerance, not bit for bit.
pub struct ManualWinNet {
    width: usize,
    /// Row-token width: [`TOKEN`] (plain) or [`T_TOKEN`] (thermometer).
    token: usize,
    row1: Dense,
    row2: Dense,
    pair: Dense,
    head1: Dense,
    head2: Dense,
    out: Dense,
    /// Trained auxiliary heads, when the model dir has them (never used in play).
    aux: Option<ManualAux>,
    /// The tempo head t: the logit gains t(x) + t(swap x) (who moves next).
    tempo: Option<Dense>,
    /// The row encoder's output for every token a game can reach:
    /// `[HALVES × HALVES × width]`, my half's [`half_index`] major.
    row_codes: Vec<f32>,
}

/// One player's half of a row token as an index into
/// [`ManualWinNet::row_codes`]: an open row with c cells passed and t ≤ c
/// marks at c(c + 1)/2 + t (0–65), a closed row with t marks at 66 + t
/// (66–78). `None` for a row no game reaches. A half depends only on
/// these (the token is direction-free).
fn half_index(total: u8, free: Option<u8>, ascending: bool) -> Option<usize> {
    let t = total as usize;
    match free {
        Some(f) if (2..=12).contains(&f) => {
            let c = if ascending { f - 2 } else { 12 - f } as usize;
            (t <= c).then(|| c * (c + 1) / 2 + t)
        }
        Some(_) => None,
        None => (t <= 12).then_some(66 + t),
    }
}

/// Number of [`half_index`] values.
const HALVES: usize = 79;

impl ManualWinNet {
    pub fn from_model(model: &WinModel<MyBackend>) -> Self {
        let mut net = ManualWinNet {
            width: model.row1.weight.shape().dims[1],
            token: model.row1.weight.shape().dims[0],
            row1: Dense::from_linear(&model.row1),
            row2: Dense::from_linear(&model.row2),
            pair: Dense::from_linear(&model.pair),
            head1: Dense::from_linear(&model.head1),
            head2: Dense::from_linear(&model.head2),
            out: Dense::from_linear(&model.out),
            aux: None,
            tempo: None,
            row_codes: Vec::new(),
        };
        net.row_codes = net.row_table();
        net
    }

    /// [`ManualWinNet::row_codes`], each entry computed as [`ManualWinNet::g`]
    /// computes it, from an ascending first row in that state.
    fn row_table(&self) -> Vec<f32> {
        let halves: Vec<(u8, Option<u8>)> = (0..=10u8)
            .flat_map(|c| (0..=c).map(move |t| (t, Some(2 + c))))
            .chain((0..=12).map(|t| (t, None)))
            .collect();
        debug_assert_eq!(halves.len(), HALVES);
        debug_assert!(halves
            .iter()
            .enumerate()
            .all(|(i, &(t, f))| half_index(t, f, true) == Some(i)));
        let board = |row| State::from_parts(0, [row, (0, Some(2)), (0, Some(12)), (0, Some(12))]);
        let (mut h, mut e) = (Vec::new(), Vec::new());
        let mut table = Vec::with_capacity(HALVES * HALVES * self.width);
        for &mine in &halves {
            for &theirs in &halves {
                let (me, opp) = (board(mine), board(theirs));
                if self.is_thermo() {
                    self.row1.apply(&thermo_input(&me, &opp)[..T_TOKEN], &mut h, true);
                } else {
                    self.row1.apply(&win_input(&me, &opp)[..TOKEN], &mut h, true);
                }
                self.row2.apply(&h, &mut e, true);
                table.extend_from_slice(&e);
            }
        }
        table
    }

    /// g(x), and t(x) when the net has a tempo head.
    fn g_t(&self, x: &[f32]) -> (f32, Option<f32>) {
        let (w, t) = (self.width, self.token);
        let (mut h, mut e) = (Vec::with_capacity(2 * w), Vec::with_capacity(w));
        let mut pooled = [vec![0.0f32; w], vec![0.0f32; w]];
        for r in 0..4 {
            self.row1.apply(&x[t * r..t * (r + 1)], &mut h, true);
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
        cat.extend_from_slice(&x[4 * t..]);
        self.head1.apply(&cat, &mut h, true);
        self.head2.apply(&h, &mut e, true);
        self.out.apply(&e, &mut h, false);
        let g = h[0];
        let t = self.tempo.as_ref().map(|t| {
            t.apply(&e, &mut h, false);
            h[0]
        });
        (g, t)
    }

    /// The logit from both orientations' (g, t).
    fn combine((gx, tx): (f32, Option<f32>), (gs, ts): (f32, Option<f32>)) -> f32 {
        match (tx, ts) {
            (Some(a), Some(b)) => (gx - gs) + (a + b),
            _ => gx - gs,
        }
    }

    pub fn is_thermo(&self) -> bool {
        self.token == T_TOKEN
    }

    /// The net with a tempo head.
    pub fn with_tempo(mut self, head: &Linear<MyBackend>) -> Self {
        self.tempo = Some(Dense::from_linear(head));
        self
    }

    pub fn has_tempo(&self) -> bool {
        self.tempo.is_some()
    }

    /// Logit z = g(x) − g(swap x) (+ t(x) + t(swap x) with a tempo head) of
    /// a plain input; a thermometer net takes [`ManualWinNet::logit_states`].
    pub fn logit(&self, x: &[f32; WIN_INPUTS]) -> f32 {
        assert!(
            !self.is_thermo(),
            "a thermometer net takes logit_states, not plain inputs"
        );
        Self::combine(self.g_t(x), self.g_t(&swap_input(x)))
    }

    /// The logit for `me` against `opp` in the net's own input encoding.
    pub fn logit_states(&self, me: &State, opp: &State) -> f32 {
        if self.is_thermo() {
            let x = thermo_input(me, opp);
            Self::combine(self.g_t(&x), self.g_t(&swap_thermo(&x)))
        } else {
            self.logit(&win_input(me, opp))
        }
    }

    /// [`ManualWinNet::logit_states`] for many positions at once, bit for
    /// bit: every input and its swap go through each layer together, in
    /// chunks of [`CHUNK`] positions so the activations stay in cache.
    pub fn logit_batch(&self, pairs: &[(&State, &State)]) -> Vec<f32> {
        pairs.chunks(CHUNK).flat_map(|c| self.logit_chunk(c)).collect()
    }

    fn logit_chunk(&self, pairs: &[(&State, &State)]) -> Vec<f32> {
        let (w, t) = (self.width, self.token);
        let len = 4 * t + if self.is_thermo() { T_GLOBALS } else { GLOBALS };
        // Input 2k is position k, input 2k + 1 its swap; padded to whole lanes.
        let m = (2 * pairs.len()).next_multiple_of(LANES);
        let mut xs = vec![0.0f32; m * len];
        for (k, (me, opp)) in pairs.iter().enumerate() {
            let (a, b) = xs[2 * k * len..(2 * k + 2) * len].split_at_mut(len);
            if self.is_thermo() {
                let x = thermo_input(me, opp);
                a.copy_from_slice(&x);
                b.copy_from_slice(&swap_thermo(&x));
            } else {
                let x = win_input(me, opp);
                a.copy_from_slice(&x);
                b.copy_from_slice(&swap_input(&x));
            }
        }
        let (mut h, mut e) = (Vec::new(), Vec::new());
        // Row encoder: looked up, or computed for a row no game reaches.
        let mut rows = vec![0.0f32; 4 * m * w];
        for (k, (me, opp)) in pairs.iter().enumerate() {
            let (mt, mf, ot, of) = (
                me.row_totals(),
                me.row_free_values(),
                opp.row_totals(),
                opp.row_free_values(),
            );
            for r in 0..4 {
                let (a, b) = (half_index(mt[r], mf[r], r < 2), half_index(ot[r], of[r], r < 2));
                for (j, halves) in [(2 * k, (a, b)), (2 * k + 1, (b, a))] {
                    let dst = &mut rows[(4 * j + r) * w..][..w];
                    if let (Some(mine), Some(theirs)) = halves {
                        dst.copy_from_slice(&self.row_codes[(mine * HALVES + theirs) * w..][..w]);
                    } else {
                        self.row1.apply(&xs[j * len + r * t..][..t], &mut h, true);
                        self.row2.apply(&h, &mut e, true);
                        dst.copy_from_slice(&e);
                    }
                }
            }
        }
        let mut pooled = vec![0.0f32; 2 * m * w];
        for (p, rows) in pooled.chunks_mut(w).zip(rows.chunks(2 * w)) {
            for (k, v) in p.iter_mut().enumerate() {
                *v = (*v + rows[k]) + rows[w + k];
            }
        }
        self.pair.apply_batch(&pooled, w, 2 * m, &mut h, true);
        let g_len = len - 4 * t;
        let mut cat = vec![0.0f32; m * (w + g_len)];
        for ((c, pp), x) in cat.chunks_mut(w + g_len).zip(h.chunks(2 * w)).zip(xs.chunks(len)) {
            for k in 0..w {
                c[k] = (c[k] + pp[k]) + pp[w + k];
            }
            c[w..].copy_from_slice(&x[4 * t..]);
        }
        self.head1.apply_batch(&cat, w + g_len, m, &mut h, true);
        self.head2.apply_batch(&h, 2 * w, m, &mut e, true);
        self.out.apply_batch(&e, w, m, &mut h, false);
        let g: Vec<f32> = h;
        match &self.tempo {
            Some(head) => {
                let mut t = Vec::new();
                head.apply_batch(&e, w, m, &mut t, false);
                (0..pairs.len())
                    .map(|k| (g[2 * k] - g[2 * k + 1]) + (t[2 * k] + t[2 * k + 1]))
                    .collect()
            }
            None => g.chunks(2).take(pairs.len()).map(|g| g[0] - g[1]).collect(),
        }
    }

    /// The net with its auxiliary heads.
    pub fn with_aux(mut self, heads: &AuxHeads<MyBackend>) -> Self {
        self.aux = Some(ManualAux {
            marks: Dense::from_linear(&heads.marks),
            strikes: Dense::from_linear(&heads.strikes),
            diff: Dense::from_linear(&heads.diff),
        });
        self
    }

    /// Each row's encoding and the head's last hidden layer for one input:
    /// [`ManualWinNet::g`] without its output layer. Kept apart from `g`,
    /// which play calls in its hot loop and reuses its buffers.
    fn trunk(&self, x: &[f32]) -> ([Vec<f32>; 4], Vec<f32>) {
        let (w, t) = (self.width, self.token);
        let mut h = Vec::with_capacity(2 * w);
        let mut rows: [Vec<f32>; 4] = Default::default();
        let mut pooled = [vec![0.0f32; w], vec![0.0f32; w]];
        for (r, e) in rows.iter_mut().enumerate() {
            self.row1.apply(&x[t * r..t * (r + 1)], &mut h, true);
            self.row2.apply(&h, e, true);
            for (p, v) in pooled[r / 2].iter_mut().zip(e.iter()) {
                *p += v;
            }
        }
        let (mut e, mut cat) = (Vec::with_capacity(w), vec![0.0f32; w]);
        for p in &pooled {
            self.pair.apply(p, &mut e, true);
            for (c, v) in cat.iter_mut().zip(&e) {
                *c += v;
            }
        }
        cat.extend_from_slice(&x[4 * t..]);
        self.head1.apply(&cat, &mut h, true);
        self.head2.apply(&h, &mut e, true);
        (rows, e)
    }

    /// The auxiliary heads' predictions for `me` against `opp`; `None`
    /// without heads.
    pub fn aux_states(&self, me: &State, opp: &State) -> Option<AuxPred> {
        let aux = self.aux.as_ref()?;
        let (x, s): (Vec<f32>, Vec<f32>) = if self.is_thermo() {
            let x = thermo_input(me, opp);
            (x.to_vec(), swap_thermo(&x).to_vec())
        } else {
            let x = win_input(me, opp);
            (x.to_vec(), swap_input(&x).to_vec())
        };
        let ((ex, hx), (es, hs)) = (self.trunk(&x), self.trunk(&s));
        let one = |d: &Dense, v: &[f32]| {
            let mut o = Vec::with_capacity(1);
            d.apply(v, &mut o, false);
            o[0]
        };
        let mut marks = [0.0f32; 8];
        for r in 0..4 {
            marks[r] = 12.0 * one(&aux.marks, &[ex[r].as_slice(), hx.as_slice()].concat());
            marks[4 + r] = 12.0 * one(&aux.marks, &[es[r].as_slice(), hs.as_slice()].concat());
        }
        Some(AuxPred {
            marks,
            strikes: [4.0 * one(&aux.strikes, &hx), 4.0 * one(&aux.strikes, &hs)],
            diff: 100.0 * (one(&aux.diff, &hx) - one(&aux.diff, &hs)),
        })
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

    /// Load `dir/config.json` (width, encoding, heads), `dir/model.mpk`, and
    /// `dir/aux.mpk` and `dir/tempo.mpk` when the config says the dir has them.
    pub fn load(dir: &str) -> Self {
        let device = Default::default();
        let cfg = WinModelConfig::load(format!("{dir}/config.json")).expect("Failed to load win model config");
        let model = cfg
            .init::<MyBackend>(&device)
            .load_file(format!("{dir}/model"), &CompactRecorder::new(), &device)
            .expect("Failed to load win model");
        let mut net = ManualWinNet::from_model(&model);
        if cfg.has_aux() {
            let heads = AuxHeads::<MyBackend>::new(cfg.width, &device)
                .load_file(format!("{dir}/aux"), &CompactRecorder::new(), &device)
                .expect("Failed to load the auxiliary heads");
            net = net.with_aux(&heads);
        }
        if cfg.is_tempo() {
            net = net.with_tempo(&load_tempo::<MyBackend>(dir, cfg.width, &device));
        }
        WinStrategy { net: Arc::new(net) }
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

    /// The logit of P(win) for each (candidate, opponent, frame): z(cand, opp)
    /// when the opponent moves next, −z(opp, cand) when I do. One batch.
    fn framed_logits(&self, items: &[(&State, State, Next)]) -> Vec<f32> {
        let pairs: Vec<(&State, &State)> = items
            .iter()
            .map(|(c, o, n)| match n {
                Next::Opp => (*c, o),
                Next::Me => (o, *c),
            })
            .collect();
        self.net
            .logit_batch(&pairs)
            .into_iter()
            .zip(items)
            .map(|(z, (_, _, n))| if *n == Next::Me { -z } else { z })
            .collect()
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
        let opps: Vec<State> = groups.iter().map(|(_, opps)| Self::opponent(opps)).collect();
        let pairs: Vec<(&State, &State)> = groups
            .iter()
            .zip(&opps)
            .flat_map(|((candidates, _), opp)| candidates.iter().map(move |c| (c, opp)))
            .collect();
        let mut values = self.net.logit_batch(&pairs).into_iter().map(probit_of_logit);
        groups
            .iter()
            .map(|(candidates, _)| values.by_ref().take(candidates.len()).collect())
            .collect()
    }

    fn evaluate_batch_next(&self, candidates: &[State], opp_states: &[State], next: Next) -> Vec<f32> {
        self.evaluate_batch_multi_next(&[(candidates, opp_states, next)])
            .pop()
            .unwrap()
    }

    /// Without a tempo head the frame cannot matter: today's values.
    fn evaluate_batch_multi_next(&self, groups: &[(&[State], &[State], Next)]) -> Vec<Vec<f32>> {
        if !self.net.has_tempo() {
            let plain: Vec<(&[State], &[State])> = groups.iter().map(|g| (g.0, g.1)).collect();
            return self.evaluate_batch_multi(&plain);
        }
        let items: Vec<(&State, State, Next)> = groups
            .iter()
            .flat_map(|(cands, opps, n)| {
                let o = Self::opponent(opps);
                cands.iter().map(move |c| (c, o, *n))
            })
            .collect();
        let mut values = self.framed_logits(&items).into_iter().map(probit_of_logit);
        groups
            .iter()
            .map(|(c, _, _)| values.by_ref().take(c.len()).collect())
            .collect()
    }
}

impl WinProb for WinStrategy {
    fn win_prob_multi(&self, groups: &[(&State, &[State])]) -> Vec<f32> {
        let opps: Vec<State> = groups.iter().map(|(_, opps)| Self::opponent(opps)).collect();
        let pairs: Vec<(&State, &State)> = groups.iter().zip(&opps).map(|((our, _), opp)| (*our, opp)).collect();
        self.net.logit_batch(&pairs).into_iter().map(sigmoid).collect()
    }

    fn win_prob_multi_next(&self, groups: &[(&State, &[State], Next)]) -> Vec<f32> {
        if !self.net.has_tempo() {
            let plain: Vec<(&State, &[State])> = groups.iter().map(|g| (g.0, g.1)).collect();
            return self.win_prob_multi(&plain);
        }
        let items: Vec<(&State, State, Next)> = groups.iter().map(|(s, o, n)| (*s, Self::opponent(o), *n)).collect();
        self.framed_logits(&items).into_iter().map(sigmoid).collect()
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

    fn random_thermo(seed: u64, width: usize) -> WinModel<MyBackend> {
        let device = Default::default();
        <MyBackend as burn::tensor::backend::Backend>::seed(&device, seed);
        WinModelConfig::new()
            .with_width(width)
            .with_thermo(Some(true))
            .init::<MyBackend>(&device)
    }

    /// Real positions: the two boards both ways, and a board with a locked row.
    fn pairs() -> Vec<(State, State)> {
        let (a, b) = boards();
        let mut c = b;
        for n in [2, 3, 4, 5, 6, 12] {
            c.apply_mark(Mark { row: 0, number: n });
        }
        vec![(a, b), (b, a), (c, a), (a, c)]
    }

    fn relabel_t(x: &[f32; WIN_INPUTS_T], code: u8) -> [f32; WIN_INPUTS_T] {
        let mut order = [0usize, 1, 2, 3];
        if code & 1 != 0 {
            order.swap(0, 1);
        }
        if code & 2 != 0 {
            order.swap(2, 3);
        }
        if code & 4 != 0 {
            order = [order[2], order[3], order[0], order[1]];
        }
        let mut y = *x;
        for (slot, &src) in order.iter().enumerate() {
            y[T_TOKEN * slot..T_TOKEN * (slot + 1)].copy_from_slice(&x[T_TOKEN * src..T_TOKEN * (src + 1)]);
        }
        y
    }

    #[test]
    fn thermometer_logit_is_antisymmetric_and_colour_invariant() {
        let model = random_thermo(21, 16);
        let xs: Vec<_> = pairs().iter().map(|(me, opp)| thermo_input(me, opp)).collect();
        let swapped: Vec<_> = xs.iter().map(swap_thermo).collect();
        let base = model_logits_t(&model, &xs);
        for (z, zs) in base.iter().zip(model_logits_t(&model, &swapped)) {
            assert!((z + zs).abs() < 1e-5, "z {z} vs swapped {zs}");
        }
        for code in 0..8u8 {
            let ys: Vec<_> = xs.iter().map(|x| relabel_t(x, code)).collect();
            for (z, zr) in base.iter().zip(model_logits_t(&model, &ys)) {
                assert!((z - zr).abs() < 1e-5, "relabelling {code}: {z} vs {zr}");
            }
        }
    }

    #[test]
    fn manual_net_matches_burn_for_a_thermometer_net() {
        let model = random_thermo(22, 16);
        let net = ManualWinNet::from_model(&model);
        assert!(net.is_thermo() && model.is_thermo());
        let xs: Vec<_> = pairs().iter().map(|(me, opp)| thermo_input(me, opp)).collect();
        for ((me, opp), zb) in pairs().iter().zip(model_logits_t(&model, &xs)) {
            let zm = net.logit_states(me, opp);
            assert!((zb - zm).abs() < 1e-4, "burn {zb} vs manual {zm}");
        }
    }

    /// Random legal positions: random marks (rows lock when they finish),
    /// some rows closed by the opponent, and 0–3 strikes per player.
    fn random_pairs(n: usize, seed: u64) -> Vec<(State, State)> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut board = || {
            let mut s = State::default();
            for _ in 0..rng.gen_range(0..40) {
                let (row, number) = (rng.gen_range(0..4), rng.gen_range(2..=12));
                if s.can_mark(row, number) {
                    s.apply_mark(Mark { row, number });
                }
            }
            s.lock(std::array::from_fn(|_| rng.gen_bool(0.1)));
            for _ in 0..rng.gen_range(0..4) {
                s.apply_strike();
            }
            s
        };
        (0..n).map(|_| (board(), board())).collect()
    }

    #[test]
    fn a_batch_gives_each_positions_logit_bit_for_bit() {
        // 37 positions: 74 inputs, not a whole number of lanes. Widths 48
        // and 4 leave outputs past the last whole tile.
        let ps = random_pairs(37, 5);
        let refs: Vec<(&State, &State)> = ps.iter().map(|(a, b)| (a, b)).collect();
        let models = [
            ("plain 64", random_model(30, 64)),
            ("thermo 64", random_thermo(31, 64)),
            ("thermo 48", random_thermo(32, 48)),
            ("thermo 4", random_thermo(33, 4)),
        ];
        for (label, model) in models {
            let net = ManualWinNet::from_model(&model);
            let got: Vec<u32> = net.logit_batch(&refs).iter().map(|z| z.to_bits()).collect();
            let want: Vec<u32> = ps.iter().map(|(a, b)| net.logit_states(a, b).to_bits()).collect();
            assert_eq!(got, want, "{label}");
            assert!(net.logit_batch(&[]).is_empty());
        }
    }

    #[test]
    fn a_batch_of_positions_outside_the_row_table_is_still_exact() {
        // More marks than cells passed: no game reaches these rows.
        let odd = State::from_parts(1, [(5, Some(3)), (0, Some(2)), (13, None), (4, Some(11))]);
        let (a, b) = boards();
        let net = ManualWinNet::from_model(&random_thermo(35, 16));
        let got = net.logit_batch(&[(&odd, &a), (&b, &odd), (&a, &b)]);
        let want = [
            net.logit_states(&odd, &a),
            net.logit_states(&b, &odd),
            net.logit_states(&a, &b),
        ];
        let got: Vec<u32> = got.iter().map(|z| z.to_bits()).collect();
        assert_eq!(got, want.map(f32::to_bits));
    }

    #[test]
    fn row_halves_are_the_same_in_both_directions() {
        // Every open (cells passed c, marks t ≤ c) and locked row state,
        // as an ascending row and as the mirrored descending row.
        let mut rows: Vec<(u8, Option<u8>, Option<u8>)> = Vec::new();
        for c in 0..=10u8 {
            for t in 0..=c {
                rows.push((t, Some(2 + c), Some(12 - c)));
            }
        }
        rows.extend((0..=12).map(|t| (t, None, None)));
        let empty = (0, Some(2));
        for (t, asc, desc) in rows {
            let a = State::from_parts(2, [(t, asc), empty, (0, Some(12)), (0, Some(12))]);
            let d = State::from_parts(2, [empty, empty, (0, Some(12)), (t, desc)]);
            let o = State::default();
            let (xa, xd) = (thermo_input(&a, &o), thermo_input(&d, &o));
            assert_eq!(&xa[..T_HALF], &xd[3 * T_TOKEN..3 * T_TOKEN + T_HALF], "{t} {asc:?}");
            let (pa, pd) = (win_input(&a, &o), win_input(&d, &o));
            assert_eq!(&pa[..TOKEN / 2], &pd[3 * TOKEN..3 * TOKEN + TOKEN / 2], "{t} {asc:?}");
        }
    }

    #[test]
    fn win_strategy_batches_give_each_positions_value() {
        let bot = WinStrategy::from_model(&random_thermo(34, 16));
        let ps = random_pairs(9, 6);
        let cands: Vec<State> = ps.iter().map(|p| p.0).collect();
        let (o1, o2) = (ps[0].1, ps[1].1);
        let groups: [(&[State], &[State]); 3] = [
            (&cands[..4], std::slice::from_ref(&o1)),
            (&cands[4..4], std::slice::from_ref(&o2)),
            (&cands[4..], std::slice::from_ref(&o2)),
        ];
        let values = bot.evaluate_batch_multi(&groups);
        for ((cs, opps), vs) in groups.iter().zip(&values) {
            let want: Vec<f32> = cs
                .iter()
                .map(|c| probit_of_logit(bot.net.logit_states(c, &opps[0])))
                .collect();
            assert_eq!(vs, &want);
        }
        let leaves: Vec<(&State, &[State])> = ps.iter().map(|(a, b)| (a, std::slice::from_ref(b))).collect();
        let want: Vec<f32> = ps.iter().map(|(a, b)| sigmoid(bot.net.logit_states(a, b))).collect();
        assert_eq!(bot.win_prob_multi(&leaves), want);
    }

    #[test]
    #[should_panic(expected = "a thermometer net takes logit_states")]
    fn a_thermometer_net_refuses_plain_inputs() {
        let net = ManualWinNet::from_model(&random_thermo(23, 8));
        let (a, b) = boards();
        net.logit(&win_input(&a, &b));
    }

    #[test]
    fn a_plain_net_gives_the_same_logit_from_states() {
        let net = ManualWinNet::from_model(&random_model(24, 16));
        assert!(!net.is_thermo());
        for (me, opp) in pairs() {
            assert_eq!(net.logit_states(&me, &opp), net.logit(&win_input(&me, &opp)));
        }
    }

    #[test]
    fn widening_keeps_the_function() {
        let device = Default::default();
        let plain = random_model(25, 16);
        let wide = widen_to_thermo(&plain, &device);
        assert!(wide.is_thermo() && !plain.is_thermo());
        let (p, t) = (ManualWinNet::from_model(&plain), ManualWinNet::from_model(&wide));
        for (me, opp) in pairs() {
            let (zp, zt) = (p.logit_states(&me, &opp), t.logit_states(&me, &opp));
            assert!((zp - zt).abs() < 1e-5, "plain {zp} vs widened {zt}");
        }
    }

    #[test]
    fn win_strategy_plays_a_thermometer_net() {
        let bot = WinStrategy::from_model(&random_thermo(26, 16));
        let (a, b) = boards();
        let values = bot.evaluate_batch(&[a, b], &[b]);
        assert_eq!(values[0], probit_of_logit(bot.net.logit_states(&a, &b)));
        assert_eq!(values[1], probit_of_logit(bot.net.logit_states(&b, &b)));
        let p = bot.win_prob_multi(&[(&a, std::slice::from_ref(&a))])[0];
        assert!((p - 0.5).abs() < 1e-6, "p = {p}");
    }

    #[test]
    fn old_model_dirs_load_and_thermometer_dirs_round_trip() {
        let base = std::env::temp_dir().join(format!("qwixxer-win-dirs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (a, b) = boards();
        for thermo in [false, true] {
            let dir = base.join(if thermo { "thermo" } else { "old" });
            std::fs::create_dir_all(&dir).unwrap();
            let dir = dir.to_string_lossy().into_owned();
            let model = if thermo {
                random_thermo(27, 8)
            } else {
                random_model(27, 8)
            };
            model
                .clone()
                .save_file(format!("{dir}/model"), &CompactRecorder::new())
                .unwrap();
            if thermo {
                WinModelConfig::new()
                    .with_width(8)
                    .with_thermo(Some(true))
                    .save(format!("{dir}/config.json"))
                    .unwrap();
            } else {
                // A config.json written before `thermo` existed.
                std::fs::write(format!("{dir}/config.json"), "{\n  \"width\": 8\n}").unwrap();
                let cfg = WinModelConfig::load(format!("{dir}/config.json")).unwrap();
                assert_eq!((cfg.width, cfg.thermo, cfg.is_thermo()), (8, None, false));
            }
            let loaded = WinStrategy::load(&dir);
            assert_eq!(loaded.net.is_thermo(), thermo);
            // The file is half precision: close, not equal.
            let want = ManualWinNet::from_model(&model).logit_states(&a, &b);
            let got = loaded.net.logit_states(&a, &b);
            assert!((want - got).abs() < 0.05, "thermo {thermo}: {want} vs {got}");
        }
        std::fs::remove_dir_all(&base).unwrap();
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

    fn random_tempo_head(seed: u64, width: usize) -> burn::nn::Linear<MyBackend> {
        let device = Default::default();
        <MyBackend as burn::tensor::backend::Backend>::seed(&device, seed);
        burn::nn::LinearConfig::new(width, 1).with_bias(true).init(&device)
    }

    #[test]
    fn a_zero_tempo_head_changes_no_logit() {
        let device = Default::default();
        for model in [random_model(40, 16), random_thermo(41, 16)] {
            let plain = ManualWinNet::from_model(&model);
            let tempo = ManualWinNet::from_model(&model).with_tempo(&zero_tempo(16, &device));
            assert!(tempo.has_tempo() && !plain.has_tempo());
            let ps = random_pairs(9, 7);
            let refs: Vec<(&State, &State)> = ps.iter().map(|(a, b)| (a, b)).collect();
            for (a, b) in &ps {
                assert_eq!(tempo.logit_states(a, b), plain.logit_states(a, b));
            }
            assert_eq!(tempo.logit_batch(&refs), plain.logit_batch(&refs));
        }
    }

    #[test]
    fn a_tempo_logit_splits_into_its_antisymmetric_and_symmetric_parts() {
        let net = ManualWinNet::from_model(&random_thermo(42, 16)).with_tempo(&random_tempo_head(43, 16));
        for (a, b) in random_pairs(9, 8) {
            let x = thermo_input(&a, &b);
            let ((gx, tx), (gs, ts)) = (net.g_t(&x), net.g_t(&swap_thermo(&x)));
            let (tx, ts) = (tx.unwrap(), ts.unwrap());
            let (zab, zba) = (net.logit_states(&a, &b), net.logit_states(&b, &a));
            assert!((zab + zba - 2.0 * (tx + ts)).abs() < 1e-4, "symmetric part");
            assert!((zab - zba - 2.0 * (gx - gs)).abs() < 1e-4, "antisymmetric part");
            assert!((tx + ts).abs() > 1e-6, "a random head adds a symmetric part");
        }
    }

    #[test]
    fn manual_tempo_net_matches_the_burn_trunk_and_head() {
        for thermo in [false, true] {
            let model = if thermo {
                random_thermo(44, 16)
            } else {
                random_model(44, 16)
            };
            let head = random_tempo_head(45, 16);
            let net = ManualWinNet::from_model(&model).with_tempo(&head);
            let ps = random_pairs(7, 9);
            let device = Default::default();
            let (xs, ss): (Vec<f32>, Vec<f32>) = if thermo {
                (
                    ps.iter().flat_map(|(a, b)| thermo_input(a, b)).collect(),
                    ps.iter().flat_map(|(a, b)| swap_thermo(&thermo_input(a, b))).collect(),
                )
            } else {
                (
                    ps.iter().flat_map(|(a, b)| win_input(a, b)).collect(),
                    ps.iter().flat_map(|(a, b)| swap_input(&win_input(a, b))).collect(),
                )
            };
            let d = if thermo { WIN_INPUTS_T } else { WIN_INPUTS };
            let n = ps.len();
            let both = Tensor::cat(
                vec![flat_tensor::<MyBackend>(&xs, d, &device), flat_tensor(&ss, d, &device)],
                0,
            );
            let (_, h) = model.trunk(both);
            let g = model.out.forward(h.clone());
            let t = head.forward(h);
            let z = g.clone().narrow(0, 0, n) - g.narrow(0, n, n) + t.clone().narrow(0, 0, n) + t.narrow(0, n, n);
            let want = z.into_data().to_vec::<f32>().unwrap();
            for ((a, b), zb) in ps.iter().zip(want) {
                let zm = net.logit_states(a, b);
                assert!((zb - zm).abs() < 1e-4, "thermo {thermo}: burn {zb} vs manual {zm}");
            }
        }
    }

    #[test]
    fn a_tempo_batch_gives_each_positions_logit_bit_for_bit() {
        let ps = random_pairs(37, 10);
        let refs: Vec<(&State, &State)> = ps.iter().map(|(a, b)| (a, b)).collect();
        for (model, w) in [
            (random_model(46, 64), 64),
            (random_thermo(47, 48), 48),
            (random_thermo(48, 4), 4),
        ] {
            let net = ManualWinNet::from_model(&model).with_tempo(&random_tempo_head(49, w));
            let got: Vec<u32> = net.logit_batch(&refs).iter().map(|z| z.to_bits()).collect();
            let want: Vec<u32> = ps.iter().map(|(a, b)| net.logit_states(a, b).to_bits()).collect();
            assert_eq!(got, want, "width {w}");
        }
    }

    #[test]
    fn zero_tempo_is_all_zero() {
        let head = zero_tempo::<MyBackend>(8, &Default::default());
        assert_eq!(head.weight.shape().dims, [8, 1]);
        assert!(head
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|&v| v == 0.0));
        assert_eq!(head.bias.unwrap().val().into_data().to_vec::<f32>().unwrap(), vec![0.0]);
    }

    /// A model dir by hand (`win_pi::save_win` needs the `dqn` feature).
    fn save_dir(dir: &str, mcfg: &WinModelConfig, model: &WinModel<MyBackend>) {
        std::fs::create_dir_all(dir).unwrap();
        mcfg.save(format!("{dir}/config.json")).unwrap();
        model
            .clone()
            .save_file(format!("{dir}/model"), &CompactRecorder::new())
            .unwrap();
    }

    #[test]
    fn tempo_dirs_round_trip_and_old_dirs_have_no_head() {
        let base = std::env::temp_dir().join(format!("qwixxer-win-tempo-dirs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let dir = base.to_string_lossy().into_owned();
        let model = random_thermo(50, 8);
        let head = random_tempo_head(51, 8);
        let mcfg = WinModelConfig::new()
            .with_width(8)
            .with_thermo(Some(true))
            .with_tempo(Some(true));
        save_dir(&dir, &mcfg, &model);
        save_tempo(&dir, &head);
        let loaded = WinStrategy::load(&dir);
        assert!(loaded.net.has_tempo());
        let (a, b) = boards();
        let want = ManualWinNet::from_model(&model).with_tempo(&head).logit_states(&a, &b);
        let got = loaded.net.logit_states(&a, &b);
        // The files are half precision: close, not equal.
        assert!((want - got).abs() < 0.05, "{want} vs {got}");
        // A config written before `tempo` existed: no head.
        std::fs::write(
            format!("{dir}/config.json"),
            "{\n  \"width\": 8,\n  \"thermo\": true\n}",
        )
        .unwrap();
        let cfg = WinModelConfig::load(format!("{dir}/config.json")).unwrap();
        assert_eq!((cfg.tempo, cfg.is_tempo()), (None, false));
        assert!(!WinStrategy::load(&dir).net.has_tempo());
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    #[should_panic(expected = "Failed to load the tempo head")]
    fn a_tempo_dir_without_its_head_fails_to_load() {
        let base = std::env::temp_dir().join(format!("qwixxer-win-tempo-missing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let dir = base.to_string_lossy().into_owned();
        let mcfg = WinModelConfig::new().with_width(8).with_tempo(Some(true));
        save_dir(&dir, &mcfg, &random_model(52, 8));
        WinStrategy::load(&dir);
    }

    use crate::strategy::Next;

    fn tempo_bot(seed: u64) -> WinStrategy {
        WinStrategy {
            net: Arc::new(
                ManualWinNet::from_model(&random_thermo(seed, 16)).with_tempo(&random_tempo_head(seed + 1, 16)),
            ),
        }
    }

    #[test]
    fn me_next_is_one_minus_the_opponents_view_for_a_tempo_net() {
        let bot = tempo_bot(60);
        for (a, b) in random_pairs(9, 11) {
            let z_ab = bot.net.logit_states(&a, &b);
            let z_ba = bot.net.logit_states(&b, &a);
            assert_eq!(
                bot.evaluate_batch_next(&[a], &[b], Next::Opp),
                vec![probit_of_logit(z_ab)]
            );
            assert_eq!(
                bot.evaluate_batch_next(&[a], &[b], Next::Me),
                vec![probit_of_logit(-z_ba)]
            );
            let leaf = [(&a, std::slice::from_ref(&b), Next::Me)];
            assert_eq!(bot.win_prob_multi_next(&leaf), vec![sigmoid(-z_ba)]);
            let leaf = [(&a, std::slice::from_ref(&b), Next::Opp)];
            assert_eq!(bot.win_prob_multi_next(&leaf), vec![sigmoid(z_ab)]);
            assert_ne!(z_ab, -z_ba, "the tempo head breaks antisymmetry");
        }
    }

    #[test]
    fn mixed_groups_equal_separate_calls() {
        let bot = tempo_bot(62);
        let ps = random_pairs(9, 12);
        let cands: Vec<State> = ps.iter().map(|p| p.0).collect();
        let (o1, o2) = (ps[0].1, ps[1].1);
        let groups: [(&[State], &[State], Next); 3] = [
            (&cands[..4], std::slice::from_ref(&o1), Next::Me),
            (&cands[4..4], std::slice::from_ref(&o2), Next::Opp),
            (&cands[4..], std::slice::from_ref(&o2), Next::Opp),
        ];
        let got = bot.evaluate_batch_multi_next(&groups);
        for ((c, o, n), vs) in groups.iter().zip(&got) {
            assert_eq!(vs, &bot.evaluate_batch_next(c, o, *n));
        }
        let leaves: Vec<(&State, &[State], Next)> = ps
            .iter()
            .enumerate()
            .map(|(i, (a, b))| {
                (
                    a,
                    std::slice::from_ref(b),
                    if i % 2 == 0 { Next::Me } else { Next::Opp },
                )
            })
            .collect();
        let want: Vec<f32> = leaves
            .iter()
            .flat_map(|l| bot.win_prob_multi_next(std::slice::from_ref(l)))
            .collect();
        assert_eq!(bot.win_prob_multi_next(&leaves), want);
    }

    #[test]
    fn next_changes_nothing_without_a_tempo_head() {
        let ps = random_pairs(9, 13);
        let cands: Vec<State> = ps.iter().map(|p| p.0).collect();
        let opp = [ps[0].1];
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        for bot in [
            WinStrategy::from_model(&random_model(64, 16)),
            WinStrategy::from_model(&random_thermo(65, 16)),
        ] {
            assert_eq!(
                bits(bot.evaluate_batch_next(&cands, &opp, Next::Me)),
                bits(bot.evaluate_batch(&cands, &opp))
            );
            let g = [(cands.as_slice(), opp.as_slice(), Next::Me)];
            assert_eq!(
                bits(bot.evaluate_batch_multi_next(&g).pop().unwrap()),
                bits(bot.evaluate_batch(&cands, &opp))
            );
            let leaves: Vec<(&State, &[State], Next)> =
                ps.iter().map(|(a, b)| (a, std::slice::from_ref(b), Next::Me)).collect();
            let plain: Vec<(&State, &[State])> = ps.iter().map(|(a, b)| (a, std::slice::from_ref(b))).collect();
            assert_eq!(bits(bot.win_prob_multi_next(&leaves)), bits(bot.win_prob_multi(&plain)));
        }
        let pair = crate::dqn::pair::PairStrategy::load("pair_model");
        assert_eq!(
            bits(pair.evaluate_batch_next(&cands, &opp, Next::Me)),
            bits(pair.evaluate_batch(&cands, &opp))
        );
    }
}
