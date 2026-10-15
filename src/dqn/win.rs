//! Win net: a 2-player P(win) network, antisymmetric in the players and
//! invariant to the 8 colour relabellings by construction, trained by TD(λ)
//! self-play on win/loss outcomes (TD-Gammon style).
//!
//! Design: docs/superpowers/specs/2026-10-15-win-net-design.md

use super::pair::board_features;
use super::MyBackend;
use crate::state::State;
use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

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
}
