//! Pairwise differential network ("pair" bot): joint two-board evaluation.
//!
//! The model sees both players' boards and predicts the distribution of the
//! *future* score differential `final_diff − current_diff` as `(μ, log σ²)`.
//! Move selection ranks candidates by `(current_diff + μ) / σ`, monotone in
//! the Gaussian P(win) against the leading opponent.
//!
//! Design: docs/superpowers/specs/2026-06-10-pair-network-design.md

use super::{
    aggregate_weighted_probability, lockable_rows, row_progress, total_progress, MyBackend, LOG_VAR_MAX, LOG_VAR_MIN,
};
use crate::state::State;
use crate::strategy::{BoardClock, Bot, EvalClock};
use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    record::CompactRecorder,
};
use std::sync::Arc;

/// Per-board feature block size: the original 20, then turns-until-roll,
/// colour-phase-pending, P(forced strike) and 4 per-row lock threats.
pub const BOARD_FEATURES: usize = 27;
/// Full pair input: two board blocks + 5 pair-level features.
pub const PAIR_FEATURES: usize = 59;
/// Index of the first pair-level feature.
pub const PAIR_LEVEL: usize = 2 * BOARD_FEATURES;
const _: () = assert!(PAIR_FEATURES == 59, "PairModelConfig's `input` default must match");

/// Per-board feature block. Same layout as the old net's per-board features,
/// plus per-board aggregates, and with the lock-rule fix: a free pointer
/// sitting on the row's terminal number (12/2) cannot be marked while the row
/// has <5 marks, so it contributes 0 to the weighted probability.
pub fn board_features(state: &State, clock: BoardClock, others: &[&State]) -> [f32; BOARD_FEATURES] {
    let totals = state.row_totals();
    let frees = state.row_free_values();
    let locked = state.locked();

    let mut f = [0.0f32; BOARD_FEATURES];
    for i in 0..4 {
        f[i] = row_progress(frees[i], i < 2);
        f[4 + i] = totals[i] as f32 / 11.0;
        f[8 + i] = if locked[i] { 1.0 } else { 0.0 };
        f[12 + i] = match frees[i] {
            Some(fr) if fr == State::row_terminal(i) && totals[i] < 5 => 0.0,
            Some(fr) => {
                let ways = 6.0 - (7.0f32 - fr as f32).abs();
                (ways / 6.0) * (totals[i] as f32 + 1.0) / 11.0
            }
            None => 0.0,
        };
    }
    f[16] = state.strikes as f32 / 3.0;
    f[17] = state.blanks() as f32 / 40.0;
    f[18] = aggregate_weighted_probability(state);
    f[19] = lockable_rows(state) as f32 / 4.0;
    // Turn position (spec 2026-10-12 §1) and dice-only threats.
    f[20] = (clock.rolls_in as f32 / 3.0).min(1.0);
    f[21] = if clock.pending { 1.0 } else { 0.0 };
    f[22] = crate::threat::forced_strike_prob(state);
    f[23..27].copy_from_slice(&crate::threat::lock_threats(state, others));
    f
}

/// Full pair input for `our` against `paired = opps[paired_idx]`. `opps` are
/// ALL opponents of `our` (the pair-level features 54–58 are computed over all
/// of them; redundant with the opponent block in 1v1). `clock.opps` is aligned
/// with `opps`. Every board's threat inputs use all *other* players, so the
/// result does not depend on the order of `opps`.
///
/// Layout: `[our 27 | paired 27 | cdiff/100, num_opps/4, max opp progress,
/// max opp strikes/3, opp lockable-rows sum/8]`.
pub fn pair_features(our: &State, paired_idx: usize, opps: &[State], clock: EvalClock) -> [f32; PAIR_FEATURES] {
    assert!(
        opps.len() < crate::strategy::clock::MAX_PLAYERS,
        "pair features support up to {} players",
        crate::strategy::clock::MAX_PLAYERS
    );
    let paired = &opps[paired_idx];
    let mut ours_others: [&State; crate::strategy::clock::MAX_PLAYERS - 1] =
        [our; crate::strategy::clock::MAX_PLAYERS - 1];
    for (i, o) in opps.iter().enumerate() {
        ours_others[i] = o;
    }
    let mut theirs_others: [&State; crate::strategy::clock::MAX_PLAYERS - 1] =
        [our; crate::strategy::clock::MAX_PLAYERS - 1];
    let mut m = 1;
    for (j, o) in opps.iter().enumerate() {
        if j != paired_idx {
            theirs_others[m] = o;
            m += 1;
        }
    }
    let mut f = [0.0f32; PAIR_FEATURES];
    f[..BOARD_FEATURES].copy_from_slice(&board_features(our, clock.us, &ours_others[..opps.len()]));
    f[BOARD_FEATURES..PAIR_LEVEL].copy_from_slice(&board_features(paired, clock.opp(paired_idx), &theirs_others[..m]));

    let cdiff = (our.count_points() - paired.count_points()) as f32;
    f[PAIR_LEVEL] = (cdiff / 100.0).clamp(-1.0, 1.0);
    f[PAIR_LEVEL + 1] = opps.len() as f32 / 4.0;
    f[PAIR_LEVEL + 2] = opps.iter().map(total_progress).fold(0.0, f32::max);
    f[PAIR_LEVEL + 3] = opps.iter().map(|s| s.strikes).max().unwrap_or(0) as f32 / 3.0;
    f[PAIR_LEVEL + 4] = opps.iter().map(lockable_rows).sum::<u8>() as f32 / 8.0;
    f
}

/// Panic with the migration command if `model` was built for another input size.
pub fn check_input_dims<B: Backend>(model: &PairModel<B>, dir: &str) {
    let d_in = model.layer1.weight.dims()[0];
    if d_in != PAIR_FEATURES {
        panic!(
            "pair model in {dir} has {d_in} inputs but this build expects {PAIR_FEATURES}; \
             convert it with `qwixxer pair-migrate --from {dir} --to <new dir>`"
        );
    }
}

/// Position of old (45-input layout) input `old` in the 59-input layout.
pub fn migrate_input_index(old: usize) -> usize {
    match old {
        0..=19 => old,
        20..=39 => old + 7,
        40..=44 => old + 14,
        _ => panic!("old pair layout has 45 inputs, got index {old}"),
    }
}

/// Widen a 45-input model to the 59-input layout: old first-layer rows move to
/// their new positions, the 14 new rows are zero, everything else is copied.
/// The new inputs therefore contribute exactly 0.0 — outputs are unchanged.
pub fn migrate_model(
    old: &PairModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
) -> PairModel<MyBackend> {
    let (w_old, _, d_in, d_out) = extract_linear(&old.layer1);
    assert_eq!(d_in, 45, "migrate_model expects a 45-input model");
    let mut w_new = vec![0.0f32; PAIR_FEATURES * d_out];
    for i in 0..45 {
        let j = migrate_input_index(i);
        w_new[j * d_out..(j + 1) * d_out].copy_from_slice(&w_old[i * d_out..(i + 1) * d_out]);
    }
    let mut new = PairModelConfig::new().init::<MyBackend>(device);
    new.layer1.weight = burn::module::Param::from_tensor(
        Tensor::<MyBackend, 1>::from_floats(w_new.as_slice(), device).reshape([PAIR_FEATURES, d_out]),
    );
    new.layer1.bias = old.layer1.bias.clone();
    new.layer2 = old.layer2.clone();
    new.output_mean = old.output_mean.clone();
    new.output_log_var = old.output_log_var.clone();
    new
}

/// Load the 45-input model in `from`, migrate it, save it to `to`.
pub fn migrate_dir(from: &str, to: &str) {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let old = PairModelConfig::new()
        .with_input(45)
        .init::<MyBackend>(&device)
        .load_file(format!("{from}/model"), &CompactRecorder::new(), &device)
        .expect("failed to load the 45-input pair model");
    let new = migrate_model(&old, &device);
    std::fs::create_dir_all(to).expect("cannot create the target directory");
    new.save_file(format!("{to}/model"), &CompactRecorder::new())
        .expect("failed to save the migrated model");
}

// ---- Model ----

#[derive(Module, Debug)]
pub struct PairModel<B: Backend> {
    pub layer1: Linear<B>,
    pub layer2: Linear<B>,
    pub output_mean: Linear<B>,
    pub output_log_var: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct PairModelConfig {
    #[config(default = 128)]
    pub hidden1: usize,
    #[config(default = 64)]
    pub hidden2: usize,
    /// Input size; older checkpoints (45) are loaded only by `pair-migrate`.
    #[config(default = 59)]
    pub input: usize,
}

impl PairModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PairModel<B> {
        PairModel {
            layer1: LinearConfig::new(self.input, self.hidden1).with_bias(true).init(device),
            layer2: LinearConfig::new(self.hidden1, self.hidden2)
                .with_bias(true)
                .init(device),
            output_mean: LinearConfig::new(self.hidden2, 1).with_bias(true).init(device),
            output_log_var: LinearConfig::new(self.hidden2, 1).with_bias(true).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> PairModel<B> {
    /// Forward pass returning `[batch, 2]`: col 0 = μ_diff (future-space),
    /// col 1 = raw `log σ²_diff` (clamped by callers / the loss).
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.layer1.forward(input));
        let x = self.activation.forward(self.layer2.forward(x));
        let mean = self.output_mean.forward(x.clone());
        let log_var = self.output_log_var.forward(x);
        Tensor::cat(vec![mean, log_var], 1)
    }
}

/// Run the model on a batch of pair-feature vectors in one forward pass.
/// Returns `(μ, log σ²)` per input row.
pub fn pair_batch_forward(
    model: &PairModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    features_list: &[[f32; PAIR_FEATURES]],
) -> Vec<(f32, f32)> {
    if features_list.is_empty() {
        return Vec::new();
    }
    let n = features_list.len();
    let flat: Vec<f32> = features_list.iter().flat_map(|f| f.iter().copied()).collect();
    let input = Tensor::<MyBackend, 1>::from_floats(flat.as_slice(), device).reshape([n, PAIR_FEATURES]);
    let output = model.forward(input);
    let values = output.into_data().to_vec::<f32>().unwrap();
    (0..n).map(|i| (values[2 * i], values[2 * i + 1])).collect()
}

// ---- Manual inference path ----

/// Plain-array forward pass extracted from a [`PairModel`]. At our sizes
/// (batches of tens to a few hundred rows through a 45→128→64→2 MLP),
/// burn's matmul spends more on cache-tile packing and tensor glue than on
/// the arithmetic; this direct implementation skips all of it. Inference
/// only — training keeps burn's autodiff. Numerics differ from burn in the
/// last ulps (different summation order), so equivalence is validated by a
/// tolerance test, not byte equality.
/// Hidden sizes, fixed to [`PairModelConfig`]'s defaults. `from_model`
/// asserts the loaded model matches; generalize to const generics only if
/// the architecture ever actually varies.
const D_H1: usize = 128;
const D_H2: usize = 64;
/// Output-tile width in [`layer_forward`]: 32 floats = 4 AVX2 vectors of
/// accumulators held in registers across each input loop.
const TILE: usize = 32;

pub struct ManualPairNet {
    /// Input-major (burn's native layout): `w1[i * D_H1..][..D_H1]` are the
    /// weights input i contributes to every neuron. This makes the inner
    /// loop an axpy over independent outputs — vectorizable — instead of a
    /// serial reduction chain, which LLVM must not reorder for f32.
    w1: Box<[f32]>,
    b1: Box<[f32]>,
    w2: Box<[f32]>,
    b2: Box<[f32]>,
    /// Heads interleaved per h2 input: `[w_mu[i], w_lv[i]]` pairs.
    w_heads: Box<[f32]>,
    b_mu: f32,
    b_lv: f32,
}

/// Extract a burn Linear's weights (input-major, as stored) plus its bias.
/// Returns `(weights, bias, d_in, d_out)`.
fn extract_linear(layer: &Linear<MyBackend>) -> (Vec<f32>, Vec<f32>, usize, usize) {
    let dims = layer.weight.shape().dims;
    let (d_in, d_out) = (dims[0], dims[1]);
    let w = layer.weight.val().into_data().to_vec::<f32>().unwrap();
    let b = layer.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().unwrap();
    (w, b, d_in, d_out)
}

/// One dense layer + ReLU, computed in output tiles of [`TILE`] floats: the
/// tile's accumulators stay in vector registers across the whole input loop
/// (one store per tile, instead of a load-modify-store per input — the axpy
/// formulation's bottleneck). Inputs that are exactly 0.0 (common: fresh
/// rows, locked flags) skip their contribution. `w` is input-major:
/// `w[i * D_OUT..][..D_OUT]` are the weights input i contributes to every
/// output.
fn layer_forward<const D_IN: usize, const D_OUT: usize>(x: &[f32; D_IN], w: &[f32], b: &[f32], out: &mut [f32; D_OUT]) {
    for t in 0..D_OUT / TILE {
        let mut acc: [f32; TILE] = b[t * TILE..(t + 1) * TILE].try_into().unwrap();
        for (i, &xi) in x.iter().enumerate() {
            if xi != 0.0 {
                let w: &[f32; TILE] = w[i * D_OUT + t * TILE..i * D_OUT + (t + 1) * TILE].try_into().unwrap();
                for k in 0..TILE {
                    acc[k] += xi * w[k];
                }
            }
        }
        for (k, &a) in acc.iter().enumerate() {
            out[t * TILE + k] = a.max(0.0);
        }
    }
}

impl ManualPairNet {
    pub fn from_model(model: &PairModel<MyBackend>) -> Self {
        let (w1, b1, d_in1, d_h1) = extract_linear(&model.layer1);
        assert_eq!((d_in1, d_h1), (PAIR_FEATURES, D_H1));
        let (w2, b2, d_in2, d_h2) = extract_linear(&model.layer2);
        assert_eq!((d_in2, d_h2), (D_H1, D_H2));
        let (w_mu, b_mu, d_mu_in, d_mu_out) = extract_linear(&model.output_mean);
        assert_eq!((d_mu_in, d_mu_out), (D_H2, 1));
        let (w_lv, b_lv, d_lv_in, d_lv_out) = extract_linear(&model.output_log_var);
        assert_eq!((d_lv_in, d_lv_out), (D_H2, 1));
        let w_heads: Vec<f32> = w_mu.iter().zip(&w_lv).flat_map(|(&m, &l)| [m, l]).collect();
        ManualPairNet {
            w1: w1.into_boxed_slice(),
            b1: b1.into_boxed_slice(),
            w2: w2.into_boxed_slice(),
            b2: b2.into_boxed_slice(),
            w_heads: w_heads.into_boxed_slice(),
            b_mu: b_mu[0],
            b_lv: b_lv[0],
        }
    }

    /// Returns `(μ, log σ²)` per row, like [`pair_batch_forward`].
    pub fn forward(&self, rows: &[[f32; PAIR_FEATURES]]) -> Vec<(f32, f32)> {
        let mut h1 = [0.0f32; D_H1];
        let mut h2 = [0.0f32; D_H2];
        rows.iter()
            .map(|x| {
                layer_forward(x, &self.w1, &self.b1, &mut h1);
                layer_forward(&h1, &self.w2, &self.b2, &mut h2);

                let mut mu = self.b_mu;
                let mut lv = self.b_lv;
                for (i, &hi) in h2.iter().enumerate() {
                    mu += hi * self.w_heads[2 * i];
                    lv += hi * self.w_heads[2 * i + 1];
                }
                (mu, lv)
            })
            .collect()
    }
}

// ---- Strategy ----

pub struct PairStrategy {
    pub model: Arc<PairModel<MyBackend>>,
    pub device: burn::backend::ndarray::NdArrayDevice,
    /// Fast inference path; rebuilt from `model` at construction.
    pub net: Arc<ManualPairNet>,
}

impl std::fmt::Debug for PairStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "PairStrategy")
    }
}

impl PairStrategy {
    pub fn load(artifact_dir: &str) -> Self {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new()
            .init::<MyBackend>(&device)
            .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
            .unwrap_or_else(|e| {
                panic!(
                    "failed to load pair model from {artifact_dir}: {e} \
                     (a model from before the 59-input layout needs `qwixxer pair-migrate`)"
                )
            });
        check_input_dims(&model, artifact_dir);
        Self::from_model(model, device)
    }

    pub fn from_model(model: PairModel<MyBackend>, device: burn::backend::ndarray::NdArrayDevice) -> Self {
        Self::from_shared(Arc::new(model), device)
    }

    pub fn from_shared(model: Arc<PairModel<MyBackend>>, device: burn::backend::ndarray::NdArrayDevice) -> Self {
        let net = Arc::new(ManualPairNet::from_model(&model));
        PairStrategy { model, device, net }
    }
}

impl Bot for PairStrategy {
    fn evaluate(&self, our_state: &State, opp_states: &[State], clock: EvalClock) -> f32 {
        self.evaluate_batch(&[*our_state], opp_states, clock)[0]
    }

    fn evaluate_batch(&self, candidates: &[State], opp_states: &[State], clock: EvalClock) -> Vec<f32> {
        self.evaluate_batch_multi(&[(candidates, opp_states, clock)])
            .pop()
            .unwrap()
    }

    fn evaluate_batch_multi(&self, groups: &[(&[State], &[State], EvalClock)]) -> Vec<Vec<f32>> {
        let default_opps = [State::default()];
        let mut leaders: Vec<isize> = Vec::with_capacity(groups.len());
        let mut feats: Vec<[f32; PAIR_FEATURES]> = Vec::new();
        for (candidates, opp_states, clock) in groups {
            // Solo fallback: rank against an empty default board. Solo play is
            // officially unsupported for the pair bot (the old DQN covers it).
            let opps: &[State] = if opp_states.is_empty() {
                &default_opps
            } else {
                opp_states
            };
            // Same tie-breaking as before (`max_by_key` keeps the last maximum).
            let leader = (0..opps.len()).max_by_key(|&i| opps[i].count_points()).unwrap();
            for c in *candidates {
                feats.push(pair_features(c, leader, opps, *clock));
            }
            leaders.push(opps[leader].count_points());
        }
        let values = self.net.forward(&feats);

        let mut out = Vec::with_capacity(groups.len());
        let mut idx = 0;
        for ((candidates, _, _), leader_points) in groups.iter().zip(leaders) {
            let group = candidates
                .iter()
                .map(|cand| {
                    let (mu, log_var) = values[idx];
                    idx += 1;
                    let cdiff = (cand.count_points() - leader_points) as f32;
                    let sigma = (0.5 * log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)).exp();
                    (cdiff + mu) / sigma
                })
                .collect();
            out.push(group);
        }
        out
    }
}

impl crate::strategy::search::WinProb for PairStrategy {
    fn win_z_multi(&self, groups: &[(&State, &[State], EvalClock)]) -> Vec<f32> {
        let default_opps = [State::default()];
        let mut feats = Vec::with_capacity(groups.len());
        let mut cdiffs = Vec::with_capacity(groups.len());
        for (our, opps, clock) in groups {
            let opps: &[State] = if opps.is_empty() { &default_opps } else { opps };
            let leader = (0..opps.len()).max_by_key(|&i| opps[i].count_points()).unwrap();
            feats.push(pair_features(our, leader, opps, *clock));
            cdiffs.push((our.count_points() - opps[leader].count_points()) as f32);
        }
        self.net
            .forward(&feats)
            .into_iter()
            .zip(cdiffs)
            .map(|((mu, log_var), cdiff)| {
                let sigma = (0.5 * log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)).exp();
                (cdiff + mu) / sigma
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Mark;

    #[test]
    fn wprob_zero_when_free_pointer_on_unlockable_terminal() {
        // Red (ascending): mark 2, 3, 11 -> free = 12 (terminal) with only 3 marks.
        let mut s = State::default();
        for n in [2u8, 3, 11] {
            s.apply_mark(Mark { row: 0, number: n });
        }
        assert_eq!(s.row_free_values()[0], Some(12));
        assert_eq!(s.row_totals()[0], 3);
        let f = board_features(&s, BoardClock::default(), &[]);
        assert_eq!(f[12], 0.0, "terminal with <5 marks must contribute 0 ways");

        // A reachable state with 5 marks whose free pointer rests on the
        // terminal (2,3,4,5,11 -> free = 12, total = 5): terminal now lockable,
        // so it contributes nonzero ways. (We rebuild because once the free
        // pointer reaches 12 the earlier numbers can no longer be marked.)
        let mut s = State::default();
        for n in [2u8, 3, 4, 5, 11] {
            s.apply_mark(Mark { row: 0, number: n });
        }
        assert_eq!(s.row_free_values()[0], Some(12));
        assert_eq!(s.row_totals()[0], 5);
        let f = board_features(&s, BoardClock::default(), &[]);
        assert!(f[12] > 0.0);

        // Row 2 (descending): descending rows mark high->low, so mark
        // 12, 11, 3 in that order -> free = 2 (terminal) with only 3 marks.
        let mut s = State::default();
        for n in [12u8, 11, 3] {
            s.apply_mark(Mark { row: 2, number: n });
        }
        assert_eq!(s.row_free_values()[2], Some(2));
        assert_eq!(s.row_totals()[2], 3);
        let f = board_features(&s, BoardClock::default(), &[]);
        assert_eq!(f[14], 0.0, "descending terminal with <5 marks must contribute 0 ways");
    }

    #[test]
    fn pair_features_fresh_1v1_golden() {
        let a = State::default();
        let b = State::default();
        let f = pair_features(&a, 0, &[b], EvalClock::after_own_turn(2));

        // Both board blocks identical for fresh boards, except turn position:
        // after our turn the opponent rolls next (0) and we roll in 1.
        assert_eq!(f[..20], f[BOARD_FEATURES..BOARD_FEATURES + 20]);
        assert_eq!(f[21..BOARD_FEATURES], f[BOARD_FEATURES + 21..PAIR_LEVEL]);
        assert!((f[20] - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(f[BOARD_FEATURES + 20], 0.0);
        // Nothing pending; forced-strike risk of a fresh board; nobody can lock.
        assert_eq!(f[21], 0.0);
        assert_eq!(f[22], crate::threat::forced_strike_prob(&a));
        assert_eq!(f[23..27], [0.0; 4]);
        // Fresh rows: progress 0, marks 0, not locked.
        for i in 0..4 {
            assert_eq!(f[i], 0.0);
            assert_eq!(f[4 + i], 0.0);
            assert_eq!(f[8 + i], 0.0);
        }
        // Fresh free pointers sit on 2 (asc) / 12 (desc): 1 way each ->
        // (1/6) * (0+1)/11 = 1/66.
        for i in 0..4 {
            assert!((f[12 + i] - 1.0 / 66.0).abs() < 1e-6);
        }
        assert_eq!(f[16], 0.0); // strikes
        assert_eq!(f[17], 0.0); // blanks
        assert!(f[18] > 0.0); // aggregate wprob
        assert_eq!(f[19], 0.0); // lockable rows
                                // Pair-level: equal scores, 1 opponent, fresh opponent stats.
        assert_eq!(f[PAIR_LEVEL], 0.0);
        assert_eq!(f[PAIR_LEVEL + 1], 0.25);
        assert_eq!(f[PAIR_LEVEL + 2], 0.0);
        assert_eq!(f[PAIR_LEVEL + 3], 0.0);
        assert_eq!(f[PAIR_LEVEL + 4], 0.0);
    }

    #[test]
    fn pair_features_swap_relationship() {
        // Asymmetric position: A has marks + a strike, B is fresh.
        let mut a = State::default();
        for n in [2u8, 3, 5] {
            a.apply_mark(Mark { row: 0, number: n });
        }
        a.apply_strike();
        let b = State::default();

        let clock = EvalClock::after_own_turn(2);
        let ab = pair_features(&a, 0, &[b], clock);
        let ba = pair_features(&b, 0, &[a], clock.reframe_to_opp(0, 1));

        // Board blocks exchange (each board keeps its own clock).
        assert_eq!(ab[..BOARD_FEATURES], ba[BOARD_FEATURES..PAIR_LEVEL]);
        assert_eq!(ab[BOARD_FEATURES..PAIR_LEVEL], ba[..BOARD_FEATURES]);
        // current_diff negates.
        assert!((ab[PAIR_LEVEL] + ba[PAIR_LEVEL]).abs() < 1e-6);
        assert!(ab[PAIR_LEVEL] != 0.0);
        // Opponent summaries describe the respective opponent.
        assert_eq!(ab[PAIR_LEVEL + 2], 0.0); // A's opponent (B) is fresh
        assert!((ba[PAIR_LEVEL + 2] - total_progress(&a)).abs() < 1e-6);
        assert!((ba[PAIR_LEVEL + 3] - 1.0 / 3.0).abs() < 1e-6); // A has 1 strike
    }

    #[test]
    fn evaluate_batch_shape_and_determinism() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device);

        let mut cand = State::default();
        cand.apply_mark(Mark { row: 1, number: 5 });
        let candidates = [State::default(), cand];
        let opps = [State::default()];

        let v1 = bot.evaluate_batch(&candidates, &opps, EvalClock::default());
        let v2 = bot.evaluate_batch(&candidates, &opps, EvalClock::default());
        assert_eq!(v1.len(), 2);
        assert_eq!(v1, v2);
        assert!(v1.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn manual_forward_matches_burn_within_tolerance() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);
        let net = ManualPairNet::from_model(&model);

        // Real-position rows plus synthetic rows covering the input range.
        let mut a = State::default();
        for n in [2u8, 4, 7] {
            a.apply_mark(Mark { row: 0, number: n });
        }
        a.apply_strike();
        let mut b = State::default();
        b.apply_mark(Mark { row: 3, number: 8 });
        let mut rows = vec![
            pair_features(&a, 0, &[b], EvalClock::after_own_turn(2)),
            pair_features(&b, 0, &[a], EvalClock::mid_turn(2, 1)),
            pair_features(&State::default(), 0, &[a, b], EvalClock::mid_turn(3, 2)),
        ];
        for k in 0..8 {
            let mut f = [0.0f32; PAIR_FEATURES];
            for (i, v) in f.iter_mut().enumerate() {
                *v = ((i * (k + 3)) % 23) as f32 / 23.0 - 0.4;
            }
            rows.push(f);
        }

        let burn_out = pair_batch_forward(&model, &device, &rows);
        let manual_out = net.forward(&rows);
        assert_eq!(burn_out.len(), manual_out.len());
        for (i, ((bm, bl), (mm, ml))) in burn_out.iter().zip(&manual_out).enumerate() {
            assert!((bm - mm).abs() < 1e-3, "row {i}: mu {bm} vs {mm}");
            assert!((bl - ml).abs() < 1e-3, "row {i}: log_var {bl} vs {ml}");
        }
    }

    #[test]
    fn evaluate_batch_multi_matches_per_group_calls() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device);

        let mut a = State::default();
        a.apply_mark(Mark { row: 0, number: 4 });
        let mut b = State::default();
        b.apply_mark(Mark { row: 2, number: 9 });
        let mut opp = State::default();
        opp.apply_mark(Mark { row: 1, number: 6 });

        let g1_c = [State::default(), a];
        let g1_o = [opp];
        let g2_c = [b];
        let g2_o = [State::default(), a]; // different leader situation

        let c = EvalClock::default();
        let multi = bot.evaluate_batch_multi(&[(&g1_c[..], &g1_o[..], c), (&g2_c[..], &g2_o[..], c)]);
        let solo1 = bot.evaluate_batch(&g1_c, &g1_o, c);
        let solo2 = bot.evaluate_batch(&g2_c, &g2_o, c);
        assert_eq!(multi.len(), 2);
        assert_eq!(multi[0], solo1);
        assert_eq!(multi[1], solo2);
    }

    #[test]
    fn evaluate_handles_empty_opponents() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device);
        let v = bot.evaluate(&State::default(), &[], EvalClock::turn_start(1, 0));
        assert!(v.is_finite());
    }

    #[test]
    fn win_prob_is_phi_of_win_z() {
        use crate::strategy::search::{phi, WinProb};
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device);
        let mut a = State::default();
        a.apply_mark(Mark { row: 0, number: 4 });
        let b = State::default();
        let c = EvalClock::after_own_turn(2);
        let groups: [(&State, &[State], EvalClock); 2] = [(&a, &[b], c), (&b, &[a], c)];
        let zs = bot.win_z_multi(&groups);
        let ps = bot.win_prob_multi(&groups);
        for (z, p) in zs.iter().zip(&ps) {
            assert_eq!(phi(*z), *p);
        }
    }

    #[test]
    fn win_prob_multi_is_probability_and_monotone_in_score() {
        use crate::strategy::search::WinProb;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device);

        // Same opponent; in the second group we are far ahead on points.
        let behind = State::default();
        let mut ahead = State::default();
        for n in 2..=8 {
            ahead.apply_mark(Mark { row: 0, number: n });
        }
        let opp = State::default();

        let c = EvalClock::after_own_turn(2);
        let p = bot.win_prob_multi(&[(&behind, &[opp], c), (&ahead, &[opp], c)]);
        assert_eq!(p.len(), 2);
        assert!(p.iter().all(|x| (0.0..=1.0).contains(x)));
        // A 28-point lead with the same opponent must not be rated worse.
        assert!(p[1] >= p[0]);
    }

    use crate::strategy::BoardClock;

    #[test]
    fn new_board_inputs_are_appended() {
        let s = State::from_parts(1, [(5, Some(11)), (0, Some(2)), (0, Some(12)), (0, Some(12))]);
        let other = State::from_parts(0, [(6, Some(12)), (0, Some(2)), (0, Some(12)), (0, Some(12))]);
        let clock = BoardClock {
            rolls_in: 2,
            pending: true,
        };
        let f = board_features(&s, clock, &[&other]);
        assert_eq!(f[20], 2.0 / 3.0);
        assert_eq!(f[21], 1.0);
        assert_eq!(f[22], crate::threat::forced_strike_prob(&s));
        assert_eq!(f[23..27], crate::threat::lock_threats(&s, &[&other]));
        assert!(f[23] > 0.0, "other player can lock red");
    }

    #[test]
    fn pair_features_are_invariant_to_opponent_order() {
        let a = State::from_parts(0, [(3, Some(7)), (0, Some(2)), (5, Some(4)), (0, Some(12))]);
        let b = State::from_parts(1, [(5, Some(12)), (2, Some(5)), (0, Some(12)), (1, Some(10))]);
        let c = State::from_parts(0, [(1, Some(4)), (6, Some(11)), (2, Some(9)), (0, Some(12))]);
        let clock = EvalClock::mid_turn(3, 2);
        let f1 = pair_features(&a, 0, &[b, c], clock);
        let swapped = EvalClock::from_boards(clock.us, &[clock.opp(1), clock.opp(0)]);
        let f2 = pair_features(&a, 1, &[c, b], swapped);
        assert_eq!(f1, f2);
    }

    #[test]
    #[should_panic(expected = "pair-migrate")]
    fn loading_an_old_45_input_model_names_the_migration() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let old = PairModelConfig::new().with_input(45).init::<MyBackend>(&device);
        check_input_dims(&old, "some_dir");
    }

    #[test]
    fn migration_index_map_follows_the_new_layout() {
        assert_eq!(migrate_input_index(0), 0);
        assert_eq!(migrate_input_index(19), 19);
        assert_eq!(migrate_input_index(20), 27);
        assert_eq!(migrate_input_index(39), 46);
        assert_eq!(migrate_input_index(40), 54);
        assert_eq!(migrate_input_index(44), 58);
    }

    #[test]
    fn migrated_model_ignores_new_inputs_and_keeps_old_outputs() {
        use rand::{Rng, SeedableRng};
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let old = PairModelConfig::new().with_input(45).init::<MyBackend>(&device);
        let new = migrate_model(&old, &device);
        check_input_dims(&new, "test");
        let mut rng = rand::rngs::SmallRng::seed_from_u64(5);
        for _ in 0..20 {
            let x45: Vec<f32> = (0..45).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let mut x59 = [0.0f32; PAIR_FEATURES];
            for (i, &v) in x45.iter().enumerate() {
                x59[migrate_input_index(i)] = v;
            }
            // Arbitrary values in the new slots must not matter.
            for i in (20..27).chain(47..54) {
                x59[i] = rng.gen_range(0.0..1.0);
            }
            let a = old.forward(Tensor::<MyBackend, 1>::from_floats(x45.as_slice(), &device).reshape([1, 45]));
            let b =
                new.forward(Tensor::<MyBackend, 1>::from_floats(x59.as_slice(), &device).reshape([1, PAIR_FEATURES]));
            let (a, b) = (
                a.into_data().to_vec::<f32>().unwrap(),
                b.into_data().to_vec::<f32>().unwrap(),
            );
            for (u, v) in a.iter().zip(&b) {
                assert!((u - v).abs() < 1e-5, "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn five_player_pair_features_do_not_panic() {
        let s: Vec<State> = (0..5u8)
            .map(|i| State::from_parts(0, [(i, Some(2 + i)), (0, Some(2)), (0, Some(12)), (0, Some(12))]))
            .collect();
        let f = pair_features(&s[0], 3, &s[1..], EvalClock::mid_turn(5, 2));
        assert!(f.iter().all(|x| x.is_finite()));
        // rolls_in 4 (seat 3 when seat 4 rolls next) clamps to 1.
        let g = pair_features(&s[0], 0, &s[1..], EvalClock::turn_start(5, 1));
        assert_eq!(g[20], 1.0);
    }
}
