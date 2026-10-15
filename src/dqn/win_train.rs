//! Win-net training: TD(λ) self-play with fresh targets, and the capacity
//! check. Only compiled with the `dqn` feature.
//!
//! Design: docs/superpowers/specs/2026-10-15-win-net-design.md

use crate::dqn::win::{input_tensor, sigmoid, swap_input, win_input, ManualWinNet, WinModel, WinStrategy, WIN_INPUTS};
use crate::dqn::MyBackend;
use crate::game::{Game, Player};
use crate::state::{Mark, State};
use crate::strategy::{active_phase1_impl, active_phase2_impl, passive_phase1_impl, Bot, Strategy};
use burn::{
    backend::Autodiff,
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{activation::log_sigmoid, ElementConversion},
};
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

pub type Ad = Autodiff<MyBackend>;

/// TD(λ) trace decay, as for the pair net.
pub const LAMBDA: f32 = 0.8;

/// One player's game: the input after each of their decisions, and the
/// outcome from their side.
#[derive(Clone, Debug)]
pub struct Chain {
    pub inputs: Vec<[f32; WIN_INPUTS]>,
    pub outcome: f32,
}

/// 1 for a win, 0 for a loss, ½ for a tie.
pub fn outcome(me: isize, opp: isize) -> f32 {
    if me > opp {
        1.0
    } else if me < opp {
        0.0
    } else {
        0.5
    }
}

/// λ-returns over one chain: `G_last = outcome`,
/// `G_t = (1 − λ)·p_{t+1} + λ·G_{t+1}`. `ps[0]` is unused.
pub fn lambda_returns(ps: &[f32], outcome: f32, lambda: f32) -> Vec<f32> {
    let n = ps.len();
    let mut g = vec![0.0f32; n];
    if n == 0 {
        return g;
    }
    g[n - 1] = outcome;
    for t in (0..n - 1).rev() {
        g[t] = (1.0 - lambda) * ps[t + 1] + lambda * g[t + 1];
    }
    g
}

/// Rebuild every chain's targets with the current net (spec "Fresh
/// targets"): stored inputs paired with their λ-returns.
pub fn fresh_targets(net: &ManualWinNet, chains: &[&Chain]) -> Vec<([f32; WIN_INPUTS], f32)> {
    chains
        .par_iter()
        .flat_map_iter(|c| {
            let ps: Vec<f32> = net.logits(&c.inputs).into_iter().map(sigmoid).collect();
            let g = lambda_returns(&ps, c.outcome, LAMBDA);
            c.inputs.iter().copied().zip(g).collect::<Vec<_>>()
        })
        .collect()
}

/// Binary cross-entropy of sigmoid(z) against soft targets t, from logits:
/// −mean(t·log σ(z) + (1 − t)·log σ(−z)).
pub fn bce_logits<B: Backend>(z: Tensor<B, 2>, t: Tensor<B, 2>) -> Tensor<B, 1> {
    let pos = log_sigmoid(z.clone());
    let neg = log_sigmoid(z.neg());
    (t.clone() * pos + t.neg().add_scalar(1.0) * neg).mean().neg()
}

/// Train on (input, soft target) pairs: Adam, batches of 1024, a seeded
/// shuffle per epoch. Returns the model and the last epoch's mean loss.
pub fn train_epochs(
    mut model: WinModel<Ad>,
    data: &[([f32; WIN_INPUTS], f32)],
    epochs: usize,
    lr: f64,
    seed: u64,
) -> (WinModel<Ad>, f32) {
    let device = Default::default();
    let mut optim = AdamConfig::new().init();
    let mut order: Vec<usize> = (0..data.len()).collect();
    let mut last = 0.0;
    for epoch in 0..epochs {
        order.shuffle(&mut SmallRng::seed_from_u64(seed.wrapping_add(epoch as u64)));
        let (mut sum, mut batches) = (0.0f32, 0usize);
        for chunk in order.chunks(1024) {
            let xs: Vec<[f32; WIN_INPUTS]> = chunk.iter().map(|&i| data[i].0).collect();
            let swapped: Vec<_> = xs.iter().map(swap_input).collect();
            let t: Vec<f32> = chunk.iter().map(|&i| data[i].1).collect();
            let z = model.forward(input_tensor::<Ad>(&xs, &device), input_tensor::<Ad>(&swapped, &device));
            let t = Tensor::<Ad, 1>::from_floats(t.as_slice(), &device).reshape([chunk.len(), 1]);
            let loss = bce_logits(z, t);
            sum += loss.clone().into_scalar().elem::<f32>();
            batches += 1;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
        }
        last = sum / batches.max(1) as f32;
    }
    (model, last)
}

// ---- Self-play recording ----

/// Plays through the shared bot pipeline and records (my post-decision
/// state, opponent) after each decision, at the pair recorder's cadence:
/// active turns once after phase 2, passive turns after every real decision
/// (skips included). ε-greedy on active decisions, greedy on passive.
struct Recorder<B: Bot> {
    bot: B,
    epsilon: f32,
    rng: SmallRng,
    recorded: Rc<RefCell<Vec<(State, State)>>>,
}

impl<B: Bot> std::fmt::Debug for Recorder<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Recorder")
    }
}

impl<B: Bot> Recorder<B> {
    fn record(&self, post: State, opps: &[State]) {
        self.recorded.borrow_mut().push((post, opps[0]));
    }

    /// Uniform over the marks plus "no mark".
    fn random(&mut self, marks: &[Mark]) -> Option<Mark> {
        let i = self.rng.gen_range(0..=marks.len());
        marks.get(i).copied()
    }
}

fn after(state: &State, mark: Option<Mark>, otherwise: State) -> State {
    match mark {
        Some(m) => {
            let mut s = *state;
            s.apply_mark(m);
            s
        }
        None => otherwise,
    }
}

impl<B: Bot> Strategy for Recorder<B> {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        if self.rng.gen::<f32>() < self.epsilon {
            let marks = state.generate_white_moves(dice[0] + dice[1]);
            return if marks.is_empty() { None } else { self.random(&marks) };
        }
        active_phase1_impl(&self.bot, state, opp_states, dice)
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        let no_mark = if has_marked {
            *state
        } else {
            let mut s = *state;
            s.apply_strike();
            s
        };
        let mark = if marks.is_empty() {
            None
        } else if self.rng.gen::<f32>() < self.epsilon {
            self.random(&marks)
        } else {
            active_phase2_impl(&self.bot, state, opp_states, dice, has_marked)
        };
        self.record(after(state, mark, no_mark), opp_states);
        mark
    }

    fn passive_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], _active: usize) -> Option<Mark> {
        if state.generate_white_moves(dice[0] + dice[1]).is_empty() {
            return None;
        }
        let mark = passive_phase1_impl(&self.bot, state, opp_states, dice);
        self.record(after(state, mark, *state), opp_states);
        mark
    }
}

/// Play one 2-player game, seat i as `bots[i]` exploring with `epsilon[i]`.
/// Returns each seat's snapshots and the final boards.
fn play_recorded<B: Bot + 'static>(
    bots: [B; 2],
    epsilon: [f32; 2],
    seed: u64,
) -> ([Vec<(State, State)>; 2], [State; 2]) {
    let bufs: [Rc<RefCell<Vec<(State, State)>>>; 2] = Default::default();
    let players: Vec<Player> = bots
        .into_iter()
        .enumerate()
        .map(|(i, bot)| {
            let recorder = Recorder {
                bot,
                epsilon: epsilon[i],
                rng: SmallRng::seed_from_u64(seed.wrapping_add(100 + i as u64)),
                recorded: bufs[i].clone(),
            };
            Player::new(
                Box::new(recorder),
                Box::new(SmallRng::seed_from_u64(seed.wrapping_add(i as u64))),
            )
        })
        .collect();
    let mut game = Game::new(players);
    game.play();
    let finals = [game.players[0].state, game.players[1].state];
    (bufs.map(|b| b.take()), finals)
}

/// One self-play game with the current net in both seats, seat 0 exploring.
pub fn self_play_game(net: &Arc<ManualWinNet>, epsilon: f32, seed: u64) -> [Chain; 2] {
    let bot = || WinStrategy { net: net.clone() };
    let (snaps, finals) = play_recorded([bot(), bot()], [epsilon, 0.0], seed);
    let scores = finals.map(|s| s.count_points());
    [0, 1].map(|i| Chain {
        inputs: snaps[i].iter().map(|(me, opp)| win_input(me, opp)).collect(),
        outcome: outcome(scores[i], scores[1 - i]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::win::{win_input, WinModelConfig};
    use crate::state::Mark;

    fn positions() -> Vec<[f32; WIN_INPUTS]> {
        let mut a = State::default();
        let mut b = State::default();
        let mut out = Vec::new();
        let ours = [(0, 2), (2, 12), (1, 4), (3, 10), (0, 6), (2, 7)];
        let theirs = [(1, 3), (3, 11), (2, 11), (0, 3), (1, 8), (3, 5)];
        for ((r, n), (q, m)) in ours.into_iter().zip(theirs) {
            a.apply_mark(Mark { row: r, number: n });
            out.push(win_input(&a, &b));
            b.apply_mark(Mark { row: q, number: m });
            out.push(win_input(&b, &a));
        }
        out
    }

    fn net(seed: u64) -> ManualWinNet {
        let device = Default::default();
        <MyBackend as Backend>::seed(&device, seed);
        ManualWinNet::from_model(&WinModelConfig::new().init::<MyBackend>(&device))
    }

    #[test]
    fn outcome_counts_ties_as_half() {
        assert_eq!(outcome(30, 20), 1.0);
        assert_eq!(outcome(20, 30), 0.0);
        assert_eq!(outcome(25, 25), 0.5);
    }

    #[test]
    fn lambda_returns_match_a_hand_computed_chain() {
        let ps = [0.9, 0.6, 0.4];
        let close = |a: Vec<f32>, b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
        assert!(close(lambda_returns(&ps, 1.0, 0.8), [0.824, 0.88, 1.0]));
        assert!(close(lambda_returns(&ps, 0.5, 0.8), [0.504, 0.48, 0.5]));
        assert!(close(lambda_returns(&ps, 0.0, 0.8), [0.184, 0.08, 0.0]));
        assert!(lambda_returns(&[], 1.0, 0.8).is_empty());
    }

    #[test]
    fn fresh_targets_follow_the_current_net() {
        let chain = Chain {
            inputs: positions(),
            outcome: 1.0,
        };
        let (old, new) = (net(1), net(2));
        let t_old = fresh_targets(&old, &[&chain]);
        let t_new = fresh_targets(&new, &[&chain]);
        let ps: Vec<f32> = new.logits(&chain.inputs).into_iter().map(sigmoid).collect();
        let expected = lambda_returns(&ps, 1.0, LAMBDA);
        assert_eq!(t_new.len(), chain.inputs.len());
        for ((x, t), (e, input)) in t_new.iter().zip(expected.iter().zip(&chain.inputs)) {
            assert_eq!(x, input);
            assert!((t - e).abs() < 1e-6);
        }
        assert!(
            t_old.iter().zip(&t_new).any(|(a, b)| (a.1 - b.1).abs() > 1e-4),
            "targets ignored the net"
        );
    }

    #[test]
    fn bce_of_an_even_logit_against_one_half_is_ln_2() {
        let device = Default::default();
        let z = Tensor::<MyBackend, 1>::from_floats([0.0f32].as_slice(), &device).reshape([1, 1]);
        let t = Tensor::<MyBackend, 1>::from_floats([0.5f32].as_slice(), &device).reshape([1, 1]);
        let loss: f32 = bce_logits(z, t).into_scalar().elem();
        assert!((loss - std::f32::consts::LN_2).abs() < 1e-5, "{loss}");
    }

    #[test]
    fn training_moves_predictions_toward_soft_targets() {
        let device = Default::default();
        <Ad as Backend>::seed(&device, 3);
        let model = WinModelConfig::new().with_width(16).init::<Ad>(&device);
        let data: Vec<_> = positions().into_iter().map(|x| (x, 0.9f32)).collect();
        let (model, loss) = train_epochs(model, &data, 300, 1e-2, 3);
        let net = ManualWinNet::from_model(&model.valid());
        let xs: Vec<_> = data.iter().map(|d| d.0).collect();
        let mean = net.logits(&xs).into_iter().map(sigmoid).sum::<f32>() / xs.len() as f32;
        assert!((mean - 0.9).abs() < 0.05, "mean prediction {mean}, loss {loss}");
    }

    #[test]
    fn self_play_game_records_both_seats_with_complementary_outcomes() {
        let net = Arc::new(net(4));
        let [c0, c1] = self_play_game(&net, 0.2, 99);
        assert!(!c0.inputs.is_empty() && !c1.inputs.is_empty());
        assert_eq!(c0.outcome + c1.outcome, 1.0);
        assert!(c0.inputs.iter().chain(&c1.inputs).flatten().all(|v| v.is_finite()));
        // Same seed, same game.
        let [d0, _] = self_play_game(&net, 0.2, 99);
        assert_eq!(c0.inputs, d0.inputs);
    }

    /// Marks and strikes only accumulate, so every snapshot of a board is
    /// bounded by that board's final state; a seat/opponent mix-up breaks it.
    fn within(snapshot: &State, final_state: &State) -> bool {
        let (a, b) = (snapshot.row_totals(), final_state.row_totals());
        (0..4).all(|r| a[r] <= b[r]) && snapshot.strikes <= final_state.strikes
    }

    #[test]
    fn recorder_pairs_each_seat_with_its_opponent() {
        let net = Arc::new(net(5));
        let bot = || WinStrategy { net: net.clone() };
        let (snaps, finals) = play_recorded([bot(), bot()], [0.0, 0.0], 7);
        for i in 0..2 {
            assert!(!snaps[i].is_empty());
            for (me, opp) in &snaps[i] {
                assert!(within(me, &finals[i]), "seat {i}: own snapshot exceeds its final board");
                assert!(
                    within(opp, &finals[1 - i]),
                    "seat {i}: opponent snapshot exceeds the opponent's final board"
                );
            }
        }
    }
}
