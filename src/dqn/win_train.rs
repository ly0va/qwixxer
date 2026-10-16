//! Win-net training: TD(λ) self-play with fresh targets, and the capacity
//! check. Only compiled with the `dqn` feature.
//!
//! Design: docs/superpowers/specs/2026-10-15-win-net-design.md

use crate::dqn::pair::PairStrategy;
use crate::dqn::win::WinModelConfig;
use crate::dqn::win::{input_tensor, sigmoid, swap_input, win_input, ManualWinNet, WinModel, WinStrategy, WIN_INPUTS};
use crate::dqn::{MyBackend, TRAIN_SEED};
use crate::game::{Game, Player};
use crate::state::{Mark, State};
use crate::strategy::search::{passive_entry, passive_first_active, phase2_entry, rollout_final_states, WinProb};
use crate::strategy::sim::SimGame;
use crate::strategy::{active_phase1_impl, active_phase2_impl, passive_phase1_impl, Bot, Strategy};
use burn::{
    backend::Autodiff,
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::{activation::log_sigmoid, ElementConversion},
};
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

pub type Ad = Autodiff<MyBackend>;

/// Default TD(λ) trace decay, as for the pair net.
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
pub fn fresh_targets(net: &ManualWinNet, chains: &[&Chain], lambda: f32) -> Vec<([f32; WIN_INPUTS], f32)> {
    chains
        .par_iter()
        .flat_map_iter(|c| {
            let ps: Vec<f32> = net.logits(&c.inputs).into_iter().map(sigmoid).collect();
            let g = lambda_returns(&ps, c.outcome, lambda);
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

/// Exploration rate for iteration `it` (0-based): start·0.95^it, never below
/// `floor` (the pair trainer's schedule is start 0.2, floor 0.07).
pub fn epsilon_at(it: usize, start: f32, floor: f32) -> f32 {
    (start * 0.95f32.powi(it as i32)).max(floor)
}

/// Learning rate for iteration `it` (0-based) of a `total`-iteration run:
/// cosine from `lr` at the first iteration to `lr_end` at the last.
pub fn lr_at(it: usize, total: usize, lr: f64, lr_end: f64) -> f64 {
    if total <= 1 {
        return lr;
    }
    let t = it.min(total - 1) as f64 / (total - 1) as f64;
    lr_end + 0.5 * (lr - lr_end) * (1.0 + (std::f64::consts::PI * t).cos())
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

// ---- Bench and self-play loop ----

/// Fixed paired game set (same pattern as the pair trainer's GA bench).
const BENCH_SEED: u64 = 0xB54C;

/// The win bot's share against Master (`pair_dir`) over `games` paired
/// 2-player games: seats alternate and each pair shares its dice; ties ½.
pub fn bench_vs_pair(win: &WinStrategy, pair_dir: &str, games: usize) -> f64 {
    let points: f64 = (0..games)
        .into_par_iter()
        .map_init(
            || PairStrategy::load(pair_dir),
            |pair, i| {
                let (pair_idx, rot) = ((i / 2) as u64, i % 2);
                let dice = |seat: u64| Box::new(SmallRng::seed_from_u64(BENCH_SEED.wrapping_add(pair_idx * 2 + seat)));
                let w: Box<dyn Strategy> = Box::new(win.clone());
                let p: Box<dyn Strategy> = Box::new(PairStrategy {
                    model: pair.model.clone(),
                    device: pair.device.clone(),
                    net: pair.net.clone(),
                });
                let players = if rot == 0 {
                    vec![Player::new(w, dice(0)), Player::new(p, dice(1))]
                } else {
                    vec![Player::new(p, dice(0)), Player::new(w, dice(1))]
                };
                let mut game = Game::new(players);
                game.play();
                let s: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
                outcome(s[rot], s[1 - rot]) as f64
            },
        )
        .sum();
    points / games.max(1) as f64
}

pub struct WinTrainCfg {
    pub dir: String,
    pub iterations: usize,
    pub games: usize,
    pub epochs: usize,
    /// Bench games against Master per iteration (0 = none).
    pub bench: usize,
    pub width: usize,
    /// Iterations of games kept in the replay buffer.
    pub buffer: usize,
    /// Resume: load `dir`'s model and continue the ε schedule from here.
    pub start_iteration: usize,
    /// Learning rate at the first iteration; cosine-decays to `lr_end` at
    /// the last (`start_iteration + iterations - 1`).
    pub lr: f64,
    pub lr_end: f64,
    /// ε at iteration 0, and the lowest ε the schedule decays to.
    pub epsilon_start: f32,
    pub epsilon_floor: f32,
    /// TD(λ) trace decay for the targets: 0 = one-step bootstrap, 1 = outcome only.
    pub lambda: f32,
    /// Warm start: begin a fresh run from this model dir (its config and
    /// weights; `width` is ignored).
    pub init: Option<String>,
    /// Master's model dir, for the per-iteration bench.
    pub pair_dir: String,
}

/// Spec "Training": per iteration, play `games` self-play games with the
/// current net, rebuild every buffered chain's targets with it, train
/// `epochs` epochs, save `model.mpk` and `iter-N.mpk`, bench against Master
/// and append to `training.csv`. Returns the final bot.
pub fn self_play_train(cfg: &WinTrainCfg) -> WinStrategy {
    // Fail before any work: a long run must not die at its first bench, nor
    // a fresh start silently overwrite a finished one.
    if cfg.bench > 0 {
        assert!(
            std::path::Path::new(&format!("{}/model.mpk", cfg.pair_dir)).exists(),
            "pair model not found at {}/model.mpk (needed for the bench; use --bench 0 to skip)",
            cfg.pair_dir
        );
    }
    assert!(
        cfg.init.is_none() || cfg.start_iteration == 0,
        "--init starts a fresh run; it cannot be combined with --start-iteration"
    );
    if cfg.start_iteration == 0 {
        let taken = ["model.mpk", "training.csv"]
            .iter()
            .any(|f| std::path::Path::new(&format!("{}/{f}", cfg.dir)).exists());
        assert!(
            !taken,
            "{} already holds a run: move it aside, or resume with --start-iteration",
            cfg.dir
        );
    }
    let device = Default::default();
    <Ad as Backend>::seed(&device, TRAIN_SEED);
    std::fs::create_dir_all(&cfg.dir).expect("Failed to create the model dir");
    let config_path = format!("{}/config.json", cfg.dir);
    let log_path = format!("{}/training.csv", cfg.dir);
    let load = |dir: &str| -> WinModel<Ad> {
        WinModelConfig::load(format!("{dir}/config.json"))
            .expect("Failed to load win model config")
            .init::<Ad>(&device)
            .load_file(format!("{dir}/model"), &CompactRecorder::new(), &device)
            .expect("Failed to load win model")
    };
    let mut model: WinModel<Ad> = if cfg.start_iteration > 0 {
        load(&cfg.dir)
    } else {
        let mcfg = match &cfg.init {
            Some(init) => {
                WinModelConfig::load(format!("{init}/config.json")).expect("Failed to load init model config")
            }
            None => WinModelConfig::new().with_width(cfg.width),
        };
        mcfg.save(&config_path).expect("Failed to save win model config");
        std::fs::write(&log_path, "iteration,games,lr,loss,share_vs_master,elapsed_s\n").expect("Failed to write log");
        match &cfg.init {
            Some(init) => load(init),
            None => mcfg.init::<Ad>(&device),
        }
    };
    let start = std::time::Instant::now();
    let bench = |model: &WinModel<Ad>| {
        (cfg.bench > 0).then(|| bench_vs_pair(&WinStrategy::from_model(&model.valid()), &cfg.pair_dir, cfg.bench))
    };
    let log = |row: String| {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&log_path)
            .expect("Failed to open log");
        writeln!(f, "{row}").ok();
    };
    let pct = |share: Option<f64>| {
        share
            .map(|s| format!("{:.2}%", s * 100.0))
            .unwrap_or_else(|| "-".into())
    };
    let csv = |share: Option<f64>| share.map(|s| format!("{s:.4}")).unwrap_or_default();
    if cfg.init.is_some() {
        // Iteration 0: the warm start's own strength, the bar to beat.
        let share = bench(&model);
        println!("Iteration   0 (init): vs Master {}", pct(share));
        log(format!("0,0,,,{},{}", csv(share), start.elapsed().as_secs()));
    }
    let mut buffer: std::collections::VecDeque<Vec<Chain>> = std::collections::VecDeque::new();

    let total = cfg.start_iteration + cfg.iterations;
    for it in cfg.start_iteration..total {
        let lr = lr_at(it, total, cfg.lr, cfg.lr_end);
        let epsilon = epsilon_at(it, cfg.epsilon_start, cfg.epsilon_floor);
        let net = Arc::new(ManualWinNet::from_model(&model.valid()));
        let chains: Vec<Chain> = (0..cfg.games)
            .into_par_iter()
            .flat_map_iter(|g| {
                let seed = TRAIN_SEED.wrapping_add(((it * cfg.games + g) as u64).wrapping_mul(1000));
                self_play_game(&net, epsilon, seed)
            })
            .collect();
        buffer.push_back(chains);
        while buffer.len() > cfg.buffer {
            buffer.pop_front();
        }
        let refs: Vec<&Chain> = buffer.iter().flatten().collect();
        let data = fresh_targets(&net, &refs, cfg.lambda);
        let (trained, loss) = train_epochs(model, &data, cfg.epochs, lr, TRAIN_SEED.wrapping_add(it as u64));
        model = trained;

        let inference = model.valid();
        inference
            .clone()
            .save_file(format!("{}/model", cfg.dir), &CompactRecorder::new())
            .expect("Failed to save win model");
        std::fs::copy(
            format!("{}/model.mpk", cfg.dir),
            format!("{}/iter-{}.mpk", cfg.dir, it + 1),
        )
        .expect("Failed to save checkpoint");

        let share = bench(&model);
        let elapsed = start.elapsed().as_secs();
        println!(
            "Iteration {:>3}: ε {epsilon:.3}, lr {lr:.1e}, {} positions, loss {loss:.4}, vs Master {}, {elapsed}s",
            it + 1,
            data.len(),
            pct(share)
        );
        log(format!(
            "{},{},{lr:.3e},{loss:.5},{},{elapsed}",
            it + 1,
            cfg.games,
            csv(share)
        ));
    }
    WinStrategy::from_model(&model.valid())
}

// ---- Capacity check ----

/// Post-move positions from Master-vs-Master games (both seats, no
/// exploration), labelled with today's net's P(win). Games are seeded from
/// `seed` in fixed blocks, so the result is deterministic.
pub fn master_positions(count: usize, seed: u64, pair_dir: &str) -> Vec<([f32; WIN_INPUTS], f32)> {
    const BLOCK: u64 = 256;
    let mut out = Vec::with_capacity(count);
    let mut block = 0u64;
    while out.len() < count {
        let chunk: Vec<Vec<([f32; WIN_INPUTS], f32)>> = (0..BLOCK)
            .into_par_iter()
            .map_init(
                || PairStrategy::load(pair_dir),
                |pair, g| {
                    let mk = || PairStrategy::from_shared(pair.model.clone(), pair.device.clone());
                    let game_seed = seed.wrapping_add((block * BLOCK + g).wrapping_mul(1000));
                    let (snaps, _) = play_recorded([mk(), mk()], [0.0, 0.0], game_seed);
                    let posts: Vec<(State, State)> = snaps.into_iter().flatten().collect();
                    let groups: Vec<(&State, &[State])> =
                        posts.iter().map(|(me, opp)| (me, std::slice::from_ref(opp))).collect();
                    let ps = pair.win_prob_multi(&groups);
                    posts
                        .iter()
                        .zip(ps)
                        .map(|((me, opp), p)| (win_input(me, opp), p))
                        .collect()
                },
            )
            .collect();
        out.extend(chunk.into_iter().flatten());
        block += 1;
    }
    out.truncate(count);
    out
}

/// One labelled snapshot: the (me, opp) the net sees, and where a rollout
/// of it starts (all-player boards with the current turn completed, whether
/// the game is over, and the seat that acts first).
struct Snap {
    seen: (State, State),
    entry: (Vec<State>, bool),
    first_active: usize,
}

/// Plays a bot greedily and records a [`Snap`] at the pair recorder's
/// cadence (as `play_recorded`): after each active phase 2, and after each
/// passive decision that had a white move to consider.
struct Snapshots<B: Bot> {
    bot: B,
    recorded: Rc<RefCell<Vec<Snap>>>,
}

impl<B: Bot> std::fmt::Debug for Snapshots<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Snapshots")
    }
}

impl<B: Bot> Strategy for Snapshots<B> {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        active_phase1_impl(&self.bot, state, opp_states, dice)
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let mark = active_phase2_impl(&self.bot, state, opp_states, dice, has_marked);
        let mut no_mark = *state;
        if !has_marked {
            no_mark.apply_strike();
        }
        let post = after(state, mark, no_mark);
        self.recorded.borrow_mut().push(Snap {
            seen: (post, opp_states[0]),
            entry: phase2_entry(post, &opp_states[..1]),
            first_active: 1,
        });
        mark
    }

    fn passive_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], active: usize) -> Option<Mark> {
        if state.generate_white_moves(dice[0] + dice[1]).is_empty() {
            return None;
        }
        let mark = passive_phase1_impl(&self.bot, state, opp_states, dice);
        self.recorded.borrow_mut().push(Snap {
            seen: (after(state, mark, *state), opp_states[0]),
            entry: passive_entry(&self.bot, state, opp_states, dice, active, mark),
            first_active: passive_first_active(active, 2),
        });
        mark
    }
}

/// P(seat 0 of each entry finishes ahead, ties ½) by `k` full-game CRN
/// rollouts with `bot` in every seat, grouped by the seat that acts first.
/// A finished entry's label is its exact result.
fn entry_labels(bot: &PairStrategy, entries: &[(Vec<State>, bool)], first: &[usize], k: usize, seed: u64) -> Vec<f32> {
    let mut labels = vec![0.0; entries.len()];
    for fa in 0..2 {
        let idx: Vec<usize> = (0..entries.len()).filter(|&i| first[i] == fa).collect();
        if idx.is_empty() {
            continue;
        }
        let group: Vec<(Vec<State>, bool)> = idx.iter().map(|&i| entries[i].clone()).collect();
        let finals = rollout_final_states(bot, &group, seed.wrapping_add(fa as u64), k, fa, false);
        for (&i, f) in idx.iter().zip(finals) {
            labels[i] = f.iter().map(|s| SimGame::outcome(s)).sum::<f32>() / f.len() as f32;
        }
    }
    labels
}

/// P(me finishes ahead, ties ½) for each turn-end (me, opp) by `k` rollouts
/// with `bot` in both seats, the opponent to act first.
pub fn rollout_labels(bot: &PairStrategy, positions: &[(State, State)], k: usize, seed: u64) -> Vec<f32> {
    let entries: Vec<(Vec<State>, bool)> = positions.iter().map(|(me, opp)| phase2_entry(*me, &[*opp])).collect();
    entry_labels(bot, &entries, &vec![1; entries.len()], k, seed)
}

/// [`master_positions`]' positions (same games and snapshots), labelled
/// `blend`·(Master's `k`-rollout P(win)) + (1 − `blend`)·(Master's static
/// P(win)): the play-out value Master+'s search consults. A passive
/// snapshot's rollout first completes the turn (the active player's phase 2
/// on the rolled dice, as search does). Deterministic per `seed`.
pub fn rollout_positions(
    count: usize,
    seed: u64,
    pair_dir: &str,
    k: usize,
    blend: f32,
) -> Vec<([f32; WIN_INPUTS], f32)> {
    const BLOCK: u64 = 64;
    let mut out = Vec::with_capacity(count);
    let mut block = 0u64;
    while out.len() < count {
        let chunk: Vec<Vec<([f32; WIN_INPUTS], f32)>> = (0..BLOCK)
            .into_par_iter()
            .map_init(
                || PairStrategy::load(pair_dir),
                |pair, g| {
                    let game_seed = seed.wrapping_add((block * BLOCK + g).wrapping_mul(1000));
                    let bufs: [Rc<RefCell<Vec<Snap>>>; 2] = Default::default();
                    let players: Vec<Player> = (0..2)
                        .map(|i| {
                            let bot = PairStrategy {
                                model: pair.model.clone(),
                                device: pair.device,
                                net: pair.net.clone(),
                            };
                            let rec = Snapshots {
                                bot,
                                recorded: bufs[i].clone(),
                            };
                            Player::new(
                                Box::new(rec),
                                Box::new(SmallRng::seed_from_u64(game_seed.wrapping_add(i as u64))),
                            )
                        })
                        .collect();
                    Game::new(players).play();
                    let snaps: Vec<Snap> = bufs.iter().flat_map(|b| b.take()).collect();
                    let groups: Vec<(&State, &[State])> = snaps
                        .iter()
                        .map(|s| (&s.seen.0, std::slice::from_ref(&s.seen.1)))
                        .collect();
                    let stat = pair.win_prob_multi(&groups);
                    let roll = if blend == 0.0 {
                        vec![0.0; snaps.len()]
                    } else {
                        let entries: Vec<(Vec<State>, bool)> = snaps.iter().map(|s| s.entry.clone()).collect();
                        let first: Vec<usize> = snaps.iter().map(|s| s.first_active).collect();
                        entry_labels(pair, &entries, &first, k, game_seed ^ 0x5011_0075)
                    };
                    snaps
                        .iter()
                        .zip(stat.into_iter().zip(roll))
                        .map(|(s, (st, r))| (win_input(&s.seen.0, &s.seen.1), blend * r + (1.0 - blend) * st))
                        .collect()
                },
            )
            .collect();
        out.extend(chunk.into_iter().flatten());
        block += 1;
    }
    out.truncate(count);
    out
}

/// Mean |sigmoid(logit) − target| over `data`.
pub fn mean_abs_error(net: &ManualWinNet, data: &[([f32; WIN_INPUTS], f32)]) -> f32 {
    let total: f32 = data.iter().map(|(x, p)| (sigmoid(net.logit(x)) - p).abs()).sum();
    total / data.len().max(1) as f32
}

pub struct FitCfg {
    pub train: usize,
    pub test: usize,
    pub epochs: usize,
    pub widths: Vec<usize>,
    pub lr: f64,
    /// Save each fitted net as a loadable model dir `<save>/w<W>`.
    pub save: Option<String>,
    /// 0: label positions with Master's static P(win). K > 0: label
    /// turn-end positions with K-rollout P(win) blended by `blend`
    /// (see [`rollout_positions`]).
    pub rollouts: usize,
    pub blend: f32,
}

/// Held-out MAE at which a width counts as big enough (spec "Capacity check").
pub const FIT_MAE: f32 = 0.02;

/// Spec "Capacity check": fit each width to today's P(win) on Master
/// positions and report held-out MAE. The fitted nets are discarded unless
/// `cfg.save` is set (they can then be benched as `win` bots).
pub fn capacity_check(cfg: &FitCfg) -> Vec<(usize, f32)> {
    let device = Default::default();
    println!("Generating {} training and {} held-out positions…", cfg.train, cfg.test);
    let positions = |count, seed| match cfg.rollouts {
        0 => master_positions(count, seed, "pair_model"),
        k => rollout_positions(count, seed, "pair_model", k, cfg.blend),
    };
    let train = positions(cfg.train, TRAIN_SEED);
    let test = positions(cfg.test, TRAIN_SEED ^ 0x7E57_0000_0000);
    let mut results = Vec::new();
    for &w in &cfg.widths {
        <Ad as Backend>::seed(&device, TRAIN_SEED);
        let model = WinModelConfig::new().with_width(w).init::<Ad>(&device);
        let (model, loss) = train_epochs(model, &train, cfg.epochs, cfg.lr, TRAIN_SEED);
        let inference = model.valid();
        let mae = mean_abs_error(&ManualWinNet::from_model(&inference), &test);
        if let Some(save) = &cfg.save {
            let dir = format!("{save}/w{w}");
            std::fs::create_dir_all(&dir).expect("Failed to create the fit model dir");
            WinModelConfig::new()
                .with_width(w)
                .save(format!("{dir}/config.json"))
                .expect("Failed to save win model config");
            inference
                .save_file(format!("{dir}/model"), &CompactRecorder::new())
                .expect("Failed to save fitted win model");
        }
        println!("  W = {w:>3}: train loss {loss:.4}, held-out MAE {mae:.4}");
        results.push((w, mae));
    }
    if cfg.rollouts > 0 {
        // Rollout labels carry their own sampling noise, so MAE has a floor
        // well above FIT_MAE; judge these nets by benching them instead.
        println!("(rollout labels: MAE includes label noise; bench the saved nets)");
    } else {
        match results.iter().find(|(_, mae)| *mae <= FIT_MAE) {
            Some((w, _)) => println!("Smallest width with MAE ≤ {FIT_MAE}: W = {w}"),
            None => println!("No width reached MAE ≤ {FIT_MAE}: decide before training"),
        }
    }
    results
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
        // A fresh net's logits are ~1e-3, so two random nets can give
        // near-identical targets: train the old one towards P = 0.9 first.
        let device = Default::default();
        let leaning: Vec<_> = chain.inputs.iter().map(|x| (*x, 0.9)).collect();
        let (old, _) = train_epochs(
            WinModelConfig::new().with_width(16).init::<Ad>(&device),
            &leaning,
            50,
            1e-2,
            1,
        );
        let old = ManualWinNet::from_model(&old.valid());
        let new = net(2);
        let t_old = fresh_targets(&old, &[&chain], LAMBDA);
        let t_new = fresh_targets(&new, &[&chain], LAMBDA);
        let ps: Vec<f32> = new.logits(&chain.inputs).into_iter().map(sigmoid).collect();
        let expected = lambda_returns(&ps, 1.0, LAMBDA);
        assert_eq!(t_new.len(), chain.inputs.len());
        for ((x, t), (e, input)) in t_new.iter().zip(expected.iter().zip(&chain.inputs)) {
            assert_eq!(x, input);
            assert!((t - e).abs() < 1e-6);
        }
        assert!(
            t_old.iter().zip(&t_new).any(|(a, b)| (a.1 - b.1).abs() > 1e-2),
            "targets ignored the net"
        );
    }

    #[test]
    fn fresh_targets_use_the_given_lambda() {
        let chain = Chain {
            inputs: positions(),
            outcome: 0.0,
        };
        let n = net(4);
        let ps: Vec<f32> = n.logits(&chain.inputs).into_iter().map(sigmoid).collect();
        // λ = 0: one-step targets, each the net's P of the next input.
        let t0: Vec<f32> = fresh_targets(&n, &[&chain], 0.0).into_iter().map(|(_, t)| t).collect();
        assert_eq!(&t0[..ps.len() - 1], &ps[1..]);
        assert_eq!(t0[ps.len() - 1], 0.0);
        // λ = 1: every target is the outcome.
        assert!(fresh_targets(&n, &[&chain], 1.0).iter().all(|(_, t)| *t == 0.0));
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

    #[test]
    fn tiny_training_run_writes_and_reloads_a_model() {
        let dir = std::env::temp_dir().join(format!("qwixxer-win-smoke-{}", std::process::id()));
        let dir = dir.to_str().unwrap().to_string();
        std::fs::remove_dir_all(&dir).ok();
        let cfg = WinTrainCfg {
            dir: dir.clone(),
            iterations: 2,
            games: 20,
            epochs: 1,
            bench: 4,
            width: 16,
            buffer: 3,
            start_iteration: 0,
            lr: 4e-4,
            lr_end: 4e-5,
            epsilon_start: 0.2,
            epsilon_floor: 0.07,
            lambda: LAMBDA,
            init: None,
            pair_dir: "pair_model".into(),
        };
        let trained = self_play_train(&cfg);
        for f in ["config.json", "model.mpk", "iter-1.mpk", "iter-2.mpk", "training.csv"] {
            assert!(std::path::Path::new(&format!("{dir}/{f}")).exists(), "missing {f}");
        }
        let log = std::fs::read_to_string(format!("{dir}/training.csv")).unwrap();
        assert_eq!(log.lines().count(), 3, "{log}");
        // The half-precision file must still be (nearly) the trained net.
        let loaded = WinStrategy::load(&dir);
        let xs: Vec<_> = (0..4)
            .flat_map(|s| self_play_game(&loaded.net, 0.0, s))
            .flat_map(|c| c.inputs)
            .collect();
        for (a, b) in trained.net.logits(&xs).into_iter().zip(loaded.net.logits(&xs)) {
            assert!((sigmoid(a) - sigmoid(b)).abs() < 1e-2, "trained {a} vs loaded {b}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn master_positions_are_labelled_with_todays_win_prob() {
        let data = master_positions(300, 21, "pair_model");
        assert_eq!(data.len(), 300);
        assert!(data.iter().all(|(_, p)| (0.0..=1.0).contains(p)));
        // Both sides of the game appear: labels on both sides of ½.
        assert!(data.iter().any(|(_, p)| *p > 0.5) && data.iter().any(|(_, p)| *p < 0.5));
        // Deterministic per seed.
        assert_eq!(master_positions(300, 21, "pair_model")[17].1, data[17].1);
    }

    fn marked(row: usize, count: usize, strikes: u8) -> State {
        let mut marks = [[false; 11]; 4];
        marks[row][..count].iter_mut().for_each(|m| *m = true);
        State::from_marks(&marks, strikes)
    }

    #[test]
    fn rollout_label_of_a_finished_position_is_its_result() {
        let pair = PairStrategy::load("pair_model");
        // Opponent on 4 strikes: the game is over; me 28 vs opp −20.
        let (me, opp) = (marked(0, 7, 0), marked(1, 0, 4));
        assert_eq!(rollout_labels(&pair, &[(me, opp)], 8, 1), vec![1.0]);
        assert_eq!(rollout_labels(&pair, &[(opp, me)], 8, 1), vec![0.0]);
    }

    #[test]
    fn rollout_label_of_a_big_lead_is_near_one_and_seeded() {
        let pair = PairStrategy::load("pair_model");
        let lead = (marked(0, 8, 0), marked(1, 0, 3));
        let even = (State::default(), State::default());
        let labels = rollout_labels(&pair, &[lead, even], 64, 7);
        assert!(labels[0] > 0.9, "{labels:?}");
        assert!((0.2..0.8).contains(&labels[1]), "{labels:?}");
        assert_eq!(rollout_labels(&pair, &[lead, even], 64, 7), labels);
    }

    #[test]
    fn rollout_positions_at_blend_zero_are_master_positions() {
        // Same games, same snapshots (turn ends and passive decisions), and
        // Master's static labels: only the rollouts differ.
        assert_eq!(
            rollout_positions(300, 21, "pair_model", 4, 0.0),
            master_positions(300, 21, "pair_model")
        );
    }

    #[test]
    fn rollout_labels_broadly_agree_with_master() {
        // Master's net estimates its own play-out value; a wrong perspective
        // in the rollout frame gives ~0.19 here, the correct frame ~0.075
        // (mostly K = 32 label noise). Passive labels sit ~0.05 below
        // Master's: the rollout plays the opponent's pending phase 2, which
        // Master's inputs cannot see.
        let roll = rollout_positions(400, 9, "pair_model", 32, 1.0);
        let stat = rollout_positions(400, 9, "pair_model", 32, 0.0);
        let mad = roll.iter().zip(&stat).map(|(r, s)| (r.1 - s.1).abs()).sum::<f32>() / roll.len() as f32;
        let bias = roll.iter().zip(&stat).map(|(r, s)| r.1 - s.1).sum::<f32>() / roll.len() as f32;
        assert!(
            mad < 0.1,
            "mean |rollout − static| = {mad}, mean rollout − static = {bias}"
        );
    }

    #[test]
    fn rollout_positions_blend_rollouts_with_masters_estimate() {
        let roll = rollout_positions(60, 5, "pair_model", 8, 1.0);
        let stat = rollout_positions(60, 5, "pair_model", 8, 0.0);
        let half = rollout_positions(60, 5, "pair_model", 8, 0.5);
        assert_eq!(roll.len(), 60);
        assert!(roll.iter().all(|(_, p)| (0.0..=1.0).contains(p)));
        for ((r, s), h) in roll.iter().zip(&stat).zip(&half) {
            assert_eq!((r.0, s.0), (h.0, h.0), "same positions for every blend");
            assert!((h.1 - (r.1 + s.1) / 2.0).abs() < 1e-5);
        }
        // Rollout labels are not Master's estimate.
        assert!(roll.iter().zip(&stat).any(|(r, s)| (r.1 - s.1).abs() > 0.05));
        assert_eq!(rollout_positions(60, 5, "pair_model", 8, 1.0), roll);
    }

    #[test]
    fn mean_abs_error_is_zero_against_the_nets_own_predictions() {
        let n = net(6);
        let xs = positions();
        let data: Vec<_> = xs.iter().map(|x| (*x, sigmoid(n.logit(x)))).collect();
        assert!(mean_abs_error(&n, &data) < 1e-6);
        let off: Vec<_> = data.iter().map(|(x, p)| (*x, (p + 0.1).min(1.0))).collect();
        assert!(mean_abs_error(&n, &off) > 0.05);
    }

    #[test]
    fn lr_schedule_is_cosine_from_lr_to_lr_end() {
        let (lr, end) = (4e-4, 4e-5);
        assert!((lr_at(0, 160, lr, end) - lr).abs() < 1e-12);
        assert!((lr_at(159, 160, lr, end) - end).abs() < 1e-12);
        // Halfway between the iterations, halfway between the rates.
        assert!((lr_at(0, 3, lr, end) - lr).abs() < 1e-12);
        assert!((lr_at(1, 3, lr, end) - (lr + end) / 2.0).abs() < 1e-12);
        // Monotone non-increasing.
        assert!((1..160).all(|i| lr_at(i, 160, lr, end) <= lr_at(i - 1, 160, lr, end)));
        // lr_end = lr is the constant schedule; a 1-iteration run uses lr.
        assert!((0..10).all(|i| lr_at(i, 10, lr, lr) == lr));
        assert_eq!(lr_at(0, 1, lr, end), lr);
    }

    #[test]
    fn epsilon_decays_from_point_two_to_the_floor() {
        assert_eq!(epsilon_at(0, 0.2, 0.07), 0.2);
        assert!((epsilon_at(1, 0.2, 0.07) - 0.19).abs() < 1e-6);
        // Today's schedule: on the 0.07 floor from iteration 21 on.
        assert!(epsilon_at(19, 0.2, 0.07) > 0.07);
        assert_eq!(epsilon_at(21, 0.2, 0.07), 0.07);
        // A lower floor keeps decaying past it, down to the new floor.
        assert!(epsilon_at(40, 0.2, 0.02) < 0.07 && epsilon_at(40, 0.2, 0.02) > 0.02);
        assert_eq!(epsilon_at(159, 0.2, 0.02), 0.02);
        assert_eq!(epsilon_at(500, 0.2, 0.0), 0.2 * 0.95f32.powi(500));
        // A lower start (warm starts) decays from there.
        assert_eq!(epsilon_at(0, 0.05, 0.02), 0.05);
        assert!((epsilon_at(1, 0.05, 0.02) - 0.0475).abs() < 1e-6);
    }

    #[test]
    fn warm_start_begins_from_the_init_model_and_benches_it_first() {
        let base = std::env::temp_dir().join(format!("qwixxer-win-warm-{}", std::process::id()));
        let base = base.to_str().unwrap().to_string();
        std::fs::remove_dir_all(&base).ok();
        let (init_dir, dir) = (format!("{base}/init"), format!("{base}/run"));
        let mut init_cfg = smoke_cfg(&init_dir);
        init_cfg.width = 8;
        self_play_train(&init_cfg);
        let init = WinStrategy::load(&init_dir);

        let mut cfg = smoke_cfg(&dir);
        cfg.init = Some(init_dir.clone());
        cfg.iterations = 0;
        cfg.bench = 4;
        let warm = self_play_train(&cfg);
        // The init net, unchanged, and its width (not cfg.width = 16).
        let xs = positions();
        assert_eq!(warm.net.logits(&xs), init.net.logits(&xs));
        assert_eq!(WinModelConfig::load(format!("{dir}/config.json")).unwrap().width, 8);
        // The log starts with the init model's bench as iteration 0.
        let log = std::fs::read_to_string(format!("{dir}/training.csv")).unwrap();
        let rows: Vec<&str> = log.lines().skip(1).collect();
        assert_eq!(rows.len(), 1, "{log}");
        assert!(rows[0].starts_with("0,0,"), "{log}");
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    #[should_panic(expected = "--init starts a fresh run")]
    fn warm_start_cannot_be_combined_with_resume() {
        let mut cfg = smoke_cfg("unused");
        cfg.init = Some("somewhere".into());
        cfg.start_iteration = 3;
        self_play_train(&cfg);
    }

    #[test]
    fn capacity_check_saves_each_fitted_net_as_a_loadable_model() {
        let dir = std::env::temp_dir().join(format!("qwixxer-win-fit-{}", std::process::id()));
        let dir = dir.to_str().unwrap().to_string();
        std::fs::remove_dir_all(&dir).ok();
        let cfg = FitCfg {
            train: 200,
            test: 50,
            epochs: 1,
            widths: vec![8, 12],
            lr: 1e-3,
            save: Some(dir.clone()),
            rollouts: 0,
            blend: 1.0,
        };
        let results = capacity_check(&cfg);
        let test = master_positions(cfg.test, TRAIN_SEED ^ 0x7E57_0000_0000, "pair_model");
        for (w, mae) in results {
            let bot = WinStrategy::load(&format!("{dir}/w{w}"));
            // The saved (half-precision) net is the fitted one.
            assert!((mean_abs_error(&bot.net, &test) - mae).abs() < 1e-2, "W = {w}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    fn smoke_cfg(dir: &str) -> WinTrainCfg {
        WinTrainCfg {
            dir: dir.to_string(),
            iterations: 1,
            games: 4,
            epochs: 1,
            bench: 0,
            width: 16,
            buffer: 3,
            start_iteration: 0,
            lr: 4e-4,
            lr_end: 4e-5,
            epsilon_start: 0.2,
            epsilon_floor: 0.07,
            lambda: LAMBDA,
            init: None,
            pair_dir: "pair_model".into(),
        }
    }

    #[test]
    #[should_panic(expected = "already holds a run")]
    fn fresh_start_refuses_to_overwrite_an_existing_run() {
        let dir = std::env::temp_dir().join(format!("qwixxer-win-guard-{}", std::process::id()));
        let dir = dir.to_str().unwrap().to_string();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(format!("{dir}/model.mpk"), b"precious").unwrap();
        self_play_train(&smoke_cfg(&dir));
    }

    #[test]
    #[should_panic(expected = "pair model not found")]
    fn missing_pair_model_fails_before_any_training() {
        let dir = std::env::temp_dir().join(format!("qwixxer-win-nopair-{}", std::process::id()));
        let dir = dir.to_str().unwrap().to_string();
        std::fs::remove_dir_all(&dir).ok();
        let mut cfg = smoke_cfg(&dir);
        cfg.bench = 4;
        cfg.pair_dir = "no_such_pair_model".into();
        self_play_train(&cfg);
    }
}
