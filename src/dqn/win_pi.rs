//! Gated policy iteration for the win net: the champion's rollouts label its
//! own games, a warm-started candidate is fitted to a window of recent
//! rounds, and a sequential paired head-to-head decides promotion.
//!
//! Design: docs/superpowers/specs/2026-10-17-win-policy-iteration-design.md,
//! docs/superpowers/specs/2026-10-17-win-candidate-distillation-design.md

use crate::dqn::pair::PairStrategy;
use crate::dqn::win::{sigmoid, ManualWinNet, WinModel, WinModelConfig, WinStrategy, WIN_INPUTS};
use crate::dqn::win_distill::diff_stats;
use crate::dqn::win_train::{bench_vs_pair, outcome, rollout_round_by, train_epochs, Ad, CandCfg};
use crate::dqn::MyBackend;
use crate::game::{Game, Player};
use crate::strategy::search::GATE_MARGIN;
use crate::strategy::{Bot, Strategy};
use burn::{module::AutodiffModule, prelude::*, record::CompactRecorder};
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::path::Path;

/// Seed purposes for [`round_seed`].
pub const POS: u64 = 1;
pub const FIT: u64 = 2;
pub const GATE: u64 = 3;

pub fn splitmix64(mut x: u64) -> u64 {
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

/// `win-gate`: `games` paired games (both seats per dice set) of the win
/// net in dir `a` against the win net in dir `b`, or Master from
/// `pair_dir` when `b` is "master". Returns (games played, a's share, z).
pub fn win_gate(a: &str, b: &str, games: usize, seed: u64, pair_dir: &str) -> (usize, f64, f64) {
    let pairs = (games / 2) as u64;
    let a = WinStrategy::load(a);
    let scores = if b == "master" {
        head_to_head(&|| a.clone(), &|| PairStrategy::load(pair_dir), 0..pairs, seed)
    } else {
        let b = WinStrategy::load(b);
        head_to_head(&|| a.clone(), &|| b.clone(), 0..pairs, seed)
    };
    let (_, share, z) = gate_decision(&scores, f64::INFINITY, true);
    (scores.len() * 2, share, z)
}

/// Learning rate of a from-scratch round 1 (no `--init`), as `win-fit`.
pub const FRESH_LR: f64 = 1e-3;

/// What labels a round's data: turn ends only, or also the top candidates
/// of the decisions win-search would search
/// (docs/superpowers/specs/2026-10-17-win-candidate-distillation-design.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Labels {
    Turn,
    Cand,
}

pub struct WinPiCfg {
    pub dir: String,
    /// Starting champion (a win model dir); `None` = Master plays round 1.
    pub init: Option<String>,
    /// Width of a from-scratch net; ignored with `init`.
    pub width: usize,
    pub rounds: usize,
    /// New turn-end positions per round.
    pub positions: usize,
    /// Rollouts per position.
    pub rollouts: usize,
    /// Rounds of positions the fit trains on.
    pub window: usize,
    pub epochs: usize,
    /// Warm-start learning rate (rounds with a win-net champion).
    pub lr: f64,
    pub gate_z: f64,
    pub gate_block: usize,
    pub gate_max: usize,
    /// Yardstick games against Master per round (0 = none).
    pub master_games: usize,
    pub seed: u64,
    pub pair_dir: String,
    pub labels: Labels,
    /// Full-game rollouts per candidate (`Labels::Cand`).
    pub cand_rollouts: usize,
    /// Copies of each candidate sample (`Labels::Cand`).
    pub cand_m: usize,
}

impl WinPiCfg {
    /// Every check that can fail, before any work.
    pub fn validate(&self) -> Result<(), String> {
        if self.rounds == 0 {
            return Err("--rounds must be at least 1".into());
        }
        if self.rollouts == 0 {
            return Err("--rollouts must be at least 1".into());
        }
        if self.window == 0 {
            return Err("--window must be at least 1".into());
        }
        if self.gate_block < 2 || self.gate_block % 2 != 0 {
            return Err("--gate-block must be even and at least 2".into());
        }
        if self.gate_max < self.gate_block {
            return Err("--gate-max must be at least --gate-block".into());
        }
        if self.cand_rollouts < 2 {
            return Err("--cand-rollouts must be at least 2".into());
        }
        if self.cand_m == 0 {
            return Err("--cand-m must be at least 1".into());
        }
        if !Path::new(&format!("{}/model.mpk", self.pair_dir)).exists() {
            return Err(format!("pair model not found at {}/model.mpk", self.pair_dir));
        }
        if let Some(init) = &self.init {
            for f in ["config.json", "model.mpk"] {
                if !Path::new(&format!("{init}/{f}")).exists() {
                    return Err(format!("--init {init}: {f} not found"));
                }
            }
            match WinModelConfig::load(format!("{init}/config.json")) {
                Ok(c) if c.is_thermo() => {
                    return Err(format!(
                        "--init {init}: win-pi trains plain nets only (this is a thermometer net)"
                    ))
                }
                Ok(_) => {}
                Err(e) => return Err(format!("--init {init}: unreadable config.json ({e:?})")),
            }
        }
        if Path::new(&self.dir).exists() {
            return Err(format!(
                "{} already exists: move it aside or pick another --dir",
                self.dir
            ));
        }
        Ok(())
    }
}

/// Write a loadable win model dir (`config.json`, `model.mpk`).
pub fn save_win(dir: &str, mcfg: &WinModelConfig, model: &WinModel<MyBackend>) {
    std::fs::create_dir_all(dir).expect("Failed to create the win model dir");
    mcfg.save(format!("{dir}/config.json"))
        .expect("Failed to save win model config");
    model
        .clone()
        .save_file(format!("{dir}/model"), &CompactRecorder::new())
        .expect("Failed to save win model");
}

pub(crate) fn load_win_ad(dir: &str) -> (WinModelConfig, WinModel<Ad>) {
    let device = Default::default();
    let mcfg = WinModelConfig::load(format!("{dir}/config.json")).expect("Failed to load win model config");
    let model = mcfg
        .init::<Ad>(&device)
        .load_file(format!("{dir}/model"), &CompactRecorder::new(), &device)
        .expect("Failed to load win model");
    (mcfg, model)
}

/// The spec's loop. Returns the final champion, or `None` when a
/// from-scratch round 1 fails to beat Master (nothing to continue from).
pub fn policy_iteration(cfg: &WinPiCfg) -> Option<WinStrategy> {
    if let Err(e) = cfg.validate() {
        panic!("{e}");
    }
    let device = Default::default();
    let start = std::time::Instant::now();
    std::fs::create_dir_all(&cfg.dir).expect("Failed to create the run dir");
    let log_path = format!("{}/training.csv", cfg.dir);
    std::fs::write(
        &log_path,
        "round,positions,cand_decisions,cand_samples,diff_rmse,diff_noise,flip_share,\
         loss,gate_games,share_vs_champion,z,promoted,share_vs_master,elapsed_s\n",
    )
    .expect("Failed to write log");
    let log = |row: String| {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&log_path)
            .expect("Failed to open log");
        writeln!(f, "{row}").ok();
    };
    let pct = |s: Option<f64>| s.map(|s| format!("{:.2}%", s * 100.0)).unwrap_or_else(|| "-".into());
    let csv = |s: Option<f64>| s.map(|s| format!("{s:.4}")).unwrap_or_default();
    let yardstick = |m: &WinModel<Ad>| {
        (cfg.master_games > 0)
            .then(|| bench_vs_pair(&WinStrategy::from_model(&m.valid()), &cfg.pair_dir, cfg.master_games))
    };
    let champion_dir = format!("{}/champion", cfg.dir);

    // `None` = Master is the champion (round 1 of a from-scratch run).
    let (mcfg, mut champion): (WinModelConfig, Option<WinModel<Ad>>) = match &cfg.init {
        Some(dir) => {
            let (mcfg, model) = load_win_ad(dir);
            (mcfg, Some(model))
        }
        None => (WinModelConfig::new().with_width(cfg.width), None),
    };
    if let Some(m) = &champion {
        save_win(&champion_dir, &mcfg, &m.valid());
        if cfg.master_games > 0 {
            let s = yardstick(m);
            println!("Round  0 (init): vs Master {}", pct(s));
            log(format!("0{},{},{}", ",".repeat(11), csv(s), start.elapsed().as_secs()));
        }
    }

    let mut window: VecDeque<Vec<([f32; WIN_INPUTS], f32)>> = VecDeque::new();
    for round in 1..=cfg.rounds {
        // 1. Generate: the champion's own turn ends, labelled by its rollouts;
        // with `--labels cand`, also the top candidates of the decisions
        // win-search would search in the same games.
        let pos_seed = round_seed(cfg.seed, round, POS);
        let cand = (cfg.labels == Labels::Cand).then_some(CandCfg {
            k: cfg.cand_rollouts,
            margin: GATE_MARGIN,
        });
        let round_data = match &champion {
            None => rollout_round_by(
                cfg.positions,
                pos_seed,
                &|| PairStrategy::load(&cfg.pair_dir),
                &cfg.pair_dir,
                cfg.rollouts,
                1.0,
                cand,
            ),
            Some(m) => {
                let win = WinStrategy::from_model(&m.valid());
                rollout_round_by(
                    cfg.positions,
                    pos_seed,
                    &|| win.clone(),
                    &cfg.pair_dir,
                    cfg.rollouts,
                    1.0,
                    cand,
                )
            }
        };

        // 2. Diagnose: the champion's error on the new candidates'
        // differences (none while Master is the champion).
        let diag = champion.as_ref().and_then(|m| {
            let net = ManualWinNet::from_model(&m.valid());
            let preds: Vec<Vec<f32>> = round_data
                .groups
                .iter()
                .map(|g| {
                    let xs: Vec<[f32; WIN_INPUTS]> = g.samples.iter().map(|s| s.0).collect();
                    net.logits(&xs).into_iter().map(sigmoid).collect()
                })
                .collect();
            diff_stats(&preds, &round_data.groups, cfg.cand_rollouts)
        });
        let cand_decisions = round_data.groups.len();
        let mut data = round_data.positions;
        let turn_ends = data.len();
        for g in &round_data.groups {
            for _ in 0..cfg.cand_m {
                data.extend_from_slice(&g.samples);
            }
        }
        let cand_samples = data.len() - turn_ends;
        let diag_csv = diag
            .as_ref()
            .map(|d| format!("{:.4},{:.4},{:.4}", d.diff_rmse, d.diff_noise, d.flip_share))
            .unwrap_or_else(|| ",,".into());
        let cand_csv = match cfg.labels {
            Labels::Turn => ",,,,".to_string(),
            Labels::Cand => format!("{cand_decisions},{cand_samples},{diag_csv}"),
        };
        let cand_note = match cfg.labels {
            Labels::Turn => String::new(),
            Labels::Cand => format!(
                ", {cand_decisions} decisions ({cand_samples} samples){}",
                diag.as_ref()
                    .map(|d| format!(
                        ", diff RMSE {:.4} (noise {:.4}), flips {:.1}%",
                        d.diff_rmse,
                        d.diff_noise,
                        d.flip_share * 100.0
                    ))
                    .unwrap_or_default()
            ),
        };

        // 3. Window: every round's data enters, whatever its gate decides.
        push_window(&mut window, data, cfg.window);
        let train: Vec<_> = window.iter().flatten().copied().collect();

        // 4. Fit: warm start from the champion, or from scratch against Master.
        let fit_seed = round_seed(cfg.seed, round, FIT);
        let (init_model, lr) = match &champion {
            Some(m) => (m.clone(), cfg.lr),
            None => {
                <Ad as Backend>::seed(&device, fit_seed);
                (mcfg.init::<Ad>(&device), FRESH_LR)
            }
        };
        let (candidate, loss) = train_epochs(init_model, &train, cfg.epochs, lr, fit_seed);
        save_win(&format!("{}/round-{round}", cfg.dir), &mcfg, &candidate.valid());

        // 5. Gate: candidate vs champion, paired, sequential.
        let cand = WinStrategy::from_model(&candidate.valid());
        let gate_seed = round_seed(cfg.seed, round, GATE);
        let g = match &champion {
            None => run_gate(
                &|| cand.clone(),
                &|| PairStrategy::load(&cfg.pair_dir),
                gate_seed,
                cfg.gate_z,
                cfg.gate_block,
                cfg.gate_max,
            ),
            Some(m) => {
                let champ = WinStrategy::from_model(&m.valid());
                run_gate(
                    &|| cand.clone(),
                    &|| champ.clone(),
                    gate_seed,
                    cfg.gate_z,
                    cfg.gate_block,
                    cfg.gate_max,
                )
            }
        };
        let verdict = if g.promoted { "promoted" } else { "kept" };
        let row = |master: Option<f64>| {
            format!(
                "{round},{},{cand_csv},{loss:.5},{},{:.4},{:.2},{},{},{}",
                train.len(),
                g.games,
                g.share,
                g.z,
                g.promoted as u8,
                csv(master),
                start.elapsed().as_secs()
            )
        };
        if g.promoted {
            save_win(&champion_dir, &mcfg, &candidate.valid());
            champion = Some(candidate);
        } else if champion.is_none() {
            println!(
                "Round {round:>2}: fresh fit lost the gate to Master ({:.2}%, z {:.2}): no win net to continue from",
                g.share * 100.0,
                g.z
            );
            log(row(None));
            return None;
        }

        // 6. Yardstick (logged only) and 7. log.
        let master = yardstick(champion.as_ref().expect("a win-net champion after round 1"));
        println!(
            "Round {round:>2}: {} positions{cand_note}, loss {loss:.4}, gate {} games {:.2}% z {:.2} → {verdict}, champion vs Master {}, {}s",
            train.len(),
            g.games,
            g.share * 100.0,
            g.z,
            pct(master),
            start.elapsed().as_secs()
        );
        log(row(master));
    }
    champion.map(|m| WinStrategy::from_model(&m.valid()))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    const HEADER: &str = "round,positions,cand_decisions,cand_samples,diff_rmse,diff_noise,flip_share,\
                          loss,gate_games,share_vs_champion,z,promoted,share_vs_master,elapsed_s";

    fn tmp(name: &str) -> String {
        let d = std::env::temp_dir().join(format!("qwixxer-win-pi-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d.to_string_lossy().into_owned()
    }

    fn tiny_cfg(dir: String, init: Option<String>) -> WinPiCfg {
        WinPiCfg {
            dir,
            init,
            width: 4,
            rounds: 2,
            positions: 30,
            rollouts: 2,
            window: 2,
            epochs: 1,
            lr: 3e-4,
            gate_z: 2.0,
            gate_block: 4,
            gate_max: 8,
            master_games: 2,
            seed: 42,
            pair_dir: "pair_model".into(),
            labels: Labels::Turn,
            cand_rollouts: 2,
            cand_m: 1,
        }
    }

    #[test]
    fn validate_refuses_bad_settings_before_any_work() {
        let base = tmp("validate");
        let ok = tiny_cfg(format!("{base}/run"), None);
        assert_eq!(ok.validate(), Ok(()));
        let err = |c: WinPiCfg| c.validate().unwrap_err();
        assert!(err(WinPiCfg {
            rollouts: 0,
            ..tiny_cfg(ok.dir.clone(), None)
        })
        .contains("--rollouts"));
        assert!(err(WinPiCfg {
            window: 0,
            ..tiny_cfg(ok.dir.clone(), None)
        })
        .contains("--window"));
        assert!(err(WinPiCfg {
            rounds: 0,
            ..tiny_cfg(ok.dir.clone(), None)
        })
        .contains("--rounds"));
        assert!(err(WinPiCfg {
            gate_block: 3,
            ..tiny_cfg(ok.dir.clone(), None)
        })
        .contains("--gate-block"));
        assert!(err(WinPiCfg {
            gate_max: 2,
            ..tiny_cfg(ok.dir.clone(), None)
        })
        .contains("--gate-max"));
        assert!(err(tiny_cfg(ok.dir.clone(), Some(format!("{base}/nope")))).contains("--init"));
        assert!(err(WinPiCfg {
            pair_dir: format!("{base}/nope"),
            ..tiny_cfg(ok.dir.clone(), None)
        })
        .contains("pair model"));
        assert!(err(WinPiCfg {
            cand_rollouts: 1,
            ..tiny_cfg(ok.dir.clone(), None)
        })
        .contains("--cand-rollouts"));
        assert!(err(WinPiCfg {
            cand_m: 0,
            ..tiny_cfg(ok.dir.clone(), None)
        })
        .contains("--cand-m"));
        std::fs::create_dir_all(&ok.dir).unwrap();
        assert!(err(tiny_cfg(ok.dir.clone(), None)).contains("already exists"));
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn win_pi_refuses_a_thermometer_init() {
        let base = tmp("thermo-init");
        let init = format!("{base}/init");
        let device = Default::default();
        let mcfg = WinModelConfig::new().with_width(4).with_thermo(Some(true));
        save_win(&init, &mcfg, &mcfg.init::<MyBackend>(&device));
        let err = tiny_cfg(format!("{base}/run"), Some(init)).validate().unwrap_err();
        assert!(err.contains("plain nets only"), "{err}");
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn two_rounds_from_an_init_net_write_the_run() {
        let base = tmp("smoke");
        let init = init_net(&base);
        let cfg = tiny_cfg(format!("{base}/run"), Some(init));
        assert!(policy_iteration(&cfg).is_some());
        for d in ["round-1", "round-2", "champion"] {
            assert!(
                std::path::Path::new(&format!("{}/{d}/model.mpk", cfg.dir)).exists(),
                "{d}"
            );
            assert!(
                std::path::Path::new(&format!("{}/{d}/config.json", cfg.dir)).exists(),
                "{d}"
            );
        }
        let csv = std::fs::read_to_string(format!("{}/training.csv", cfg.dir)).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], HEADER);
        assert_eq!(lines.len(), 4, "{csv}");
        assert!(lines[1].starts_with("0,,,,,,,,,,,,"), "{}", lines[1]);
        assert_eq!(lines[1].split(',').count(), 14, "{}", lines[1]);
        for (i, row) in lines[2..].iter().enumerate() {
            let f: Vec<&str> = row.split(',').collect();
            assert_eq!(f.len(), 14, "{row}");
            assert_eq!(f[0], (i + 1).to_string());
            // --labels turn: no candidate columns.
            assert!(f[2..7].iter().all(|c| c.is_empty()), "{row}");
            assert!(f[11] == "0" || f[11] == "1", "{row}");
            assert!(!f[12].is_empty(), "{row}");
        }
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn a_fresh_start_that_loses_to_master_stops_cleanly() {
        let base = tmp("fresh");
        // A barely trained W=4 net cannot beat Master over 8 games with z > 2.
        let cfg = WinPiCfg {
            master_games: 0,
            ..tiny_cfg(format!("{base}/run"), None)
        };
        assert!(policy_iteration(&cfg).is_none());
        assert!(std::path::Path::new(&format!("{}/round-1/model.mpk", cfg.dir)).exists());
        assert!(!std::path::Path::new(&format!("{}/champion", cfg.dir)).exists());
        let csv = std::fs::read_to_string(format!("{}/training.csv", cfg.dir)).unwrap();
        let rows: Vec<&str> = csv.lines().skip(1).collect();
        assert_eq!(rows.len(), 1, "{csv}");
        assert_eq!(rows[0].split(',').nth(11), Some("0"));
        std::fs::remove_dir_all(&base).unwrap();
    }

    fn init_net(base: &str) -> String {
        let init = format!("{base}/init");
        let device = Default::default();
        <MyBackend as Backend>::seed(&device, 3);
        let mcfg = WinModelConfig::new().with_width(4);
        let model: WinModel<MyBackend> = mcfg.init(&device);
        save_win(&init, &mcfg, &model);
        init
    }

    #[test]
    fn candidate_rounds_log_their_decisions_and_headroom() {
        let base = tmp("cand");
        let cfg = WinPiCfg {
            labels: Labels::Cand,
            cand_m: 2,
            rounds: 1,
            ..tiny_cfg(format!("{base}/run"), Some(init_net(&base)))
        };
        assert!(policy_iteration(&cfg).is_some());
        let csv = std::fs::read_to_string(format!("{}/training.csv", cfg.dir)).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3, "{csv}");
        let f: Vec<&str> = lines[2].split(',').collect();
        assert_eq!(f.len(), 14, "{}", lines[2]);
        let decisions: usize = f[2].parse().unwrap();
        let samples: usize = f[3].parse().unwrap();
        assert!(decisions > 0, "{}", lines[2]);
        // 2–3 candidates per decision, each emitted cand_m = 2 times.
        assert!(samples >= 4 * decisions && samples <= 6 * decisions, "{}", lines[2]);
        assert_eq!(samples % 2, 0);
        assert!(f[4..7].iter().all(|c| !c.is_empty()), "{}", lines[2]);
        // The window total counts the turn ends and the candidate samples.
        assert_eq!(f[1].parse::<usize>().unwrap(), cfg.positions + samples);
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn a_fresh_candidate_run_has_no_champion_to_diagnose() {
        let base = tmp("fresh-cand");
        let cfg = WinPiCfg {
            labels: Labels::Cand,
            master_games: 0,
            ..tiny_cfg(format!("{base}/run"), None)
        };
        assert!(policy_iteration(&cfg).is_none());
        let csv = std::fs::read_to_string(format!("{}/training.csv", cfg.dir)).unwrap();
        let row = csv.lines().nth(1).unwrap();
        let f: Vec<&str> = row.split(',').collect();
        assert!(!f[2].is_empty() && !f[3].is_empty(), "{row}");
        assert!(f[4..7].iter().all(|c| c.is_empty()), "{row}");
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn win_gate_of_a_net_against_itself_is_even() {
        let base = tmp("gate");
        let dir = format!("{base}/net");
        let device = Default::default();
        let mcfg = WinModelConfig::new().with_width(4).with_thermo(Some(true));
        save_win(&dir, &mcfg, &mcfg.init::<MyBackend>(&device));
        assert_eq!(win_gate(&dir, &dir, 8, 7, "pair_model"), (8, 0.5, 0.0));
        let (games, share, _) = win_gate(&dir, "master", 4, 7, "pair_model");
        assert!(games == 4 && (0.0..=1.0).contains(&share));
        std::fs::remove_dir_all(&base).unwrap();
    }
}
