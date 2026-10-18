use clap::{Parser, Subcommand, ValueEnum};
use game::Player;
use qwixxer::*;

/// Pools freed memory instead of returning it to the OS: the search bot's
/// rollout driver and burn's per-forward buffers otherwise cause constant
/// cross-thread munmap/TLB-shootdown churn (~17% of cycles in
/// smp_call_function_many_cond before this).
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::Arc;

#[derive(Debug, Clone, ValueEnum)]
enum BotType {
    Ga,
    Dqn,
    Pair,
    PairSearch,
    /// Pair-search with an experimental `SearchConfig` read from the
    /// `QWIXXER_SEARCH` env var (e.g. `passive,lock=256,strat,k=144`).
    PairSearchExp,
    /// Second experimental pair-search, configured by `QWIXXER_SEARCH_B`
    /// (for A/B benches between two non-production configs).
    PairSearchExpB,
    /// Static pair bot with the model from the `QWIXXER_ALT_MODEL` directory
    /// (for head-to-head checkpoint comparisons).
    PairAlt,
    /// Production pair-search on the `QWIXXER_ALT_MODEL` checkpoint.
    PairSearchAlt,
    /// Static win net (2-player only), from `win_model/`.
    Win,
    /// Production pair-search driven by the win net (2-player only).
    WinSearch,
    Mcts,
    Opportunist,
    Conservative,
    Random,
}

impl std::fmt::Display for BotType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            BotType::Ga => write!(f, "GA"),
            BotType::Dqn => write!(f, "DQN"),
            BotType::Pair => write!(f, "PAIR"),
            BotType::PairSearch => write!(f, "PAIR-SEARCH"),
            BotType::PairSearchExp => write!(f, "PAIR-SEARCH[{}]", exp_search_config("QWIXXER_SEARCH")),
            BotType::PairSearchExpB => write!(f, "PAIR-SEARCH-B[{}]", exp_search_config("QWIXXER_SEARCH_B")),
            BotType::PairAlt => write!(f, "PAIR({})", alt_model_dir()),
            BotType::PairSearchAlt => write!(f, "PAIR-SEARCH({})", alt_model_dir()),
            BotType::Win => write!(f, "WIN({})", win_model_dir()),
            BotType::WinSearch => write!(f, "WIN-SEARCH({})", win_model_dir()),
            BotType::Mcts => write!(f, "MCTS"),
            BotType::Opportunist => write!(f, "Opportunist"),
            BotType::Conservative => write!(f, "Conservative"),
            BotType::Random => write!(f, "Random"),
        }
    }
}

fn make_strategy(bot: &BotType) -> Box<dyn strategy::Strategy> {
    let genes = Arc::new(bot::default_genes());
    match bot {
        BotType::Ga => {
            let champion =
                bot::DNA::load_weights("champion.txt", genes).expect("No champion.txt found. Run `train ga` first.");
            Box::new(champion)
        }
        BotType::Dqn => Box::new(dqn::DqnStrategy::load("dqn_model")),
        BotType::Pair => Box::new(dqn::pair::PairStrategy::load("pair_model")),
        BotType::PairSearch => Box::new(strategy::search::SearchBot::new(dqn::pair::PairStrategy::load(
            "pair_model",
        ))),
        BotType::PairAlt => Box::new(dqn::pair::PairStrategy::load(&alt_model_dir())),
        BotType::PairSearchAlt => Box::new(strategy::search::SearchBot::new(dqn::pair::PairStrategy::load(
            &alt_model_dir(),
        ))),
        BotType::PairSearchExp | BotType::PairSearchExpB => Box::new(strategy::search::SearchBot::with_config(
            dqn::pair::PairStrategy::load("pair_model"),
            exp_search_config(exp_env(bot)),
        )),
        BotType::Win => Box::new(dqn::win::WinStrategy::load(&win_model_dir())),
        BotType::WinSearch => Box::new(strategy::search::SearchBot::new(dqn::win::WinStrategy::load(
            &win_model_dir(),
        ))),
        BotType::Mcts => {
            let champion =
                bot::DNA::load_weights("champion.txt", genes).expect("No champion.txt found. Run `train ga` first.");
            Box::new(mcts::MonteCarlo::with_ga(200, champion))
        }
        BotType::Opportunist => Box::<strategy::Opportunist>::default(),
        BotType::Conservative => Box::<strategy::Conservative>::default(),
        BotType::Random => Box::new(strategy::Random),
    }
}

/// Model directory for `pair-alt` / `pair-search-alt`, from `QWIXXER_ALT_MODEL`.
fn alt_model_dir() -> String {
    std::env::var("QWIXXER_ALT_MODEL")
        .expect("pair-alt / pair-search-alt need QWIXXER_ALT_MODEL=<dir containing model.mpk>")
}

/// Model directory for `win` / `win-search`: `QWIXXER_WIN_MODEL`, else `win_model`.
fn win_model_dir() -> String {
    std::env::var("QWIXXER_WIN_MODEL").unwrap_or_else(|_| "win_model".into())
}

/// Env var holding the `SearchConfig` spec of an experimental pair-search.
fn exp_env(bot: &BotType) -> &'static str {
    match bot {
        BotType::PairSearchExpB => "QWIXXER_SEARCH_B",
        _ => "QWIXXER_SEARCH",
    }
}

/// `SearchConfig` for `pair-search-exp[-b]`, from env var `var`.
fn exp_search_config(var: &str) -> strategy::search::SearchConfig {
    let spec =
        std::env::var(var).unwrap_or_else(|_| panic!("{var} must be set, e.g. {var}=passive,lock=256,strat,k=144"));
    strategy::search::SearchConfig::parse(&spec).unwrap_or_else(|e| panic!("{var}: {e}"))
}

struct StrategyTemplates {
    dqn: Option<dqn::DqnStrategy>,
    pair: Option<dqn::pair::PairStrategy>,
    pair_alt: Option<dqn::pair::PairStrategy>,
    win: Option<dqn::win::WinStrategy>,
    champion: Option<bot::DNA>,
}

impl StrategyTemplates {
    fn new(bots: &[BotType]) -> Self {
        let needs_dqn = bots.iter().any(|b| matches!(b, BotType::Dqn));
        let needs_pair = bots.iter().any(|b| {
            matches!(
                b,
                BotType::Pair | BotType::PairSearch | BotType::PairSearchExp | BotType::PairSearchExpB
            )
        });
        let needs_pair_alt = bots
            .iter()
            .any(|b| matches!(b, BotType::PairAlt | BotType::PairSearchAlt));
        let needs_win = bots.iter().any(|b| matches!(b, BotType::Win | BotType::WinSearch));
        let needs_champion = bots.iter().any(|b| matches!(b, BotType::Ga | BotType::Mcts));
        let genes = Arc::new(bot::default_genes());
        StrategyTemplates {
            dqn: if needs_dqn {
                Some(dqn::DqnStrategy::load("dqn_model"))
            } else {
                None
            },
            pair: if needs_pair {
                Some(dqn::pair::PairStrategy::load("pair_model"))
            } else {
                None
            },
            pair_alt: if needs_pair_alt {
                Some(dqn::pair::PairStrategy::load(&alt_model_dir()))
            } else {
                None
            },
            win: if needs_win {
                Some(dqn::win::WinStrategy::load(&win_model_dir()))
            } else {
                None
            },
            champion: if needs_champion {
                Some(bot::DNA::load_weights("champion.txt", genes).expect("No champion.txt found. Run `evolve` first."))
            } else {
                None
            },
        }
    }

    fn create(&self, bot: &BotType) -> Box<dyn strategy::Strategy> {
        match bot {
            BotType::Ga => Box::new(self.champion.as_ref().unwrap().clone()),
            BotType::Dqn => {
                let t = self.dqn.as_ref().unwrap();
                Box::new(dqn::DqnStrategy::from_shared(t.model.clone(), t.device.clone()))
            }
            BotType::Pair => {
                let t = self.pair.as_ref().unwrap();
                Box::new(dqn::pair::PairStrategy::from_shared(t.model.clone(), t.device.clone()))
            }
            BotType::PairSearch => {
                let t = self.pair.as_ref().unwrap();
                Box::new(strategy::search::SearchBot::new(dqn::pair::PairStrategy::from_shared(
                    t.model.clone(),
                    t.device.clone(),
                )))
            }
            BotType::PairSearchExp | BotType::PairSearchExpB => {
                let t = self.pair.as_ref().unwrap();
                Box::new(strategy::search::SearchBot::with_config(
                    dqn::pair::PairStrategy::from_shared(t.model.clone(), t.device.clone()),
                    exp_search_config(exp_env(bot)),
                ))
            }
            BotType::PairAlt => {
                let t = self.pair_alt.as_ref().unwrap();
                Box::new(dqn::pair::PairStrategy::from_shared(t.model.clone(), t.device.clone()))
            }
            BotType::PairSearchAlt => {
                let t = self.pair_alt.as_ref().unwrap();
                Box::new(strategy::search::SearchBot::new(dqn::pair::PairStrategy::from_shared(
                    t.model.clone(),
                    t.device.clone(),
                )))
            }
            BotType::Win => Box::new(self.win.as_ref().unwrap().clone()),
            BotType::WinSearch => Box::new(strategy::search::SearchBot::new(self.win.as_ref().unwrap().clone())),
            BotType::Mcts => Box::new(mcts::MonteCarlo::with_ga(200, self.champion.as_ref().unwrap().clone())),
            BotType::Opportunist => Box::<strategy::Opportunist>::default(),
            BotType::Conservative => Box::<strategy::Conservative>::default(),
            BotType::Random => Box::new(strategy::Random),
        }
    }
}

#[allow(dead_code)]
fn bot_name(bot: &BotType, index: usize, total: usize) -> String {
    if total > 1 {
        format!("{} #{}", bot, index + 1)
    } else {
        bot.to_string()
    }
}

#[derive(Parser)]
#[command(name = "qwixxer", about = "Qwixx bot framework")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Play interactively against bots
    Play {
        /// Bot types to play against
        #[arg(default_value = "mcts")]
        bots: Vec<BotType>,
        /// Show bot decisions and boards
        #[arg(short, long)]
        verbose: bool,
    },
    /// Benchmark bots against each other
    Bench {
        /// Bot types (2+)
        bots: Vec<BotType>,
        /// Number of games
        #[arg(short, long, default_value = "1000")]
        num_games: usize,
        /// Base seed for paired dice streams (same seed = identical game set)
        #[arg(short, long, default_value = "42")]
        seed: u64,
    },
    /// Single-player score benchmark
    Solo {
        /// Number of games per strategy
        #[arg(short, long, default_value = "10000")]
        num_games: usize,
    },
    /// Evolve the GA champion via genetic algorithm
    Evolve,
    /// DQN self-play reinforcement learning
    #[cfg(feature = "dqn")]
    DqnSelfplay {
        /// Number of iterations
        #[arg(short, long, default_value = "40")]
        iterations: usize,
        /// Benchmark games per iteration (0 to disable)
        #[arg(short, long, default_value = "0")]
        bench: usize,
        /// Save per-iteration checkpoints as iter-N.mpk
        #[arg(short, long)]
        checkpoints: bool,
        /// Starting iteration offset (for epsilon schedule when resuming)
        #[arg(short, long, default_value = "0")]
        start_iteration: usize,
    },
    /// Pair-network self-play reinforcement learning
    #[cfg(feature = "dqn")]
    PairTrain {
        /// Number of iterations
        #[arg(short, long, default_value = "40")]
        iterations: usize,
        /// Games per iteration
        #[arg(short, long, default_value = "20000")]
        games: usize,
        /// Training epochs per iteration over the replay buffer
        #[arg(short, long, default_value = "10")]
        epochs: usize,
        /// Benchmark games per iteration (0 to disable)
        #[arg(short, long, default_value = "0")]
        bench: usize,
        /// Save per-iteration checkpoints as iter-N.mpk
        #[arg(short, long)]
        checkpoints: bool,
        /// Starting iteration offset (for epsilon schedule when resuming)
        #[arg(long, default_value = "0")]
        start_iteration: usize,
        /// Generate games with the search bot as every player's policy
        /// (expert iteration)
        #[arg(long)]
        search: bool,
        /// Emit rollout-derived value targets at gated and lock-forced
        /// decisions (search-value distillation)
        #[arg(long)]
        distill: bool,
        /// Full-game rollouts per distill candidate
        #[arg(long, default_value = "32")]
        distill_k: usize,
        /// Distill samples per (candidate, opponent) at gated decisions
        #[arg(long, default_value = "2")]
        distill_m: usize,
        /// Distill samples per (candidate, opponent) at lock firings
        /// (weighted higher: firings are ~15x rarer than gated decisions)
        #[arg(long, default_value = "16")]
        distill_m_lock: usize,
        /// Probability of declining a forced safe lock during generation
        /// (0 = current behavior; the run recipe uses 0.05)
        #[arg(long, default_value = "0.0")]
        epsilon_lock: f32,
        /// Also emit distill targets at gated passive (white-sum) decisions,
        /// which production search searches since Phase 21
        #[arg(long)]
        distill_passive: bool,
    },
    /// Win-net self-play training (2-player TD(λ) on win/loss, fresh targets)
    #[cfg(feature = "dqn")]
    WinTrain {
        /// Number of iterations
        #[arg(long, default_value = "160")]
        iterations: usize,
        /// Games per iteration
        #[arg(long, default_value = "5000")]
        games: usize,
        /// Training epochs per iteration over the replay buffer
        #[arg(long, default_value = "2")]
        epochs: usize,
        /// Bench games against Master per iteration (0 to disable)
        #[arg(long, default_value = "200000")]
        bench: usize,
        /// Learning rate at the first iteration
        #[arg(long, default_value = "4e-4")]
        lr: f64,
        /// Learning rate at the last iteration (cosine decay from --lr; set equal to --lr for constant)
        #[arg(long, default_value = "4e-5")]
        lr_end: f64,
        /// Exploration rate at the first iteration (decays ×0.95 per iteration)
        #[arg(long, default_value = "0.2")]
        epsilon_start: f32,
        /// Lowest exploration rate the schedule decays to
        #[arg(long, default_value = "0.07")]
        epsilon_floor: f32,
        /// TD(λ) trace decay: 0 = bootstrap from the next position only, 1 = game outcome only
        #[arg(long, default_value = "0.8")]
        lambda: f32,
        /// Warm start: begin a fresh run from this model dir (e.g. win_fit/w64); --width is then ignored
        #[arg(long)]
        init: Option<String>,
        /// Encoder width W (pick it with `win-fit`)
        #[arg(long, default_value = "32")]
        width: usize,
        /// Iterations of games kept in the replay buffer
        #[arg(long, default_value = "3")]
        buffer: usize,
        /// Resume from win_model/ at this iteration (continues the ε schedule)
        #[arg(long, default_value = "0")]
        start_iteration: usize,
    },
    /// Win-net capacity check: fit each width to today's P(win) on Master positions
    #[cfg(feature = "dqn")]
    WinFit {
        /// Training positions
        #[arg(long, default_value = "300000")]
        train: usize,
        /// Held-out positions
        #[arg(long, default_value = "50000")]
        test: usize,
        /// Epochs per width
        #[arg(long, default_value = "20")]
        epochs: usize,
        /// Widths to try
        #[arg(long, value_delimiter = ',', default_value = "16,32,64")]
        widths: Vec<usize>,
        /// Save each fitted net to <SAVE>/w<W> (bench it with QWIXXER_WIN_MODEL=<SAVE>/w<W> ... bench win pair)
        #[arg(long)]
        save: Option<String>,
        /// Label with K full-game Master rollouts per (turn-end) position instead of Master's static P(win); 0 = static
        #[arg(long, default_value = "0")]
        rollouts: usize,
        /// With --rollouts: target = blend·rollout P + (1 − blend)·Master's static P
        #[arg(long, default_value = "1.0")]
        blend: f32,
        /// With --rollouts: this win model dir plays the games and rollouts instead of Master (policy iteration)
        #[arg(long)]
        player: Option<String>,
    },
    /// Win-net value gap: the net's P(win) vs its own many-rollout P(win) on its self-play turn ends
    #[cfg(feature = "dqn")]
    WinGap {
        /// Win model dir to measure
        #[arg(long, default_value = "win_model")]
        model: String,
        /// Turn-end positions
        #[arg(long, default_value = "3000")]
        positions: usize,
        /// Rollouts per position
        #[arg(long, default_value = "1024")]
        rollouts: usize,
        #[arg(long, default_value = "42")]
        seed: u64,
    },
    /// Win-net policy iteration: rollout labels from the champion's own games, gated promotion
    #[cfg(feature = "dqn")]
    WinPi {
        /// Output dir (must not exist): round-<i>/, champion/, training.csv
        #[arg(long, default_value = "win_pi")]
        dir: String,
        /// Starting champion (a win model dir, e.g. win_model); without it Master plays round 1
        #[arg(long)]
        init: Option<String>,
        /// Width of a from-scratch net (ignored with --init)
        #[arg(long, default_value = "64")]
        width: usize,
        #[arg(long, default_value = "10")]
        rounds: usize,
        /// New turn-end positions per round
        #[arg(long, default_value = "300000")]
        positions: usize,
        /// Rollouts per position
        #[arg(long, default_value = "64")]
        rollouts: usize,
        /// Rounds of positions each fit trains on
        #[arg(long, default_value = "2")]
        window: usize,
        /// Epochs per fit
        #[arg(long, default_value = "20")]
        epochs: usize,
        /// Warm-start learning rate (a from-scratch round 1 uses 1e-3)
        #[arg(long, default_value = "3e-4")]
        lr: f64,
        /// Promote when z = (share − ½)/se exceeds this
        #[arg(long, default_value = "2.0")]
        gate_z: f64,
        /// Gate games per block (even)
        #[arg(long, default_value = "100000")]
        gate_block: usize,
        /// Gate games at most
        #[arg(long, default_value = "1000000")]
        gate_max: usize,
        /// Champion vs Master games per round, logged only (0 = skip)
        #[arg(long, default_value = "200000")]
        master_games: usize,
        #[arg(long, default_value = "42")]
        seed: u64,
        /// turn = turn-end rollout labels; cand = also the top candidates of the decisions win-search would search
        #[arg(long, default_value = "turn", value_parser = ["turn", "cand"])]
        labels: String,
        /// Full-game rollouts per candidate (--labels cand)
        #[arg(long, default_value = "64")]
        cand_rollouts: usize,
        /// Copies of each candidate sample (--labels cand)
        #[arg(long, default_value = "1")]
        cand_m: usize,
    },
    /// Win-net dataset: a player's turn ends and gated candidates with win and auxiliary rollout labels, held out by game
    #[cfg(feature = "dqn")]
    WinData {
        /// Win model dir that plays the games and the rollouts
        #[arg(long)]
        player: String,
        /// Output dir (must not exist): meta.json, data.bin
        #[arg(long)]
        out: String,
        /// Turn ends to reach (whole games are kept)
        #[arg(long, default_value = "400000")]
        positions: usize,
        /// Rollouts per turn end
        #[arg(long, default_value = "64")]
        rollouts: usize,
        /// Rollouts per candidate
        #[arg(long, default_value = "64")]
        cand_rollouts: usize,
        /// Share of games held out
        #[arg(long, default_value = "0.1")]
        holdout: f64,
        #[arg(long, default_value = "42")]
        seed: u64,
    },
}

fn run_play(bots: Vec<BotType>, verbose: bool) {
    let bot_names: Vec<String> = bots.iter().map(|b| b.to_string()).collect();
    println!("Playing against: {}\n", bot_names.join(", "));

    let mut players: Vec<Player> = bots
        .iter()
        .map(|b| Player::new(make_strategy(b), Box::new(SmallRng::from_entropy())))
        .collect();
    players.push(Player::new(
        Box::<strategy::Interactive>::default(),
        Box::new(SmallRng::from_entropy()),
    ));

    let mut game = game::Game::new(players);
    game.verbose = verbose;
    game.play();
    game.print_game_over();
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Dice-stream seed for one seat of one game pair. Distinct `(pair, seat)`
/// inputs map to distinct values within a run (seats < 8); hashing `base`
/// keeps the stream sets of nearby base seeds disjoint, so different `--seed`
/// runs are independent samples. `seed_from_u64` expands the result into a
/// decorrelated stream state.
fn seat_dice_seed(base: u64, pair: usize, seat: usize) -> u64 {
    splitmix64(base).wrapping_add(pair as u64 * 8 + seat as u64)
}

fn run_bench(bots: Vec<BotType>, num_games: usize, seed: u64) {
    if bots.len() < 2 {
        eprintln!("Need at least 2 bots to benchmark");
        return;
    }

    let num_players = bots.len();
    // Games come in pairs of num_players rotations sharing per-seat dice
    // streams, so round up to complete the final pair.
    let num_games = num_games.div_ceil(num_players) * num_players;
    println!(
        "Benchmarking {} ({num_games} games, rotating seats, paired dice, seed {seed}):\n",
        bots.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(" vs ")
    );
    let bench_start = std::time::Instant::now();

    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    let bench_game = |templates: &StrategyTemplates, i: usize| {
        let pair = i / num_players;
        let rotation = i % num_players;
        let players: Vec<Player> = (0..num_players)
            .map(|j| {
                let bot_idx = (j + num_players - rotation) % num_players;
                Player::new(
                    templates.create(&bots[bot_idx]),
                    Box::new(SmallRng::seed_from_u64(seat_dice_seed(seed, pair, j))),
                )
            })
            .collect();

        let mut game = game::Game::new(players);
        game.play();

        let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
        let max = *scores.iter().max().unwrap();
        let num_winners = scores.iter().filter(|&&s| s == max).count();

        let per_bot: Vec<(usize, isize)> = (0..num_players)
            .map(|j| {
                let bot_idx = (j + num_players - rotation) % num_players;
                (bot_idx, scores[j])
            })
            .collect();

        let is_tie = num_winners > 1;
        (per_bot, is_tie)
    };

    #[cfg(feature = "parallel")]
    let results: Vec<(Vec<(usize, isize)>, bool)> = (0..num_games)
        .into_par_iter()
        .map_init(|| StrategyTemplates::new(&bots), |t, i| bench_game(t, i))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let results: Vec<(Vec<(usize, isize)>, bool)> = {
        let templates = StrategyTemplates::new(&bots);
        (0..num_games).map(|i| bench_game(&templates, i)).collect()
    };

    // Aggregate
    let mut wins = vec![0u32; num_players];
    let mut total_pts = vec![0i64; num_players];
    let mut ties = 0u32;

    for (per_bot, is_tie) in &results {
        let max = per_bot.iter().map(|(_, s)| *s).max().unwrap();
        let num_winners = per_bot.iter().filter(|(_, s)| *s == max).count();
        for &(bot_idx, score) in per_bot {
            total_pts[bot_idx] += score as i64;
            if score == max && num_winners == 1 {
                wins[bot_idx] += 1;
            }
        }
        if *is_tie {
            ties += 1;
        }
    }

    // Per-player stats
    for (i, bot) in bots.iter().enumerate() {
        println!(
            "  {:<16} {:>5} wins ({:>4.1}%)  avg {:.1} pts",
            format!("{} #{}", bot, i + 1),
            wins[i],
            wins[i] as f64 / num_games as f64 * 100.0,
            total_pts[i] as f64 / num_games as f64
        );
    }
    if ties > 0 {
        println!(
            "  {:<16} {:>5}       ({:>4.1}%)",
            "Ties",
            ties,
            ties as f64 / num_games as f64 * 100.0
        );
    }

    // Aggregate stats per strategy (when multiple bots share a strategy)
    let unique_strategies: Vec<String> = bots
        .iter()
        .map(|b| b.to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    if unique_strategies.len() < num_players {
        println!("\n  By strategy:");

        // For each game, determine which STRATEGY won.
        // A tie between bots of the same strategy counts as a win for that strategy.
        let mut strat_wins: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let mut strat_pts: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        let mut strat_count: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let mut strat_ties = 0u32;

        for bot in &bots {
            let name = bot.to_string();
            strat_count.entry(name.clone()).or_insert(0);
            strat_wins.entry(name.clone()).or_insert(0);
            strat_pts.entry(name.clone()).or_insert(0);
        }
        for (i, bot) in bots.iter().enumerate() {
            let name = bot.to_string();
            *strat_count.entry(name).or_insert(0) += 1;
            let _ = i; // count only
        }

        for (per_bot, _) in &results {
            let max = per_bot.iter().map(|(_, s)| *s).max().unwrap();

            // Accumulate points per strategy
            for &(bot_idx, score) in per_bot {
                let name = bots[bot_idx].to_string();
                *strat_pts.entry(name).or_insert(0) += score as i64;
            }

            // Find which strategies have the max score
            let winning_strategies: std::collections::BTreeSet<String> = per_bot
                .iter()
                .filter(|(_, s)| *s == max)
                .map(|(idx, _)| bots[*idx].to_string())
                .collect();

            if winning_strategies.len() == 1 {
                let name = winning_strategies.into_iter().next().unwrap();
                *strat_wins.entry(name).or_insert(0) += 1;
            } else {
                strat_ties += 1;
            }
        }

        for strat in &unique_strategies {
            let count = *strat_count.get(strat).unwrap() as f64;
            let sw = *strat_wins.get(strat).unwrap_or(&0);
            let sp = *strat_pts.get(strat).unwrap_or(&0);
            println!(
                "  {:<16} {:>5} wins ({:>4.1}%)  avg {:.1} pts",
                strat,
                sw,
                sw as f64 / num_games as f64 * 100.0,
                sp as f64 / (num_games as f64 * count),
            );
        }
        if strat_ties > 0 {
            println!(
                "  {:<16} {:>5}       ({:>4.1}%)",
                "Ties",
                strat_ties,
                strat_ties as f64 / num_games as f64 * 100.0
            );
        }
    }

    // 99% CI on winrate for 2-strategy matchups (covers plain 1v1 and
    // aggregated NvN). Computed over pair means: the games of one pair share
    // per-seat dice streams, so their outcomes are correlated and the pair is
    // the independent sampling unit. This both keeps the CI honest and
    // captures the variance reduction from pairing.
    if unique_strategies.len() == 2 {
        // Winning strategy per game (None = tie between the two strategies).
        let game_winners: Vec<Option<String>> = results
            .iter()
            .map(|(per_bot, _)| {
                let max = per_bot.iter().map(|(_, s)| *s).max().unwrap();
                let winners: std::collections::BTreeSet<String> = per_bot
                    .iter()
                    .filter(|(_, s)| *s == max)
                    .map(|(idx, _)| bots[*idx].to_string())
                    .collect();
                if winners.len() == 1 {
                    winners.into_iter().next()
                } else {
                    None
                }
            })
            .collect();

        let count_wins = |name: &str| game_winners.iter().filter(|w| w.as_deref() == Some(name)).count();
        let (s0, s1) = (&unique_strategies[0], &unique_strategies[1]);
        let leader = if count_wins(s0) >= count_wins(s1) { s0 } else { s1 };

        let pair_means: Vec<f64> = game_winners
            .chunks(num_players)
            .map(|chunk| {
                chunk.iter().filter(|w| w.as_deref() == Some(leader.as_str())).count() as f64 / chunk.len() as f64
            })
            .collect();
        let num_pairs = pair_means.len() as f64;
        let p = pair_means.iter().sum::<f64>() / num_pairs;
        let var = pair_means.iter().map(|a| (a - p).powi(2)).sum::<f64>() / (num_pairs - 1.0);
        let se_paired = (var / num_pairs).sqrt();
        let se_naive = (p * (1.0 - p) / num_games as f64).sqrt();
        let z = 2.576;
        println!(
            "\n  99% CI (paired): {} wins {:.2}% - {:.2}%",
            leader,
            (p - z * se_paired) * 100.0,
            (p + z * se_paired) * 100.0
        );
        if se_paired > 0.0 {
            println!(
                "  Pairing efficiency: {:.2}x (naive SE {:.3}%, paired SE {:.3}%)",
                (se_naive / se_paired).powi(2),
                se_naive * 100.0,
                se_paired * 100.0
            );
        }

        // Dice-pairing attribution: the games of one pairing share per-seat
        // dice, so a pairing SWEPT by one strategy (it wins every rotation)
        // is a skill signal, while a SPLIT pairing (each strategy wins at
        // least once) was decided by seat/dice. The split share is a direct
        // read on how much of the matchup is dice-dominated.
        let mut swept = [0usize; 2];
        let mut split = 0usize;
        let mut with_ties = 0usize;
        for chunk in game_winners.chunks(num_players) {
            let w0 = chunk.iter().filter(|w| w.as_deref() == Some(s0.as_str())).count();
            let w1 = chunk.iter().filter(|w| w.as_deref() == Some(s1.as_str())).count();
            if w0 == chunk.len() {
                swept[0] += 1;
            } else if w1 == chunk.len() {
                swept[1] += 1;
            } else if w0 > 0 && w1 > 0 {
                split += 1;
            } else {
                with_ties += 1; // no sweep, no split: at least one tie involved
            }
        }
        let np = game_winners.chunks(num_players).len() as f64;
        println!(
            "  Dice pairings ({:.0}): swept by {} {} ({:.1}%), by {} {} ({:.1}%), split {} ({:.1}%), with ties {} ({:.1}%)",
            np,
            s0,
            swept[0],
            swept[0] as f64 / np * 100.0,
            s1,
            swept[1],
            swept[1] as f64 / np * 100.0,
            split,
            split as f64 / np * 100.0,
            with_ties,
            with_ties as f64 / np * 100.0
        );
    }

    let elapsed = bench_start.elapsed();
    println!(
        "\n  Elapsed: {}m{:02}s ({:.0} games/s)",
        elapsed.as_secs() / 60,
        elapsed.as_secs() % 60,
        num_games as f64 / elapsed.as_secs_f64()
    );
}

fn run_solo(num_games: usize) {
    let genes = Arc::new(bot::default_genes());
    let champion = bot::DNA::load_weights("champion.txt", genes).ok();

    let all_bots: Vec<(BotType, bool)> = vec![
        (BotType::Random, true),
        (BotType::Conservative, true),
        (BotType::Opportunist, true),
        (BotType::Ga, champion.is_some()),
        (BotType::Dqn, std::path::Path::new("dqn_model/model.mpk").exists()),
    ];

    println!("Single-player scores over {num_games} games:\n");
    for (bot, available) in &all_bots {
        if !available {
            continue;
        }
        let mut total = 0i64;
        let mut min = i64::MAX;
        let mut max = i64::MIN;
        for _ in 0..num_games {
            let mut game = game::Game::new(vec![Player::new(
                make_strategy(bot),
                Box::new(SmallRng::from_entropy()),
            )]);
            game.play();
            let pts = game.players[0].state.count_points() as i64;
            total += pts;
            min = min.min(pts);
            max = max.max(pts);
        }
        let avg = total as f64 / num_games as f64;
        println!("  {:<16} avg {avg:>5.1}  min {min:>4}  max {max:>4}", bot.to_string());
    }
}

fn run_train() {
    println!("Training GA bot (population=100, 200 generations)...\n");
    let mut pop = bot::Population::new(100, bot::default_genes(), 42);
    pop.evolve(200);

    let _champion = pop.current_champion().clone();
    println!("\nBenchmarking champion vs Opportunist...\n");
    run_bench(vec![BotType::Ga, BotType::Opportunist], 100_000, 42);
}

#[cfg(feature = "dqn")]
fn run_dqn_selfplay(iterations: usize, bench_games: usize, checkpoints: bool, start_iteration: usize) {
    dqn::train::self_play_train(
        "dqn_model",
        iterations,
        20000,
        10,
        bench_games,
        checkpoints,
        start_iteration,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seat_dice_seeds_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for pair in 0..1000 {
            for seat in 0..5 {
                assert!(seen.insert(seat_dice_seed(42, pair, seat)));
            }
        }
    }

    #[test]
    fn seeded_games_are_deterministic() {
        let play = || {
            let players = vec![
                Player::new(
                    Box::<strategy::Opportunist>::default(),
                    Box::new(SmallRng::seed_from_u64(seat_dice_seed(42, 7, 0))),
                ),
                Player::new(
                    Box::<strategy::Conservative>::default(),
                    Box::new(SmallRng::seed_from_u64(seat_dice_seed(42, 7, 1))),
                ),
            ];
            let mut game = game::Game::new(players);
            game.play();
            game.players.iter().map(|p| p.state.count_points()).collect::<Vec<_>>()
        };
        assert_eq!(play(), play());
    }
}

fn main() {
    // Our matmuls are tiny (≤ a few hundred rows of a 45→128→64 MLP) and run
    // inside rayon-parallel games; matrixmultiply's own thread pool on top is
    // pure scheduler churn (measured 2x slowdown on search benches). Respect
    // an explicit user override.
    if std::env::var_os("MATMUL_NUM_THREADS").is_none() {
        std::env::set_var("MATMUL_NUM_THREADS", "1");
    }

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Play { bots, verbose }) => run_play(bots, verbose),
        Some(Commands::Bench { bots, num_games, seed }) => run_bench(bots, num_games, seed),
        Some(Commands::Solo { num_games }) => run_solo(num_games),
        Some(Commands::Evolve) => run_train(),
        #[cfg(feature = "dqn")]
        Some(Commands::DqnSelfplay {
            iterations,
            bench,
            checkpoints,
            start_iteration,
        }) => run_dqn_selfplay(iterations, bench, checkpoints, start_iteration),
        #[cfg(feature = "dqn")]
        Some(Commands::PairTrain {
            iterations,
            games,
            epochs,
            bench,
            checkpoints,
            start_iteration,
            search,
            distill,
            distill_k,
            distill_m,
            distill_m_lock,
            epsilon_lock,
            distill_passive,
        }) => dqn::pair_train::self_play_train(
            "pair_model",
            iterations,
            games,
            epochs,
            bench,
            checkpoints,
            start_iteration,
            search,
            dqn::pair_train::DistillCfg {
                enabled: distill,
                k: distill_k,
                m_gated: distill_m,
                m_lock: distill_m_lock,
                epsilon_lock,
                passive: distill_passive,
            },
        ),
        #[cfg(feature = "dqn")]
        Some(Commands::WinTrain {
            iterations,
            games,
            epochs,
            bench,
            lr,
            lr_end,
            epsilon_start,
            epsilon_floor,
            lambda,
            init,
            width,
            buffer,
            start_iteration,
        }) => {
            dqn::win_train::self_play_train(&dqn::win_train::WinTrainCfg {
                dir: "win_model".into(),
                iterations,
                games,
                epochs,
                bench,
                width,
                buffer,
                start_iteration,
                lr,
                lr_end,
                epsilon_start,
                epsilon_floor,
                lambda,
                init,
                pair_dir: "pair_model".into(),
            });
        }
        #[cfg(feature = "dqn")]
        Some(Commands::WinFit {
            train,
            test,
            epochs,
            widths,
            save,
            rollouts,
            blend,
            player,
        }) => {
            dqn::win_train::capacity_check(&dqn::win_train::FitCfg {
                train,
                test,
                epochs,
                widths,
                lr: 1e-3,
                save,
                rollouts,
                blend,
                player,
            });
        }
        #[cfg(feature = "dqn")]
        Some(Commands::WinGap {
            model,
            positions,
            rollouts,
            seed,
        }) => {
            let start = std::time::Instant::now();
            println!("{model}: {positions} turn ends of its own games, {rollouts} rollouts each…");
            let r = dqn::win_train::value_gap(&model, positions, rollouts, seed);
            println!("  MAE {:.4}, bias {:+.4}, raw RMSE {:.4}", r.mae, r.bias, r.rmse_raw);
            println!(
                "  label noise (SD) {:.4} at K = {rollouts}; a K = 64 training label's: {:.4}",
                r.noise_var.sqrt(),
                r.noise_sd_64
            );
            println!("  net RMSE (label noise removed): {:.4}", r.rmse);
            println!(
                "  best constant logit offset {:+.3}; net RMSE after it (noise removed): {:.4}",
                r.offset, r.rmse_offset
            );
            println!("  calibration (predicted bin: n, mean predicted, mean rollout):");
            for (i, (n, p, m)) in r.calibration.iter().enumerate() {
                if *n > 0 {
                    println!(
                        "    [{:.1}, {:.1}): {n:>5}, {p:.3}, {m:.3}",
                        i as f32 / 10.0,
                        (i + 1) as f32 / 10.0
                    );
                }
            }
            println!("  {}s", start.elapsed().as_secs());
        }
        #[cfg(feature = "dqn")]
        Some(Commands::WinPi {
            dir,
            init,
            width,
            rounds,
            positions,
            rollouts,
            window,
            epochs,
            lr,
            gate_z,
            gate_block,
            gate_max,
            master_games,
            seed,
            labels,
            cand_rollouts,
            cand_m,
        }) => {
            let cfg = dqn::win_pi::WinPiCfg {
                dir,
                init,
                width,
                rounds,
                positions,
                rollouts,
                window,
                epochs,
                lr,
                gate_z,
                gate_block,
                gate_max,
                master_games,
                seed,
                pair_dir: "pair_model".into(),
                labels: if labels == "cand" {
                    dqn::win_pi::Labels::Cand
                } else {
                    dqn::win_pi::Labels::Turn
                },
                cand_rollouts,
                cand_m,
            };
            if let Err(e) = cfg.validate() {
                eprintln!("win-pi: {e}");
                std::process::exit(2);
            }
            dqn::win_pi::policy_iteration(&cfg);
        }
        #[cfg(feature = "dqn")]
        Some(Commands::WinData {
            player,
            out,
            positions,
            rollouts,
            cand_rollouts,
            holdout,
            seed,
        }) => {
            let cfg = dqn::win_data::DataCfg {
                player,
                out,
                positions,
                rollouts,
                cand_rollouts,
                holdout,
                seed,
            };
            if let Err(e) = cfg.validate() {
                eprintln!("win-data: {e}");
                std::process::exit(2);
            }
            let start = std::time::Instant::now();
            let win = dqn::win::WinStrategy::load(&cfg.player);
            let ds = dqn::win_data::generate(&cfg, &|| win.clone());
            dqn::win_data::save(&cfg.out, &ds).expect("Failed to write the dataset");
            let m = &ds.meta;
            println!(
                "win-data: {} games; train {} turn ends + {} decisions ({} samples); held out {} + {} ({} samples); {}s",
                m.games,
                m.train_turns,
                m.train_decisions,
                ds.train.cand_samples(),
                m.heldout_turns,
                m.heldout_decisions,
                ds.heldout.cand_samples(),
                start.elapsed().as_secs()
            );
        }
        None => {
            // Default: play against MCTS
            run_play(vec![BotType::Mcts], false);
        }
    }
}
