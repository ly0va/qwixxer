//! Feature screen (spike, throwaway): would a candidate input feature fix the
//! static pair net's *decision* errors?
//!
//! 1. Static pair self-play (1v1, production pipeline). At every value-picked
//!    decision with ≥2 candidates, record the context and the net's top-M
//!    candidates (z, P = Φ(z)).
//! 2. Per candidate: complete the turn (the search/distill entry helpers),
//!    then two independent batches of K full-game CRN rollouts under the
//!    rollout policy. Batch A labels W_A = P(win); batch B gives an
//!    independent W_B and the oracle game-length quantities.
//! 3. Per non-pick candidate c, the net's error on the *difference* to its
//!    pick: y = ΔW_A − ΔP. Regress y on Δf for each feature group (train on
//!    the first half of games, report on the second half): R² relative to the
//!    explainable (non-rollout-noise) variance, and the held-out win-prob gain
//!    per player-game of re-picking with P + β·Δf.
//!
//! The `ceiling` group (ΔW_B, an independent rollout estimate of the label)
//! bounds what any feature can explain; `phase24` (Phase 24's threat inputs)
//! is the negative control — expected ≈ 0.
//!
//! Run: cargo run --release --example feature_screen -- -n 400 -k 64

mod threat;

use clap::Parser;
use qwixxer::dqn::pair::PairStrategy;
use qwixxer::game::{Game, Player};
use qwixxer::state::{Mark, State};
use qwixxer::strategy::bot_impl::{
    active_phase1_choices, active_phase2_choices, argmax, passive_phase1_choices, Decision,
};
use qwixxer::strategy::search::{
    context_seed, passive_entry, passive_first_active, phase1_entry, phase2_entry, phi, sample_player_seed,
};
use qwixxer::strategy::sim::{BatchedRollouts, SimGame};
use qwixxer::strategy::{Bot, Strategy};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;

#[derive(Parser)]
struct Cli {
    /// Self-play games (both seats' decisions are screened).
    #[arg(short, long, default_value = "400")]
    n: usize,
    /// Rollouts per candidate per batch.
    #[arg(short, long, default_value = "64")]
    k: usize,
    /// Candidates per decision (the net's top M).
    #[arg(short, long, default_value = "4")]
    m: usize,
    #[arg(short, long, default_value = "0")]
    seed: u64,
    /// Per-candidate CSV for offline analysis.
    #[arg(long, default_value = "feature_screen.csv")]
    out: String,
}

/// One recorded decision: phase 0 = passive phase 1, 1 = active phase 1,
/// 2 = active phase 2. `cands` = (mark, post-state, z), best z first.
#[derive(Clone)]
struct Ctx {
    game: usize,
    phase: u8,
    state: State,
    opps: Vec<State>,
    eval_opps: Vec<State>,
    dice: [u8; 6],
    active_player: usize,
    cands: Vec<(Option<Mark>, State, f32)>,
    pick: usize,
}

struct Recorder {
    bot: PairStrategy,
    game: usize,
    m: usize,
    ctxs: Rc<RefCell<Vec<Ctx>>>,
}

impl std::fmt::Debug for Recorder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Recorder")
    }
}

impl Recorder {
    /// Production static pick over `choices`; records the context when ≥2
    /// distinct moves remain. Plans sharing a phase-1 mark collapse to their
    /// best value (the decision is the mark; phase 2 is re-decided).
    #[allow(clippy::too_many_arguments)]
    fn decide(
        &self,
        phase: u8,
        choices: Vec<(Option<Mark>, State)>,
        state: &State,
        opps: &[State],
        eval_opps: &[State],
        dice: [u8; 6],
        active_player: usize,
    ) -> Option<Mark> {
        let posts: Vec<State> = choices.iter().map(|(_, s)| *s).collect();
        let z = self.bot.evaluate_batch(&posts, eval_opps);
        let pick_mark = choices[argmax(&z)].0;
        let mut cands: Vec<(Option<Mark>, State, f32)> = Vec::new();
        for ((m, s), &v) in choices.iter().zip(&z) {
            match cands.iter_mut().find(|c| c.0 == *m) {
                Some(c) if v > c.2 => *c = (*m, *s, v),
                Some(_) => {}
                None => cands.push((*m, *s, v)),
            }
        }
        cands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        cands.truncate(self.m);
        if cands.len() >= 2 {
            if let Some(pick) = cands.iter().position(|c| c.0 == pick_mark) {
                self.ctxs.borrow_mut().push(Ctx {
                    game: self.game,
                    phase,
                    state: *state,
                    opps: opps.to_vec(),
                    eval_opps: eval_opps.to_vec(),
                    dice,
                    active_player,
                    cands,
                    pick,
                });
            }
        }
        pick_mark
    }
}

impl Strategy for Recorder {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let (d, sim_opp) = active_phase1_choices(&self.bot, state, opp_states, dice);
        match d {
            Decision::Forced(m) => m,
            Decision::Choices(c) => self.decide(1, c, state, opp_states, &sim_opp, dice, 0),
        }
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        match active_phase2_choices(state, opp_states, dice, has_marked) {
            Decision::Forced(m) => m,
            Decision::Choices(c) => self.decide(2, c, state, opp_states, opp_states, dice, 0),
        }
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        active_player: usize,
    ) -> Option<Mark> {
        match passive_phase1_choices(state, opp_states, dice) {
            Decision::Forced(m) => m,
            Decision::Choices(c) => self.decide(0, c, state, opp_states, opp_states, dice, active_player),
        }
    }
}

fn collect(template: &PairStrategy, game: usize, seed: u64, m: usize) -> Vec<Ctx> {
    let ctxs = Rc::new(RefCell::new(Vec::new()));
    let players: Vec<Player> = (0..2u64)
        .map(|seat| {
            let rec = Recorder {
                bot: PairStrategy::from_shared(template.model.clone(), template.device),
                game,
                m,
                ctxs: ctxs.clone(),
            };
            let dice = Box::new(SmallRng::seed_from_u64(
                seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (game as u64 * 2 + seat),
            ));
            Player::new(Box::new(rec), dice)
        })
        .collect();
    let mut g = Game::new(players);
    g.play();
    drop(g);
    Rc::try_unwrap(ctxs)
        .unwrap_or_else(|_| panic!("ctxs still shared"))
        .into_inner()
}

/// Per-sample rollout results for one entry: (outcome for player 0, turns
/// until game over, final states).
type Samples = Vec<(f32, u32)>;

fn rollouts(
    bot: &PairStrategy,
    entries: &[(Vec<State>, bool)],
    seed: u64,
    k: usize,
    first_active: usize,
) -> Vec<Samples> {
    let n = entries[0].0.len();
    let mut sims = Vec::new();
    let mut owner = Vec::new();
    for (ei, (states, ended)) in entries.iter().enumerate() {
        if *ended {
            continue;
        }
        for s in 0..k {
            sims.push(SimGame {
                states: states.clone(),
                active: first_active,
                rngs: (0..n)
                    .map(|p| SmallRng::seed_from_u64(sample_player_seed(seed, s, p)))
                    .collect(),
                over: false,
            });
            owner.push(ei);
        }
    }
    let mut turns_at_end = vec![0u32; sims.len()];
    let mut driver = BatchedRollouts::new(bot, sims);
    let mut turns = 0u32;
    while !driver.all_over() {
        driver.step_turn();
        turns += 1;
        for (i, sim) in driver.sims.iter().enumerate() {
            if sim.over && turns_at_end[i] == 0 {
                turns_at_end[i] = turns;
            }
        }
        assert!(turns <= 200);
    }
    let mut out: Vec<Samples> = entries
        .iter()
        .map(|(states, ended)| {
            if *ended {
                vec![(SimGame::outcome(states), 0); k]
            } else {
                Vec::with_capacity(k)
            }
        })
        .collect();
    for (i, sim) in driver.sims.iter().enumerate() {
        out[owner[i]].push((SimGame::outcome(&sim.states), turns_at_end[i]));
    }
    out
}

/// Probability-weighted markable space left in each row (2d6 sum weights;
/// the terminal counts only once the row has 5 marks). Locked rows: 0.
fn dice_mass(s: &State) -> [f32; 4] {
    let frees = s.row_free_values();
    let totals = s.row_totals();
    let locked = s.locked();
    let mut out = [0.0f32; 4];
    for r in 0..4 {
        let Some(f) = frees[r] else { continue };
        if locked[r] {
            continue;
        }
        let asc = r < 2;
        let term = State::row_terminal(r);
        let vals: Vec<u8> = if asc {
            (f..=12).collect()
        } else {
            (2..=f).rev().collect()
        };
        for v in vals {
            if v == term && totals[r] < 5 {
                continue;
            }
            out[r] += (6.0 - (7.0f32 - v as f32).abs()) / 36.0;
        }
    }
    out
}

/// Per-candidate measurements.
#[derive(Clone, Default)]
struct CandRow {
    z: f32,
    p: f32,
    w_a: f32,
    w_b: f32,
    /// Batch-A per-sample outcomes (for the CRN-paired noise of ΔW_A).
    o_a: Vec<f32>,
    turns: f32,
    end2: f32,
    end4: f32,
    mass: [f32; 4],
    fs_us: f32,
    fs_opp: f32,
    thr_us: f32,
    thr_opp: f32,
    lockcal: f32,
}

struct Screened {
    game: usize,
    phase: u8,
    cdiff: f32,
    pick: usize,
    rows: Vec<CandRow>,
}

fn screen_ctx(bot: &PairStrategy, ctx: &Ctx, k: usize) -> Screened {
    let n = ctx.opps.len() + 1;
    let (entries, first_active): (Vec<(Vec<State>, bool)>, usize) = match ctx.phase {
        0 => (
            ctx.cands
                .iter()
                .map(|(m, _, _)| passive_entry(bot, &ctx.state, &ctx.opps, ctx.dice, ctx.active_player, *m))
                .collect(),
            passive_first_active(ctx.active_player, n),
        ),
        1 => (
            ctx.cands
                .iter()
                .map(|(m, _, _)| phase1_entry(bot, &ctx.state, &ctx.eval_opps, ctx.dice, *m))
                .collect(),
            1 % n,
        ),
        _ => (
            ctx.cands
                .iter()
                .map(|(_, post, _)| phase2_entry(*post, &ctx.opps))
                .collect(),
            1 % n,
        ),
    };
    let base = context_seed(&ctx.state, &ctx.opps, ctx.dice);
    let seed_a = base ^ 0xA5A5_0000_0000_0001;
    let seed_b = base.rotate_left(17) ^ 0x5B5B_0000_0000_0002;
    let ra = rollouts(bot, &entries, seed_a, k, first_active);
    let rb = rollouts(bot, &entries, seed_b, k, first_active);
    let opp = &ctx.eval_opps[0];
    let rows = ctx
        .cands
        .iter()
        .enumerate()
        .map(|(i, (_, post, z))| {
            let mean = |v: &Samples, f: &dyn Fn(&(f32, u32)) -> f32| v.iter().map(f).sum::<f32>() / v.len() as f32;
            let has_lock = post.count_locked() > 0 || opp.count_locked() > 0;
            CandRow {
                z: *z,
                p: phi(*z),
                w_a: mean(&ra[i], &|s| s.0),
                w_b: mean(&rb[i], &|s| s.0),
                o_a: ra[i].iter().map(|s| s.0).collect(),
                turns: mean(&rb[i], &|s| s.1 as f32),
                end2: mean(&rb[i], &|s| (s.1 <= 2) as u8 as f32),
                end4: mean(&rb[i], &|s| (s.1 <= 4) as u8 as f32),
                mass: dice_mass(post),
                fs_us: threat::forced_strike_prob(post),
                fs_opp: threat::forced_strike_prob(opp),
                thr_us: threat::lock_threats(post, &[opp]).iter().sum(),
                thr_opp: threat::lock_threats(opp, &[post]).iter().sum(),
                lockcal: if has_lock { phi(0.75 * z) - phi(*z) } else { 0.0 },
            }
        })
        .collect();
    let pick_post = &ctx.cands[ctx.pick].1;
    Screened {
        game: ctx.game,
        phase: ctx.phase,
        cdiff: (pick_post.count_points() - opp.count_points()) as f32,
        pick: ctx.pick,
        rows,
    }
}

/// Least squares with intercept; returns [b0, b1..]. Tiny ridge for safety.
fn ols(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let d = x[0].len() + 1;
    let mut a = vec![vec![0.0f64; d + 1]; d];
    for (xi, &yi) in x.iter().zip(y) {
        let row: Vec<f64> = std::iter::once(1.0).chain(xi.iter().copied()).collect();
        for r in 0..d {
            for c in 0..d {
                a[r][c] += row[r] * row[c];
            }
            a[r][d] += row[r] * yi;
        }
    }
    for (r, row) in a.iter_mut().enumerate() {
        row[r] += 1e-9;
    }
    for col in 0..d {
        let piv = (col..d)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())
            .unwrap();
        a.swap(col, piv);
        let p = a[col][col];
        if p.abs() < 1e-12 {
            continue;
        }
        for c in col..=d {
            a[col][c] /= p;
        }
        for r in 0..d {
            if r != col {
                let f = a[r][col];
                for c in col..=d {
                    a[r][c] -= f * a[col][c];
                }
            }
        }
    }
    (0..d).map(|r| a[r][d]).collect()
}

type FeatFn = fn(&CandRow, &CandRow, f32) -> Vec<f64>;

fn d(a: f32, b: f32) -> f64 {
    (a - b) as f64
}

fn groups() -> Vec<(&'static str, FeatFn)> {
    vec![
        ("ceiling (indep. rollout W_B)", |c, p, _| vec![d(c.w_b, p.w_b)]),
        ("length (oracle E[turns], P(end<=2,4))", |c, p, _| {
            vec![d(c.turns, p.turns), d(c.end2, p.end2), d(c.end4, p.end4)]
        }),
        ("length x behind/ahead", |c, p, cd| {
            let s = (cd / 10.0).tanh() as f64;
            vec![
                d(c.turns, p.turns),
                d(c.end2, p.end2),
                d(c.end4, p.end4),
                s * d(c.turns, p.turns),
                s * d(c.end2, p.end2),
                s * d(c.end4, p.end4),
            ]
        }),
        ("dice mass per row", |c, p, _| {
            (0..4).map(|r| d(c.mass[r], p.mass[r])).collect()
        }),
        ("phase24 threats (neg. control)", |c, p, _| {
            vec![
                d(c.fs_us, p.fs_us),
                d(c.fs_opp, p.fs_opp),
                d(c.thr_us, p.thr_us),
                d(c.thr_opp, p.thr_opp),
            ]
        }),
        ("lock-aware recal (a=0.75)", |c, p, _| vec![d(c.lockcal, p.lockcal)]),
    ]
}

fn main() {
    let cli = Cli::parse();
    let t0 = std::time::Instant::now();

    let ctxs: Vec<Ctx> = (0..cli.n)
        .into_par_iter()
        .map_init(
            || PairStrategy::load("pair_model"),
            |t, g| collect(t, g, cli.seed, cli.m),
        )
        .flatten_iter()
        .collect();
    eprintln!("{} decisions from {} games ({:.0?})", ctxs.len(), cli.n, t0.elapsed());

    let done = std::sync::atomic::AtomicUsize::new(0);
    let screened: Vec<Screened> = ctxs
        .par_iter()
        .map_init(
            || PairStrategy::load("pair_model"),
            |bot, ctx| {
                let s = screen_ctx(bot, ctx, cli.k);
                let i = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if i % 2000 == 0 {
                    eprintln!("  {i}/{} screened ({:.0?})", ctxs.len(), t0.elapsed());
                }
                s
            },
        )
        .collect();
    eprintln!("screened in {:.0?}", t0.elapsed());

    // Per-candidate CSV.
    if let Ok(mut f) = std::fs::File::create(&cli.out) {
        writeln!(f, "game,phase,cdiff,is_pick,z,p,w_a,w_b,turns,end2,end4,mass0,mass1,mass2,mass3,fs_us,fs_opp,thr_us,thr_opp,lockcal").ok();
        for s in &screened {
            for (i, r) in s.rows.iter().enumerate() {
                writeln!(
                    f,
                    "{},{},{},{},{:.5},{:.5},{:.5},{:.5},{:.3},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.5},{:.5},{:.5},{:.5},{:.5}",
                    s.game, s.phase, s.cdiff, (i == s.pick) as u8, r.z, r.p, r.w_a, r.w_b, r.turns, r.end2, r.end4,
                    r.mass[0], r.mass[1], r.mass[2], r.mass[3], r.fs_us, r.fs_opp, r.thr_us, r.thr_opp, r.lockcal
                )
                .ok();
            }
        }
    }

    let half = cli.n / 2;
    let test_games = (cli.n - half) as f64;
    let player_games = 2.0 * test_games;

    // Regression rows: (train?, y, noise var, x per group).
    struct Row {
        train: bool,
        y: f64,
        noise: f64,
    }
    let gs = groups();
    let mut rows: Vec<Row> = Vec::new();
    let mut xs: Vec<Vec<Vec<f64>>> = vec![Vec::new(); gs.len()];
    for s in &screened {
        let p = &s.rows[s.pick];
        for (i, c) in s.rows.iter().enumerate() {
            if i == s.pick {
                continue;
            }
            let diffs: Vec<f64> = c.o_a.iter().zip(&p.o_a).map(|(a, b)| (a - b) as f64).collect();
            let mu = diffs.iter().sum::<f64>() / diffs.len() as f64;
            let var = diffs.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (diffs.len() as f64 - 1.0).max(1.0);
            rows.push(Row {
                train: s.game < half,
                y: d(c.w_a, p.w_a) - d(c.p, p.p),
                noise: var / diffs.len() as f64,
            });
            for (gi, (_, f)) in gs.iter().enumerate() {
                xs[gi].push(f(c, p, s.cdiff));
            }
        }
    }

    let test_idx: Vec<usize> = (0..rows.len()).filter(|&i| !rows[i].train).collect();
    let train_idx: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].train).collect();
    let yt: Vec<f64> = test_idx.iter().map(|&i| rows[i].y).collect();
    let ymean = yt.iter().sum::<f64>() / yt.len() as f64;
    let var_y = yt.iter().map(|y| (y - ymean).powi(2)).sum::<f64>() / yt.len() as f64;
    let noise = test_idx.iter().map(|&i| rows[i].noise).sum::<f64>() / test_idx.len() as f64;
    let explainable = (var_y - noise).max(1e-12);

    // Realized gain of the static pick vs an alternative policy, per player-game.
    let test_ctx: Vec<&Screened> = screened.iter().filter(|s| s.game >= half).collect();
    let gain_of = |choose: &dyn Fn(&Screened) -> usize| -> (f64, f64) {
        let mut per_game = vec![0.0f64; cli.n];
        for s in &test_ctx {
            let j = choose(s);
            per_game[s.game] += d(s.rows[j].w_a, s.rows[s.pick].w_a);
        }
        let v: Vec<f64> = per_game[half..].to_vec();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt();
        // per game covers both seats → divide by 2 for per player-game
        (mean / 2.0, sd / 2.0 / (v.len() as f64).sqrt())
    };

    println!();
    println!(
        "{} decisions, {} candidate pairs ({} test); K={} per batch; test player-games {}",
        screened.len(),
        rows.len(),
        test_idx.len(),
        cli.k,
        player_games
    );
    println!(
        "net's Δ-error on test: var {:.5}, rollout noise {:.5}, explainable {:.5} ({:.0}%)",
        var_y,
        noise,
        explainable,
        100.0 * explainable / var_y
    );
    let (g_or, se_or) = gain_of(&|s: &Screened| {
        (0..s.rows.len())
            .max_by(|&a, &b| s.rows[a].w_b.partial_cmp(&s.rows[b].w_b).unwrap())
            .unwrap()
    });
    println!(
        "re-pick by independent rollouts (W_B argmax): {:+.4} ± {:.4} win-prob per player-game",
        g_or, se_or
    );
    println!();
    println!(
        "{:<40} {:>9} {:>11} {:>22}",
        "feature group", "R² (raw)", "R² / expl.", "held-out gain ± SE"
    );
    for (gi, (name, f)) in gs.iter().enumerate() {
        let xtr: Vec<Vec<f64>> = train_idx.iter().map(|&i| xs[gi][i].clone()).collect();
        let ytr: Vec<f64> = train_idx.iter().map(|&i| rows[i].y).collect();
        let b = ols(&xtr, &ytr);
        let pred = |x: &[f64]| b[0] + x.iter().zip(&b[1..]).map(|(a, c)| a * c).sum::<f64>();
        let sse = test_idx
            .iter()
            .map(|&i| (rows[i].y - pred(&xs[gi][i])).powi(2))
            .sum::<f64>()
            / test_idx.len() as f64;
        let r2 = 1.0 - sse / var_y;
        let r2e = (var_y - sse) / explainable;
        let (g, se) = gain_of(&|s: &Screened| {
            let p = &s.rows[s.pick];
            (0..s.rows.len())
                .max_by(|&a, &c| {
                    let sc = |j: usize| {
                        if j == s.pick {
                            p.p as f64
                        } else {
                            let x = f(&s.rows[j], p, s.cdiff);
                            s.rows[j].p as f64 + x.iter().zip(&b[1..]).map(|(a, c)| a * c).sum::<f64>()
                        }
                    };
                    sc(a).partial_cmp(&sc(c)).unwrap()
                })
                .unwrap()
        });
        println!("{:<40} {:>9.4} {:>11.3} {:>+13.4} ± {:.4}", name, r2, r2e, g, se);
    }
    eprintln!("total {:.0?}", t0.elapsed());
}
