//! THROWAWAY probe: where does the win net, freed from the safe-lock rule,
//! decline the lock, and what does each decline cost?
//!
//! Plays the model against itself with the rule on (production). At every
//! decision where the rule fires, asks the same net without the rule
//! (`NoLock`) for its pick. Where the pick differs, the lock and the pick
//! are each labelled with K full-game CRN rollouts by the rule-on net: one
//! decision changed, the rest of the game played as production plays it.
//! Active phase 1 completes the turn's phase 2 without the rule (as the
//! freed bot would); a passive decision completes the opponent's turn
//! (which the static value under option (a) does not see).
//!
//! Run: cargo test --release --lib dqn::lock_probe -- --ignored --nocapture
//! Env: LOCK_MODEL (win_model), LOCK_GAMES (20000), LOCK_K (512), LOCK_SEED (1).

use crate::dqn::win::WinStrategy;
use crate::dqn::win_distill::{label_decision, CandDecision};
use crate::game::{Game, Player};
use crate::state::{Mark, State};
use crate::strategy::bot_impl::{
    active_phase1_choices, active_phase1_impl, active_phase2_impl, is_safe_lock, passive_phase1_impl,
};
use crate::strategy::search::{context_seed, passive_entry, passive_first_active, phase2_entry, WinProb};
use crate::strategy::sim::SimGame;
use crate::strategy::{Next, NoLock, Strategy};
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ctx {
    /// Passive phase 1 (white sum).
    Pp1,
    /// Active phase 1 (white sum, then the turn's colour phase).
    Ap1,
    /// Active phase 2 (colour).
    Ap2,
}

const CTXS: [(Ctx, &str); 3] = [
    (Ctx::Pp1, "passive p1"),
    (Ctx::Ap1, "active p1"),
    (Ctx::Ap2, "active p2"),
];

#[derive(Clone, Debug)]
struct Event {
    ctx: Ctx,
    state: State,
    opps: Vec<State>,
    dice: [u8; 6],
    has_marked: bool,
    active_player: usize,
    lock: Option<Mark>,
    alt: Option<Mark>,
}

#[derive(Debug)]
struct Spy {
    bot: WinStrategy,
    log: Rc<RefCell<Vec<Event>>>,
}

impl Spy {
    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        ctx: Ctx,
        state: &State,
        opps: &[State],
        dice: [u8; 6],
        has_marked: bool,
        active_player: usize,
        lock: Option<Mark>,
        alt: Option<Mark>,
    ) {
        self.log.borrow_mut().push(Event {
            ctx,
            state: *state,
            opps: opps.to_vec(),
            dice,
            has_marked,
            active_player,
            lock,
            alt,
        });
    }

    fn free(&self) -> NoLock<WinStrategy> {
        NoLock(self.bot.clone())
    }
}

impl Strategy for Spy {
    fn active_phase1(&mut self, state: &State, opps: &[State], dice: [u8; 6]) -> Option<Mark> {
        let pick = active_phase1_impl(&self.bot, state, opps, dice);
        if is_safe_lock(state, pick) {
            let alt = active_phase1_impl(&self.free(), state, opps, dice);
            self.record(Ctx::Ap1, state, opps, dice, false, 0, pick, alt);
        }
        pick
    }

    fn active_phase2(&mut self, state: &State, opps: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let pick = active_phase2_impl(&self.bot, state, opps, dice, has_marked);
        if is_safe_lock(state, pick) {
            let alt = active_phase2_impl(&self.free(), state, opps, dice, has_marked);
            self.record(Ctx::Ap2, state, opps, dice, has_marked, 0, pick, alt);
        }
        pick
    }

    fn passive_phase1(&mut self, state: &State, opps: &[State], dice: [u8; 6], active_player: usize) -> Option<Mark> {
        let pick = passive_phase1_impl(&self.bot, state, opps, dice);
        if is_safe_lock(state, pick) {
            let alt = passive_phase1_impl(&self.free(), state, opps, dice);
            self.record(Ctx::Pp1, state, opps, dice, false, active_player, pick, alt);
        }
        pick
    }
}

fn env(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The entry (all-player states, ended) after choosing `m` at `e`.
fn entry(bot: &WinStrategy, e: &Event, m: Option<Mark>, sim_opp: &[State]) -> (Vec<State>, bool) {
    match e.ctx {
        Ctx::Pp1 => passive_entry(bot, &e.state, &e.opps, e.dice, e.active_player, m),
        Ctx::Ap2 => phase2_entry(ap2_post(e, m), &e.opps),
        Ctx::Ap1 => {
            // search::phase1_entry, with this turn's phase 2 played without the rule.
            let mut our = e.state;
            if let Some(x) = m {
                our.apply_mark(x);
            }
            let mut all: Vec<State> = std::iter::once(our).chain(sim_opp.iter().copied()).collect();
            SimGame::propagate_locks(&mut all);
            if SimGame::game_over(&all) {
                return (all, true);
            }
            let view = SimGame::opp_view(&all, 0);
            match active_phase2_impl(&NoLock(bot.clone()), &all[0], &view, e.dice, m.is_some()) {
                Some(x) => all[0].apply_mark(x),
                None if m.is_none() => all[0].apply_strike(),
                None => {}
            }
            SimGame::propagate_locks(&mut all);
            let ended = SimGame::game_over(&all);
            (all, ended)
        }
    }
}

fn ap2_post(e: &Event, m: Option<Mark>) -> State {
    let mut post = e.state;
    match m {
        Some(x) => post.apply_mark(x),
        None if !e.has_marked => post.apply_strike(),
        None => {}
    }
    post
}

/// The net's own P(win) for the lock and for its pick, as the freed bot
/// valued them (probit is monotone, so the same argmax). Active phase 1:
/// the completed turn of each arm (the lock's own plans can all be pruned
/// as dominated, by a plan that locks the row in the colour phase).
fn statics(bot: &WinStrategy, e: &Event, arms: &[(Vec<State>, bool); 2]) -> [f64; 2] {
    let one = |post: &State, opps: &[State], next: Next| bot.win_prob_multi_next(&[(post, opps, next)])[0] as f64;
    let marks = [e.lock, e.alt];
    match e.ctx {
        Ctx::Pp1 => marks.map(|m| {
            let mut post = e.state;
            if let Some(x) = m {
                post.apply_mark(x);
            }
            one(&post, &e.opps, Next::Me)
        }),
        Ctx::Ap2 => marks.map(|m| one(&ap2_post(e, m), &e.opps, Next::Opp)),
        Ctx::Ap1 => [0, 1].map(|i| {
            let (all, ended) = &arms[i];
            if *ended {
                SimGame::outcome(all) as f64
            } else {
                one(&all[0], &all[1..], Next::Opp)
            }
        }),
    }
}

struct Res {
    ctx: Ctx,
    cdiff: isize,
    /// Rollout value (for the decider) of its pick minus the lock's.
    gap: f64,
    /// Standard error of `gap` (CRN pairs).
    se: f64,
    /// The net's static gap, same sign.
    sgap: f64,
    /// The pick locks a row later in the same turn (active phase 1: the
    /// colour phase).
    relock: bool,
}

fn adjudicate(bot: &WinStrategy, e: &Event, k: usize) -> Res {
    let sim_opp = match e.ctx {
        Ctx::Ap1 => active_phase1_choices(bot, &e.state, &e.opps, e.dice).1,
        _ => e.opps.clone(),
    };
    let arms = [entry(bot, e, e.lock, &sim_opp), entry(bot, e, e.alt, &sim_opp)];
    let relock = arms[1].0[0].count_locked() > e.state.count_locked();
    let s = statics(bot, e, &arms);
    let d = CandDecision {
        entries: arms.to_vec(),
        first_active: match e.ctx {
            Ctx::Pp1 => passive_first_active(e.active_player, 2),
            _ => 1,
        },
        seed: context_seed(&e.state, &e.opps, e.dice),
    };
    let g = label_decision(bot, &d, k);
    let mine = |y: f32| if g.swapped { 1.0 - y as f64 } else { y as f64 };
    Res {
        ctx: e.ctx,
        cdiff: e.state.count_points() - e.opps[0].count_points(),
        gap: mine(g.samples[1].1) - mine(g.samples[0].1),
        se: (g.pair_var[0] as f64 / k as f64).sqrt(),
        sgap: s[1] - s[0],
        relock,
    }
}

fn summary(label: &str, rs: &[&Res], games: f64) {
    let n = rs.len();
    if n == 0 {
        eprintln!("  {label:<14} n 0");
        return;
    }
    let nf = n as f64;
    let mean = rs.iter().map(|r| r.gap).sum::<f64>() / nf;
    let sd = (rs.iter().map(|r| (r.gap - mean).powi(2)).sum::<f64>() / (nf - 1.0).max(1.0)).sqrt();
    let better = rs.iter().filter(|r| r.gap > 2.0 * r.se).count();
    let worse = rs.iter().filter(|r| r.gap < -2.0 * r.se).count();
    let smean = rs.iter().map(|r| r.sgap).sum::<f64>() / nf;
    let err2 = rs.iter().map(|r| (r.sgap - r.gap).powi(2)).sum::<f64>() / nf;
    let noise2 = rs.iter().map(|r| r.se * r.se).sum::<f64>() / nf;
    eprintln!(
        "  {label:<14} n {n:>6}  pick − lock {:+.4} ± {:.4}  (pick better {:.0}%, lock better {:.0}%)  \
         static {:+.4}, own error {:.3}  →  {:+.3}pp per game for one freed seat",
        mean,
        sd / nf.sqrt(),
        100.0 * better as f64 / nf,
        100.0 * worse as f64 / nf,
        smean,
        (err2 - noise2).max(0.0).sqrt(),
        100.0 * rs.iter().map(|r| r.gap).sum::<f64>() / (2.0 * games),
    );
}

#[test]
#[ignore]
fn lock_probe() {
    let model = std::env::var("LOCK_MODEL").unwrap_or_else(|_| "win_model".into());
    let games = env("LOCK_GAMES", 20_000);
    let k = env("LOCK_K", 512) as usize;
    let seed = env("LOCK_SEED", 1);
    let bot = WinStrategy::load(&model);
    eprintln!("model {model}: {games} games of itself (rule on), {k} rollouts per arm");

    let t0 = Instant::now();
    let events: Vec<Event> = (0..games)
        .into_par_iter()
        .flat_map_iter(|g| {
            let log = Rc::new(RefCell::new(Vec::new()));
            let dice = |seat: u64| {
                Box::new(SmallRng::seed_from_u64(
                    seed.wrapping_mul(1_000_003).wrapping_add(g * 2 + seat),
                ))
            };
            let spy = |log: &Rc<RefCell<Vec<Event>>>| -> Box<dyn Strategy> {
                Box::new(Spy {
                    bot: bot.clone(),
                    log: log.clone(),
                })
            };
            let mut game = Game::new(vec![Player::new(spy(&log), dice(0)), Player::new(spy(&log), dice(1))]);
            game.play();
            let out = log.borrow().clone();
            out
        })
        .collect();
    let declines: Vec<&Event> = events.iter().filter(|e| e.alt != e.lock).collect();
    eprintln!("played in {:.0}s", t0.elapsed().as_secs_f64());

    let gf = games as f64;
    eprintln!("\nfirings (both seats) and how often the freed net declines:");
    for (ctx, label) in CTXS {
        let f = events.iter().filter(|e| e.ctx == ctx).count();
        let d = declines.iter().filter(|e| e.ctx == ctx).count();
        eprintln!(
            "  {label:<14} {f:>6} firings ({:.3}/game), {d:>6} declined ({:.1}%)",
            f as f64 / gf,
            100.0 * d as f64 / f.max(1) as f64
        );
    }

    let t0 = Instant::now();
    let res: Vec<Res> = declines.par_iter().map(|e| adjudicate(&bot, e, k)).collect();
    eprintln!(
        "\nadjudicated {} declines in {:.0}s",
        res.len(),
        t0.elapsed().as_secs_f64()
    );

    eprintln!("\nper decline (decider's win share, tie ½; own error = √(static-vs-rollout² − noise²)):");
    for (ctx, label) in CTXS {
        let rs: Vec<&Res> = res.iter().filter(|r| r.ctx == ctx).collect();
        summary(label, &rs, gf);
    }
    summary("all", &res.iter().collect::<Vec<_>>(), gf);
    let (re, other): (Vec<&Res>, Vec<&Res>) = res.iter().filter(|r| r.ctx == Ctx::Ap1).partition(|r| r.relock);
    eprintln!("\nactive p1, by whether the pick locks in the colour phase instead:");
    summary("locks later", &re, gf);
    summary("no lock", &other, gf);
    eprintln!("\nby score margin (decider's points − opponent's), all contexts:");
    let bins: [(&str, fn(isize) -> bool); 5] = [
        ("≤ −10", |c| c <= -10),
        ("−9 .. −1", |c| (-9..=-1).contains(&c)),
        ("0", |c| c == 0),
        ("1 .. 9", |c| (1..=9).contains(&c)),
        ("≥ 10", |c| c >= 10),
    ];
    for (label, f) in bins {
        let rs: Vec<&Res> = res.iter().filter(|r| f(r.cdiff)).collect();
        summary(label, &rs, gf);
    }
    eprintln!("\nby context, behind (< 0) vs not:");
    for (ctx, label) in CTXS {
        for (side, behind) in [("behind", true), ("level+", false)] {
            let rs: Vec<&Res> = res.iter().filter(|r| r.ctx == ctx && (r.cdiff < 0) == behind).collect();
            summary(&format!("{label} {side}"), &rs, gf);
        }
    }
}
