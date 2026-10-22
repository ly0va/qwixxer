//! How close do cheaper estimates of a position's P(win) come to the
//! play-out estimate the explorer shows (Master vs Master, no passive
//! lookahead, the roller first, played to the end)?
//!
//! The estimates: the win net alone (depth 0), and truncated play-outs that
//! play `d` turns with random dice and value the board the net's way (a
//! finished game by its result), averaged over many samples.
//!
//! Positions: turn starts of win-net self-play (lookahead on), seen from
//! the player about to roll and from the other one. The play-out reference
//! and every estimate's sampling noise are subtracted from the spread, so
//! "own error" is the estimate's systematic error.
//!
//! Run: cargo run --release --example net_vs_playouts -- 600 4000 1000

use qwixxer::dqn::win::WinStrategy;
use qwixxer::state::State;
use qwixxer::strategy::search::WinProb;
use qwixxer::strategy::sim::{BatchedRollouts, SimGame};
use qwixxer::strategy::Next;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// (you, opponent, you roll next).
type Pos = (State, State, bool);

/// The depths measured; `None` plays to the end.
const DEPTHS: [Option<usize>; 7] = [Some(1), Some(2), Some(3), Some(4), Some(6), Some(8), None];

fn positions(win: &WinStrategy, n: usize) -> Vec<Pos> {
    let mut rng = SmallRng::seed_from_u64(77);
    let sims = (0..n as u64)
        .map(|g| SimGame {
            states: vec![State::default(); 2],
            active: (g % 2) as usize,
            rngs: (0..2).map(|p| SmallRng::seed_from_u64(9_000 + 2 * g + p)).collect(),
            over: false,
        })
        .collect();
    let mut d = BatchedRollouts::new(win, sims);
    d.passive_lookahead = true;
    let mut seen: Vec<Vec<Pos>> = vec![vec![]; n];
    while !d.all_over() {
        for (i, s) in d.sims.iter().enumerate().filter(|(_, s)| !s.over) {
            seen[i].push((s.states[0], s.states[1], s.active == 0));
        }
        d.step_turn();
    }
    // One turn start per game, uniformly over its length, alternating
    // whose point of view.
    seen.iter()
        .enumerate()
        .map(|(i, v)| {
            let (a, b, me) = v[rng.gen_range(0..v.len())];
            if i % 2 == 0 {
                (a, b, me)
            } else {
                (b, a, !me)
            }
        })
        .collect()
}

/// One estimate from `samples` games of at most `depth` turns: the mean
/// value for "you", the per-sample variance, and the turns simulated.
fn truncated(
    win: &WinStrategy,
    (you, opp, mine): Pos,
    samples: usize,
    depth: Option<usize>,
    seed: u64,
) -> (f64, f64, usize) {
    let sims = (0..samples as u64)
        .map(|k| SimGame {
            states: vec![you, opp],
            active: if mine { 0 } else { 1 },
            rngs: (0..2).map(|p| SmallRng::seed_from_u64(seed ^ (k << 2) ^ p)).collect(),
            over: false,
        })
        .collect();
    let mut d = BatchedRollouts::new(win, sims);
    let mut turns = 0;
    let mut played = 0;
    while !d.all_over() && depth.is_none_or(|k| played < k) {
        turns += d.sims.iter().filter(|s| !s.over).count();
        d.step_turn();
        played += 1;
    }
    let live: Vec<&SimGame> = d.sims.iter().filter(|s| !s.over).collect();
    let groups: Vec<(&State, &[State], Next)> = live
        .iter()
        .map(|s| {
            (
                &s.states[0],
                &s.states[1..],
                if s.active == 0 { Next::Me } else { Next::Opp },
            )
        })
        .collect();
    let mut leaf = win.win_prob_multi_next(&groups).into_iter();
    let o: Vec<f64> = d
        .sims
        .iter()
        .map(|s| if s.over { SimGame::outcome(&s.states) } else { leaf.next().unwrap() } as f64)
        .collect();
    let m = o.iter().sum::<f64>() / o.len() as f64;
    let v = o.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (o.len() - 1) as f64;
    (m, v, turns)
}

/// Per position: the reference (mean, variance of the mean, marks), and per
/// depth (mean, per-sample variance, turns per sample); depth 0 first.
struct Row {
    reference: (f64, f64, u32),
    est: Vec<(f64, f64, f64)>,
}

fn main() {
    let arg = |i: usize, d: usize| std::env::args().nth(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (n, games, samples) = (arg(1, 300), arg(2, 3000), arg(3, 1000));
    let win = WinStrategy::load("win_model");
    let ps = positions(&win, n);
    let t = std::time::Instant::now();
    let rows: Vec<Row> = ps
        .par_iter()
        .enumerate()
        .map(|(i, &(you, opp, mine))| {
            let next = if mine { Next::Me } else { Next::Opp };
            let net = win.win_prob_multi_next(&[(&you, std::slice::from_ref(&opp), next)])[0] as f64;
            let (m, v, _) = truncated(&win, (you, opp, mine), games, None, 0xC0FFEE ^ i as u64);
            let marks = (you.row_totals().iter().sum::<u8>() + opp.row_totals().iter().sum::<u8>()) as u32;
            // Independent dice from the reference's.
            let est = std::iter::once((net, 0.0, 0.0))
                .chain(DEPTHS.iter().map(|&d| {
                    let (e, var, turns) = truncated(&win, (you, opp, mine), samples, d, 0xD1CE ^ (i as u64) << 20);
                    (e, var, turns as f64 / samples as f64)
                }))
                .collect();
            Row {
                reference: (m, v / games as f64, marks),
                est,
            }
        })
        .collect();
    println!(
        "{n} positions; reference {games} full play-outs, each estimate {samples} samples; {:.0}s\n",
        t.elapsed().as_secs_f64()
    );
    let names: Vec<String> = std::iter::once("net (depth 0)".to_string())
        .chain(
            DEPTHS
                .iter()
                .map(|d| d.map_or("to the end".to_string(), |k| format!("depth {k}"))),
        )
        .collect();
    println!(
        "estimate        own error  bias     max|diff|  sd/sample  turns/sample  samples for ±1pt  turns for ±1pt"
    );
    for (j, name) in names.iter().enumerate() {
        let all: Vec<&Row> = rows.iter().collect();
        let (own, bias, max) = own_error(&all, j, samples);
        let var = rows.iter().map(|r| r.est[j].1).sum::<f64>() / n as f64;
        let turns = rows.iter().map(|r| r.est[j].2).sum::<f64>() / n as f64;
        // Samples for a standard error of 0.01, on top of the own error.
        let need = (var / 1e-4).ceil();
        println!(
            "{name:14}  {own:.4}     {bias:+.4}  {max:.3}      {:.3}      {turns:5.1}         {:>6}            {:>7}",
            var.sqrt(),
            need.max(1.0),
            (need * turns).round()
        );
    }
    for (title, buckets) in [(
        "by stage (marks on both sheets)",
        [(0, 10), (10, 20), (20, 30), (30, 99)],
    )] {
        println!("\nown error {title}:");
        print!("  {:14}", "");
        for (lo, hi) in buckets {
            print!("  {:>8}", format!("{lo}–{}", hi - 1));
        }
        println!();
        for (j, name) in names.iter().enumerate() {
            print!("  {name:14}");
            for (lo, hi) in buckets {
                let v: Vec<&Row> = rows
                    .iter()
                    .filter(|r| r.reference.2 >= lo && r.reference.2 < hi)
                    .collect();
                print!("  {:8.4}", own_error(&v, j, samples).0);
            }
            println!();
        }
    }
}

/// Systematic error of estimate `j` against the reference, with both
/// sampling noises removed: (own error, mean bias, max |diff|).
fn own_error(rows: &[&Row], j: usize, samples: usize) -> (f64, f64, f64) {
    let n = rows.len() as f64;
    let diff = |r: &Row| r.est[j].0 - r.reference.0;
    let bias = rows.iter().map(|r| diff(r)).sum::<f64>() / n;
    let ms = rows.iter().map(|r| diff(r).powi(2)).sum::<f64>() / n;
    let noise = rows
        .iter()
        .map(|r| r.reference.1 + r.est[j].1 / samples as f64)
        .sum::<f64>()
        / n;
    let max = rows.iter().map(|r| diff(r).abs()).fold(0.0, f64::max);
    ((ms - noise).max(0.0).sqrt(), bias, max)
}
