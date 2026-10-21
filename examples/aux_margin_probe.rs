//! How well does the win net's aux head predict the final score difference,
//! against the pair net's margin (current difference + μ) that the web
//! explorer shows today?
//!
//! Plays N win-net self-play games (passive lookahead on) and, at the start
//! of every turn, records both seats' view of the position. Each prediction
//! is compared with the game's actual final difference.
//!
//! Run: cargo run --release --example aux_margin_probe -- 5000

use qwixxer::dqn::pair::{pair_features, PairStrategy};
use qwixxer::dqn::win::WinStrategy;
use qwixxer::dqn::{LOG_VAR_MAX, LOG_VAR_MIN};
use qwixxer::state::State;
use qwixxer::strategy::sim::{BatchedRollouts, SimGame};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// One seat's view of a position: (me, opponent, game index, seat, seat moves next).
type Seen = (State, State, usize, usize, bool);

struct Row {
    marks: u32,
    mover: bool,
    actual: f32,
    aux: f32,
    pair: f32,
    sigma: f32,
}

fn main() {
    let games: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(5000);
    let win = WinStrategy::load("win_model");
    let pair = PairStrategy::load("pair_model");
    let start = std::time::Instant::now();
    let rows: Vec<Row> = (0..games)
        .collect::<Vec<_>>()
        .par_chunks(250)
        .flat_map_iter(|chunk| {
            let sims = chunk
                .iter()
                .map(|&g| SimGame {
                    states: vec![State::default(); 2],
                    active: g % 2,
                    rngs: (0..2)
                        .map(|p| SmallRng::seed_from_u64(31_000 + 2 * g as u64 + p))
                        .collect(),
                    over: false,
                })
                .collect();
            let mut driver = BatchedRollouts::new(&win, sims);
            driver.passive_lookahead = true;
            let mut seen: Vec<Seen> = Vec::new();
            while !driver.all_over() {
                for (i, sim) in driver.sims.iter().enumerate().filter(|(_, s)| !s.over) {
                    seen.push((sim.states[0], sim.states[1], i, 0, sim.active == 0));
                    seen.push((sim.states[1], sim.states[0], i, 1, sim.active == 1));
                }
                driver.step_turn();
            }
            let finals: Vec<[i32; 2]> = driver
                .sims
                .iter()
                .map(|s| [s.states[0].count_points() as i32, s.states[1].count_points() as i32])
                .collect();
            let feats: Vec<_> = seen
                .iter()
                .map(|(me, opp, _, _, _)| pair_features(me, opp, &[*opp]))
                .collect();
            let pair_out = pair.net.forward(&feats);
            seen.iter()
                .zip(pair_out)
                .map(|(&(me, opp, i, seat, mover), (mu, log_var))| {
                    let now = (me.count_points() - opp.count_points()) as f32;
                    Row {
                        mover,
                        marks: (me.row_totals().iter().sum::<u8>() + opp.row_totals().iter().sum::<u8>()) as u32,
                        actual: (finals[i][seat] - finals[i][1 - seat]) as f32,
                        aux: win.net.aux_states(&me, &opp).expect("win_model has aux heads").diff,
                        pair: now + mu,
                        sigma: (0.5 * log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)).exp(),
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    println!(
        "{games} win-net self-play games, {} positions (both seats), {:.0}s\n",
        rows.len(),
        start.elapsed().as_secs_f64()
    );
    report(&rows);
}

fn stats(rows: &[&Row], pred: impl Fn(&Row) -> f32) -> (f64, f64) {
    let n = rows.len().max(1) as f64;
    let bias = rows.iter().map(|r| (pred(r) - r.actual) as f64).sum::<f64>() / n;
    let rmse = (rows.iter().map(|r| ((pred(r) - r.actual) as f64).powi(2)).sum::<f64>() / n).sqrt();
    (bias, rmse)
}

fn report(rows: &[Row]) {
    let all: Vec<&Row> = rows.iter().collect();
    let gap = |r: &Row| (r.aux - r.pair) as f64;
    let n = all.len() as f64;
    let (ab, ar) = stats(&all, |r| r.aux);
    let (pb, pr) = stats(&all, |r| r.pair);
    println!("overall: aux bias {ab:+.2}, RMSE {ar:.2} | pair bias {pb:+.2}, RMSE {pr:.2}");
    println!(
        "aux − pair: mean {:+.2}, RMS {:.2}",
        all.iter().map(|r| gap(r)).sum::<f64>() / n,
        (all.iter().map(|r| gap(r).powi(2)).sum::<f64>() / n).sqrt()
    );
    let z2 = all
        .iter()
        .map(|r| ((r.actual - r.pair) / r.sigma).powi(2) as f64)
        .sum::<f64>()
        / n;
    for (name, m) in [("to move next", true), ("waiting", false)] {
        let v: Vec<&Row> = rows.iter().filter(|r| r.mover == m).collect();
        let (ab, ar) = stats(&v, |r| r.aux);
        let (pb, pr) = stats(&v, |r| r.pair);
        println!("  seat {name}: aux bias {ab:+.2}, RMSE {ar:.2} | pair bias {pb:+.2}, RMSE {pr:.2}");
    }
    println!("pair σ calibration: mean ((actual − pair)/σ)² = {z2:.3} (1 = calibrated)\n");

    println!("by stage (marks on both boards): positions, aux bias / RMSE, pair bias / RMSE, mean pair σ");
    let mut stage: BTreeMap<u32, Vec<&Row>> = BTreeMap::new();
    for r in rows {
        stage.entry(r.marks / 10 * 10).or_default().push(r);
    }
    for (s, v) in &stage {
        let (ab, ar) = stats(v, |r| r.aux);
        let (pb, pr) = stats(v, |r| r.pair);
        let sig = v.iter().map(|r| r.sigma as f64).sum::<f64>() / v.len() as f64;
        println!(
            "  {:>2}–{:<2} {:>8}  {:+6.2} / {:5.2}   {:+6.2} / {:5.2}   {:5.2}",
            s,
            s + 9,
            v.len(),
            ab,
            ar,
            pb,
            pr,
            sig
        );
    }
    println!();

    println!("calibration by aux prediction: bucket, positions, mean aux, mean actual");
    let mut cal: BTreeMap<i32, Vec<&Row>> = BTreeMap::new();
    for r in rows {
        cal.entry(((r.aux / 10.0).round() as i32).clamp(-6, 6))
            .or_default()
            .push(r);
    }
    for (b, v) in &cal {
        let m = v.len() as f64;
        println!(
            "  {:>+4} {:>8}  {:+7.2}  {:+7.2}",
            b * 10,
            v.len(),
            v.iter().map(|r| r.aux as f64).sum::<f64>() / m,
            v.iter().map(|r| r.actual as f64).sum::<f64>() / m
        );
    }
}
