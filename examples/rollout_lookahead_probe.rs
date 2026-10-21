//! How much would the passive lookahead inside rollouts change win-search?
//!
//! Plays N win-search self-play games. At every decision search makes, the
//! shortlist is also rolled out `k_samples` times per candidate with plain
//! passive decisions (production) and with the lookahead, on the same dice
//! (decisions never draw dice), so every difference comes from the passive
//! decisions alone. Play itself is production win-search.
//!
//! Run: cargo run --release --example rollout_lookahead_probe -- 200

use qwixxer::dqn::win::WinStrategy;
use qwixxer::game::{Game, Player};
use qwixxer::strategy::search::{phi, LookaheadProbe, SearchBot, SearchStats, BLEND};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    let games: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let net = WinStrategy::load("win_model");
    let start = std::time::Instant::now();
    let rows: Vec<LookaheadProbe> = (0..games)
        .into_par_iter()
        .flat_map_iter(|g| {
            let stats: Vec<Rc<RefCell<SearchStats>>> = (0..2)
                .map(|_| {
                    Rc::new(RefCell::new(SearchStats {
                        lookahead_probe: Some(Vec::new()),
                        ..Default::default()
                    }))
                })
                .collect();
            let players = (0..2)
                .map(|p| {
                    let mut bot = SearchBot::new(net.clone());
                    bot.stats = Some(stats[p].clone());
                    Player::new(
                        Box::new(bot),
                        Box::new(SmallRng::seed_from_u64(77_000 + 2 * g + p as u64)),
                    )
                })
                .collect();
            Game::new(players).play();
            stats
                .into_iter()
                .flat_map(|s| s.borrow_mut().lookahead_probe.take().unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    report(&rows, games, start.elapsed().as_secs_f64());
}

/// Standard error of the mean of `a[k] − b[k]`.
fn paired_se(a: &[f32], b: &[f32]) -> f64 {
    let d: Vec<f64> = a.iter().zip(b).map(|(&x, &y)| (x - y) as f64).collect();
    let m = d.iter().sum::<f64>() / d.len() as f64;
    let var = d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (d.len() - 1) as f64;
    (var / d.len() as f64).sqrt()
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p) as usize]
}

fn report(rows: &[LookaheadProbe], games: u64, secs: f64) {
    let lambda = BLEND as f64;
    let blended = |statics: &[f32], m: &[f32]| -> Vec<f64> {
        statics
            .iter()
            .zip(m)
            .map(|(&s, &r)| (1.0 - lambda) * phi(s) as f64 + lambda * r as f64)
            .collect()
    };
    let (mut shift, mut shift_se, mut gap_shift, mut gap_se) = (vec![], vec![], vec![], vec![]);
    let (mut flips, mut gain_look, mut loss_lite) = (0usize, 0.0f64, 0.0f64);
    let (mut lite_secs, mut look_secs) = (0.0, 0.0);
    for r in rows {
        lite_secs += r.lite_secs;
        look_secs += r.look_secs;
        for i in 0..r.lite.len() {
            shift.push((r.look[i] - r.lite[i]).abs() as f64);
            shift_se.push(paired_se(&r.look_samples[i], &r.lite_samples[i]));
        }
        let (bl, bk) = (blended(&r.statics, &r.lite), blended(&r.statics, &r.look));
        let (pl, pk) = (argmax(&bl), argmax(&bk));
        // The production pick against the best other candidate.
        let other = (0..bl.len())
            .filter(|&i| i != pl)
            .fold(None, |b: Option<usize>, i| match b {
                Some(j) if bl[j] >= bl[i] => Some(j),
                _ => Some(i),
            });
        if let Some(o) = other {
            gap_shift.push(((bk[pl] - bk[o]) - (bl[pl] - bl[o])).abs());
            gap_se.push(lambda * paired_se(&r.lite_samples[pl], &r.lite_samples[o]));
        }
        if pl != pk {
            flips += 1;
            gain_look += bk[pk] - bk[pl];
            loss_lite += bl[pl] - bl[pk];
        }
    }
    let n = rows.len().max(1);
    let sort = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    let (shift, gap_shift) = (sort(shift), sort(gap_shift));
    let seat_games = 2.0 * games as f64;
    println!(
        "games {games} (both seats probed), {secs:.0}s; searched decisions {}",
        rows.len()
    );
    println!(
        "candidate value |look − lite| (win prob): mean {:.4}, p50 {:.4}, p90 {:.4}, p99 {:.4}, max {:.4}",
        shift.iter().sum::<f64>() / shift.len().max(1) as f64,
        pct(&shift, 0.5),
        pct(&shift, 0.9),
        pct(&shift, 0.99),
        pct(&shift, 1.0)
    );
    println!(
        "  paired SE of that shift (same dice): mean {:.4}",
        shift_se.iter().sum::<f64>() / shift_se.len().max(1) as f64
    );
    println!(
        "blended pick-vs-runner-up gap |shift|: mean {:.4}, p90 {:.4}, p99 {:.4}; production SE of the gap at k=512: mean {:.4}",
        gap_shift.iter().sum::<f64>() / gap_shift.len().max(1) as f64,
        pct(&gap_shift, 0.9),
        pct(&gap_shift, 0.99),
        gap_se.iter().sum::<f64>() / gap_se.len().max(1) as f64
    );
    println!(
        "pick flips (full-k blended argmax): {flips} of {} ({:.2}%), {:.3} per seat-game",
        rows.len(),
        100.0 * flips as f64 / n as f64,
        flips as f64 / seat_games
    );
    println!(
        "on flips, lookahead's own estimate of its gain: {:.4} total, {:.5} per seat-game (≈ win-rate points ×100: {:.3}pp)",
        gain_look,
        gain_look / seat_games,
        100.0 * gain_look / seat_games
    );
    println!(
        "  plain rollouts' estimate of the same switch: −{:.4} total ({:.3}pp per seat-game)",
        loss_lite,
        100.0 * loss_lite / seat_games
    );
    println!(
        "rollout time: plain {lite_secs:.1}s, lookahead {look_secs:.1}s (×{:.2})",
        look_secs / lite_secs
    );
}
