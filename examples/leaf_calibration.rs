//! Calibration of the search leaf value: is the pair net's win probability
//! Φ((cdiff + μ)/σ) — what `SearchBot` scores surviving rollouts with, next to
//! exact 1/½/0 outcomes for ended ones — actually calibrated?
//!
//! Plays N 1v1 static-pair self-play games with the Lite simulator (search's
//! own rollout policy). At every turn start, both players' leaf values are
//! recorded with the final outcome (ties = ½, as search scores them).
//! Reports reliability tables (overall, by side to move — the features carry
//! no turn indicator, while search's leaves sit at a fixed side to move — and
//! by game phase), plus a recalibration fit outcome ~ Φ(a·z + b), where z is
//! the net's own z-score (a = 1, b = 0 means calibrated).
//!
//! Run: cargo run --release --example leaf_calibration -- [games]

use qwixxer::dqn::pair::PairStrategy;
use qwixxer::state::State;
use qwixxer::strategy::search::WinProb;
use qwixxer::strategy::sim::{play_sim_turn, Fidelity, SimGame};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Standard normal CDF in f64 (same Abramowitz–Stegun 7.1.26 erf as `phi`).
fn phi64(z: f64) -> f64 {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    0.5 * (1.0 + if z < 0.0 { -y } else { y })
}

fn pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Inverse of [`phi`] (bisection; `p` clamped away from 0 and 1).
fn probit(p: f64) -> f64 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    let (mut lo, mut hi) = (-8.0, 8.0);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if phi64(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Maximum-likelihood fit of y ~ Φ(a·z + b) (y in [0, 1]; ties as ½) by
/// Fisher scoring. Returns (a, b).
fn fit_probit(zs: &[f64], ys: &[f64]) -> (f64, f64) {
    let (mut a, mut b) = (1.0, 0.0);
    for _ in 0..50 {
        // Score s and Fisher information I for θ = (a, b), x = (z, 1).
        let (mut s0, mut s1, mut i00, mut i01, mut i11) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (&z, &y) in zs.iter().zip(ys) {
            let eta = a * z + b;
            let p = phi64(eta).clamp(1e-9, 1.0 - 1e-9);
            let d = pdf(eta);
            let g = (y - p) * d / (p * (1.0 - p));
            let w = d * d / (p * (1.0 - p));
            s0 += g * z;
            s1 += g;
            i00 += w * z * z;
            i01 += w * z;
            i11 += w;
        }
        let det = i00 * i11 - i01 * i01;
        let da = (i11 * s0 - i01 * s1) / det;
        let db = (i00 * s1 - i01 * s0) / det;
        a += da;
        b += db;
        if da.abs() < 1e-9 && db.abs() < 1e-9 {
            break;
        }
    }
    (a, b)
}

/// One leaf-value observation: the net's z-score and probability for a
/// player at a turn start, and that player's final outcome.
struct Obs {
    z: f64,
    p: f64,
    y: f64,
    to_move: bool,
    phase: &'static str,
}

fn phase_of(s: &[State; 2]) -> &'static str {
    let locked = s[0].count_locked().max(s[1].count_locked()) >= 1;
    let strikes = s[0].strikes.max(s[1].strikes);
    match (locked, strikes) {
        (true, 3..) => "lock+3strikes",
        (true, 2) => "lock+2strikes",
        (true, _) => "lock",
        (false, 3..) => "3strikes",
        (false, _) => "neither",
    }
}

fn play_game(bot: &PairStrategy, g: u64) -> Vec<Obs> {
    let mut sim = SimGame {
        states: vec![State::default(); 2],
        active: (g % 2) as usize,
        rngs: (0..2u64)
            .map(|p| SmallRng::seed_from_u64(0xCA11B ^ (g << 8) ^ p))
            .collect(),
        over: false,
    };
    // (z, p, to_move, phase) per player per turn start.
    let mut pending: Vec<(usize, f64, f64, bool, &'static str)> = Vec::new();
    while !sim.over {
        let st = [sim.states[0], sim.states[1]];
        let opp0 = [st[1]];
        let opp1 = [st[0]];
        let ps = bot.win_prob_multi(&[(&st[0], &opp0[..]), (&st[1], &opp1[..])]);
        let phase = phase_of(&st);
        for j in 0..2 {
            let p = ps[j] as f64;
            pending.push((j, probit(p), p, sim.active == j, phase));
        }
        play_sim_turn(bot, &mut sim, Fidelity::Lite);
    }
    let pts = [sim.states[0].count_points(), sim.states[1].count_points()];
    let outcome = |j: usize| match pts[j].cmp(&pts[1 - j]) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Equal => 0.5,
        std::cmp::Ordering::Less => 0.0,
    };
    pending
        .into_iter()
        .map(|(j, z, p, to_move, phase)| Obs {
            z,
            p,
            y: outcome(j),
            to_move,
            phase,
        })
        .collect()
}

/// (Brier, log loss, ECE over 10 equal-width bins) for probabilities `p`.
fn scores(obs: &[&Obs], p_of: impl Fn(&Obs) -> f64) -> (f64, f64, f64) {
    let n = obs.len() as f64;
    let mut brier = 0.0;
    let mut ll = 0.0;
    let mut bins = [(0.0f64, 0.0f64, 0usize); 10];
    for o in obs {
        let p = p_of(o).clamp(1e-6, 1.0 - 1e-6);
        brier += (p - o.y).powi(2);
        ll -= o.y * p.ln() + (1.0 - o.y) * (1.0 - p).ln();
        let b = ((p * 10.0) as usize).min(9);
        bins[b].0 += p;
        bins[b].1 += o.y;
        bins[b].2 += 1;
    }
    let ece = bins
        .iter()
        .filter(|b| b.2 > 0)
        .map(|b| (b.0 - b.1).abs() / n)
        .sum::<f64>();
    (brier / n, ll / n, ece)
}

fn summarize(label: &str, obs: &[&Obs]) {
    if obs.is_empty() {
        return;
    }
    let zs: Vec<f64> = obs.iter().map(|o| o.z).collect();
    let ys: Vec<f64> = obs.iter().map(|o| o.y).collect();
    let (a, b) = fit_probit(&zs, &ys);
    let (br, ll, ece) = scores(obs, |o| o.p);
    let (_, ll_fit, ece_fit) = scores(obs, |o| phi64(a * o.z + b));
    let mean_p = obs.iter().map(|o| o.p).sum::<f64>() / obs.len() as f64;
    let mean_y = ys.iter().sum::<f64>() / ys.len() as f64;
    println!(
        "{label:<18} {:>9} {:>7.4} {:>7.4} {:>+8.4} {:>7.4} {:>7.4} {:>7.4} | a {:>6.3} b {:>+7.3} | ll {:>7.4} ece {:>7.4}",
        obs.len(),
        mean_p,
        mean_y,
        mean_y - mean_p,
        br,
        ll,
        ece,
        a,
        b,
        ll_fit,
        ece_fit
    );
}

fn main() {
    let games: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let t0 = std::time::Instant::now();
    let obs: Vec<Obs> = (0..games)
        .into_par_iter()
        .map_init(|| PairStrategy::load("pair_model"), |bot, g| play_game(bot, g))
        .flatten_iter()
        .collect();
    println!(
        "{games} static-pair self-play games (Lite), {} leaf observations, {:.1}s",
        obs.len(),
        t0.elapsed().as_secs_f64()
    );
    println!("(observations within a game are correlated; treat differences < ~0.005 as noise)\n");

    let all: Vec<&Obs> = obs.iter().collect();
    println!("reliability (all):   bin      n   mean p   observed   gap");
    for b in 0..10 {
        let lo = b as f64 / 10.0;
        let sel: Vec<&&Obs> = all.iter().filter(|o| o.p >= lo && (o.p < lo + 0.1 || b == 9)).collect();
        if sel.is_empty() {
            continue;
        }
        let mp = sel.iter().map(|o| o.p).sum::<f64>() / sel.len() as f64;
        let my = sel.iter().map(|o| o.y).sum::<f64>() / sel.len() as f64;
        println!(
            "                    {lo:.1}-{:.1} {:>8} {mp:>7.3} {my:>9.3} {:>+8.3}",
            lo + 0.1,
            sel.len(),
            my - mp
        );
    }
    println!();
    println!(
        "{:<18} {:>9} {:>7} {:>7} {:>8} {:>7} {:>7} {:>7} | fit Φ(a·z + b)         | after fit",
        "slice", "n", "mean p", "obs", "obs-p", "brier", "logloss", "ece"
    );
    summarize("all", &all);
    let to_move: Vec<&Obs> = obs.iter().filter(|o| o.to_move).collect();
    let waiting: Vec<&Obs> = obs.iter().filter(|o| !o.to_move).collect();
    summarize("about to roll", &to_move);
    summarize("opponent to roll", &waiting);
    for phase in ["neither", "lock", "lock+2strikes", "lock+3strikes", "3strikes"] {
        let sel: Vec<&Obs> = obs.iter().filter(|o| o.phase == phase).collect();
        summarize(phase, &sel);
    }
    // Cells for a search-side correction: (any locked row?) × (side to move).
    println!();
    for locked in [false, true] {
        for to_move in [true, false] {
            let sel: Vec<&Obs> = obs
                .iter()
                .filter(|o| o.phase.starts_with("lock") == locked && o.to_move == to_move)
                .collect();
            let label = format!(
                "{}/{}",
                if locked { "locked" } else { "unlocked" },
                if to_move { "to roll" } else { "waiting" }
            );
            summarize(&label, &sel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwixxer::strategy::search::phi;

    #[test]
    fn probit_inverts_phi() {
        for z in [-3.0, -1.2, 0.0, 0.4, 2.5] {
            let p = phi(z as f32) as f64;
            assert!((probit(p) - z).abs() < 1e-3, "z {z}: got {}", probit(p));
        }
    }

    #[test]
    fn fit_recovers_known_parameters() {
        // Synthetic data with a known miscalibration: y ~ Bernoulli(Φ(1.3 z − 0.2)).
        use rand::Rng;
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut zs, mut ys) = (Vec::new(), Vec::new());
        for _ in 0..200_000 {
            let z: f64 = rng.gen_range(-3.0..3.0);
            let p = phi((1.3 * z - 0.2) as f32) as f64;
            zs.push(z);
            ys.push(if rng.gen::<f64>() < p { 1.0 } else { 0.0 });
        }
        let (a, b) = fit_probit(&zs, &ys);
        assert!((a - 1.3).abs() < 0.03 && (b + 0.2).abs() < 0.03, "a {a}, b {b}");
    }

    #[test]
    fn fit_is_identity_on_calibrated_data() {
        use rand::Rng;
        let mut rng = SmallRng::seed_from_u64(2);
        let (mut zs, mut ys) = (Vec::new(), Vec::new());
        for _ in 0..200_000 {
            let z: f64 = rng.gen_range(-2.5..2.5);
            zs.push(z);
            ys.push(if rng.gen::<f64>() < phi(z as f32) as f64 {
                1.0
            } else {
                0.0
            });
        }
        let (a, b) = fit_probit(&zs, &ys);
        assert!((a - 1.0).abs() < 0.03 && b.abs() < 0.03, "a {a}, b {b}");
    }
}
