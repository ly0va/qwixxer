//! Where does win-search overrule the static win net?
//!
//! Plays N win-search self-play games (production `SearchConfig::win()`),
//! logs every search (both seats) and breaks down the ones whose pick
//! differs from the static pick: by decision context, gate, game stage,
//! score margin, strikes and locks, and by what kind of move each side
//! chose. Ends with sample positions.
//!
//! Run: cargo run --release --example search_disagreements -- 1000

use qwixxer::dqn::win::WinStrategy;
use qwixxer::game::{Game, Player};
use qwixxer::state::{Mark, State};
use qwixxer::strategy::search::{SearchBot, SearchConfig, SearchContext, SearchRecord, SearchStats};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

fn main() {
    let games: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let net = WinStrategy::load("win_model");
    let start = std::time::Instant::now();
    let log: Vec<SearchRecord> = (0..games)
        .into_par_iter()
        .flat_map_iter(|g| {
            let stats: Vec<Rc<RefCell<SearchStats>>> = (0..2)
                .map(|_| {
                    Rc::new(RefCell::new(SearchStats {
                        search_log: Some(Vec::new()),
                        ..Default::default()
                    }))
                })
                .collect();
            let players = (0..2)
                .map(|p| {
                    let mut bot = SearchBot::with_config(net.clone(), SearchConfig::win());
                    bot.stats = Some(stats[p].clone());
                    Player::new(
                        Box::new(bot),
                        Box::new(SmallRng::seed_from_u64(91_000 + 2 * g + p as u64)),
                    )
                })
                .collect();
            Game::new(players).play();
            stats
                .into_iter()
                .flat_map(|s| s.borrow_mut().search_log.take().unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    println!(
        "{games} self-play games, {:.0}s: {} searches (both seats)\n",
        start.elapsed().as_secs_f64(),
        log.len()
    );
    report(&log, games);
}

fn ctx_name(c: SearchContext) -> &'static str {
    match c {
        SearchContext::ActiveP1 => "active p1",
        SearchContext::ActiveP2 => "active p2",
        SearchContext::Passive => "passive",
    }
}

fn marks(s: &State) -> u8 {
    s.row_totals().iter().sum()
}

/// Cells passed over by `m` (0 = the next free number).
fn skipped(pre: &State, m: Mark) -> u8 {
    let free = pre.row_free_values()[m.row].expect("a markable row is open");
    if m.row < 2 {
        m.number - free
    } else {
        free - m.number
    }
}

/// The kind of move a candidate is. Active phase 1 is the white mark of the
/// plan, with what the plan's colour phase adds.
fn kind(r: &SearchRecord, i: usize) -> String {
    let pre = &r.state;
    let post = &r.posts[i];
    let base = match r.marks[i] {
        None => match r.context {
            SearchContext::Passive => "skip".to_string(),
            SearchContext::ActiveP1 => "no white".to_string(),
            SearchContext::ActiveP2 if post.strikes > pre.strikes => "strike".to_string(),
            SearchContext::ActiveP2 => "skip".to_string(),
        },
        Some(m) if pre.would_lock_row(m) => "lock".to_string(),
        Some(m) => match skipped(pre, m) {
            k @ 0..=2 => format!("skip{k}"),
            _ => "skip3+".to_string(),
        },
    };
    if r.context == SearchContext::ActiveP1 {
        let colour = marks(post) as i32 - marks(pre) as i32 - r.marks[i].is_some() as i32;
        let struck = post.strikes > pre.strikes;
        let tail = match (colour > 0, struck) {
            (true, _) => "+colour",
            (false, true) => "+strike",
            (false, false) => "",
        };
        format!("{base}{tail}")
    } else {
        base
    }
}

fn board(s: &State) -> String {
    let names = ['R', 'Y', 'G', 'B'];
    let rows: Vec<String> = (0..4)
        .map(|i| match s.row_free_values()[i] {
            None => format!("{}{}L", names[i], s.row_totals()[i]),
            Some(f) => format!("{}{}>{f}", names[i], s.row_totals()[i]),
        })
        .collect();
    format!("{} x{} {:>3}pts", rows.join(" "), s.strikes, s.count_points())
}

fn mark_name(m: Option<Mark>) -> String {
    match m {
        None => "-".into(),
        Some(m) => format!("{}{}", ['R', 'Y', 'G', 'B'][m.row], m.number),
    }
}

#[derive(Default, Clone, Copy)]
struct Count {
    searched: u32,
    overruled: u32,
}

fn table(title: &str, rows: &BTreeMap<String, Count>, total_over: u32) {
    println!("{title}");
    println!(
        "  {:<24} {:>8} {:>9} {:>7} {:>9}",
        "", "searched", "overruled", "rate", "share"
    );
    for (k, c) in rows {
        println!(
            "  {:<24} {:>8} {:>9} {:>6.1}% {:>8.1}%",
            k,
            c.searched,
            c.overruled,
            100.0 * c.overruled as f64 / c.searched.max(1) as f64,
            100.0 * c.overruled as f64 / total_over.max(1) as f64
        );
    }
    println!();
}

fn by(log: &[SearchRecord], key: impl Fn(&SearchRecord) -> String) -> BTreeMap<String, Count> {
    let mut m: BTreeMap<String, Count> = BTreeMap::new();
    for r in log {
        let c = m.entry(key(r)).or_default();
        c.searched += 1;
        c.overruled += (r.pick != 0) as u32;
    }
    m
}

fn quantiles(mut v: Vec<f32>) -> String {
    if v.is_empty() {
        return "-".into();
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
    format!(
        "p10 {:.3}  p50 {:.3}  p90 {:.3}  max {:.3}",
        q(0.1),
        q(0.5),
        q(0.9),
        q(1.0)
    )
}

fn report(log: &[SearchRecord], games: u64) {
    let over: Vec<&SearchRecord> = log.iter().filter(|r| r.pick != 0).collect();
    let n_over = over.len() as u32;
    println!(
        "overruled: {n_over} of {} searches ({:.1}%), {:.2} per seat-game\n",
        log.len(),
        100.0 * n_over as f64 / log.len().max(1) as f64,
        n_over as f64 / (2 * games) as f64
    );
    table("by context", &by(log, |r| ctx_name(r.context).into()), n_over);
    table(
        "by gate",
        &by(log, |r| {
            match (r.close, r.endgame) {
                (true, true) => "close + endgame",
                (true, false) => "close only",
                (false, true) => "endgame only",
                (false, false) => "neither (forced)",
            }
            .into()
        }),
        n_over,
    );
    table(
        "by stage (marks on both boards)",
        &by(log, |r| {
            let m = marks(&r.state) + r.opps.iter().map(marks).sum::<u8>();
            format!("{:>2}-{:<2}", m / 10 * 10, m / 10 * 10 + 9)
        }),
        n_over,
    );
    table(
        "by score margin (me − opp)",
        &by(log, |r| {
            let d = r.state.count_points() - r.opps[0].count_points();
            match d {
                ..=-20 => "a: ≤ −20",
                -19..=-8 => "b: −19..−8",
                -7..=-1 => "c: −7..−1",
                0..=7 => "d: 0..7",
                8..=19 => "e: 8..19",
                _ => "f: ≥ 20",
            }
            .into()
        }),
        n_over,
    );
    table(
        "by strikes (me, opp)",
        &by(log, |r| format!("{} , {}", r.state.strikes, r.opps[0].strikes)),
        n_over,
    );
    table(
        "by locked rows",
        &by(log, |r| {
            format!("{}", r.state.count_locked().max(r.opps[0].count_locked()))
        }),
        n_over,
    );
    table(
        "by how decided (best static value, probit; |z| 2.5 = P(win) 0.6% from 0 or 1)",
        &by(log, |r| {
            match r.statics[0] {
                z if z <= -2.5 => "a: z ≤ −2.5",
                z if z <= -1.0 => "b: −2.5 < z ≤ −1",
                z if z < 1.0 => "c: |z| < 1",
                z if z < 2.5 => "d: 1 ≤ z < 2.5",
                _ => "e: z ≥ 2.5",
            }
            .into()
        }),
        n_over,
    );
    println!("search margin of overrules (blended P(win), pick − static pick) by gate");
    for (name, f) in [
        ("close only", (true, false)),
        ("close + endgame", (true, true)),
        ("endgame only", (false, true)),
    ] {
        let m: Vec<f32> = over
            .iter()
            .filter(|r| (r.close, r.endgame) == f)
            .map(|r| r.scores[r.pick] - r.scores[0])
            .collect();
        let sum: f32 = m.iter().sum();
        println!("  {:<16} n {:>5}  sum {:>7.2}  {}", name, m.len(), sum, quantiles(m));
    }
    println!();
    table("by shortlist size", &by(log, |r| format!("{}", r.marks.len())), n_over);

    // What each side picked.
    let mut moves: BTreeMap<(String, String), u32> = BTreeMap::new();
    for r in &over {
        *moves
            .entry((
                ctx_name(r.context).into(),
                format!("{} → {}", kind(r, 0), kind(r, r.pick)),
            ))
            .or_default() += 1;
    }
    let mut moves: Vec<_> = moves.into_iter().collect();
    moves.sort_by(|a, b| b.1.cmp(&a.1));
    println!("overrules by move kind (static → search), top 25");
    for ((ctx, k), n) in moves.iter().take(25) {
        println!(
            "  {:<10} {:<34} {:>5} ({:.1}%)",
            ctx,
            k,
            n,
            100.0 * *n as f64 / n_over as f64
        );
    }
    println!();

    // Which kinds search moves away from / towards, against how often
    // they are the static pick at all.
    let mut static_kind: BTreeMap<String, (u32, u32, u32)> = BTreeMap::new(); // (static pick, overruled away, chosen by search)
    for r in log {
        static_kind.entry(kind(r, 0)).or_default().0 += 1;
        if r.pick != 0 {
            static_kind.entry(kind(r, 0)).or_default().1 += 1;
            static_kind.entry(kind(r, r.pick)).or_default().2 += 1;
        }
    }
    println!("per move kind: static pick in searches, overruled away from it, chosen by search instead");
    for (k, (s, away, to)) in &static_kind {
        println!(
            "  {:<20} {:>7} {:>6} ({:>4.1}%) {:>6}",
            k,
            s,
            away,
            100.0 * *away as f64 / (*s).max(1) as f64,
            to
        );
    }
    println!();

    println!(
        "static gap, top two (probit):   all searches {}",
        quantiles(log.iter().map(|r| r.statics[0] - r.statics[1]).collect())
    );
    println!(
        "                                overruled    {}",
        quantiles(over.iter().map(|r| r.statics[0] - r.statics[r.pick]).collect())
    );
    println!(
        "search margin, pick − static pick (blended P(win)): {}",
        quantiles(over.iter().map(|r| r.scores[r.pick] - r.scores[0]).collect())
    );
    println!();

    // How many candidates sit within a static gap of the best, over every
    // candidate (the shortlist keeps the top `cands` of them).
    let gate = SearchConfig::win().gate;
    for margin in [gate, 0.1, 0.2] {
        let mut hist = [0usize; 7]; // 1..=6, 7+
        for r in log {
            let n = r.all_statics.iter().filter(|&&v| r.all_statics[0] - v < margin).count();
            hist[n.clamp(1, 7) - 1] += 1;
        }
        let row: Vec<String> = hist
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                format!(
                    "{}{}: {} ({:.1}%)",
                    i + 1,
                    if i == 6 { "+" } else { "" },
                    c,
                    100.0 * c as f64 / log.len() as f64
                )
            })
            .collect();
        println!("candidates within {margin} of the best: {}", row.join(", "));
    }
    println!();

    // The third shortlisted candidate by its static gap to the best: how
    // often search picks it, what that is worth against the pick among the
    // top two alone, and the samples it costs.
    let thirds: Vec<&SearchRecord> = log.iter().filter(|r| r.statics.len() >= 3).collect();
    println!(
        "third candidate ({} of {} searches): gap bucket, searches, picked, gain over top-two pick, mean samples (third / top two)",
        thirds.len(),
        log.len()
    );
    let edges = [0.0f32, 0.05, 0.1, 0.2, 0.5, f32::INFINITY];
    for w in edges.windows(2) {
        let b: Vec<_> = thirds
            .iter()
            .filter(|r| (w[0]..w[1]).contains(&(r.statics[0] - r.statics[2])))
            .collect();
        let picked: Vec<_> = b.iter().filter(|r| r.pick == 2).collect();
        let gain: f64 = picked
            .iter()
            .map(|r| (r.scores[2] - r.scores[0].max(r.scores[1])) as f64)
            .sum();
        let n = b.len().max(1) as f64;
        println!(
            "  {:>4.2}..{:<5.2} {:>7} {:>6} {:>8.3} {:>7.1} / {:.1}",
            w[0],
            w[1],
            b.len(),
            picked.len(),
            gain,
            b.iter().map(|r| r.samples[2] as f64).sum::<f64>() / n,
            b.iter()
                .map(|r| (r.samples[0] + r.samples[1]) as f64 / 2.0)
                .sum::<f64>()
                / n
        );
    }
    println!();

    // Endgame-only searches (not close) by the static top-two gap: what a
    // gap cap on the endgame gate would keep.
    let eg: Vec<&SearchRecord> = log.iter().filter(|r| r.endgame && !r.close).collect();
    println!(
        "endgame only: static gap, top two  all {}",
        quantiles(eg.iter().map(|r| r.statics[0] - r.statics[1]).collect())
    );
    println!(
        "                                   overruled {}",
        quantiles(
            eg.iter()
                .filter(|r| r.pick != 0)
                .map(|r| r.statics[0] - r.statics[1])
                .collect()
        )
    );
    println!("  gap bucket      searches  overrules  margin sum");
    let edges = [0.05f32, 0.1, 0.2, 0.3, 0.5, 1.0, f32::INFINITY];
    for w in edges.windows(2) {
        let b: Vec<_> = eg
            .iter()
            .filter(|r| (w[0]..w[1]).contains(&(r.statics[0] - r.statics[1])))
            .collect();
        let o: Vec<_> = b.iter().filter(|r| r.pick != 0).collect();
        println!(
            "  {:>4.2}..{:<6.2} {:>9} {:>10} {:>11.2}",
            w[0],
            w[1],
            b.len(),
            o.len(),
            o.iter().map(|r| (r.scores[r.pick] - r.scores[0]) as f64).sum::<f64>()
        );
    }
    println!();

    let mut rng = SmallRng::seed_from_u64(1);
    let mut sample: Vec<&&SearchRecord> = over.iter().collect();
    sample.shuffle(&mut rng);
    println!("sample overrules (rows: count>next free, L locked; x strikes):");
    for r in sample.iter().take(16) {
        println!(
            "  {} dice W{} {} | R{} Y{} G{} B{}",
            ctx_name(r.context),
            r.dice[0],
            r.dice[1],
            r.dice[2],
            r.dice[3],
            r.dice[4],
            r.dice[5]
        );
        println!("    me  {}", board(&r.state));
        println!("    opp {}", board(&r.opps[0]));
        for i in 0..r.marks.len() {
            println!(
                "    {} {:<4} {:<16} static {:+.3}  score {:.3}",
                if i == r.pick {
                    "S"
                } else if i == 0 {
                    "N"
                } else {
                    " "
                },
                mark_name(r.marks[i]),
                kind(r, i),
                r.statics[i],
                r.scores[i]
            );
        }
    }
}
