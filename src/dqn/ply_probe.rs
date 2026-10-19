//! THROWAWAY probe (do not commit): does the static net's error concentrate
//! where it disagrees with its own exact 1-ply lookahead?
//!
//! 1-ply value of a turn-end position (the other player to move) = exact
//! expectation over the next turn's dice of a rule-free greedy turn on the
//! net (passive: the bot's own decision on the white sum; active: best of
//! strike / white-only / white-then-colour / colour-only, exact over the 21
//! white outcomes and the independent colour dice). Ended positions are exact.

use crate::dqn::win::WinStrategy;
use crate::dqn::win_data::{self, label_decision_full, DecisionData, Sample};
use crate::dqn::win_distill::CandDecision;
use crate::state::{Mark, State};
use crate::strategy::bot_impl::{eval_decision, passive_phase1_choices};
use crate::strategy::search::WinProb;
use crate::strategy::sim::SimGame;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Val {
    Net(usize),
    Exact(f32),
}

#[derive(Default)]
struct Collector {
    map: HashMap<([u8; 13], [u8; 13]), usize>,
    pairs: Vec<(State, State)>,
}

impl Collector {
    fn add(&mut self, a: State, p: State) -> Val {
        let both = [a, p];
        if SimGame::game_over(&both) {
            return Val::Exact(SimGame::outcome(&both));
        }
        let key = (a.to_bytes(), p.to_bytes());
        let n = self.pairs.len();
        Val::Net(*self.map.entry(key).or_insert_with(|| {
            self.pairs.push((a, p));
            n
        }))
    }

    fn eval(&self, net: &WinStrategy) -> Vec<f32> {
        let groups: Vec<(&State, &[State])> = self.pairs.iter().map(|(a, p)| (a, std::slice::from_ref(p))).collect();
        if groups.is_empty() {
            return vec![];
        }
        net.win_prob_multi(&groups)
    }
}

struct Table {
    k: Vec<Val>,
    g: [[Vec<Val>; 6]; 4],
}

fn pair(a: State, p: State) -> (State, State) {
    let mut s = [a, p];
    SimGame::propagate_locks(&mut s);
    (s[0], s[1])
}

fn table(col: &mut Collector, a0: State, p1: State, w1: u8, w2: u8) -> Table {
    let mut t = Table {
        k: vec![],
        g: Default::default(),
    };
    let wms: Vec<Option<Mark>> = std::iter::once(None)
        .chain(a0.generate_white_moves(w1 + w2).into_iter().map(Some))
        .collect();
    for wm in wms {
        let mut a = a0;
        if let Some(m) = wm {
            a.apply_mark(m);
        }
        let (a1, p) = pair(a, p1);
        if SimGame::game_over(&[a1, p]) {
            t.k.push(col.add(a1, p));
            continue;
        }
        if wm.is_none() {
            let mut s = a1;
            s.apply_strike();
            t.k.push(col.add(s, p));
        } else {
            t.k.push(col.add(a1, p));
        }
        for c in 0..4 {
            for d in 1..=6u8 {
                let mut ns = vec![w1 + d];
                if w2 != w1 {
                    ns.push(w2 + d);
                }
                for n in ns {
                    if a1.can_mark(c, n) {
                        let mut s = a1;
                        s.apply_mark(Mark { row: c, number: n });
                        let (a2, p2) = pair(s, p);
                        t.g[c][d as usize - 1].push(col.add(a2, p2));
                    }
                }
            }
        }
    }
    t
}

fn resolve(v: &[f32], vals: &[Val]) -> f32 {
    vals.iter()
        .map(|x| match *x {
            Val::Net(i) => v[i],
            Val::Exact(e) => e,
        })
        .fold(f32::NEG_INFINITY, f32::max)
}

fn expected_max(k: f32, g: &[[f32; 6]; 4]) -> f64 {
    let mut vals: Vec<f32> = std::iter::once(k)
        .chain(g.iter().flatten().copied().filter(|&x| x > k))
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals.dedup();
    let cdf = |x: f32| -> f64 {
        g.iter()
            .map(|row| row.iter().filter(|&&y| y <= x).count() as f64 / 6.0)
            .product()
    };
    let (mut e, mut prev) = (0.0, 0.0);
    for &x in &vals {
        let c = cdf(x);
        e += x as f64 * (c - prev);
        prev = c;
    }
    e
}

fn passive_after(net: &WinStrategy, a0: State, p0: State, s: u8) -> State {
    let dice = [s - s / 2, s / 2, 1, 1, 1, 1];
    let opp = [a0];
    let mut p = p0;
    if let Some(m) = eval_decision(net, passive_phase1_choices(&p0, &opp, dice), &opp) {
        p.apply_mark(m);
    }
    p
}

/// P(win) for `me`, `opp` to move next: (static, 1-ply).
fn values(net: &WinStrategy, me: State, opp: State) -> (f32, f32) {
    let both = [me, opp];
    if SimGame::game_over(&both) {
        let o = SimGame::outcome(&both);
        return (o, o);
    }
    let stat = net.win_prob_multi(&[(&me, std::slice::from_ref(&opp))])[0];
    let (a0, p0) = (opp, me);
    let passive: Vec<State> = (2..=12).map(|s| passive_after(net, a0, p0, s)).collect();
    let mut col = Collector::default();
    let mut tables = vec![];
    for x in 1..=6u8 {
        for y in x..=6u8 {
            let wgt = if x == y { 1.0 } else { 2.0 } / 36.0;
            tables.push((wgt, table(&mut col, a0, passive[(x + y - 2) as usize], x, y)));
        }
    }
    let v = col.eval(net);
    let mut e = 0.0;
    for (wgt, t) in &tables {
        let k = resolve(&v, &t.k);
        let g: [[f32; 6]; 4] = std::array::from_fn(|c| std::array::from_fn(|d| resolve(&v, &t.g[c][d])));
        e += wgt * expected_max(k, &g);
    }
    (stat, (1.0 - e) as f32)
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

struct Dec {
    s0: Vec<f32>,
    s1: Vec<f32>,
    y: Vec<f32>,
    pv: Vec<f32>,
    swapped: bool,
}

impl Dec {
    fn mine(&self, v: &[f32], i: usize) -> f32 {
        if self.swapped {
            1.0 - v[i]
        } else {
            v[i]
        }
    }
    fn pick(&self, v: &[f32]) -> usize {
        (1..v.len()).fold(0, |b, i| if self.mine(v, i) > self.mine(v, b) { i } else { b })
    }
    /// Largest change of a candidate difference from static to 1-ply.
    fn disagreement(&self) -> f32 {
        (1..self.y.len())
            .map(|i| ((self.s1[i] - self.s1[0]) - (self.s0[i] - self.s0[0])).abs())
            .fold(0.0, f32::max)
    }
}

/// Per-group sums: (count, Σ static err², Σ 1-ply err², Σ noise var, gain sum, gain sq sum, decisions).
#[derive(Default, Clone, Copy)]
struct Acc {
    n: f64,
    e0: f64,
    e1: f64,
    noise: f64,
    gain: f64,
    gain2: f64,
    groups: f64,
}

impl Acc {
    fn own(&self, e: f64) -> f64 {
        ((e - self.noise) / self.n).max(0.0).sqrt()
    }
}

fn print_table(title: &str, accs: &[(String, Acc)], with_gain: bool) {
    let total: f64 = accs.iter().map(|(_, a)| a.e0 - a.noise).sum();
    eprintln!("\n{title}");
    eprintln!(
        "{:<22} {:>8} {:>7} {:>10} {:>10} {:>10} {:>9}{}",
        "group",
        "items",
        "share",
        "rmse",
        "own static",
        "own 1-ply",
        "err² share",
        if with_gain { "   1-ply gain (se)" } else { "" }
    );
    let all: f64 = accs.iter().map(|(_, a)| a.groups).sum();
    for (name, a) in accs {
        let gain = if with_gain && a.groups > 1.0 {
            let m = a.gain / a.groups;
            let v = (a.gain2 / a.groups - m * m) / (a.groups - 1.0);
            format!("   {:+.4} ({:.4})", m, v.max(0.0).sqrt())
        } else {
            String::new()
        };
        eprintln!(
            "{:<22} {:>8} {:>7.3} {:>10.4} {:>10.4} {:>10.4} {:>9.3}{}",
            name,
            a.groups,
            a.groups / all,
            (a.e0 / a.n).sqrt(),
            a.own(a.e0),
            a.own(a.e1),
            (a.e0 - a.noise) / total,
            gain
        );
    }
}

/// Rank-quantile buckets (cut points as shares of the items, top first).
const CUTS: [(f64, &str); 5] = [
    (0.02, "top 2%"),
    (0.05, "top 2-5%"),
    (0.15, "top 5-15%"),
    (0.40, "top 15-40%"),
    (1.00, "bottom 60%"),
];

fn bucket_of(rank: usize, n: usize) -> usize {
    let q = rank as f64 / n as f64;
    CUTS.iter().position(|&(c, _)| q < c).unwrap_or(CUTS.len() - 1)
}

#[test]
#[ignore]
fn ply_probe() {
    let model = std::env::var("PLY_MODEL").unwrap_or_else(|_| "win_model".into());
    let data = std::env::var("PLY_DATA").unwrap_or_else(|_| "win_data_r1".into());
    let net = WinStrategy::load(&model);
    let ds = win_data::load(&data).unwrap();
    let (k_turn, k_cand) = (ds.meta.rollouts as f64, ds.meta.cand_rollouts as f64);
    let decs: Vec<&DecisionData> = ds.heldout.decisions.iter().take(env("PLY_DEC", usize::MAX)).collect();
    let turns: Vec<&Sample> = ds.heldout.turns.iter().take(env("PLY_TURNS", usize::MAX)).collect();
    eprintln!(
        "model {model}, data {data}: {} held-out decisions, {} turn ends",
        decs.len(),
        turns.len()
    );

    // Cost: 1-ply of every candidate vs its 64-rollout labels, one thread.
    let n_time = decs.len().min(env("PLY_TIME", 100));
    let t0 = Instant::now();
    let mut cands = 0;
    for d in &decs[..n_time] {
        for s in &d.samples {
            std::hint::black_box(values(&net, s.me, s.opp));
            cands += 1;
        }
    }
    let t_ply = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    for (i, d) in decs[..n_time].iter().enumerate() {
        let cd = CandDecision {
            entries: d.samples.iter().map(|s| (vec![s.me, s.opp], false)).collect(),
            first_active: 1,
            seed: i as u64,
        };
        std::hint::black_box(label_decision_full(&net, &cd, k_cand as usize));
    }
    let t_roll = t0.elapsed().as_secs_f64();
    eprintln!(
        "one thread, {cands} candidates: 1-ply {:.2} ms each, {} rollouts {:.1} ms each ({:.0}x)",
        t_ply / cands as f64 * 1e3,
        k_cand,
        t_roll / cands as f64 * 1e3,
        t_roll / t_ply
    );

    let t0 = Instant::now();
    let res: Vec<Dec> = decs
        .par_iter()
        .map(|d| {
            let (s0, s1) = d.samples.iter().map(|s| values(&net, s.me, s.opp)).unzip();
            Dec {
                s0,
                s1,
                y: d.samples.iter().map(|s| s.win).collect(),
                pv: d.pair_var.clone(),
                swapped: d.swapped,
            }
        })
        .collect();
    let tv: Vec<(f32, f32, f32, f32)> = turns
        .par_iter()
        .map(|s| {
            let (a, b) = values(&net, s.me, s.opp);
            (a, b, s.win, s.win_var)
        })
        .collect();
    eprintln!("evaluated in {:.0?}", t0.elapsed());

    // Decisions, by whether the 1-ply pick differs, then by disagreement rank.
    let add = |a: &mut Acc, d: &Dec| {
        for i in 1..d.y.len() {
            let dy = d.y[i] - d.y[0];
            let e0 = (d.s0[i] - d.s0[0]) - dy;
            let e1 = (d.s1[i] - d.s1[0]) - dy;
            a.n += 1.0;
            a.e0 += (e0 * e0) as f64;
            a.e1 += (e1 * e1) as f64;
            a.noise += d.pv[i - 1] as f64 / k_cand;
        }
        let g = (d.mine(&d.y, d.pick(&d.s1)) - d.mine(&d.y, d.pick(&d.s0))) as f64;
        a.gain += g;
        a.gain2 += g * g;
        a.groups += 1.0;
    };
    let mut by_flip = [Acc::default(); 2];
    for d in &res {
        add(&mut by_flip[(d.pick(&d.s0) != d.pick(&d.s1)) as usize], d);
    }
    print_table(
        "DECISIONS by pick (pair differences; gain = label value of 1-ply pick − static pick, for the mover)",
        &[
            ("same pick".into(), by_flip[0]),
            ("1-ply picks other".into(), by_flip[1]),
        ],
        true,
    );
    let mut order: Vec<usize> = (0..res.len()).collect();
    order.sort_by(|&a, &b| res[b].disagreement().partial_cmp(&res[a].disagreement()).unwrap());
    let mut by_rank = vec![Acc::default(); CUTS.len()];
    for (r, &i) in order.iter().enumerate() {
        add(&mut by_rank[bucket_of(r, order.len())], &res[i]);
    }
    let rows: Vec<(String, Acc)> = CUTS
        .iter()
        .zip(&by_rank)
        .map(|(&(c, n), a)| {
            let at = order[((c * order.len() as f64) as usize).min(order.len() - 1)];
            (format!("{n} (>{:.3})", res[at].disagreement()), *a)
        })
        .collect();
    print_table("DECISIONS by |Δ diff| static vs 1-ply (largest first)", &rows, true);

    // Turn ends, by |1-ply − static|.
    let live: Vec<&(f32, f32, f32, f32)> = tv.iter().filter(|t| t.3 > 0.0 || (t.2 > 0.0 && t.2 < 1.0)).collect();
    let nl = live.len() as f64;
    let mean = |f: &dyn Fn(&(f32, f32, f32, f32)) -> f32| live.iter().map(|t| f(t) as f64).sum::<f64>() / nl;
    eprintln!(
        "\nturn ends: mean label {:.4}, static {:.4}, 1-ply {:.4}; mean (1-ply − static) {:+.4}, mean |..| {:.4}",
        mean(&|t| t.2),
        mean(&|t| t.0),
        mean(&|t| t.1),
        mean(&|t| t.1 - t.0),
        mean(&|t| (t.1 - t.0).abs())
    );
    let mut order: Vec<usize> = (0..live.len()).collect();
    let dis = |t: &(f32, f32, f32, f32)| (t.1 - t.0).abs();
    order.sort_by(|&a, &b| dis(live[b]).partial_cmp(&dis(live[a])).unwrap());
    let mut by_rank = vec![Acc::default(); CUTS.len()];
    for (r, &i) in order.iter().enumerate() {
        let (s0, s1, y, v) = *live[i];
        let a = &mut by_rank[bucket_of(r, order.len())];
        a.n += 1.0;
        a.e0 += ((s0 - y) * (s0 - y)) as f64;
        a.e1 += ((s1 - y) * (s1 - y)) as f64;
        a.noise += v as f64 / k_turn;
        a.groups += 1.0;
    }
    let rows: Vec<(String, Acc)> = CUTS
        .iter()
        .zip(&by_rank)
        .map(|(&(c, n), a)| {
            let at = order[((c * order.len() as f64) as usize).min(order.len() - 1)];
            (format!("{n} (>{:.3})", dis(live[at])), *a)
        })
        .collect();
    print_table("TURN ENDS by |1-ply − static| (largest first; win error)", &rows, false);
}
