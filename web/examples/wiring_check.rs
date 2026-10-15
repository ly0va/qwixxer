//! Checks the web bots end to end, natively.
//! 1. Master+ (web `make_bot`) decides exactly like the CLI's production
//!    pair-search (`SearchBot::new(PairStrategy::load("pair_model"))`).
//! 2. Win rates of each web bot against the GA champion, with the GA playing
//!    the human seat through the web `Session` API (bot starts every game).
//!
//! Run from the repo root: `cargo run --release --manifest-path web/Cargo.toml --example wiring_check -- 400`
//! Optional second argument: comma-separated bots for part 2 only, skipping
//! part 1 (e.g. `-- 3000 newbie,conservative`; `conservative` is the unpruned
//! CLI bot, for comparison with Newbie).

use qwixxer::dqn::pair::PairStrategy;
use qwixxer::state::{Mark, State};
use qwixxer::strategy::search::{SearchBot, SearchConfig};
use qwixxer::strategy::{Conservative, Strategy};
use qwixxer_web::bots::{make_bot, BotKind, MASTER_PLUS_SPEC};
use qwixxer_web::session::{Phase, Session, SheetSnap, Snapshot, Sq};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn state_of(s: &SheetSnap, locked: [bool; 4]) -> State {
    let mut st = State::from_marks(&s.marks, s.strikes);
    st.lock(locked);
    st
}

fn states(snap: &Snapshot) -> (State, State) {
    (state_of(&snap.you, snap.locked), state_of(&snap.bot, snap.locked))
}

/// Part 1: identical decisions on positions from real games.
fn same_decisions(positions: usize) {
    assert_eq!(SearchConfig::parse(MASTER_PLUS_SPEC).unwrap(), SearchConfig::default());
    println!("Master+ config: {}", SearchConfig::parse(MASTER_PLUS_SPEC).unwrap());
    let mut web = make_bot(BotKind::MasterPlus);
    let mut cli = SearchBot::new(PairStrategy::load("pair_model"));
    let mut student = make_bot(BotKind::Student);
    let mut rng = SmallRng::seed_from_u64(7);
    let (mut checked, mut differ) = (0, 0);
    // Positions come from Student-vs-Student play through the Session.
    while checked < positions {
        let mut s = Session::new(make_bot(BotKind::Student), rng.gen());
        while s.view().phase != Phase::Over && checked < positions {
            let (you, bot) = states(&s.snapshot());
            let dice: [u8; 6] = core::array::from_fn(|_| rng.gen_range(1..=6));
            let a = (
                web.active_phase1(&bot, &[you], dice),
                web.passive_phase1(&bot, &[you], dice, 0),
            );
            let b = (
                cli.active_phase1(&bot, &[you], dice),
                cli.passive_phase1(&bot, &[you], dice, 0),
            );
            checked += 1;
            if a != b {
                differ += 1;
            }
            advance(&mut s, &mut student);
        }
    }
    println!("decisions compared: {checked}, different: {differ}");
}

/// Plays the human seat with `human` (a Strategy) through the Session API.
fn advance(s: &mut Session, human: &mut Box<dyn Strategy>) {
    let v = s.view();
    match v.phase {
        Phase::BotToRoll => s.bot_roll().unwrap(),
        Phase::YouToRoll => s.you_roll().unwrap(),
        Phase::Passive => {
            let snap = s.snapshot();
            let (you, bot) = states(&snap);
            let pick = human.passive_phase1(&you, &[bot], snap.dice.unwrap(), 0).map(Sq::from);
            s.submit_passive(pick).unwrap();
        }
        Phase::Active => {
            let snap = s.snapshot();
            let (you, bot) = states(&snap);
            let dice = snap.dice.unwrap();
            // Phase 1 on the pre-roll boards; phase 2 after both phase-1 marks and locks.
            let w = human.active_phase1(&you, &[bot], dice);
            let mut you2 = you;
            let mut bot2 = bot;
            if let Some(m) = w {
                you2.apply_mark(m);
            }
            if let Some(bw) = snap.bot_white {
                bot2.apply_mark(Mark::from(bw));
            }
            let lk: [bool; 4] = core::array::from_fn(|r| you2.locked()[r] || bot2.locked()[r]);
            you2.lock(lk);
            bot2.lock(lk);
            let opts = v.options.unwrap();
            let ended = lk.iter().filter(|&&l| l).count() >= 2;
            let c = if ended {
                None
            } else {
                human.active_phase2(&you2, &[bot2], dice, w.is_some())
            };
            let picks: Vec<Sq> = w.into_iter().chain(c).map(Sq::from).collect();
            if picks.is_empty() {
                s.submit_active(&[], opts.strike).unwrap();
            } else {
                s.submit_active(&picks, false).expect("human pick must be legal");
            }
        }
        Phase::Over => {}
    }
}

/// The bots part 2 can measure: id, display name, constructor.
const BOTS: [(&str, &str, fn() -> Box<dyn Strategy>); 5] = [
    ("master_plus", "Master+", || make_bot(BotKind::MasterPlus)),
    ("master", "Master", || make_bot(BotKind::Master)),
    ("student", "Student", || make_bot(BotKind::Student)),
    ("newbie", "Newbie", || make_bot(BotKind::Newbie)),
    ("conservative", "Cons.", || Box::<Conservative>::default()),
];

/// Part 2: bot vs GA (human seat) through the Session, `games` per bot.
fn win_rates(games: usize, only: Option<&str>) {
    let wanted = |id: &str| match only {
        Some(list) => list.split(',').any(|x| x == id),
        None => id != "conservative",
    };
    for (_, name, make) in BOTS.into_iter().filter(|(id, _, _)| wanted(id)) {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let per = games.div_ceil(threads);
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                std::thread::spawn(move || {
                    let (mut bot_w, mut ga_w, mut ties) = (0, 0, 0);
                    let mut ga = make_bot(BotKind::Student);
                    for g in 0..per {
                        let mut s = Session::new(make(), (t * 100_000 + g) as u64);
                        while s.view().phase != Phase::Over {
                            advance(&mut s, &mut ga);
                        }
                        let v = s.view();
                        match v.bot.score.cmp(&v.you.score) {
                            std::cmp::Ordering::Greater => bot_w += 1,
                            std::cmp::Ordering::Less => ga_w += 1,
                            std::cmp::Ordering::Equal => ties += 1,
                        }
                    }
                    (bot_w, ga_w, ties)
                })
            })
            .collect();
        let (mut b, mut g, mut t) = (0, 0, 0);
        for h in handles {
            let (x, y, z) = h.join().unwrap();
            b += x;
            g += y;
            t += z;
        }
        let n = (b + g + t) as f64;
        println!(
            "{name:8} vs GA: bot {:.1}%  GA {:.1}%  ties {:.1}%  ({} games; bot always starts)",
            100.0 * b as f64 / n,
            100.0 * g as f64 / n,
            100.0 * t as f64 / n,
            n
        );
    }
}

fn main() {
    let games: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(400);
    let only = std::env::args().nth(2);
    if only.is_none() {
        same_decisions(300);
    }
    win_rates(games, only.as_deref());
}
