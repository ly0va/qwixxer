//! Browser/Node latency benchmark: plays games with random human choices and
//! times every call into the bot.

use crate::bots::{make_bot, win, BotKind};
use crate::session::{Phase, Session};
use qwixxer::state::{Mark, State};
use qwixxer::strategy::search::{SearchBot, SearchConfig};
use qwixxer::strategy::Strategy;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
}

#[derive(Debug)]
struct Timed {
    inner: Box<dyn Strategy>,
    ms: Rc<RefCell<Vec<f64>>>,
}

impl Timed {
    fn time<T>(&mut self, f: impl FnOnce(&mut Box<dyn Strategy>) -> T) -> T {
        let t = now();
        let r = f(&mut self.inner);
        self.ms.borrow_mut().push(now() - t);
        r
    }
}

impl Strategy for Timed {
    fn active_phase1(&mut self, s: &State, o: &[State], d: [u8; 6]) -> Option<Mark> {
        self.time(|b| b.active_phase1(s, o, d))
    }
    fn active_phase2(&mut self, s: &State, o: &[State], d: [u8; 6], h: bool) -> Option<Mark> {
        self.time(|b| b.active_phase2(s, o, d, h))
    }
    fn passive_phase1(&mut self, s: &State, o: &[State], d: [u8; 6], a: usize) -> Option<Mark> {
        self.time(|b| b.passive_phase1(s, o, d, a))
    }
}

/// A web bot by id, or the win net (`win`, `win_search`), which the UI
/// does not offer yet.
fn bench_bot(bot: &str) -> Result<fn() -> Box<dyn Strategy>, String> {
    Ok(match bot {
        "win" => || Box::new(win()),
        "win_search" => || Box::new(SearchBot::with_config(win(), SearchConfig::win())),
        _ => match BotKind::parse(bot)? {
            BotKind::MasterPlus => || make_bot(BotKind::MasterPlus),
            BotKind::Master => || make_bot(BotKind::Master),
            BotKind::Student => || make_bot(BotKind::Student),
            BotKind::Newbie => || make_bot(BotKind::Newbie),
        },
    })
}

#[wasm_bindgen]
pub fn bench(bot: &str, games: u32, seed: u32) -> Result<String, JsError> {
    let make = bench_bot(bot).map_err(|e| JsError::new(&e))?;
    let ms = Rc::new(RefCell::new(Vec::new()));
    let mut rng = SmallRng::seed_from_u64(seed as u64);
    for g in 0..games {
        let timed = Timed {
            inner: make(),
            ms: ms.clone(),
        };
        let mut s = Session::new(Box::new(timed), seed as u64 * 1000 + g as u64);
        loop {
            match s.view().phase {
                Phase::BotToRoll => s.bot_roll().unwrap(),
                Phase::Passive => {
                    let pick = s.view().passive.choose(&mut rng).copied();
                    s.submit_passive(pick).unwrap();
                }
                Phase::YouToRoll => s.you_roll().unwrap(),
                Phase::Active => {
                    let o = s.view().options.unwrap();
                    match o.selections.choose(&mut rng) {
                        Some(sel) => s.submit_active(&sel.clone(), false).unwrap(),
                        None => s.submit_active(&[], o.strike).unwrap(),
                    };
                }
                Phase::Over => break,
            }
        }
    }
    let mut v = ms.borrow().clone();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    Ok(format!(
        "{{\"decisions\":{},\"mean_ms\":{:.2},\"p50_ms\":{:.2},\"p95_ms\":{:.2},\"max_ms\":{:.2}}}",
        v.len(),
        mean,
        q(0.5),
        q(0.95),
        q(1.0)
    ))
}
