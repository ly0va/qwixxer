//! WASM entry points for the web UI: `WasmGame` (one game, JSON in/out),
//! `WasmExplorer` (the state explorer) and `bench`. Game logic lives in
//! `session.rs` / `selection.rs`.

pub mod bench;
pub mod bots;
pub mod explorer;
pub mod gamecode;
pub mod review;
pub mod selection;
pub mod session;

use bots::{make_bot, BotKind};
use session::{Session, Snapshot, Sq};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmGame {
    session: Session,
    bot: BotKind,
}

fn js(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen]
impl WasmGame {
    #[wasm_bindgen(constructor)]
    pub fn new(bot: &str, seed: u32) -> Result<WasmGame, JsError> {
        let kind = BotKind::parse(bot).map_err(js)?;
        Ok(WasmGame {
            session: Session::new(make_bot(kind), seed as u64),
            bot: kind,
        })
    }

    pub fn restore(bot: &str, snapshot: &str, seed: u32) -> Result<WasmGame, JsError> {
        let kind = BotKind::parse(bot).map_err(js)?;
        let snap: Snapshot = serde_json::from_str(snapshot).map_err(js)?;
        Ok(WasmGame {
            session: Session::restore(snap, make_bot(kind), seed as u64).map_err(js)?,
            bot: kind,
        })
    }

    pub fn view(&self) -> String {
        serde_json::to_string(&self.session.view()).unwrap()
    }

    pub fn snapshot(&self) -> String {
        serde_json::to_string(&self.session.snapshot()).unwrap()
    }

    /// The game code (spec `2026-10-24-game-review-design.md`); an error
    /// for a game saved before rolls were recorded.
    pub fn code(&self) -> Result<String, JsError> {
        self.session.code(self.bot).map_err(js)
    }

    pub fn bot_roll(&mut self) -> Result<(), JsError> {
        self.session.bot_roll().map_err(js)
    }

    pub fn submit_passive(&mut self, pick_json: &str) -> Result<String, JsError> {
        let pick: Option<Sq> = serde_json::from_str(pick_json).map_err(js)?;
        let rec = self.session.submit_passive(pick).map_err(js)?;
        Ok(serde_json::to_string(&rec).unwrap())
    }

    pub fn you_roll(&mut self) -> Result<(), JsError> {
        self.session.you_roll().map_err(js)
    }

    pub fn submit_active(&mut self, picks_json: &str, strike: bool) -> Result<String, JsError> {
        let picks: Vec<Sq> = serde_json::from_str(picks_json).map_err(js)?;
        let rec = self.session.submit_active(&picks, strike).map_err(js)?;
        Ok(serde_json::to_string(&rec).unwrap())
    }
}

/// The state explorer: stateless calls, one explorer per worker.
#[wasm_bindgen]
pub struct WasmExplorer {
    inner: explorer::Explorer,
    /// The last game `review_roll` replayed, by code: a review asks for
    /// every roll of one game in turn.
    reviewed: Option<(String, gamecode::ReviewGame)>,
}

impl Default for WasmExplorer {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmExplorer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmExplorer {
        WasmExplorer {
            inner: explorer::Explorer::new(),
            reviewed: None,
        }
    }

    pub fn explore(&mut self, position: &str) -> Result<String, JsError> {
        let pos: explorer::Position = serde_json::from_str(position).map_err(js)?;
        let answer = self.inner.explore(&pos).map_err(js)?;
        Ok(serde_json::to_string(&answer).unwrap())
    }

    /// Play-out totals for games `start .. start + count` (spec
    /// `2026-10-15-explorer-analysis-design.md`), as JSON.
    pub fn analyze(&self, position: &str, start: u32, count: u32) -> Result<String, JsError> {
        let pos: explorer::Position = serde_json::from_str(position).map_err(js)?;
        let a = self.inner.analyze(&pos, start, count).map_err(js)?;
        Ok(serde_json::to_string(&a).unwrap())
    }

    /// Your P(win) after the roller's best turn for the position's roll,
    /// and white pairs `first .. first + count`'s share of its average over
    /// all rolls (spec `2026-10-22-explorer-roll-luck-design.md`), as JSON.
    pub fn luck(&self, position: &str, first: u32, count: u32) -> Result<String, JsError> {
        let pos: explorer::Position = serde_json::from_str(position).map_err(js)?;
        let l = self.inner.luck(&pos, first, count).map_err(js)?;
        Ok(serde_json::to_string(&l).unwrap())
    }

    /// Your move's 1-ply value on your roll (spec
    /// `2026-10-24-game-review-design.md`, "Explorer"): `marks` as JSON.
    pub fn move_one_ply(&self, position: &str, marks: &str, strike: bool) -> Result<f32, JsError> {
        let pos: explorer::Position = serde_json::from_str(position).map_err(js)?;
        let marks: Vec<Sq> = serde_json::from_str(marks).map_err(js)?;
        self.inner.move_one_ply(&pos, &marks, strike).map_err(js)
    }

    /// The position after playing your move (`marks` as JSON, or a strike)
    /// with the opponent's predicted part, as JSON.
    pub fn play(&self, position: &str, marks: &str, strike: bool) -> Result<String, JsError> {
        let pos: explorer::Position = serde_json::from_str(position).map_err(js)?;
        let marks: Vec<Sq> = serde_json::from_str(marks).map_err(js)?;
        let played = self.inner.play(&pos, &marks, strike).map_err(js)?;
        Ok(serde_json::to_string(&played).unwrap())
    }

    /// The game a code describes, replayed (spec
    /// `2026-10-24-game-review-design.md`), as JSON; an error for a code
    /// that is malformed or breaks the rules.
    pub fn review_game(&self, code: &str) -> Result<String, JsError> {
        let g = gamecode::replay(code).map_err(js)?;
        Ok(serde_json::to_string(&g).unwrap())
    }

    /// Roll `k`'s review analysis (luck, the 1-ply move costs), as JSON.
    pub fn review_roll(&mut self, code: &str, k: u32) -> Result<String, JsError> {
        if self.reviewed.as_ref().is_none_or(|(c, _)| c != code) {
            self.reviewed = Some((code.to_string(), gamecode::replay(code).map_err(js)?));
        }
        let g = &self.reviewed.as_ref().unwrap().1;
        let a = self.inner.review_roll(g, k as usize).map_err(js)?;
        Ok(serde_json::to_string(&a).unwrap())
    }
}
