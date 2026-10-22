//! WASM entry points for the web UI: `WasmGame` (one game, JSON in/out),
//! `WasmExplorer` (the state explorer) and `bench`. Game logic lives in
//! `session.rs` / `selection.rs`.

pub mod bench;
pub mod bots;
pub mod explorer;
pub mod selection;
pub mod session;

use bots::{make_bot, BotKind};
use session::{Session, Snapshot, Sq};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmGame {
    session: Session,
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
        })
    }

    pub fn restore(bot: &str, snapshot: &str, seed: u32) -> Result<WasmGame, JsError> {
        let kind = BotKind::parse(bot).map_err(js)?;
        let snap: Snapshot = serde_json::from_str(snapshot).map_err(js)?;
        Ok(WasmGame {
            session: Session::restore(snap, make_bot(kind), seed as u64).map_err(js)?,
        })
    }

    pub fn view(&self) -> String {
        serde_json::to_string(&self.session.view()).unwrap()
    }

    pub fn snapshot(&self) -> String {
        serde_json::to_string(&self.session.snapshot()).unwrap()
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

    /// The move list's top P(win) for the position's roll and its average
    /// over all rolls (spec `2026-10-22-explorer-roll-luck-design.md`), as
    /// JSON.
    pub fn luck(&self, position: &str) -> Result<String, JsError> {
        let pos: explorer::Position = serde_json::from_str(position).map_err(js)?;
        let l = self.inner.luck(&pos).map_err(js)?;
        Ok(serde_json::to_string(&l).unwrap())
    }
}
