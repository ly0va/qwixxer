//! The three web bots (spec "Implementation decisions").

use qwixxer::bot::{self, DNA};
use qwixxer::dqn::pair::PairStrategy;
use qwixxer::strategy::search::{SearchBot, SearchConfig};
use qwixxer::strategy::Strategy;
use std::sync::Arc;

const CHAMPION: &[u8] = include_bytes!("../../champion.txt");
const PAIR_MODEL: &[u8] = include_bytes!("../../pair_model/model.mpk");

/// Master+ search config: "" = native production (`SearchConfig::default()`).
/// Task 4's browser benchmark may set a lighter spec here.
pub const MASTER_PLUS_SPEC: &str = "";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotKind {
    MasterPlus,
    Master,
    Student,
}

impl BotKind {
    pub fn parse(s: &str) -> Result<BotKind, String> {
        match s {
            "master_plus" => Ok(BotKind::MasterPlus),
            "master" => Ok(BotKind::Master),
            "student" => Ok(BotKind::Student),
            _ => Err(format!("unknown bot {s:?} (master_plus, master, student)")),
        }
    }
}

pub fn make_bot(kind: BotKind) -> Box<dyn Strategy> {
    match kind {
        BotKind::MasterPlus => Box::new(SearchBot::with_config(
            PairStrategy::load_from_bytes(PAIR_MODEL),
            SearchConfig::parse(MASTER_PLUS_SPEC).expect("MASTER_PLUS_SPEC"),
        )),
        BotKind::Master => Box::new(PairStrategy::load_from_bytes(PAIR_MODEL)),
        BotKind::Student => Box::new(DNA::load_weights_from_bytes(CHAMPION, Arc::new(bot::default_genes()))),
    }
}
