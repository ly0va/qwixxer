//! The four web bots: Master+, Master, Student (spec "Implementation
//! decisions") and Newbie, the hand-written Conservative.

use qwixxer::bot::{self, DNA};
use qwixxer::dqn::pair::PairStrategy;
use qwixxer::strategy::search::{SearchBot, SearchConfig};
use qwixxer::strategy::{Conservative, Strategy};
use std::sync::Arc;

const CHAMPION: &[u8] = include_bytes!("../../champion.txt");
pub(crate) const PAIR_MODEL: &[u8] = include_bytes!("../../pair_model/model.mpk");

/// Master+ search config: "" = native production (`SearchConfig::default()`).
/// Task 4's browser benchmark may set a lighter spec here.
pub const MASTER_PLUS_SPEC: &str = "";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotKind {
    MasterPlus,
    Master,
    Student,
    Newbie,
}

impl BotKind {
    /// Every web bot, strongest first (the order of the explorer's tags).
    pub const ALL: [BotKind; 4] = [BotKind::MasterPlus, BotKind::Master, BotKind::Student, BotKind::Newbie];

    /// The id `parse` accepts (and the explorer reports).
    pub fn id(self) -> &'static str {
        match self {
            BotKind::MasterPlus => "master_plus",
            BotKind::Master => "master",
            BotKind::Student => "student",
            BotKind::Newbie => "newbie",
        }
    }

    pub fn parse(s: &str) -> Result<BotKind, String> {
        match s {
            "master_plus" => Ok(BotKind::MasterPlus),
            "master" => Ok(BotKind::Master),
            "student" => Ok(BotKind::Student),
            "newbie" => Ok(BotKind::Newbie),
            _ => Err(format!("unknown bot {s:?} (master_plus, master, student, newbie)")),
        }
    }
}

/// Master+ as its concrete type (the explorer asks it to explain decisions).
pub fn master_plus() -> SearchBot<PairStrategy> {
    SearchBot::with_config(
        PairStrategy::load_from_bytes(PAIR_MODEL),
        SearchConfig::parse(MASTER_PLUS_SPEC).expect("MASTER_PLUS_SPEC"),
    )
}

pub fn make_bot(kind: BotKind) -> Box<dyn Strategy> {
    match kind {
        BotKind::MasterPlus => Box::new(master_plus()),
        BotKind::Master => Box::new(PairStrategy::load_from_bytes(PAIR_MODEL)),
        BotKind::Student => Box::new(DNA::load_weights_from_bytes(CHAMPION, Arc::new(bot::default_genes()))),
        BotKind::Newbie => Box::<Conservative>::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Phase, Session};

    #[test]
    fn every_bot_kind_parses_and_finishes_a_game() {
        for id in ["master_plus", "master", "student", "newbie"] {
            let kind = BotKind::parse(id).unwrap();
            let mut s = Session::new(make_bot(kind), 3);
            // The human always passes or takes the first legal option.
            for _ in 0..400 {
                match s.view().phase {
                    Phase::BotToRoll => s.bot_roll().unwrap(),
                    Phase::Passive => {
                        s.submit_passive(None).unwrap();
                    }
                    Phase::YouToRoll => s.you_roll().unwrap(),
                    Phase::Active => {
                        let o = s.view().options.unwrap();
                        match o.selections.first() {
                            Some(sel) => s.submit_active(&sel.clone(), false).unwrap(),
                            None => s.submit_active(&[], o.strike).unwrap(),
                        };
                    }
                    Phase::Over => break,
                }
            }
            assert_eq!(s.view().phase, Phase::Over, "{id} game did not finish");
        }
        assert!(BotKind::parse("opportunist").is_err());
    }
}
