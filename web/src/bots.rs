//! The four web bots: Master+ (win-search), Master (the win net), Student
//! (the GA champion) and Newbie, the hand-written Conservative.

use qwixxer::bot::{self, DNA};
use qwixxer::dqn::win::WinStrategy;
use qwixxer::strategy::search::{SearchBot, SearchConfig};
use qwixxer::strategy::{Conservative, Strategy};
use std::sync::Arc;

const CHAMPION: &[u8] = include_bytes!("../../champion.txt");

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

/// Master+ as its concrete type (the explorer asks it to explain decisions):
/// native production win-search.
pub fn master_plus() -> SearchBot<WinStrategy> {
    SearchBot::with_config(win(), SearchConfig::win())
}

/// Master: the win net, as native `win` plays (passive lookahead, no
/// forced safe locks).
pub fn win() -> WinStrategy {
    WinStrategy::load_from_bytes(
        include_bytes!("../../win_model/config.json"),
        include_bytes!("../../win_model/model.mpk"),
        Some(include_bytes!("../../win_model/aux.mpk")),
        Some(include_bytes!("../../win_model/tempo.mpk")),
    )
}

pub fn make_bot(kind: BotKind) -> Box<dyn Strategy> {
    match kind {
        BotKind::MasterPlus => Box::new(master_plus()),
        BotKind::Master => Box::new(win()),
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
