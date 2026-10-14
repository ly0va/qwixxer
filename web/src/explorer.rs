//! The state explorer (spec `docs/superpowers/specs/2026-10-14-state-explorer-design.md`):
//! any two-sheet position, evaluated by the pair net, and for a roll every
//! legal move of yours ranked by your chance to win after it, tagged with
//! the move each web bot would make in your seat.

use crate::bots::{make_bot, BotKind, PAIR_MODEL};
use crate::selection::{active_options, n_locked, passive_options, union};
use crate::session::{SheetSnap, Sq};
use qwixxer::dqn::pair::{pair_features, PairStrategy};
use qwixxer::dqn::{LOG_VAR_MAX, LOG_VAR_MIN};
use qwixxer::state::{Mark, State};
use qwixxer::strategy::search::WinProb;
use qwixxer::strategy::Strategy;
use serde::{Deserialize, Serialize};
use std::slice::from_ref;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Roller {
    You,
    Opponent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub you: SheetSnap,
    pub opponent: SheetSnap,
    /// Rows closed by tapping their lock. Marking a row's end number with
    /// five or more marks closes it too (the engine's rule), unioned in.
    pub locked: [bool; 4],
    /// All six dice, or none.
    pub dice: Option<[u8; 6]>,
    pub roller: Roller,
}

impl Position {
    pub fn validate(&self) -> Result<(), String> {
        for s in [&self.you, &self.opponent] {
            if s.strikes > 4 {
                return Err(format!("strikes {} > 4", s.strikes));
            }
        }
        if let Some(d) = self.dice {
            if d.iter().any(|v| !(1..=6).contains(v)) {
                return Err(format!("bad dice {d:?}"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Eval {
    /// The pair net's P(you win) (Master's and Master+'s evaluator).
    pub p_win: f32,
    /// Expected final score difference, you − opponent: current difference + μ.
    pub margin: f32,
    pub sigma: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Over {
    pub you: isize,
    pub opponent: isize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MoveOut {
    /// Marks in application order (white sum first).
    pub marks: Vec<Sq>,
    pub strike: bool,
    pub pass: bool,
    /// P(you win) after the move.
    pub p_win: f32,
    /// Ids of the bots that would make this move in your seat.
    pub bots: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Answer {
    pub over: Option<Over>,
    pub eval: Eval,
    pub moves: Option<Vec<MoveOut>>,
}

/// One full turn of yours: marks in application order, a strike, or (both
/// empty) a pass.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub marks: Vec<Mark>,
    pub strike: bool,
}

impl Turn {
    fn key(&self) -> (Vec<(usize, u8)>, bool) {
        let mut k: Vec<(usize, u8)> = self.marks.iter().map(|m| (m.row, m.number)).collect();
        k.sort_unstable();
        (k, self.strike)
    }
}

/// Both sheets as engine states with every closed row applied to both.
pub fn states(pos: &Position) -> (State, State, [bool; 4]) {
    let mut you = State::from_marks(&pos.you.marks, pos.you.strikes);
    let mut opp = State::from_marks(&pos.opponent.marks, pos.opponent.strikes);
    let locked = union(pos.locked, union(you.locked(), opp.locked()));
    you.lock(locked);
    opp.lock(locked);
    (you, opp, locked)
}

/// Your legal turns, from the game's own rules (`selection.rs`). On your
/// roll the opponent's simultaneous white-sum mark is not modelled (spec
/// "Moves"): it is treated as a pass.
pub fn legal_turns(you: &State, opp: &State, dice: [u8; 6], roller: Roller) -> Vec<Turn> {
    let pass = Turn {
        marks: vec![],
        strike: false,
    };
    match roller {
        Roller::You => {
            let o = active_options(you, opp, None, dice);
            let mut t: Vec<Turn> = o
                .selections
                .into_iter()
                .map(|marks| Turn { marks, strike: false })
                .collect();
            if o.strike {
                t.push(Turn {
                    marks: vec![],
                    strike: true,
                });
            }
            if o.pass {
                t.push(pass);
            }
            t
        }
        Roller::Opponent => {
            let mut t: Vec<Turn> = passive_options(you, dice)
                .into_iter()
                .map(|m| Turn {
                    marks: vec![m],
                    strike: false,
                })
                .collect();
            t.push(pass);
            t
        }
    }
}

/// Both sheets after your turn, with any row it closed applied to both.
pub fn after(you: &State, opp: &State, turn: &Turn) -> (State, State) {
    let mut y = *you;
    for &m in &turn.marks {
        y.apply_mark(m);
    }
    if turn.strike {
        y.apply_strike();
    }
    let mut o = *opp;
    let locks = union(y.locked(), o.locked());
    y.lock(locks);
    o.lock(locks);
    (y, o)
}

/// The turn `bot` would take in your seat: its own two-phase decision, as
/// the game asks it (see `web/examples/wiring_check.rs`).
pub fn bot_turn(bot: &mut dyn Strategy, you: &State, opp: &State, dice: [u8; 6], roller: Roller) -> Turn {
    match roller {
        // The active player (the opponent) is index 0 of the opponents.
        Roller::Opponent => Turn {
            marks: bot.passive_phase1(you, from_ref(opp), dice, 0).into_iter().collect(),
            strike: false,
        },
        Roller::You => {
            let w = bot.active_phase1(you, from_ref(opp), dice);
            let mut marks: Vec<Mark> = w.into_iter().collect();
            let mut y = *you;
            if let Some(m) = w {
                y.apply_mark(m);
            }
            let locks = union(y.locked(), opp.locked());
            if n_locked(locks) < 2 {
                let mut o = *opp;
                y.lock(locks);
                o.lock(locks);
                if let Some(c) = bot.active_phase2(&y, from_ref(&o), dice, w.is_some()) {
                    marks.push(c);
                }
            }
            let strike = marks.is_empty() && active_options(you, opp, None, dice).strike;
            Turn { marks, strike }
        }
    }
}

pub struct Explorer {
    net: PairStrategy,
    bots: Vec<(BotKind, Box<dyn Strategy>)>,
}

impl Default for Explorer {
    fn default() -> Self {
        Self::new()
    }
}

impl Explorer {
    pub fn new() -> Self {
        Explorer {
            net: PairStrategy::load_from_bytes(PAIR_MODEL),
            bots: BotKind::ALL.iter().map(|&k| (k, make_bot(k))).collect(),
        }
    }

    pub fn explore(&mut self, pos: &Position) -> Result<Answer, String> {
        pos.validate()?;
        let (you, opp, locked) = states(pos);
        let eval = self.eval(&you, &opp);
        let over = (you.strikes >= 4 || opp.strikes >= 4 || n_locked(locked) >= 2).then(|| Over {
            you: you.count_points(),
            opponent: opp.count_points(),
        });
        let moves = match (&over, pos.dice) {
            (None, Some(dice)) => Some(self.moves(&you, &opp, dice, pos.roller)),
            _ => None,
        };
        Ok(Answer { over, eval, moves })
    }

    fn eval(&self, you: &State, opp: &State) -> Eval {
        let p_win = self.net.win_prob_multi(&[(you, from_ref(opp))])[0];
        let (mu, log_var) = self.net.net.forward(&[pair_features(you, opp, from_ref(opp))])[0];
        Eval {
            p_win,
            margin: (you.count_points() - opp.count_points()) as f32 + mu,
            sigma: (0.5 * log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)).exp(),
        }
    }

    fn moves(&mut self, you: &State, opp: &State, dice: [u8; 6], roller: Roller) -> Vec<MoveOut> {
        let turns = legal_turns(you, opp, dice, roller);
        let afters: Vec<(State, [State; 1])> = turns
            .iter()
            .map(|t| {
                let (y, o) = after(you, opp, t);
                (y, [o])
            })
            .collect();
        let groups: Vec<(&State, &[State])> = afters.iter().map(|(y, o)| (y, &o[..])).collect();
        let p = self.net.win_prob_multi(&groups);
        let mut out: Vec<MoveOut> = turns
            .iter()
            .zip(p)
            .map(|(t, p_win)| MoveOut {
                marks: t.marks.iter().map(|&m| Sq::from(m)).collect(),
                strike: t.strike,
                pass: t.marks.is_empty() && !t.strike,
                p_win,
                bots: vec![],
            })
            .collect();
        let keys: Vec<_> = turns.iter().map(Turn::key).collect();
        for (kind, bot) in self.bots.iter_mut() {
            let pick = bot_turn(bot.as_mut(), you, opp, dice, roller).key();
            // A pick that matches no listed move is an engine bug: it tags
            // nothing, and development builds report it (the random-position
            // test pins that it never happens).
            if let Some(i) = keys.iter().position(|k| *k == pick) {
                out[i].bots.push(kind.id());
            } else {
                debug_assert!(false, "{} picked {:?}, which is not a listed move", kind.id(), pick);
            }
        }
        out.sort_by(|a, b| b.p_win.partial_cmp(&a.p_win).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn sheet() -> SheetSnap {
        SheetSnap {
            marks: [[false; 11]; 4],
            strikes: 0,
        }
    }
    fn pos(dice: Option<[u8; 6]>, roller: Roller) -> Position {
        Position {
            you: sheet(),
            opponent: sheet(),
            locked: [false; 4],
            dice,
            roller,
        }
    }
    fn key(m: &MoveOut) -> (Vec<(usize, u8)>, bool) {
        let mut k: Vec<(usize, u8)> = m.marks.iter().map(|s| (s.row, s.n)).collect();
        k.sort_unstable();
        (k, m.strike)
    }

    #[test]
    fn empty_position_is_even_and_has_no_moves_without_dice() {
        let a = Explorer::new().explore(&pos(None, Roller::You)).unwrap();
        assert!(a.over.is_none());
        assert!(a.moves.is_none());
        assert!((0.3..0.7).contains(&a.eval.p_win), "p_win {}", a.eval.p_win);
        assert!(a.eval.sigma > 0.0);
    }

    #[test]
    fn your_roll_lists_every_legal_turn_ranked_with_each_bot_once() {
        let dice = [3, 4, 2, 5, 3, 1];
        let a = Explorer::new().explore(&pos(Some(dice), Roller::You)).unwrap();
        let moves = a.moves.unwrap();
        let o = active_options(&State::default(), &State::default(), None, dice);
        assert_eq!(moves.len(), o.selections.len() + 1); // + strike
        assert!(moves.iter().any(|m| m.strike && m.marks.is_empty() && !m.pass));
        assert!(moves.windows(2).all(|w| w[0].p_win >= w[1].p_win));
        for id in ["master_plus", "master", "student", "newbie"] {
            assert_eq!(moves.iter().filter(|m| m.bots.contains(&id)).count(), 1, "{id}");
        }
        let mut keys: Vec<_> = moves.iter().map(key).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), moves.len());
    }

    #[test]
    fn opponents_roll_offers_white_sum_marks_and_pass() {
        let a = Explorer::new()
            .explore(&pos(Some([3, 4, 2, 5, 3, 1]), Roller::Opponent))
            .unwrap();
        let moves = a.moves.unwrap();
        assert_eq!(moves.len(), 5); // white 7 in each of the 4 rows, or pass
        assert!(moves.iter().all(|m| m.marks.len() <= 1 && !m.strike));
        assert_eq!(moves.iter().filter(|m| m.pass).count(), 1);
        assert!(moves.iter().filter(|m| !m.pass).all(|m| m.marks[0].n == 7));
    }

    #[test]
    fn game_over_positions_report_the_result_and_no_moves() {
        let mut p = pos(Some([1, 1, 1, 1, 1, 1]), Roller::You);
        p.opponent.strikes = 4;
        p.you.marks[0][0] = true; // red 2: 1 point
        let a = Explorer::new().explore(&p).unwrap();
        assert_eq!(a.over, Some(Over { you: 1, opponent: -20 }));
        assert!(a.moves.is_none());

        let mut p = pos(None, Roller::You);
        p.locked = [true, false, true, false];
        assert!(Explorer::new().explore(&p).unwrap().over.is_some());
    }

    #[test]
    fn opponents_roll_with_every_white_sum_square_blocked_offers_only_pass() {
        let mut p = pos(Some([3, 4, 2, 5, 3, 1]), Roller::Opponent);
        // Index 6 is red/yellow 8 and green/blue 6: each lies past 7.
        for r in 0..4 {
            p.you.marks[r][6] = true;
        }
        let moves = Explorer::new().explore(&p).unwrap().moves.unwrap();
        assert_eq!(moves.len(), 1);
        assert!(moves[0].pass && moves[0].marks.is_empty() && !moves[0].strike);
        assert_eq!(moves[0].bots, ["master_plus", "master", "student", "newbie"]);
    }

    #[test]
    fn a_row_closed_by_marking_its_end_number_offers_no_marks() {
        let mut p = pos(Some([6, 6, 6, 6, 6, 6]), Roller::You);
        // Red 2, 3, 4, 5, 12: five marks ending on 12 close red for both.
        for i in [0, 1, 2, 3, 10] {
            p.opponent.marks[0][i] = true;
        }
        let moves = Explorer::new().explore(&p).unwrap().moves.unwrap();
        assert!(moves.iter().all(|m| m.marks.iter().all(|s| s.row != 0)));
    }

    #[test]
    fn rejects_out_of_range_values() {
        let mut p = pos(None, Roller::You);
        p.you.strikes = 5;
        assert!(Explorer::new().explore(&p).is_err());
        let p = pos(Some([1, 2, 3, 4, 5, 7]), Roller::You);
        assert!(Explorer::new().explore(&p).is_err());
    }

    /// Spec "Edge cases": arbitrary (unreachable) marks never panic, and
    /// every bot's pick is one of the listed moves.
    #[test]
    fn random_positions_never_panic_and_every_pick_is_listed() {
        let mut ex = Explorer::new();
        let mut rng = SmallRng::seed_from_u64(11);
        for _ in 0..150 {
            let mut p = pos(
                None,
                if rng.gen_bool(0.5) {
                    Roller::You
                } else {
                    Roller::Opponent
                },
            );
            for s in [&mut p.you, &mut p.opponent] {
                for r in 0..4 {
                    for i in 0..11 {
                        s.marks[r][i] = rng.gen_bool(0.3);
                    }
                }
                s.strikes = rng.gen_range(0..=3);
            }
            p.locked[rng.gen_range(0..4)] = rng.gen_bool(0.2);
            p.dice = Some(core::array::from_fn(|_| rng.gen_range(1..=6)));
            let a = ex.explore(&p).unwrap();
            assert!(a.eval.p_win.is_finite());
            if let Some(moves) = a.moves {
                assert!(!moves.is_empty());
                for id in ["master_plus", "master", "student", "newbie"] {
                    assert_eq!(
                        moves.iter().filter(|m| m.bots.contains(&id)).count(),
                        1,
                        "{id} in {p:?}"
                    );
                }
            }
        }
    }
}
