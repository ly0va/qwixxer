//! The state explorer (spec `docs/superpowers/specs/2026-10-14-state-explorer-design.md`):
//! any two-sheet position, evaluated by the pair net, and for a roll every
//! legal move of yours ranked by the pair net's chance to win after it (as
//! Master+ judges it), tagged with the move each web bot would make in your
//! seat.

use crate::bots::{make_bot, master_plus, BotKind, PAIR_MODEL};
use crate::selection::{active_options, n_locked, passive_options, union};
use crate::session::{SheetSnap, Sq};
use qwixxer::dqn::pair::{pair_features, PairStrategy};
use qwixxer::dqn::{LOG_VAR_MAX, LOG_VAR_MIN};
use qwixxer::state::{Mark, State};
use qwixxer::strategy::bot_impl::{
    active_phase1_choices, active_phase2_choices, eval_decision, passive_phase1_choices, Decision,
};
use qwixxer::strategy::search::{context_seed, phi, sample_player_seed, SearchBot, WinProb};
use qwixxer::strategy::sim::{BatchedRollouts, SimGame};
use qwixxer::strategy::{Bot, Strategy};
use rand::rngs::SmallRng;
use rand::SeedableRng;
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
    /// The pair net's P(you win) after the move, judged as Master+ judges
    /// it (see `Explorer::moves`).
    pub p_win: f32,
    /// Ids of the bots that would make this move in your seat.
    pub bots: Vec<&'static str>,
    /// On Master+'s move when it isn't ranked first, why Master+ still
    /// plays it: "rolled_out" (its search preferred it) or "rule" (a
    /// meta-rule forced it).
    pub note: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Answer {
    pub over: Option<Over>,
    pub eval: Eval,
    pub moves: Option<Vec<MoveOut>>,
    /// On your roll (dice set, game not over): the white-sum mark the
    /// opponent is predicted to make at the same time (`mark: null` = pass).
    pub opponent_white: Option<OpponentWhite>,
}

/// Spec "Engine" (`2026-10-15-explorer-analysis-design.md`): totals of
/// play-out games `start .. start + count` of a position. Batches add up.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Analysis {
    pub games: u32,
    /// Games you win (higher final score) and games tied.
    pub wins: u32,
    pub ties: u32,
    /// Sum, and sum of squares, of your final score minus the opponent's.
    pub diff_sum: f64,
    pub diff_sq_sum: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpponentWhite {
    pub mark: Option<Sq>,
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
/// roll they are judged with the opponent's simultaneous white-sum mark
/// `opp_white` (a lock it makes can end the game after phase 1).
pub fn legal_turns(you: &State, opp: &State, dice: [u8; 6], roller: Roller, opp_white: Option<Mark>) -> Vec<Turn> {
    let pass = Turn {
        marks: vec![],
        strike: false,
    };
    match roller {
        Roller::You => {
            let o = active_options(you, opp, opp_white, dice);
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

/// The opponent's white-sum mark on your roll, predicted as the bots predict
/// it (`bot_impl::simulate_opp_phase1` with the pair net: Master's choice).
pub fn predict_opp_white(net: &PairStrategy, you: &State, opp: &State, dice: [u8; 6]) -> Option<Mark> {
    eval_decision(net, passive_phase1_choices(opp, from_ref(you), dice), from_ref(you))
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
/// the game asks it (see `web/examples/wiring_check.rs`), with the
/// opponent's predicted white-sum mark `opp_white` on your roll. The flag
/// says whether a meta-rule forced its phase 2 (a safe lock).
pub fn bot_turn(
    bot: &mut dyn Strategy,
    you: &State,
    opp: &State,
    opp_white: Option<Mark>,
    dice: [u8; 6],
    roller: Roller,
) -> (Turn, bool) {
    match roller {
        // The active player (the opponent) is index 0 of the opponents.
        Roller::Opponent => (
            Turn {
                marks: bot.passive_phase1(you, from_ref(opp), dice, 0).into_iter().collect(),
                strike: false,
            },
            false,
        ),
        Roller::You => {
            let w = bot.active_phase1(you, from_ref(opp), dice);
            finish_turn(bot, you, opp, opp_white, dice, w)
        }
    }
}

/// Your turn after the phase-1 mark `w`, with the opponent's simultaneous
/// mark `opp_white`: locks propagate, then (unless two rows are closed) the
/// phase-2 decision `phase2(you, opponent, has_marked)` against the
/// opponent's marked sheet; nothing marked is a strike. Also returns whether
/// a meta-rule forced phase 2 and whether `phase2` reported a search.
fn finish_turn_with(
    you: &State,
    opp: &State,
    opp_white: Option<Mark>,
    dice: [u8; 6],
    w: Option<Mark>,
    mut phase2: impl FnMut(&State, &State, bool) -> (Option<Mark>, bool),
) -> (Turn, bool, bool) {
    let mut marks: Vec<Mark> = w.into_iter().collect();
    let (mut forced, mut searched) = (false, false);
    let mut y = *you;
    if let Some(m) = w {
        y.apply_mark(m);
    }
    let mut o = *opp;
    if let Some(m) = opp_white {
        o.apply_mark(m);
    }
    let locks = union(y.locked(), o.locked());
    if n_locked(locks) < 2 {
        y.lock(locks);
        o.lock(locks);
        forced = matches!(
            active_phase2_choices(&y, from_ref(&o), dice, w.is_some()),
            Decision::Forced(Some(_))
        );
        let (c, s) = phase2(&y, &o, w.is_some());
        searched = s;
        marks.extend(c);
    }
    let strike = marks.is_empty() && active_options(you, opp, opp_white, dice).strike;
    (Turn { marks, strike }, forced, searched)
}

/// [`finish_turn_with`] for any bot: its own `active_phase2`.
fn finish_turn(
    bot: &mut dyn Strategy,
    you: &State,
    opp: &State,
    opp_white: Option<Mark>,
    dice: [u8; 6],
    w: Option<Mark>,
) -> (Turn, bool) {
    let (turn, forced, _) = finish_turn_with(you, opp, opp_white, dice, w, |y, o, hm| {
        (bot.active_phase2(y, from_ref(o), dice, hm), false)
    });
    (turn, forced)
}

pub struct Explorer {
    net: PairStrategy,
    /// Master+ ranks the moves (and is tagged); the other bots are only tagged.
    plus: SearchBot<PairStrategy>,
    others: Vec<(BotKind, Box<dyn Strategy>)>,
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
            plus: master_plus(),
            others: BotKind::ALL[1..].iter().map(|&k| (k, make_bot(k))).collect(),
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
        let (moves, opponent_white) = match (&over, pos.dice) {
            (None, Some(dice)) => {
                let opp_white = match pos.roller {
                    Roller::You => Some(predict_opp_white(&self.net, &you, &opp, dice)),
                    Roller::Opponent => None,
                };
                let moves = self.moves(&you, &opp, dice, pos.roller, opp_white.flatten());
                (Some(moves), opp_white.map(|m| OpponentWhite { mark: m.map(Sq::from) }))
            }
            _ => (None, None),
        };
        Ok(Answer {
            over,
            eval,
            moves,
            opponent_white,
        })
    }

    /// Plays games `start .. start + count` from the position before the
    /// roll (its dice ignored), the roller first, both seats Master (the
    /// pair net, batched across games). Game k's dice depend only on the
    /// position and k, so batches compose and replay exactly.
    pub fn analyze(&self, pos: &Position, start: u32, count: u32) -> Result<Analysis, String> {
        pos.validate()?;
        let (you, opp, locked) = states(pos);
        if you.strikes >= 4 || opp.strikes >= 4 || n_locked(locked) >= 2 {
            return Err("the game is over".into());
        }
        let first = match pos.roller {
            Roller::You => 0,
            Roller::Opponent => 1,
        };
        let seed = context_seed(&you, from_ref(&opp), [0; 6]) ^ first as u64;
        let sims = (start..start + count)
            .map(|k| SimGame {
                states: vec![you, opp],
                active: first,
                rngs: (0..2)
                    .map(|p| SmallRng::seed_from_u64(sample_player_seed(seed, k as usize, p)))
                    .collect(),
                over: false,
            })
            .collect();
        let mut driver = BatchedRollouts::new(&self.net, sims);
        let mut turns = 0;
        while !driver.all_over() {
            driver.step_turn();
            turns += 1;
            assert!(turns <= 200, "play-out exceeded 200 turns");
        }
        let mut a = Analysis {
            games: 0,
            wins: 0,
            ties: 0,
            diff_sum: 0.0,
            diff_sq_sum: 0.0,
        };
        for sim in &driver.sims {
            let d = (sim.states[0].count_points() - sim.states[1].count_points()) as f64;
            a.games += 1;
            a.wins += (d > 0.0) as u32;
            a.ties += (d == 0.0) as u32;
            a.diff_sum += d;
            a.diff_sq_sum += d * d;
        }
        Ok(a)
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

    /// Spec "Moves": every legal turn with the pair net's P(win) after it,
    /// judged as Master judges it (on your roll against the opponent's sheet
    /// with its predicted white-sum mark `opp_white`; on theirs against their
    /// sheet), highest first. When Master+'s or Master's move isn't first, a
    /// note says why: Master+'s search preferred it ("rolled_out"), or a
    /// meta-rule forced it ("rule").
    fn moves(
        &mut self,
        you: &State,
        opp: &State,
        dice: [u8; 6],
        roller: Roller,
        opp_white: Option<Mark>,
    ) -> Vec<MoveOut> {
        let turns = legal_turns(you, opp, dice, roller, opp_white);
        let keys: Vec<_> = turns.iter().map(Turn::key).collect();
        let mut opp_after = *opp;
        if let Some(m) = opp_white {
            opp_after.apply_mark(m);
        }
        // Whether a meta-rule forced Master's (pair-net pipeline) phase-1 or
        // passive decision; Master+ shares that pipeline.
        let (explained, rule_first) = match roller {
            Roller::You => (
                self.plus.explain_active_phase1(you, from_ref(opp), dice),
                matches!(
                    active_phase1_choices(&self.net, you, from_ref(opp), dice).0,
                    Decision::Forced(Some(_))
                ),
            ),
            Roller::Opponent => (
                self.plus.explain_passive_phase1(you, from_ref(opp), dice, 0),
                matches!(
                    passive_phase1_choices(you, from_ref(opp), dice),
                    Decision::Forced(Some(_))
                ),
            ),
        };

        let posts: Vec<State> = turns
            .iter()
            .map(|t| {
                let mut y = *you;
                for &m in &t.marks {
                    y.apply_mark(m);
                }
                if t.strike {
                    y.apply_strike();
                }
                y
            })
            .collect();
        let z = self.net.evaluate_batch(&posts, from_ref(&opp_after));
        let mut out: Vec<MoveOut> = turns
            .iter()
            .zip(z)
            .map(|(t, z)| MoveOut {
                marks: t.marks.iter().map(|&m| Sq::from(m)).collect(),
                strike: t.strike,
                pass: t.marks.is_empty() && !t.strike,
                p_win: phi(z),
                bots: vec![],
                note: None,
            })
            .collect();

        // Each bot's pick, and the note its row gets if it isn't first.
        // Master+'s colour mark may come from a search of its own.
        let plus = &mut self.plus;
        let (plus_turn, plus_forced2, plus_searched2) = match roller {
            Roller::You => finish_turn_with(you, opp, opp_white, dice, explained.pick, |y, o, hm| {
                let e = plus.explain_active_phase2(y, from_ref(o), dice, hm);
                (e.pick, !e.searched.is_empty())
            }),
            Roller::Opponent => (
                Turn {
                    marks: explained.pick.into_iter().collect(),
                    strike: false,
                },
                false,
                false,
            ),
        };
        let plus_note = if !explained.searched.is_empty() || plus_searched2 {
            Some("rolled_out")
        } else if explained.forced || plus_forced2 {
            Some("rule")
        } else {
            None
        };
        let mut picks = vec![(BotKind::MasterPlus, plus_turn.key(), plus_note)];
        for (kind, bot) in self.others.iter_mut() {
            let (turn, forced2) = bot_turn(bot.as_mut(), you, opp, opp_white, dice, roller);
            let note = (*kind == BotKind::Master && (rule_first || forced2)).then_some("rule");
            picks.push((*kind, turn.key(), note));
        }
        let mut notes: Vec<Option<&'static str>> = vec![None; out.len()];
        for (kind, pick, note) in picks {
            // A pick that matches no listed move is an engine bug: it tags
            // nothing, and development builds report it (the random-position
            // test pins that it never happens).
            if let Some(i) = keys.iter().position(|k| *k == pick) {
                out[i].bots.push(kind.id());
                notes[i] = notes[i].or(note);
            } else {
                debug_assert!(false, "{} picked {:?}, which is not a listed move", kind.id(), pick);
            }
        }
        for (m, n) in out.iter_mut().zip(notes) {
            m.note = n;
        }

        // Highest first; on a tie Master+'s move, then Master's, leads.
        let rank = |m: &MoveOut| match m.bots.first() {
            Some(&"master_plus") => 2,
            Some(&"master") => 1,
            _ => 0,
        };
        out.sort_by(|a, b| {
            b.p_win
                .partial_cmp(&a.p_win)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(rank(b).cmp(&rank(a)))
        });
        // A note explains a move that isn't first; the first row needs none.
        out[0].note = None;
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
        assert!(moves.iter().all(|m| (0.0..=1.0).contains(&m.p_win)));
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

    /// A reported position: Master prefers the strike, Master+'s colour-mark
    /// search prefers yellow 7, so Master+'s row (second) explains itself.
    #[test]
    fn master_plus_off_rank_one_by_its_colour_search_gets_a_note() {
        let mut p = pos(Some([1, 3, 1, 4, 1, 1]), Roller::You);
        for i in [0, 2, 3] {
            p.you.marks[0][i] = true; // red 2, 4, 5
        }
        p.you.marks[1][3] = true; // yellow 5
        let moves = Explorer::new().explore(&p).unwrap().moves.unwrap();
        let plus = moves.iter().position(|m| m.bots.contains(&"master_plus")).unwrap();
        assert!(plus > 0, "{moves:?}");
        assert_eq!(moves[plus].note, Some("rolled_out"));
    }

    /// Spec "Edge cases": arbitrary (unreachable) marks never panic, and
    /// every bot's pick is one of the listed moves.
    #[test]
    fn random_positions_never_panic_and_every_pick_is_listed() {
        let mut ex = Explorer::new();
        let mut rng = SmallRng::seed_from_u64(11);
        let mut unexplained: Vec<String> = vec![];
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
                assert!(moves.windows(2).all(|w| w[0].p_win >= w[1].p_win));
                // Notes sit only on Master+'s or Master's move, below rank 1.
                assert!(moves[0].note.is_none());
                assert!(moves
                    .iter()
                    .all(|m| m.note.is_none() || m.bots.iter().any(|b| *b == "master_plus" || *b == "master")));
                // Master+ is first unless its search or a rule moved it.
                let plus = moves.iter().position(|m| m.bots.contains(&"master_plus")).unwrap();
                if plus != 0 && moves[plus].bots[0] == "master_plus" {
                    assert!(moves[plus].note.is_some(), "Master+ off rank 1 without a note: {p:?}");
                }
                // Master (the static net) is first unless a rule moved it.
                let master = moves.iter().position(|m| m.bots.contains(&"master")).unwrap();
                if master != 0 && moves[master].note.is_none() {
                    unexplained.push(format!("{:.3} vs {:.3}", moves[master].p_win, moves[0].p_win));
                }
                for id in ["master_plus", "master", "student", "newbie"] {
                    assert_eq!(
                        moves.iter().filter(|m| m.bots.contains(&id)).count(),
                        1,
                        "{id} in {p:?}"
                    );
                }
            }
        }
        assert!(
            unexplained.is_empty(),
            "Master off rank 1 without a note: {unexplained:?}"
        );
    }
    #[test]
    fn analysis_batches_compose_and_replay() {
        let ex = Explorer::new();
        let p = pos(None, Roller::You);
        let a = ex.analyze(&p, 0, 40).unwrap();
        let b = ex.analyze(&p, 40, 60).unwrap();
        let all = ex.analyze(&p, 0, 100).unwrap();
        assert_eq!(all.games, 100);
        assert_eq!(
            (a.games + b.games, a.wins + b.wins, a.ties + b.ties),
            (all.games, all.wins, all.ties)
        );
        // Diffs are integers, so f64 sums are exact.
        assert_eq!(a.diff_sum + b.diff_sum, all.diff_sum);
        assert_eq!(a.diff_sq_sum + b.diff_sq_sum, all.diff_sq_sum);
        assert_eq!(ex.analyze(&p, 0, 100).unwrap(), all);
    }

    #[test]
    fn analysis_ignores_the_dice_but_not_the_roller() {
        let ex = Explorer::new();
        let mut p = pos(None, Roller::You);
        p.you.marks[0][2] = true; // red 4
        let base = ex.analyze(&p, 0, 60).unwrap();
        let mut with_dice = p.clone();
        with_dice.dice = Some([6, 6, 1, 2, 3, 4]);
        assert_eq!(ex.analyze(&with_dice, 0, 60).unwrap(), base);
        let mut theirs = p.clone();
        theirs.roller = Roller::Opponent;
        assert_ne!(ex.analyze(&theirs, 0, 60).unwrap(), base);
    }

    #[test]
    fn analysis_refuses_a_finished_game() {
        let ex = Explorer::new();
        let mut p = pos(None, Roller::You);
        p.opponent.strikes = 4;
        assert!(ex.analyze(&p, 0, 10).is_err());
        let mut p = pos(None, Roller::You);
        p.locked = [true, true, false, false];
        assert!(ex.analyze(&p, 0, 10).is_err());
    }

    #[test]
    fn analysis_of_the_empty_position_is_roughly_even() {
        let a = Explorer::new().analyze(&pos(None, Roller::You), 0, 300).unwrap();
        let p = (a.wins as f64 + a.ties as f64 / 2.0) / a.games as f64;
        assert!((0.35..0.65).contains(&p), "p {p}");
        assert!(a.diff_sq_sum > 0.0);
    }
}
