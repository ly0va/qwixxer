//! Game review (spec `docs/superpowers/specs/2026-10-24-game-review-design.md`,
//! "The numbers"): per roll of a replayed game, luck as the explorer
//! measures it, the played move's 1-ply value, and what each player's
//! choice cost. Every value is your (the player's) P(win).

use crate::explorer::{legal_turns, predict_opp_white, states, Explorer, Position, Roller, Turn, WHITE_PAIRS};
use crate::gamecode::ReviewGame;
use crate::selection::n_locked;
use crate::session::{SheetSnap, Sq, Who};
use qwixxer::state::Mark;
use serde::Serialize;

/// A move of yours and the position it leaves (everything else as it happened).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BestMove {
    pub marks: Vec<Sq>,
    pub strike: bool,
    pub after: Position,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RollAnalysis {
    /// Roll 1 only: E[f] of the first roll, the graph's start point.
    pub start: Option<f32>,
    /// Roll 1 only: the variance of the first roll's f (of its luck).
    pub start_var: Option<f32>,
    /// f(this roll): your P(win) after the roller's best turn (the luck chip's "this roll").
    pub this_roll: f32,
    /// The played move's 1-ply value: the next roll's luck average (or 0, ½, 1 at the end).
    pub after_move: f32,
    /// The variance of the next roll's f (of its luck), 0 at the end.
    pub after_var: f32,
    /// What your choice this roll cost you (your turn, or your white-sum mark on the bot's roll).
    pub you_cost: f32,
    pub you_best: BestMove,
    /// What the bot's choice this roll cost it (in your favour).
    pub bot_cost: f32,
}

/// `pos` (dice dropped, the next roller) after the roller's `turn` and the
/// other player's white-sum mark `other`.
fn applied(pos: &Position, turn: &Turn, other: Option<Mark>) -> Position {
    let you_roll = pos.roller == Roller::You;
    let mut after = pos.clone();
    after.dice = None;
    after.roller = if you_roll { Roller::Opponent } else { Roller::You };
    let (rs, os): (&mut SheetSnap, &mut SheetSnap) = if you_roll {
        (&mut after.you, &mut after.opponent)
    } else {
        (&mut after.opponent, &mut after.you)
    };
    for &m in &turn.marks {
        rs.mark(m.into());
    }
    if turn.strike {
        rs.strikes += 1;
    }
    if let Some(m) = other {
        os.mark(m.into());
    }
    after
}

impl Explorer {
    /// E[f] of the next roll from `after`, your P(win); a finished game is
    /// worth 1, 0 or ½ by points.
    pub(crate) fn one_ply(&self, after: &Position) -> f32 {
        self.one_ply_moments(after).0
    }

    /// [`Explorer::one_ply`] and the variance of the next roll's f (0 at the end).
    pub(crate) fn one_ply_moments(&self, after: &Position) -> (f32, f32) {
        let (you, opp, locked) = states(after);
        if you.strikes >= 4 || opp.strikes >= 4 || n_locked(locked) >= 2 {
            let (a, b) = (you.count_points(), opp.count_points());
            let v = if a > b {
                1.0
            } else if a < b {
                0.0
            } else {
                0.5
            };
            return (v, 0.0);
        }
        let (m1, m2) = self.top_sums(&you, &opp, after.roller, 0..WHITE_PAIRS as usize);
        (m1 as f32, (m2 - m1 * m1).max(0.0) as f32)
    }

    /// The explorer's 1-ply value of your move on your roll: [`Explorer::one_ply`]
    /// after it and the opponent's predicted white-sum mark. With the bot's
    /// actual mark as predicted it is the review's value of that move.
    pub fn move_one_ply(&self, pos: &Position, marks: &[Sq], strike: bool) -> Result<f32, String> {
        pos.validate()?;
        if pos.roller != Roller::You {
            return Err("1-ply values are for your roll".into());
        }
        let dice = pos.dice.ok_or("no dice")?;
        let (you, opp, locked) = states(pos);
        if you.strikes >= 4 || opp.strikes >= 4 || n_locked(locked) >= 2 {
            return Err("the game is over".into());
        }
        let opp_white = predict_opp_white(&self.net, &you, &opp, dice);
        let turn = Turn {
            marks: marks.iter().map(|&q| q.into()).collect(),
            strike,
        };
        if !legal_turns(&you, &opp, dice, Roller::You, opp_white)
            .iter()
            .any(|t| t.key() == turn.key())
        {
            return Err("not a legal move".into());
        }
        Ok(self.one_ply(&applied(pos, &turn, opp_white)))
    }

    /// Spec "The numbers" for roll `k` of `game`.
    pub fn review_roll(&self, game: &ReviewGame, k: usize) -> Result<RollAnalysis, String> {
        let roll = game.rolls.get(k).ok_or_else(|| format!("no roll {}", k + 1))?;
        let before = &roll.before;
        let dice = before.dice.ok_or("no dice")?;
        let rec = &roll.record;
        let this_roll = self.luck(before, 0, 0)?.this_roll;
        let (start, start_var) = if k == 0 {
            let (m, v) = self.one_ply_moments(before);
            (Some(m), Some(v))
        } else {
            (None, None)
        };
        let (after_move, after_var) = self.one_ply_moments(&roll.after);

        let (you, bot, _) = states(before);
        let you_roll = rec.roller == Who::You;
        let (r, o) = if you_roll { (you, bot) } else { (bot, you) };
        let played = if you_roll {
            Turn {
                marks: rec.you.iter().map(|&q| q.into()).collect(),
                strike: rec.you_strike,
            }
        } else {
            Turn {
                marks: rec
                    .bot_white
                    .iter()
                    .chain(rec.bot_colour.iter())
                    .map(|&q| q.into())
                    .collect(),
                strike: rec.bot_strike,
            }
        };
        let other: Option<Mark> = if you_roll {
            rec.bot_white.map(Into::into)
        } else {
            rec.you.first().map(|&q| q.into())
        };

        // The roller: the static top 3 (against the other's actual white
        // mark) plus the played turn, each by its 1-ply value.
        let turns = legal_turns(&r, &o, dice, Roller::You, other);
        let (_, vals) = self.turn_values(&r, &o, &turns, Roller::You, other, dice);
        let mut ix: Vec<usize> = (0..turns.len()).collect();
        ix.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));
        let mut cands: Vec<Turn> = ix.iter().take(3).map(|&i| turns[i].clone()).collect();
        if !cands.iter().any(|t| t.key() == played.key()) {
            cands.push(played.clone());
        }
        let mut seen: Vec<(Position, f32)> = vec![];
        let valued: Vec<(Turn, Position, f32)> = cands
            .into_iter()
            .map(|t| {
                let pos = applied(before, &t, other);
                let v = if t.key() == played.key() {
                    after_move
                } else if let Some((_, v)) = seen
                    .iter()
                    .find(|(p, _)| p.you == pos.you && p.opponent == pos.opponent)
                {
                    *v
                } else {
                    let v = self.one_ply(&pos);
                    seen.push((pos.clone(), v));
                    v
                };
                (t, pos, v)
            })
            .collect();
        let better = |a: f32, b: f32| if you_roll { a > b } else { a < b };
        let best = valued
            .iter()
            .fold(&valued[0], |acc, c| if better(c.2, acc.2) { c } else { acc });
        let roller_cost = (best.2 - after_move).abs();

        // The other player's white-sum mark, as the explorer ranks it.
        let passive_cost = |me, them| {
            let (turns, vals) = self.move_values(me, them, dice, Roller::Opponent);
            let want = Turn {
                marks: other.into_iter().collect(),
                strike: false,
            }
            .key();
            let i = turns
                .iter()
                .position(|t| t.key() == want)
                .expect("the played mark is legal");
            let top = (0..turns.len()).fold(0, |a, j| if vals[j] > vals[a] { j } else { a });
            (vals[top] - vals[i], turns[top].clone())
        };

        let (you_cost, you_best, bot_cost) = if you_roll {
            let (bot_cost, _) = passive_cost(&bot, &you);
            let mv = BestMove {
                marks: best.0.marks.iter().map(|&m| Sq::from(m)).collect(),
                strike: best.0.strike,
                after: best.1.clone(),
            };
            (roller_cost, mv, bot_cost)
        } else {
            let (you_cost, mine) = passive_cost(&you, &bot);
            // Your best mark with the bot's turn as played; a colour mark in
            // a row your mark has just closed is dropped.
            let mut after = applied(before, &played, mine.marks.first().copied());
            if let Some(c) = rec.bot_colour {
                let closed = mine
                    .marks
                    .first()
                    .is_some_and(|m| m.row == c.row && m.number == if c.row < 2 { 12 } else { 2 });
                if closed {
                    after.opponent.marks[c.row][crate::session::cell_index(c.row, c.n)] = false;
                }
            }
            let mv = BestMove {
                marks: mine.marks.iter().map(|&m| Sq::from(m)).collect(),
                strike: false,
                after,
            };
            (you_cost, mv, roller_cost)
        };
        Ok(RollAnalysis {
            start,
            start_var,
            this_roll,
            after_move,
            after_var,
            you_cost,
            you_best,
            bot_cost,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::{make_bot, BotKind};
    use crate::explorer::rolls;
    use crate::gamecode::{canonical_action, decode, encode, pack, replay, unpack, Header};
    use crate::session::{Phase, Session};
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    /// A finished self-played game (Master bot, random choices of yours) and its code.
    fn game(seed: u64) -> (String, ReviewGame) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut s = Session::new(make_bot(BotKind::Master), seed);
        loop {
            match s.view().phase {
                Phase::BotToRoll => s.bot_roll().unwrap(),
                Phase::YouToRoll => s.you_roll().unwrap(),
                Phase::Passive => {
                    let pick = s.view().passive.choose(&mut rng).copied();
                    s.submit_passive(pick).unwrap();
                }
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
        let code = s.code(BotKind::Master).unwrap();
        let g = replay(&code).unwrap();
        (code, g)
    }

    #[test]
    fn after_move_is_the_next_rolls_luck_average() {
        let ex = Explorer::new();
        let (_, g) = game(1);
        for k in 0..g.rolls.len() - 1 {
            let a = ex.review_roll(&g, k).unwrap();
            let next = ex.luck(&g.rolls[k + 1].before, 0, WHITE_PAIRS).unwrap();
            assert_eq!(a.after_move, next.average as f32, "roll {k}");
        }
    }

    #[test]
    fn start_is_roll_ones_luck_average() {
        let ex = Explorer::new();
        let (_, g) = game(2);
        let a0 = ex.review_roll(&g, 0).unwrap();
        assert_eq!(
            a0.start,
            Some(ex.luck(&g.rolls[0].before, 0, WHITE_PAIRS).unwrap().average as f32)
        );
        assert_eq!(ex.review_roll(&g, 1).unwrap().start, None);
    }

    #[test]
    fn this_roll_is_the_explorer_luck() {
        let ex = Explorer::new();
        let (_, g) = game(3);
        for k in [0, 1, 5] {
            let a = ex.review_roll(&g, k).unwrap();
            assert_eq!(
                a.this_roll,
                ex.luck(&g.rolls[k].before, 0, WHITE_PAIRS).unwrap().this_roll
            );
        }
    }

    #[test]
    fn one_ply_and_the_luck_variance_match_brute_force() {
        let ex = Explorer::new();
        let (_, g) = game(4);
        let pos = &g.rolls[6].after;
        let (you, opp, _) = states(pos);
        let fs: Vec<(f64, f64)> = rolls()
            .map(|(d, w)| (w, ex.top(&you, &opp, d, pos.roller) as f64))
            .collect();
        let mean: f64 = fs.iter().map(|(w, f)| w * f).sum();
        let var: f64 = fs.iter().map(|(w, f)| w * (f - mean) * (f - mean)).sum();
        assert!(
            (ex.one_ply(pos) as f64 - mean).abs() < 1e-5,
            "{} vs {mean}",
            ex.one_ply(pos)
        );
        let a = ex.review_roll(&g, 6).unwrap();
        assert!((a.after_var as f64 - var).abs() < 1e-5, "{} vs {var}", a.after_var);
        assert!(var > 0.0);
    }

    #[test]
    fn roll_one_has_the_start_variance_and_the_last_roll_none_after() {
        let ex = Explorer::new();
        let (_, g) = game(12);
        let a0 = ex.review_roll(&g, 0).unwrap();
        let v0 = a0.start_var.unwrap();
        assert!(v0 > 0.0 && v0 < 0.25, "{v0}");
        assert_eq!(ex.review_roll(&g, 1).unwrap().start_var, None);
        assert_eq!(ex.review_roll(&g, g.rolls.len() - 1).unwrap().after_var, 0.0);
    }

    #[test]
    fn costs_are_non_negative_and_best_is_a_candidate() {
        let ex = Explorer::new();
        let (_, g) = game(5);
        for k in 0..g.rolls.len() {
            let a = ex.review_roll(&g, k).unwrap();
            assert!(a.you_cost >= 0.0 && a.bot_cost >= 0.0, "roll {k}: {a:?}");
            for q in &a.you_best.marks {
                let row = &a.you_best.after.you.marks[q.row];
                assert!(
                    row[crate::session::cell_index(q.row, q.n)],
                    "roll {k}: {q:?} not on the best sheet"
                );
            }
        }
    }

    #[test]
    fn passive_cost_matches_the_explorer_list() {
        let ex = Explorer::new();
        let (_, g) = game(6);
        let k = g
            .rolls
            .iter()
            .position(|r| r.before.roller == Roller::Opponent)
            .unwrap();
        let roll = &g.rolls[k];
        let (you, bot, _) = states(&roll.before);
        let (turns, vals) = ex.move_values(&you, &bot, roll.before.dice.unwrap(), Roller::Opponent);
        let played = Turn {
            marks: roll.record.you.iter().map(|&q| q.into()).collect(),
            strike: false,
        }
        .key();
        let i = turns.iter().position(|t| t.key() == played).unwrap();
        let top = vals.iter().copied().fold(f32::MIN, f32::max);
        assert_eq!(ex.review_roll(&g, k).unwrap().you_cost, top - vals[i]);
    }

    #[test]
    fn the_last_roll_is_terminal() {
        let ex = Explorer::new();
        let (_, g) = game(7);
        let a = ex.review_roll(&g, g.rolls.len() - 1).unwrap();
        assert!([0.0, 0.5, 1.0].contains(&a.after_move), "{}", a.after_move);
    }

    /// On your roll, the static top 3 and the best of their explorer 1-ply values.
    fn explorer_best(ex: &Explorer, pos: &Position) -> (Vec<Sq>, bool, f32) {
        let (you, bot, _) = states(pos);
        let (turns, vals) = ex.move_values(&you, &bot, pos.dice.unwrap(), Roller::You);
        let mut ix: Vec<usize> = (0..turns.len()).collect();
        ix.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));
        ix.iter()
            .take(3)
            .map(|&i| {
                let marks: Vec<Sq> = turns[i].marks.iter().map(|&m| Sq::from(m)).collect();
                let v = ex.move_one_ply(pos, &marks, turns[i].strike).unwrap();
                (marks, turns[i].strike, v)
            })
            .fold(None::<(Vec<Sq>, bool, f32)>, |acc, c| match acc {
                Some(a) if a.2 >= c.2 => Some(a),
                _ => Some(c),
            })
            .unwrap()
    }

    #[test]
    fn explorer_one_ply_agrees_with_the_review_when_the_prediction_held() {
        let ex = Explorer::new();
        let mut checked = 0;
        for seed in [9, 10] {
            let (_, g) = game(seed);
            for (k, roll) in g.rolls.iter().enumerate() {
                let pos = &roll.before;
                if pos.roller != Roller::You {
                    continue;
                }
                let (you, bot, _) = states(pos);
                let predicted = crate::explorer::predict_opp_white(&ex.net, &you, &bot, pos.dice.unwrap());
                if predicted != roll.record.bot_white.map(Into::into) {
                    continue;
                }
                let a = ex.review_roll(&g, k).unwrap();
                let (marks, strike, v) = explorer_best(&ex, pos);
                assert_eq!(v, a.after_move + a.you_cost, "seed {seed} roll {k}");
                if a.you_cost > 0.0 {
                    assert_eq!(
                        (marks, strike),
                        (a.you_best.marks.clone(), a.you_best.strike),
                        "seed {seed} roll {k}"
                    );
                }
                checked += 1;
            }
        }
        assert!(checked >= 10, "only {checked} rolls checked");
    }

    #[test]
    fn explorer_one_ply_picks_the_reviews_best_on_the_reported_roll() {
        // The user's game: on roll 8 the static top (B7+R5, played) and the
        // 1-ply best (B7+Y6) differ.
        let ex = Explorer::new();
        let g = replay("1BD6DXn-KjBvbXBFuDA4zZShp6WEFNQ8nZNdrYgHXVkOwcA3vvXyvSWrebDhIDYUjtC9MNlc1gD2Udm8w_").unwrap();
        let a = ex.review_roll(&g, 7).unwrap();
        let (marks, strike, v) = explorer_best(&ex, &g.rolls[7].before);
        assert_eq!((marks, strike), (a.you_best.marks.clone(), a.you_best.strike));
        assert_eq!(v, a.after_move + a.you_cost);
    }

    #[test]
    fn move_one_ply_is_for_a_legal_move_on_your_roll() {
        let ex = Explorer::new();
        let (_, g) = game(11);
        let theirs = g.rolls.iter().find(|r| r.before.roller == Roller::Opponent).unwrap();
        assert!(ex.move_one_ply(&theirs.before, &[], false).is_err());
        let mine = g.rolls.iter().find(|r| r.before.roller == Roller::You).unwrap();
        let bogus = [Sq { row: 0, n: 2 }, Sq { row: 0, n: 3 }, Sq { row: 0, n: 4 }];
        assert!(ex.move_one_ply(&mine.before, &bogus, false).is_err());
        assert!(ex.move_one_ply(&mine.before, &[], true).is_ok());
    }

    #[test]
    fn playing_the_reported_best_costs_nothing() {
        let ex = Explorer::new();
        let (code, g) = game(8);
        let k = (0..g.rolls.len())
            .find(|&k| g.rolls[k].before.roller == Roller::You && ex.review_roll(&g, k).unwrap().you_cost > 0.0)
            .expect("a costly move of yours");
        let a = ex.review_roll(&g, k).unwrap();
        let roll = &g.rolls[k];
        let (you, bot, _) = states(&roll.before);
        let dice = roll.before.dice.unwrap();
        let best = Turn {
            marks: a.you_best.marks.iter().map(|&q| q.into()).collect(),
            strike: a.you_best.strike,
        };
        let action = canonical_action(&you, &bot, dice, &best, roll.record.bot_white.map(Into::into)).unwrap();
        let (h, values) = decode(&code).unwrap();
        let mut v = values[..k].to_vec();
        v.push(pack(dice, action));
        assert_eq!(unpack(v[k]).unwrap().0, dice);
        let g2 = replay(&encode(
            &Header {
                bot_first: h.bot_first,
                bot: h.bot,
            },
            &v,
        ))
        .unwrap();
        assert_eq!(ex.review_roll(&g2, k).unwrap().you_cost, 0.0);
    }
}
