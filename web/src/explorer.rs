//! The state explorer (spec `docs/superpowers/specs/2026-10-14-state-explorer-design.md`):
//! any two-sheet position, evaluated by the win net, and for a roll every
//! legal move of yours ranked by the win net's chance to win after it (as
//! Master judges it), tagged with the move each web bot would make in your
//! seat.

use crate::bots::{make_bot, master_plus, win, BotKind};
use crate::selection::{active_options, n_locked, passive_options, union};
use crate::session::{SheetSnap, Sq};
use qwixxer::dqn::win::WinStrategy;
use qwixxer::state::{post_state_dominates, Mark, State};
use qwixxer::strategy::bot_impl::{
    active_phase1_choices, active_phase2_choices_with, passive_lookahead_values, passive_phase1_choices_with,
    passive_phase1_impl, passive_turn_predictions, Decision,
};
use qwixxer::strategy::search::{context_seed, phi, sample_player_seed, SearchBot, WinProb};
use qwixxer::strategy::sim::{BatchedRollouts, SimGame};
use qwixxer::strategy::{Bot, Next, Strategy};
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
    /// The win net's P(you win) before the roll, for whoever rolls next
    /// (Master's and Master+'s evaluator).
    pub p_win: f32,
    /// Expected final score difference, you − opponent (the net's aux head).
    pub margin: f32,
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
    /// The win net's P(you win) after the move, judged as Master judges it
    /// (see `Explorer::moves`).
    pub p_win: f32,
    /// Ids of the bots that would make this move in your seat.
    pub bots: Vec<&'static str>,
    /// On Master+'s move when it isn't ranked first, why Master+ still
    /// plays it: "rolled_out" (its search preferred it) or "rule" (a
    /// meta-rule forced it, or pruned every move above it as dominated).
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
    /// On their roll (dice set, game not over): their predicted turn after
    /// your top-ranked move, as Master's lookahead predicts it.
    pub opponent_turn: Option<OpponentTurn>,
}

/// Spec "Engine" (`2026-10-15-explorer-analysis-design.md`): totals of
/// play-outs `start .. start + count` of a position. Batches add up.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Analysis {
    pub games: u32,
    /// Sum, and sum of squares, of each play-out's value for you: a
    /// finished game's result (1 win, ½ tie, 0 loss), else the win net's
    /// P(you win) after [`PLAY_OUT_TURNS`] turns.
    pub win_sum: f64,
    pub win_sq_sum: f64,
    /// Sum of each play-out's final score difference, you − opponent: a
    /// finished game's, else the net's predicted one (its aux head).
    pub diff_sum: f64,
}

/// Spec "Engine" (`2026-10-22-explorer-roll-luck-design.md`): your P(win)
/// after the roller's best turn for this roll, and the share of its average
/// over all rolls that a chunk of white pairs contributes. The chunks' shares add up
/// to the average; luck is this roll minus it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Luck {
    pub this_roll: f32,
    pub average: f64,
}

/// The unordered white pairs, the unit luck is computed in.
pub const WHITE_PAIRS: u32 = 21;

/// Every distinct roll with its probability: the white dice unordered (21
/// pairs, 1/36 or 2/36), times the four colour dice (1,296 outcomes).
#[cfg(test)]
pub(crate) fn rolls() -> impl Iterator<Item = ([u8; 6], f64)> {
    rolls_of(0..WHITE_PAIRS as usize)
}

/// [`rolls`] for the white pairs `pairs` (numbered in `rolls`' order).
fn rolls_of(pairs: std::ops::Range<usize>) -> impl Iterator<Item = ([u8; 6], f64)> {
    (1..=6u8)
        .flat_map(|a| (a..=6u8).map(move |b| (a, b)))
        .enumerate()
        .filter(move |(i, _)| pairs.contains(i))
        .flat_map(|(_, (a, b))| {
            let w = if a == b { 1.0 } else { 2.0 } / 46_656.0;
            (0..1296u32).map(move |c| {
                let f = |k: u32| (c / 6u32.pow(k) % 6) as u8 + 1;
                ([a, b, f(0), f(1), f(2), f(3)], w)
            })
        })
}

/// Turns a play-out plays before the net values it. Phase 28: at 4 the
/// estimate matches full play-outs (no measurable systematic error) with
/// about 15× fewer simulated turns for the same ±.
pub const PLAY_OUT_TURNS: usize = 4;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpponentWhite {
    pub mark: Option<Sq>,
}

/// The rolling opponent's turn: its white-sum mark, then its colour mark
/// or a strike (neither when the game ends after the white sum).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpponentTurn {
    pub white: Option<Sq>,
    pub colour: Option<Sq>,
    pub strike: bool,
}

/// One full turn of yours: marks in application order, a strike, or (both
/// empty) a pass.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub marks: Vec<Mark>,
    pub strike: bool,
}

impl Turn {
    pub(crate) fn key(&self) -> (Vec<(usize, u8)>, bool) {
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

/// The opponent's white-sum mark on your roll, predicted as Master predicts
/// it (`bot_impl::simulate_opp_phase1`: its plain passive decision).
pub fn predict_opp_white(net: &WinStrategy, you: &State, opp: &State, dice: [u8; 6]) -> Option<Mark> {
    passive_phase1_impl(net, opp, from_ref(you), dice)
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
/// says whether a meta-rule forced its phase 2, judged with the safe-lock
/// rule on or off (`force_lock`).
pub fn bot_turn(
    bot: &mut dyn Strategy,
    you: &State,
    opp: &State,
    opp_white: Option<Mark>,
    dice: [u8; 6],
    roller: Roller,
    force_lock: bool,
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
            finish_turn(bot, you, opp, (opp_white, dice), w, force_lock)
        }
    }
}

/// Your turn after the phase-1 mark `w`, with the opponent's simultaneous
/// mark `opp_white`: locks propagate, then (unless two rows are closed) the
/// phase-2 decision `phase2(you, opponent, has_marked)` against the
/// opponent's marked sheet; nothing marked is a strike. Also returns whether
/// a meta-rule (with the safe-lock rule if `force_lock`) forced phase 2 and
/// whether `phase2` reported a search.
fn finish_turn_with(
    you: &State,
    opp: &State,
    (opp_white, dice): (Option<Mark>, [u8; 6]),
    w: Option<Mark>,
    force_lock: bool,
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
            active_phase2_choices_with(&y, from_ref(&o), dice, w.is_some(), force_lock),
            Decision::Forced(Some(_))
        );
        let (c, s) = phase2(&y, &o, w.is_some());
        searched = s;
        marks.extend(c);
    }
    let strike = marks.is_empty() && active_options(you, opp, opp_white, dice).strike;
    (Turn { marks, strike }, forced, searched)
}

/// For each move, whether pruning explains it sitting below others: the
/// bots prune a move another one dominates row by row, though the net may
/// value it higher. True when some move is valued above it and every such
/// move is dominated by a listed one (`posts`, each move's resulting sheet).
fn pruned_above(p_wins: &[f32], posts: &[State]) -> Vec<bool> {
    let pruned = |j: usize| posts.iter().any(|k| post_state_dominates(k, &posts[j]));
    p_wins
        .iter()
        .map(|&p| {
            let mut above = (0..p_wins.len()).filter(|&j| p_wins[j] > p).peekable();
            above.peek().is_some() && above.all(pruned)
        })
        .collect()
}

/// [`finish_turn_with`] for any bot: its own `active_phase2`.
fn finish_turn(
    bot: &mut dyn Strategy,
    you: &State,
    opp: &State,
    (opp_white, dice): (Option<Mark>, [u8; 6]),
    w: Option<Mark>,
    force_lock: bool,
) -> (Turn, bool) {
    let (turn, forced, _) = finish_turn_with(you, opp, (opp_white, dice), w, force_lock, |y, o, hm| {
        (bot.active_phase2(y, from_ref(o), dice, hm), false)
    });
    (turn, forced)
}

pub struct Explorer {
    /// Master's net: values the position and ranks the moves.
    pub(crate) net: WinStrategy,
    /// Master+ explains its own pick; the other bots are only tagged.
    plus: SearchBot<WinStrategy>,
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
            net: win(),
            plus: master_plus(),
            others: BotKind::ALL[1..].iter().map(|&k| (k, make_bot(k))).collect(),
        }
    }

    pub fn explore(&mut self, pos: &Position) -> Result<Answer, String> {
        pos.validate()?;
        let (you, opp, locked) = states(pos);
        let eval = self.eval(&you, &opp, pos.roller);
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
        let opponent_turn = match (&moves, pos.roller, pos.dice) {
            (Some(moves), Roller::Opponent, Some(dice)) => {
                let best: Option<Mark> = moves[0].marks.first().map(|&q| q.into());
                let t = passive_turn_predictions(&self.net, &you, from_ref(&opp), dice, 0, &[best])[0].2;
                Some(OpponentTurn {
                    white: t.white.map(Sq::from),
                    colour: t.colour.map(Sq::from),
                    strike: t.strike,
                })
            }
            _ => None,
        };
        Ok(Answer {
            over,
            eval,
            moves,
            opponent_white,
            opponent_turn,
        })
    }

    /// Plays `start .. start + count` from the position before the roll (its
    /// dice ignored) for [`PLAY_OUT_TURNS`] turns, the roller first, both
    /// seats Master (the win net, batched across games) without its passive
    /// lookahead, which barely moves the numbers and costs well over twice
    /// the time; then values each by the net. Play-out k's dice depend only
    /// on the position and k, so batches compose and replay exactly.
    pub fn analyze(&self, pos: &Position, start: u32, count: u32) -> Result<Analysis, String> {
        self.play_outs(pos, start, count, false)
    }

    /// [`Explorer::analyze`], with or without Master's passive lookahead.
    fn play_outs(&self, pos: &Position, start: u32, count: u32, lookahead: bool) -> Result<Analysis, String> {
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
        driver.passive_lookahead = lookahead;
        for _ in 0..PLAY_OUT_TURNS {
            driver.step_turn();
        }
        let live: Vec<(&State, &[State], Next)> = driver
            .sims
            .iter()
            .filter(|s| !s.over)
            .map(|s| {
                (
                    &s.states[0],
                    &s.states[1..],
                    if s.active == 0 { Next::Me } else { Next::Opp },
                )
            })
            .collect();
        let mut leaf = self.net.win_prob_multi_next(&live).into_iter();
        let mut a = Analysis {
            games: 0,
            win_sum: 0.0,
            win_sq_sum: 0.0,
            diff_sum: 0.0,
        };
        for sim in &driver.sims {
            let (you, opp) = (&sim.states[0], &sim.states[1]);
            let (w, d) = if sim.over {
                (
                    SimGame::outcome(&sim.states),
                    (you.count_points() - opp.count_points()) as f32,
                )
            } else {
                (leaf.next().unwrap(), self.margin(you, opp))
            };
            a.games += 1;
            a.win_sum += w as f64;
            a.win_sq_sum += (w * w) as f64;
            a.diff_sum += d as f64;
        }
        Ok(a)
    }

    fn eval(&self, you: &State, opp: &State, roller: Roller) -> Eval {
        let next = match roller {
            Roller::You => Next::Me,
            Roller::Opponent => Next::Opp,
        };
        Eval {
            p_win: self.net.win_prob_multi_next(&[(you, from_ref(opp), next)])[0],
            margin: self.margin(you, opp),
        }
    }

    /// The net's predicted final score difference, you − opponent.
    fn margin(&self, you: &State, opp: &State) -> f32 {
        self.net
            .net
            .aux_states(you, opp)
            .expect("the win model has aux heads")
            .diff
    }

    /// Spec "Moves": each legal turn of yours, your sheet after it, and the
    /// win net's P(win) after it, judged as Master judges it (see
    /// [`Explorer::moves`]).
    fn valued_turns(
        &self,
        you: &State,
        opp: &State,
        dice: [u8; 6],
        roller: Roller,
        opp_white: Option<Mark>,
    ) -> (Vec<Turn>, Vec<State>, Vec<f32>) {
        let turns = legal_turns(you, opp, dice, roller, opp_white);
        let (posts, p_wins) = self.turn_values(you, opp, &turns, roller, opp_white, dice);
        (turns, posts, p_wins)
    }

    /// Your sheet after each of `turns` and the P(win) after it, as
    /// [`Explorer::valued_turns`] judges it. On your roll the values don't
    /// depend on the dice.
    pub(crate) fn turn_values(
        &self,
        you: &State,
        opp: &State,
        turns: &[Turn],
        roller: Roller,
        opp_white: Option<Mark>,
        dice: [u8; 6],
    ) -> (Vec<State>, Vec<f32>) {
        let mut opp_after = *opp;
        if let Some(m) = opp_white {
            opp_after.apply_mark(m);
        }
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
        let z = match roller {
            Roller::You => self.net.evaluate_batch_next(&posts, from_ref(&opp_after), Next::Opp),
            Roller::Opponent => {
                let marks: Vec<Option<Mark>> = turns.iter().map(|t| t.marks.first().copied()).collect();
                passive_lookahead_values(&self.net, you, from_ref(opp), dice, 0, &marks)
            }
        };
        (posts, z.into_iter().map(phi).collect())
    }

    /// Each legal turn and the P(win) the move list shows for it, the
    /// opponent's predicted white-sum mark included on your roll.
    pub(crate) fn move_values(&self, you: &State, opp: &State, dice: [u8; 6], roller: Roller) -> (Vec<Turn>, Vec<f32>) {
        let opp_white = match roller {
            Roller::You => predict_opp_white(&self.net, you, opp, dice),
            Roller::Opponent => None,
        };
        let (turns, _, p_wins) = self.valued_turns(you, opp, dice, roller, opp_white);
        (turns, p_wins)
    }

    /// f in spec `2026-10-22-explorer-roll-luck-design.md`: your P(win)
    /// after the roller's best turn. On your roll the move list's top; on
    /// theirs one minus the top of their own move list (sheets swapped).
    pub(crate) fn top(&self, you: &State, opp: &State, dice: [u8; 6], roller: Roller) -> f32 {
        let best = |a, b| {
            self.move_values(a, b, dice, Roller::You)
                .1
                .into_iter()
                .fold(f32::MIN, f32::max)
        };
        match roller {
            Roller::You => best(you, opp),
            Roller::Opponent => 1.0 - best(opp, you),
        }
    }

    /// Spec "Engine" (`2026-10-22-explorer-roll-luck-design.md`): the move
    /// list's top P(win) for the position's roll, and the white pairs
    /// `first .. first + count`'s share of its average over every roll.
    pub fn luck(&self, pos: &Position, first: u32, count: u32) -> Result<Luck, String> {
        pos.validate()?;
        if first + count > WHITE_PAIRS {
            return Err(format!("white pairs {first}..{} past {WHITE_PAIRS}", first + count));
        }
        let pairs = first as usize..(first + count) as usize;
        let dice = pos.dice.ok_or("no dice")?;
        let (you, opp, locked) = states(pos);
        if you.strikes >= 4 || opp.strikes >= 4 || n_locked(locked) >= 2 {
            return Err("the game is over".into());
        }
        Ok(Luck {
            this_roll: self.top(&you, &opp, dice, pos.roller),
            average: self.average_top(&you, &opp, pos.roller, pairs),
        })
    }

    /// E[f]: the white pairs `pairs`' share of f's average over every roll.
    pub(crate) fn average_top(&self, you: &State, opp: &State, roller: Roller, pairs: std::ops::Range<usize>) -> f64 {
        match roller {
            Roller::You => self.average_top_yours(you, opp, pairs),
            Roller::Opponent => {
                let weight: f64 = rolls_of(pairs.clone()).map(|r| r.1).sum();
                weight - self.average_top_yours(opp, you, pairs)
            }
        }
    }

    /// E[f] on your roll (on theirs with the sheets swapped). Per white
    /// pair the opponent's predicted white-sum mark is fixed and a turn's
    /// value depends on your sheet after it, so each distinct turn is
    /// valued once; each colour outcome takes the best of the turns it
    /// makes legal.
    fn average_top_yours(&self, you: &State, opp: &State, pairs: std::ops::Range<usize>) -> f64 {
        let mut total = 0.0;
        let mut outcomes = rolls_of(pairs).peekable();
        while let Some(&(first, w)) = outcomes.peek() {
            let opp_white = predict_opp_white(&self.net, you, opp, first);
            let mut keys = std::collections::HashMap::new();
            let mut distinct: Vec<Turn> = vec![];
            let mut legal: Vec<Vec<usize>> = Vec::with_capacity(1296);
            while let Some(&(d, _)) = outcomes.peek() {
                if d[..2] != first[..2] {
                    break;
                }
                outcomes.next();
                let ix = legal_turns(you, opp, d, Roller::You, opp_white)
                    .into_iter()
                    .map(|t| {
                        *keys.entry(t.key()).or_insert_with(|| {
                            distinct.push(t);
                            distinct.len() - 1
                        })
                    })
                    .collect();
                legal.push(ix);
            }
            let values = self.turn_values(you, opp, &distinct, Roller::You, opp_white, first).1;
            for ix in &legal {
                total += w * ix.iter().map(|&i| values[i]).fold(f32::MIN, f32::max) as f64;
            }
        }
        total
    }

    /// [`Explorer::average_top`] by brute force over every distinct roll.
    #[cfg(test)]
    fn luck_reference(&self, you: &State, opp: &State, roller: Roller) -> f64 {
        rolls().map(|(d, w)| w * self.top(you, opp, d, roller) as f64).sum()
    }

    /// Spec "Moves": every legal turn with the win net's P(win) after it,
    /// judged as Master judges it (on your roll after your turn, against the
    /// opponent's sheet with its predicted white-sum mark `opp_white`; on
    /// theirs at the end of their predicted turn, Master's lookahead),
    /// highest first. When Master+'s or Master's move isn't first, a
    /// note says why: Master+'s search preferred it ("rolled_out"), or a
    /// meta-rule forced it or pruned every move above it ("rule").
    fn moves(
        &mut self,
        you: &State,
        opp: &State,
        dice: [u8; 6],
        roller: Roller,
        opp_white: Option<Mark>,
    ) -> Vec<MoveOut> {
        let (turns, posts, p_wins) = self.valued_turns(you, opp, dice, roller, opp_white);
        let keys: Vec<_> = turns.iter().map(Turn::key).collect();
        // Whether a meta-rule forced Master's phase-1 or passive decision;
        // Master+ shares that pipeline.
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
                    passive_phase1_choices_with(you, from_ref(opp), dice, self.net.force_lock()),
                    Decision::Forced(Some(_))
                ),
            ),
        };

        let mut out: Vec<MoveOut> = turns
            .iter()
            .zip(&p_wins)
            .map(|(t, &p_win)| MoveOut {
                marks: t.marks.iter().map(|&m| Sq::from(m)).collect(),
                strike: t.strike,
                pass: t.marks.is_empty() && !t.strike,
                p_win,
                bots: vec![],
                note: None,
            })
            .collect();

        // Each bot's pick, and the note its row gets if it isn't first.
        // Master+'s colour mark may come from a search of its own.
        let plus = &mut self.plus;
        let force_lock = self.net.force_lock();
        let (plus_turn, plus_forced2, plus_searched2) = match roller {
            Roller::You => finish_turn_with(you, opp, (opp_white, dice), explained.pick, force_lock, |y, o, hm| {
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
            let (turn, forced2) = bot_turn(bot.as_mut(), you, opp, opp_white, dice, roller, force_lock);
            let note = (*kind == BotKind::Master && (rule_first || forced2)).then_some("rule");
            picks.push((*kind, turn.key(), note));
        }
        let pruned_above = pruned_above(&p_wins, &posts);
        let mut notes: Vec<Option<&'static str>> = vec![None; out.len()];
        for (kind, pick, note) in picks {
            // A pick that matches no listed move is an engine bug: it tags
            // nothing, and development builds report it (the random-position
            // test pins that it never happens).
            if let Some(i) = keys.iter().position(|k| *k == pick) {
                let pruned = matches!(kind, BotKind::MasterPlus | BotKind::Master) && pruned_above[i];
                out[i].bots.push(kind.id());
                notes[i] = notes[i].or(note).or(pruned.then_some("rule"));
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
    }

    #[test]
    fn the_position_is_valued_by_the_win_net_for_whoever_rolls_next() {
        let mut ex = Explorer::new();
        let mut p = pos(None, Roller::You);
        p.you.marks[0][2] = true; // red 4
        let (you, opp, _) = states(&p);
        let mine = ex.explore(&p).unwrap().eval;
        p.roller = Roller::Opponent;
        let theirs = ex.explore(&p).unwrap().eval;
        let net = crate::bots::win();
        let expect = |next| net.win_prob_multi_next(&[(&you, from_ref(&opp), next)])[0];
        assert_eq!(mine.p_win, expect(Next::Me));
        assert_eq!(theirs.p_win, expect(Next::Opp));
        assert!(mine.p_win > theirs.p_win, "rolling next is worth something");
        assert_eq!(mine.margin, net.net.aux_states(&you, &opp).unwrap().diff);
    }

    #[test]
    fn their_roll_ranks_your_moves_by_masters_lookahead() {
        let dice = [3, 4, 2, 5, 3, 1];
        let p = pos(Some(dice), Roller::Opponent);
        let moves = Explorer::new().explore(&p).unwrap().moves.unwrap();
        let (you, opp, _) = states(&p);
        let marks: Vec<Option<Mark>> = moves.iter().map(|m| m.marks.first().map(|&q| q.into())).collect();
        let values = passive_lookahead_values(&crate::bots::win(), &you, from_ref(&opp), dice, 0, &marks);
        for (m, v) in moves.iter().zip(values) {
            assert_eq!(m.p_win, phi(v), "{m:?}");
        }
        assert!(moves[0].bots.contains(&"master"), "{moves:?}");
    }

    #[test]
    fn their_roll_predicts_their_turn_after_your_best_move() {
        let dice = [3, 4, 2, 5, 3, 1];
        let p = pos(Some(dice), Roller::Opponent);
        let a = Explorer::new().explore(&p).unwrap();
        assert!(a.opponent_white.is_none());
        let (you, opp, _) = states(&p);
        let best: Option<Mark> = a.moves.unwrap()[0].marks.first().map(|&q| q.into());
        let net = crate::bots::win();
        let t = passive_turn_predictions(&net, &you, from_ref(&opp), dice, 0, &[best])[0].2;
        assert_eq!(
            a.opponent_turn,
            Some(OpponentTurn {
                white: t.white.map(Sq::from),
                colour: t.colour.map(Sq::from),
                strike: t.strike,
            })
        );
        let mine = Explorer::new().explore(&pos(Some(dice), Roller::You)).unwrap();
        assert!(mine.opponent_turn.is_none() && mine.opponent_white.is_some());
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

    /// A position where Master+'s white-sum pick is Master's but its
    /// colour-mark search overrules the net, so Master+'s row (below rank 1)
    /// explains itself.
    #[test]
    fn master_plus_off_rank_one_by_its_colour_search_gets_a_note() {
        let dice = [5, 2, 6, 3, 2, 6];
        let mut p = pos(Some(dice), Roller::You);
        let marks = |s: &mut SheetSnap, rows: [&[usize]; 4], strikes: u8| {
            for (r, idx) in rows.iter().enumerate() {
                for &i in *idx {
                    s.marks[r][i] = true;
                }
            }
            s.strikes = strikes;
        };
        marks(&mut p.you, [&[], &[1, 2, 4, 5], &[2, 3, 7], &[2]], 1);
        marks(&mut p.opponent, [&[0, 4, 5], &[], &[4], &[2, 7]], 1);
        let mut ex = Explorer::new();
        let (you, opp, _) = states(&p);
        assert!(ex
            .plus
            .explain_active_phase1(&you, from_ref(&opp), dice)
            .searched
            .is_empty());
        let moves = ex.explore(&p).unwrap().moves.unwrap();
        let plus = moves.iter().position(|m| m.bots.contains(&"master_plus")).unwrap();
        assert!(plus > 0, "{moves:?}");
        assert_eq!(moves[plus].note, Some("rolled_out"));
    }

    /// A move tied with the top has nothing above it to explain: no note.
    #[test]
    fn a_move_tied_with_the_top_is_not_explained_by_pruning() {
        let mark = |n: u8| {
            let mut s = State::default();
            s.apply_mark(Mark { row: 0, number: n });
            s
        };
        // Red 10 dominates red 11; red 5 and red 11 are valued equally.
        let posts = [mark(5), mark(11), mark(10)];
        assert_eq!(pruned_above(&[0.9, 0.9, 0.8], &posts), [false, false, false]);
        // Red 11 above red 5 is dominated: pruning explains red 5.
        assert_eq!(pruned_above(&[0.8, 0.9, 0.7], &posts), [true, false, false]);
    }

    /// White red 11 leaves the same marks as colour red 10 with one option
    /// fewer, so the bots prune it, though the net values it higher on this
    /// (unreachable) board: their row, below it, gets the rule note.
    #[test]
    fn a_pruned_dominated_move_above_the_bots_pick_gets_a_rule_note() {
        let dice = [6, 5, 5, 3, 5, 1];
        let mut p = pos(Some(dice), Roller::You);
        let marks = |s: &mut SheetSnap, rows: [&[usize]; 4], strikes: u8| {
            for (r, idx) in rows.iter().enumerate() {
                for &i in *idx {
                    s.marks[r][i] = true;
                }
            }
            s.strikes = strikes;
        };
        marks(&mut p.you, [&[2, 7], &[0, 3, 7, 10], &[0, 2, 3, 4, 10], &[1, 5, 10]], 1);
        marks(&mut p.opponent, [&[6, 9], &[6, 7], &[4, 6, 8, 9], &[2, 4, 5, 6]], 3);
        let moves = Explorer::new().explore(&p).unwrap().moves.unwrap();
        assert_eq!(moves[0].marks, [Sq { row: 0, n: 11 }], "{moves:?}");
        let master = moves.iter().position(|m| m.bots.contains(&"master")).unwrap();
        assert_eq!(moves[master].marks, [Sq { row: 0, n: 10 }], "{moves:?}");
        assert_eq!(moves[master].note, Some("rule"));
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
                // Master (the static net) is first, or tied first (a won game
                // is worth exactly 1), unless a rule moved it.
                let master = moves.iter().position(|m| m.bots.contains(&"master")).unwrap();
                if master != 0 && moves[master].note.is_none() && moves[master].p_win != moves[0].p_win {
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
    /// Rows of a sheet marked at these indices (in row order).
    fn marked(rows: [&[usize]; 4], strikes: u8) -> SheetSnap {
        let mut s = sheet();
        for (r, idx) in rows.iter().enumerate() {
            for &i in *idx {
                s.marks[r][i] = true;
            }
        }
        s.strikes = strikes;
        s
    }

    /// Positions for the luck tests, each with both rollers: the empty
    /// one, a midgame, a late one with green closed, and one where the
    /// opponent's white 2 closes blue, ending the game after phase 1 on
    /// your roll.
    fn luck_positions() -> Vec<Position> {
        let mid = |dice| Position {
            you: marked([&[0, 2, 3], &[1, 4], &[0, 1], &[2]], 1),
            opponent: marked([&[1, 2], &[0, 3, 5], &[2], &[0, 1]], 0),
            locked: [false; 4],
            dice: Some(dice),
            roller: Roller::You,
        };
        let late = |dice| Position {
            you: marked([&[0, 1, 2, 3, 4, 5, 6], &[0, 1, 2, 3, 4], &[0, 1], &[0, 1, 2, 3]], 2),
            opponent: marked(
                [&[0, 1, 2, 3, 4, 5], &[0, 1, 2, 3, 4, 5, 6], &[0, 2], &[0, 1, 2, 3, 4]],
                3,
            ),
            locked: [false, false, true, false],
            dice: Some(dice),
            roller: Roller::You,
        };
        let mut end = late([1, 1, 4, 2, 6, 3]);
        end.opponent.marks[3] = [true, true, true, true, true, true, false, false, false, false, false];
        let base = [
            pos(Some([3, 4, 2, 5, 6, 1]), Roller::You),
            mid([2, 5, 6, 3, 1, 4]),
            late([6, 5, 2, 2, 4, 6]),
            end,
        ];
        base.iter()
            .flat_map(|p| {
                let mut o = p.clone();
                o.roller = Roller::Opponent;
                [p.clone(), o]
            })
            .collect()
    }

    /// f is the roller's best turn: on your roll the move list's top; on
    /// theirs one minus the top of their own move list (sheets swapped).
    #[test]
    fn luck_this_roll_is_the_rollers_top() {
        for p in luck_positions() {
            let mut q = p.clone();
            if p.roller == Roller::Opponent {
                (q.you, q.opponent) = (p.opponent.clone(), p.you.clone());
                q.roller = Roller::You;
            }
            let top = Explorer::new().explore(&q).unwrap().moves.unwrap()[0].p_win;
            let want = if p.roller == Roller::You { top } else { 1.0 - top };
            assert_eq!(
                Explorer::new().luck(&p, 0, WHITE_PAIRS).unwrap().this_roll,
                want,
                "{p:?}"
            );
        }
    }

    #[test]
    fn the_average_is_exact() {
        let ex = Explorer::new();
        for p in luck_positions() {
            let (you, opp, _) = states(&p);
            let want = ex.luck_reference(&you, &opp, p.roller);
            let got = ex.luck(&p, 0, WHITE_PAIRS).unwrap().average;
            assert!((got - want).abs() < 1e-5, "{p:?}: {got} vs {want}");
        }
    }

    /// Luck comes in chunks of white pairs so the move list never waits
    /// long behind it; the chunks add up to the whole.
    #[test]
    fn luck_chunks_add_up() {
        let ex = Explorer::new();
        for p in [&luck_positions()[2], &luck_positions()[3]] {
            let all = ex.luck(p, 0, WHITE_PAIRS).unwrap();
            let parts: Vec<Luck> = [(0, 3), (3, 10), (13, 8)]
                .iter()
                .map(|&(a, n)| ex.luck(p, a, n).unwrap())
                .collect();
            assert!(parts.iter().all(|l| l.this_roll == all.this_roll));
            let sum: f64 = parts.iter().map(|l| l.average).sum();
            assert!((sum - all.average).abs() < 1e-9, "{sum} vs {}", all.average);
        }
        assert!(ex.luck(&luck_positions()[0], 20, 2).is_err());
    }

    #[test]
    fn the_rolls_cover_every_roll_once() {
        let v: Vec<_> = rolls().collect();
        assert_eq!(v.len(), 21 * 1296);
        assert!((v.iter().map(|r| r.1).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn luck_refuses_no_dice_and_a_finished_game() {
        let ex = Explorer::new();
        assert!(ex.luck(&pos(None, Roller::You), 0, WHITE_PAIRS).is_err());
        let mut p = pos(Some([1, 2, 3, 4, 5, 6]), Roller::You);
        p.opponent.strikes = 4;
        assert!(ex.luck(&p, 0, WHITE_PAIRS).is_err());
        let mut p = pos(Some([1, 2, 3, 4, 5, 6]), Roller::You);
        p.locked = [true, true, false, false];
        assert!(ex.luck(&p, 0, WHITE_PAIRS).is_err());
    }

    #[test]
    fn analysis_batches_compose_and_replay() {
        let ex = Explorer::new();
        let p = pos(None, Roller::You);
        let a = ex.analyze(&p, 0, 40).unwrap();
        let b = ex.analyze(&p, 40, 60).unwrap();
        let all = ex.analyze(&p, 0, 100).unwrap();
        assert_eq!((all.games, a.games + b.games), (100, 100));
        let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
        assert!(close(a.win_sum + b.win_sum, all.win_sum));
        assert!(close(a.win_sq_sum + b.win_sq_sum, all.win_sq_sum));
        assert!(close(a.diff_sum + b.diff_sum, all.diff_sum));
        assert_eq!(ex.analyze(&p, 0, 100).unwrap(), all);
    }

    /// From the empty position no game can end within PLAY_OUT_TURNS turns,
    /// so every sample is the net's value, strictly between a loss and a
    /// win, and its margin the aux head's, not a whole number.
    #[test]
    fn analysis_stops_after_a_few_turns_and_asks_the_net() {
        let a = Explorer::new().analyze(&pos(None, Roller::You), 0, 50).unwrap();
        assert!(a.win_sq_sum < a.win_sum && a.win_sq_sum > a.win_sum * a.win_sum / 50.0);
        assert_ne!(a.diff_sum, a.diff_sum.round());
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

    /// Play-outs leave out Master's passive lookahead: it barely moves the
    /// numbers and costs well over twice the time.
    #[test]
    fn analysis_plays_out_without_the_passive_lookahead() {
        let ex = Explorer::new();
        let p = pos(None, Roller::You);
        let a = ex.analyze(&p, 0, 60).unwrap();
        assert_eq!(a, ex.play_outs(&p, 0, 60, false).unwrap());
        assert_ne!(a, ex.play_outs(&p, 0, 60, true).unwrap());
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
        let p = a.win_sum / a.games as f64;
        assert!((0.35..0.65).contains(&p), "p {p}");
        assert!((a.diff_sum / a.games as f64).abs() < 10.0);
    }
}
