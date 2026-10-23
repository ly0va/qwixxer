//! The game code (spec `docs/superpowers/specs/2026-10-24-game-review-design.md`,
//! "The game code"): a finished or unfinished game against the bot as a
//! short URL-safe string. It stores each roll's dice and choices, not the
//! sheets; replaying them through the rules rebuilds the game and rejects
//! any code that isn't one.
//!
//! Layout: the version `1`, a header character (bit 0: the bot rolls first;
//! bits 1–2: the bot level, its index in [`BotKind::ALL`]), then 4
//! characters (24 bits, big-endian) per roll: `dice × 225 + action`, see
//! [`pack`] and [`turn_of`].

use crate::bots::BotKind;
use crate::explorer::{legal_turns, states, Over, Position, Roller, Turn};
use crate::selection::{n_locked, passive_options};
use crate::session::{RollRecord, SheetSnap, Sq, Who};
use qwixxer::state::{Mark, State};
use serde::Serialize;

pub const VERSION: char = '1';
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
/// Distinct actions: roller white (5) × roller colour (9) × other white (5).
const ACTIONS: u32 = 225;
const DICE: u32 = 46_656;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Header {
    pub bot_first: bool,
    pub bot: BotKind,
}

/// One roll as a number below 46,656 × 225: the dice as base-6 digits
/// (d − 1, white 1 first, then white 2, red, yellow, green, blue), then
/// the action.
pub fn pack(dice: [u8; 6], action: u32) -> u32 {
    dice.iter().fold(0, |v, &d| v * 6 + (d - 1) as u32) * ACTIONS + action
}

pub fn unpack(v: u32) -> Result<([u8; 6], u32), String> {
    if v >= DICE * ACTIONS {
        return Err(format!("roll value {v} out of range"));
    }
    let mut d = v / ACTIONS;
    let mut dice = [0u8; 6];
    for i in (0..6).rev() {
        dice[i] = (d % 6) as u8 + 1;
        d /= 6;
    }
    Ok((dice, v % ACTIONS))
}

fn char_value(c: u8) -> Option<u32> {
    ALPHABET.iter().position(|&a| a == c).map(|i| i as u32)
}

pub fn encode(h: &Header, rolls: &[u32]) -> String {
    let bot = BotKind::ALL.iter().position(|&b| b == h.bot).unwrap() as u32;
    let mut s = String::from(VERSION);
    s.push(ALPHABET[(h.bot_first as u32 | bot << 1) as usize] as char);
    for &v in rolls {
        for shift in [18, 12, 6, 0] {
            s.push(ALPHABET[((v >> shift) & 63) as usize] as char);
        }
    }
    s
}

/// The header and the roll values; checks the syntax only ([`replay`]
/// checks the rules).
pub fn decode(code: &str) -> Result<(Header, Vec<u32>), String> {
    let b = code.as_bytes();
    if b.first() != Some(&(VERSION as u8)) {
        return Err("unknown code version".into());
    }
    let header = b.get(1).and_then(|&c| char_value(c)).ok_or("bad code header")?;
    if header >> 3 != 0 {
        return Err("bad code header".into());
    }
    let body = &b[2..];
    if body.len() % 4 != 0 {
        return Err("the code is cut off".into());
    }
    let rolls = body
        .chunks(4)
        .map(|c| {
            let v = c
                .iter()
                .try_fold(0u32, |v, &ch| char_value(ch).map(|x| v << 6 | x))
                .ok_or("bad character in the code")?;
            unpack(v)?;
            Ok(v)
        })
        .collect::<Result<Vec<u32>, String>>()?;
    Ok((
        Header {
            bot_first: header & 1 == 1,
            bot: BotKind::ALL[(header >> 1) as usize],
        },
        rolls,
    ))
}

/// The action's marks: the roller's (white sum first, then white die +
/// colour die) and the other player's white sum. `action = rollerWhite ×
/// 45 + rollerColour × 5 + otherWhite`; whites 0 = none, 1–4 = row 0–3;
/// colour 0 = none, else 1 + row × 2 + white die (0 or 1).
pub fn turn_of(dice: [u8; 6], action: u32) -> (Vec<Mark>, Option<Mark>) {
    let (rw, rc, ow) = (action / 45, action / 5 % 9, action % 5);
    let sum = dice[0] + dice[1];
    let mut marks = vec![];
    if rw > 0 {
        marks.push(Mark {
            row: rw as usize - 1,
            number: sum,
        });
    }
    if rc > 0 {
        let row = (rc as usize - 1) / 2;
        marks.push(Mark {
            row,
            number: dice[(rc as usize - 1) % 2] + dice[2 + row],
        });
    }
    let other = (ow > 0).then(|| Mark {
        row: ow as usize - 1,
        number: sum,
    });
    (marks, other)
}

/// The legal turn with these marks (no marks: the strike or pass the rules
/// allow), with the other player's white mark legal too.
fn resolve(r: &State, o: &State, dice: [u8; 6], marks: Vec<Mark>, other: Option<Mark>) -> Option<Turn> {
    if other.is_some_and(|m| !passive_options(o, dice).contains(&m)) {
        return None;
    }
    let legal = legal_turns(r, o, dice, Roller::You, other);
    if marks.is_empty() {
        return legal.into_iter().find(|t| t.marks.is_empty());
    }
    let want = Turn { marks, strike: false }.key();
    legal.into_iter().find(|t| t.key() == want)
}

/// The lowest action that replays to `turn` (same marks as a set, same
/// strike or pass) with the other player's white mark `other_white`, if
/// any does: each game has exactly one code.
pub fn canonical_action(
    roller: &State,
    other: &State,
    dice: [u8; 6],
    turn: &Turn,
    other_white: Option<Mark>,
) -> Option<u32> {
    let want = turn.key();
    (0..ACTIONS).find(|&a| {
        let (marks, ow) = turn_of(dice, a);
        ow == other_white && resolve(roller, other, dice, marks, ow).is_some_and(|t| t.key() == want)
    })
}

impl Position {
    /// Both sheets empty, no dice.
    pub fn start(roller: Roller) -> Position {
        let empty = SheetSnap {
            marks: [[false; 11]; 4],
            strikes: 0,
        };
        Position {
            you: empty,
            opponent: empty,
            locked: [false; 4],
            dice: None,
            roller,
        }
    }
}

/// One roll of a replayed game: the position before its marks (with the
/// dice and the roller), what happened, and the position after (no dice,
/// the next roller). Rows close from the marks, so `locked` stays false.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReviewRoll {
    pub before: Position,
    pub record: RollRecord,
    pub after: Position,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReviewGame {
    /// [`BotKind::id`].
    pub bot: &'static str,
    pub bot_first: bool,
    pub rolls: Vec<ReviewRoll>,
    /// Points when the game is over; `None` for an unfinished game.
    pub over: Option<Over>,
}

/// One roll from `pos` (its dice ignored): the record, the position after,
/// and the final points if the game ended. Rejects illegal and
/// non-canonical actions.
pub(crate) fn replay_roll(
    pos: &Position,
    dice: [u8; 6],
    action: u32,
) -> Result<(RollRecord, Position, Option<Over>), String> {
    let (you, bot, locked_before) = states(pos);
    let you_roll = pos.roller == Roller::You;
    let (r, o) = if you_roll { (you, bot) } else { (bot, you) };
    let (marks, other) = turn_of(dice, action);
    let turn = resolve(&r, &o, dice, marks.clone(), other)
        .ok_or_else(|| format!("illegal move (action {action}, dice {dice:?})"))?;
    if canonical_action(&r, &o, dice, &turn, other) != Some(action) {
        return Err(format!("action {action} isn't written the canonical way"));
    }
    let mut after = pos.clone();
    after.dice = None;
    after.roller = if you_roll { Roller::Opponent } else { Roller::You };
    {
        let (rs, os) = if you_roll {
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
    }
    let (y, b, locked_after) = states(&after);
    let locked: Vec<usize> = (0..4).filter(|&r| locked_after[r] && !locked_before[r]).collect();
    let (rw, rc) = (action / 45 > 0, action / 5 % 9 > 0);
    let record = if you_roll {
        RollRecord {
            roller: Who::You,
            you: marks.iter().map(|&m| Sq::from(m)).collect(),
            you_strike: turn.strike,
            bot_white: other.map(Sq::from),
            bot_colour: None,
            bot_strike: false,
            locked,
        }
    } else {
        RollRecord {
            roller: Who::Bot,
            you: other.into_iter().map(Sq::from).collect(),
            you_strike: false,
            bot_white: rw.then(|| Sq::from(marks[0])),
            bot_colour: rc.then(|| Sq::from(*marks.last().unwrap())),
            bot_strike: turn.strike,
            locked,
        }
    };
    let over = (y.strikes >= 4 || b.strikes >= 4 || n_locked(locked_after) >= 2).then(|| Over {
        you: y.count_points(),
        opponent: b.count_points(),
    });
    Ok((record, after, over))
}

/// The game a code describes, roll by roll; `Err` if the code is malformed
/// or breaks the rules, or a roll follows the end of the game.
pub fn replay(code: &str) -> Result<ReviewGame, String> {
    let (h, values) = decode(code)?;
    let mut pos = Position::start(if h.bot_first { Roller::Opponent } else { Roller::You });
    let mut rolls = vec![];
    let mut over = None;
    for (k, &v) in values.iter().enumerate() {
        if over.is_some() {
            return Err(format!("roll {} comes after the game ended", k + 1));
        }
        let (dice, action) = unpack(v)?;
        let mut before = pos.clone();
        before.dice = Some(dice);
        let (record, after, o) = replay_roll(&pos, dice, action).map_err(|e| format!("roll {}: {e}", k + 1))?;
        rolls.push(ReviewRoll {
            before,
            record,
            after: after.clone(),
        });
        pos = after;
        over = o;
    }
    Ok(ReviewGame {
        bot: h.bot.id(),
        bot_first: h.bot_first,
        rolls,
        over,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::make_bot;
    use crate::session::{Phase, Session, Snapshot};
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    fn m(row: usize, number: u8) -> Mark {
        Mark { row, number }
    }

    #[test]
    fn pack_unpack_round_trip_at_the_edges() {
        assert_eq!(pack([1; 6], 0), 0);
        assert_eq!(pack([6; 6], 224), 10_497_599);
        assert_eq!(unpack(10_497_599).unwrap(), ([6; 6], 224));
        assert_eq!(
            unpack(pack([3, 4, 2, 5, 6, 1], 117)).unwrap(),
            ([3, 4, 2, 5, 6, 1], 117)
        );
        assert!(unpack(10_497_600).is_err());
    }

    #[test]
    fn encode_decode_round_trip() {
        let h = Header {
            bot_first: true,
            bot: BotKind::Student,
        };
        let rolls = [0, 10_497_599, 12345];
        let code = encode(&h, &rolls);
        assert!(code.starts_with('1'));
        assert_eq!(code.len(), 2 + 12);
        let (h2, r2) = decode(&code).unwrap();
        assert_eq!((h2.bot_first, h2.bot), (true, BotKind::Student));
        assert_eq!(r2, rolls);
    }

    #[test]
    fn decode_rejects_bad_codes() {
        for bad in ["", "2A", "1", "1AAAA", "1A!AAA"] {
            assert!(decode(bad).is_err(), "{bad:?}");
        }
        // Header with bit 3 set ('I' = 8).
        assert!(decode("1I").is_err());
        // A roll value past the last one.
        assert!(decode("1A____").is_err());
    }

    #[test]
    fn turn_of_reads_the_fields() {
        let dice = [3, 4, 2, 5, 6, 1];
        // Roller: yellow white sum 7, then white die 1 (4) + yellow 5 = 9; other: green 7.
        let action = 2 * 45 + (1 + 1 * 2 + 1) * 5 + 3;
        let (marks, other) = turn_of(dice, action);
        assert_eq!(marks, vec![m(1, 7), m(1, 9)]);
        assert_eq!(other, Some(m(2, 7)));
    }

    #[test]
    fn equal_white_dice_take_white_die_0() {
        let s = State::default();
        let dice = [4, 4, 2, 5, 6, 1];
        // Red 6 = white 4 + red 2, from either white die.
        let turn = Turn {
            marks: vec![m(0, 6)],
            strike: false,
        };
        let a = canonical_action(&s, &s, dice, &turn, None).unwrap();
        assert_eq!(a, (1 + 0 * 2 + 0) * 5);
    }

    /// One self-played game: the position before each roll (with its dice
    /// and roller), the session's record, and the final snapshot.
    fn play(seed: u64) -> (Vec<(Position, RollRecord)>, Snapshot) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut s = Session::new(make_bot(BotKind::Master), seed);
        let mut out = vec![];
        loop {
            let snap = s.snapshot();
            let before = |roller| Position {
                you: snap.you,
                opponent: snap.bot,
                locked: [false; 4],
                dice: snap.dice,
                roller,
            };
            match s.view().phase {
                Phase::BotToRoll => s.bot_roll().unwrap(),
                Phase::YouToRoll => s.you_roll().unwrap(),
                Phase::Passive => {
                    let pos = before(Roller::Opponent);
                    let pick = s.view().passive.choose(&mut rng).copied();
                    out.push((pos, s.submit_passive(pick).unwrap()));
                }
                Phase::Active => {
                    let pos = before(Roller::You);
                    let o = s.view().options.unwrap();
                    let rec = match o.selections.choose(&mut rng) {
                        Some(sel) => s.submit_active(&sel.clone(), false).unwrap(),
                        None => s.submit_active(&[], o.strike).unwrap(),
                    };
                    out.push((pos, rec));
                }
                Phase::Over => return (out, s.snapshot()),
            }
        }
    }

    /// The code of a played game, built roll by roll from the records.
    fn code_of(rolls: &[(Position, RollRecord)]) -> String {
        let values: Vec<u32> = rolls
            .iter()
            .map(|(pos, rec)| {
                let (you, bot, _) = states(pos);
                let dice = pos.dice.unwrap();
                let (r, o, turn, other) = match rec.roller {
                    Who::You => (
                        you,
                        bot,
                        Turn {
                            marks: rec.you.iter().map(|&q| q.into()).collect(),
                            strike: rec.you_strike,
                        },
                        rec.bot_white.map(Mark::from),
                    ),
                    Who::Bot => (
                        bot,
                        you,
                        Turn {
                            marks: rec
                                .bot_white
                                .iter()
                                .chain(rec.bot_colour.iter())
                                .map(|&q| q.into())
                                .collect(),
                            strike: rec.bot_strike,
                        },
                        rec.you.first().map(|&q| q.into()),
                    ),
                };
                pack(dice, canonical_action(&r, &o, dice, &turn, other).unwrap())
            })
            .collect();
        encode(
            &Header {
                bot_first: true,
                bot: BotKind::Master,
            },
            &values,
        )
    }

    fn sorted<T: Ord + Clone>(v: &[T]) -> Vec<T> {
        let mut v = v.to_vec();
        v.sort();
        v
    }

    /// Records match as sets: the order of marks within a roll isn't part
    /// of the code (ledger ruling, pre-flight).
    pub(crate) fn same_record(a: &RollRecord, b: &RollRecord) -> bool {
        let key = |r: &RollRecord| {
            (
                r.roller,
                sorted(&r.you.iter().map(|q| (q.row, q.n)).collect::<Vec<_>>()),
                r.you_strike,
                sorted(
                    &r.bot_white
                        .iter()
                        .chain(r.bot_colour.iter())
                        .map(|q| (q.row, q.n))
                        .collect::<Vec<_>>(),
                ),
                r.bot_strike,
                sorted(&r.locked),
            )
        };
        key(a) == key(b)
    }

    #[test]
    fn replay_round_trips_self_play() {
        for seed in 0..20 {
            let (rolls, end) = play(seed);
            let g = replay(&code_of(&rolls)).unwrap();
            assert_eq!(g.rolls.len(), rolls.len(), "seed {seed}");
            assert_eq!(g.bot, "master");
            for (k, (r, (pos, rec))) in g.rolls.iter().zip(&rolls).enumerate() {
                assert!(
                    same_record(&r.record, rec),
                    "seed {seed} roll {k}: {:?} vs {rec:?}",
                    r.record
                );
                assert_eq!(r.before.you, pos.you, "seed {seed} roll {k}");
                assert_eq!(r.before.dice, pos.dice);
                assert_eq!(r.before.roller, pos.roller);
            }
            let last = &g.rolls.last().unwrap().after;
            assert_eq!((last.you, last.opponent), (end.you, end.bot), "seed {seed}");
            assert!(g.over.is_some(), "seed {seed}");
        }
    }

    #[test]
    fn replay_rejects_illegal_and_trailing_rolls() {
        let (rolls, _) = play(3);
        let code = code_of(&rolls);
        // A roll after the game ended.
        assert!(replay(&format!("{code}AAAA")).is_err());
        // Red white sum 12 on an empty sheet: closing a row needs 5 marks.
        let dice = [6, 6, 1, 1, 1, 1];
        assert!(replay_roll(&Position::start(Roller::You), dice, 1 * 45).is_err());
        // Red 6 via white die 1 with equal white dice: not the canonical action.
        let dice = [4, 4, 2, 5, 6, 1];
        assert!(replay_roll(&Position::start(Roller::Opponent), dice, (1 + 1) * 5).is_err());
        assert!(replay_roll(&Position::start(Roller::Opponent), dice, 1 * 5).is_ok());
    }

    #[test]
    fn a_prefix_is_an_unfinished_game() {
        let (rolls, _) = play(5);
        let code = code_of(&rolls);
        let g = replay(&code[..2 + 4 * 5]).unwrap();
        assert_eq!(g.rolls.len(), 5);
        assert!(g.over.is_none());
    }

    fn marked(rows: [&[usize]; 4]) -> SheetSnap {
        let mut s = SheetSnap {
            marks: [[false; 11]; 4],
            strikes: 0,
        };
        for (r, idx) in rows.iter().enumerate() {
            for &i in *idx {
                s.marks[r][i] = true;
            }
        }
        s
    }

    #[test]
    fn phase_one_end_skips_the_colour() {
        // You have five red marks, the bot five yellow: white sum 12 lets
        // you close red and the bot close yellow in phase 1, ending the game.
        let pos = Position {
            you: marked([&[0, 1, 2, 3, 4], &[], &[], &[]]),
            opponent: marked([&[], &[0, 1, 2, 3, 4], &[], &[]]),
            locked: [false; 4],
            dice: None,
            roller: Roller::You,
        };
        let dice = [6, 6, 1, 1, 1, 1];
        let (rec, after, over) = replay_roll(&pos, dice, 1 * 45 + 2).unwrap();
        assert_eq!(rec.locked.len(), 2);
        assert!(over.is_some());
        assert!(after.you.marks[0][10] && after.opponent.marks[1][10]);
        // The same roll with a colour mark (blue 7: white 6 + blue 1) is illegal.
        assert!(replay_roll(&pos, dice, 1 * 45 + (1 + 3 * 2) * 5 + 2).is_err());
    }
}
