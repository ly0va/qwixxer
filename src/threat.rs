//! Dice-only threat quantities for the pair net's inputs (spec §1):
//! P(forced strike) and per-row P(locked by someone else within a round).
//! Every quantity depends only on each row's (direction, total, free), so
//! per-row tables are built once and lookups are cheap.

use crate::state::{Mark, State};

use std::sync::OnceLock;

/// Row-state index: direction (asc = rows 0/1), total 0..=12, free 0 = locked
/// or 2..=12. Table entries for impossible combinations are never read.
fn row_index(asc: bool, total: u8, free: Option<u8>) -> usize {
    ((asc as usize) * 13 + total as usize) * 13 + free.unwrap_or(0) as usize
}

const ROW_STATES: usize = 2 * 13 * 13;

/// A State whose row `r` (0 = asc, 2 = desc) has the given (total, free).
fn probe_state(asc: bool, total: u8, free: Option<u8>) -> (State, usize) {
    let r = if asc { 0 } else { 2 };
    let mut parts = [(0, Some(2)), (0, Some(2)), (0, Some(12)), (0, Some(12))];
    parts[r] = (total, free);
    (State::from_parts(0, parts), r)
}

/// Per row state and ordered white pair (w1 - 1) * 6 + (w2 - 1):
/// (white sum markable, number of colour values c with neither W1+c nor W2+c markable).
fn strike_table() -> &'static Vec<[(bool, u8); 36]> {
    static T: OnceLock<Vec<[(bool, u8); 36]>> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = vec![[(false, 6u8); 36]; ROW_STATES];
        for asc in [false, true] {
            for total in 0..=12u8 {
                for free in std::iter::once(None).chain((2..=12u8).map(Some)) {
                    let (s, r) = probe_state(asc, total, free);
                    let e = &mut t[row_index(asc, total, free)];
                    for w1 in 1..=6u8 {
                        for w2 in 1..=6u8 {
                            let none = (1..=6u8)
                                .filter(|&c| !s.can_mark(r, w1 + c) && !s.can_mark(r, w2 + c))
                                .count();
                            e[((w1 - 1) * 6 + (w2 - 1)) as usize] = (s.can_mark(r, w1 + w2), none as u8);
                        }
                    }
                }
            }
        }
        t
    })
}

/// Per row state: (P(own active roll can lock the row), P(white sum hits the
/// terminal OR own active roll can lock)), over (W1, W2, own colour die).
fn lock_table() -> &'static Vec<(f32, f32)> {
    static T: OnceLock<Vec<(f32, f32)>> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = vec![(0.0f32, 0.0f32); ROW_STATES];
        for asc in [false, true] {
            for total in 0..=12u8 {
                for free in std::iter::once(None).chain((2..=12u8).map(Some)) {
                    let (s, r) = probe_state(asc, total, free);
                    let term = State::row_terminal(r);
                    let (mut act, mut uni) = (0u32, 0u32);
                    for w1 in 1..=6u8 {
                        for w2 in 1..=6u8 {
                            for c in 1..=6u8 {
                                let ws = w1 + w2;
                                let colour_hits = w1 + c == term || w2 + c == term;
                                let direct = s.can_mark(r, term) && (ws == term || colour_hits);
                                let via_white = ws != term && s.can_mark(r, ws) && colour_hits && {
                                    let mut s2 = s;
                                    s2.apply_mark(Mark { row: r, number: ws });
                                    s2.can_mark(r, term)
                                };
                                let a = direct || via_white;
                                act += a as u32;
                                uni += (a || ws == term) as u32;
                            }
                        }
                    }
                    t[row_index(asc, total, free)] = (act as f32 / 216.0, uni as f32 / 216.0);
                }
            }
        }
        t
    })
}

fn row_key(s: &State, r: usize) -> usize {
    row_index(r < 2, s.row_totals()[r], s.row_free_values()[r])
}

/// P(a full roll [W1, W2, R, Y, G, B] offers `s` no legal mark at all).
pub fn forced_strike_prob(s: &State) -> f32 {
    let t = strike_table();
    let rows: [&[(bool, u8); 36]; 4] = std::array::from_fn(|r| &t[row_key(s, r)]);
    let mut p = 0.0f32;
    for w in 0..36 {
        if rows.iter().any(|row| row[w].0) {
            continue;
        }
        p += rows.iter().map(|row| row[w].1 as f32 / 6.0).product::<f32>() / 36.0;
    }
    p
}

/// Per row r: P(some player other than `board`'s gets an opportunity to lock
/// r during the next round — one roll by each player). `others` = all other
/// players' states, in any order. 0 for a locked row.
pub fn lock_threats(board: &State, others: &[&State]) -> [f32; 4] {
    let t = lock_table();
    std::array::from_fn(|r| {
        if board.row_free_values()[r].is_none() {
            return 0.0;
        }
        let term = State::row_terminal(r);
        let any_white = others.iter().any(|s| s.can_mark(r, term));
        // Board's own roll: only others' white-sum locks (P(sum = terminal) = 1/36).
        let mut none = if any_white { 1.0 - 1.0 / 36.0 } else { 1.0 };
        for s in others {
            let (act, uni) = t[row_key(s, r)];
            none *= 1.0 - if any_white { uni } else { act };
        }
        1.0 - none
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_rolls() -> impl Iterator<Item = [u8; 6]> {
        (0..46656u32).map(|mut i| {
            let mut d = [0u8; 6];
            for x in d.iter_mut() {
                *x = (i % 6) as u8 + 1;
                i /= 6;
            }
            d
        })
    }

    /// Fresh, mid, near-lock (5 marks), 4-marks double-lock case, a row
    /// locked by someone else (free None, low total), and a dense board.
    fn boards() -> Vec<State> {
        vec![
            State::default(),
            State::from_parts(1, [(3, Some(7)), (2, Some(5)), (4, Some(8)), (1, Some(11))]),
            State::from_parts(0, [(5, Some(11)), (6, Some(12)), (5, Some(3)), (2, Some(2))]),
            State::from_parts(2, [(4, Some(10)), (4, Some(9)), (4, Some(4)), (4, Some(5))]),
            State::from_parts(3, [(2, None), (7, Some(12)), (0, Some(12)), (5, Some(2))]),
            State::from_parts(3, [(4, Some(12)), (3, Some(12)), (4, Some(2)), (2, Some(2))]),
        ]
    }

    #[test]
    fn forced_strike_matches_brute_force() {
        for s in boards() {
            let none = all_rolls()
                .filter(|&d| s.generate_white_moves(d[0] + d[1]).is_empty() && s.generate_color_moves(d).is_empty())
                .count();
            let expected = none as f32 / 46656.0;
            assert!(
                (forced_strike_prob(&s) - expected).abs() < 1e-6,
                "{s:?}: {} vs {expected}",
                forced_strike_prob(&s)
            );
        }
    }

    /// Reference: per roll by `roller`, can any player other than `b` lock row
    /// r with these dice, using the real move generator (white mark, or white
    /// then colour on the roller's own turn)?
    fn reference_threat(all: &[State], b: usize, r: usize) -> f32 {
        let t = State::row_terminal(r);
        let lock = Mark { row: r, number: t };
        if all[b].row_free_values()[r].is_none() {
            return 0.0;
        }
        let mut none = 1.0f64;
        for roller in 0..all.len() {
            let hits = all_rolls()
                .filter(|&d| {
                    (0..all.len()).filter(|&k| k != b).any(|k| {
                        let s = all[k];
                        let whites = s.generate_white_moves(d[0] + d[1]);
                        if whites.contains(&lock) {
                            return true;
                        }
                        if k != roller {
                            return false;
                        }
                        std::iter::once(None).chain(whites.iter().map(|&m| Some(m))).any(|m1| {
                            let mut s2 = s;
                            if let Some(m) = m1 {
                                s2.apply_mark(m);
                            }
                            s2.generate_color_moves(d).contains(&lock)
                        })
                    })
                })
                .count();
            none *= 1.0 - hits as f64 / 46656.0;
        }
        (1.0 - none) as f32
    }

    #[test]
    fn lock_threats_match_brute_force() {
        let bs = boards();
        let tables: [&[usize]; 3] = [&[2, 1], &[2, 3, 5], &[1, 2, 3, 4]];
        for idxs in tables {
            let all: Vec<State> = idxs.iter().map(|&i| bs[i]).collect();
            for b in 0..all.len() {
                let others: Vec<&State> = (0..all.len()).filter(|&k| k != b).map(|k| &all[k]).collect();
                let got = lock_threats(&all[b], &others);
                for r in 0..4 {
                    let want = reference_threat(&all, b, r);
                    assert!(
                        (got[r] - want).abs() < 1e-5,
                        "players {idxs:?} b {b} row {r}: {} vs {want}",
                        got[r]
                    );
                }
            }
        }
    }

    #[test]
    fn own_lockability_does_not_count_as_threat() {
        // Board can lock red itself; nobody else can: threat is 0.
        let me = State::from_parts(0, [(6, Some(12)), (0, Some(2)), (0, Some(12)), (0, Some(12))]);
        let other = State::default();
        assert_eq!(lock_threats(&me, &[&other])[0], 0.0);
    }

    #[test]
    fn locked_row_has_no_threat_and_offers_no_marks() {
        let locked = State::from_parts(0, [(1, None), (0, Some(2)), (0, Some(12)), (0, Some(12))]);
        let lockable = State::from_parts(0, [(1, None), (6, Some(12)), (0, Some(12)), (0, Some(12))]);
        assert_eq!(lock_threats(&locked, &[&lockable])[0], 0.0);
        assert!(lock_threats(&locked, &[&lockable])[1] > 0.0);
    }
}
