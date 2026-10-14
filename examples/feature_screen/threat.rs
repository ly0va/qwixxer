//! Copied from branch `pair-inputs` (src/threat.rs, tests dropped) for the
//! Phase 24 negative-control features. Throwaway spike code.
//! Dice-only threat quantities for the pair net's inputs (spec §1):
//! P(forced strike) and per-row P(locked by someone else within a round).
//! Every quantity depends only on each row's (direction, total, free), so
//! per-row tables are built once and lookups are cheap.

use qwixxer::state::{Mark, State};

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
        // P(no lock opportunity) per roll: the board's own roll (only others'
        // white-sum locks, P(sum = terminal) = 1/36) and each other player's.
        // Factors are multiplied in sorted order so the result is bit-identical
        // for any order of `others`.
        let mut factors = [1.0f32; 6];
        factors[0] = if any_white { 1.0 - 1.0 / 36.0 } else { 1.0 };
        for (i, s) in others.iter().enumerate() {
            let (act, uni) = t[row_key(s, r)];
            factors[i + 1] = 1.0 - if any_white { uni } else { act };
        }
        let used = &mut factors[..others.len() + 1];
        used.sort_by(|a, b| a.partial_cmp(b).unwrap());
        1.0 - used.iter().product::<f32>()
    })
}
