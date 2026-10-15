//! Win net: a 2-player P(win) network, antisymmetric in the players and
//! invariant to the 8 colour relabellings by construction, trained by TD(λ)
//! self-play on win/loss outcomes (TD-Gammon style).
//!
//! Design: docs/superpowers/specs/2026-10-15-win-net-design.md

use super::pair::board_features;
use crate::state::State;

/// Row-token width: a row's 4 features for me, then the same 4 for the opponent.
pub const TOKEN: usize = 8;
/// Global inputs: 4 per player (strikes, blanks, aggregate weight, lockable
/// rows), then the current difference / 100.
pub const GLOBALS: usize = 9;
/// Full input: 4 row tokens (red, yellow, green, blue) + globals.
pub const WIN_INPUTS: usize = 4 * TOKEN + GLOBALS;

/// Input for "me" against "opp": today's two 20-feature board blocks
/// regrouped by row. The row features are direction-free, so every colour
/// relabelling only permutes the 4 tokens.
pub fn win_input(me: &State, opp: &State) -> [f32; WIN_INPUTS] {
    let (a, b) = (board_features(me), board_features(opp));
    let mut x = [0.0f32; WIN_INPUTS];
    for r in 0..4 {
        for k in 0..4 {
            x[TOKEN * r + k] = a[4 * k + r];
            x[TOKEN * r + 4 + k] = b[4 * k + r];
        }
    }
    x[32..36].copy_from_slice(&a[16..20]);
    x[36..40].copy_from_slice(&b[16..20]);
    let cdiff = (me.count_points() - opp.count_points()) as f32;
    x[40] = (cdiff / 100.0).clamp(-1.0, 1.0);
    x
}

/// The same position from the opponent's side: swap the halves of every
/// token and of the globals, negate the difference.
pub fn swap_input(x: &[f32; WIN_INPUTS]) -> [f32; WIN_INPUTS] {
    let mut s = *x;
    for r in 0..4 {
        for k in 0..4 {
            s[TOKEN * r + k] = x[TOKEN * r + 4 + k];
            s[TOKEN * r + 4 + k] = x[TOKEN * r + k];
        }
    }
    for k in 0..4 {
        s[32 + k] = x[36 + k];
        s[36 + k] = x[32 + k];
    }
    s[40] = -x[40];
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Mark;

    /// Two different mid-game boards.
    pub(super) fn boards() -> (State, State) {
        let mut a = State::default();
        for m in [
            Mark { row: 0, number: 2 },
            Mark { row: 0, number: 5 },
            Mark { row: 2, number: 12 },
        ] {
            a.apply_mark(m);
        }
        a.apply_strike();
        let mut b = State::default();
        for m in [
            Mark { row: 1, number: 4 },
            Mark { row: 3, number: 9 },
            Mark { row: 3, number: 7 },
        ] {
            b.apply_mark(m);
        }
        (a, b)
    }

    #[test]
    fn input_regroups_board_features_into_row_tokens_and_globals() {
        let (a, b) = boards();
        let (fa, fb) = (board_features(&a), board_features(&b));
        let x = win_input(&a, &b);
        for r in 0..4 {
            for k in 0..4 {
                assert_eq!(x[TOKEN * r + k], fa[4 * k + r], "me row {r} feature {k}");
                assert_eq!(x[TOKEN * r + 4 + k], fb[4 * k + r], "opp row {r} feature {k}");
            }
        }
        assert_eq!(&x[32..36], &fa[16..20]);
        assert_eq!(&x[36..40], &fb[16..20]);
        let cdiff = (a.count_points() - b.count_points()) as f32;
        assert_eq!(x[40], (cdiff / 100.0).clamp(-1.0, 1.0));
    }

    #[test]
    fn swapped_input_equals_input_with_players_exchanged() {
        let (a, b) = boards();
        assert_eq!(swap_input(&win_input(&a, &b)), win_input(&b, &a));
        assert_eq!(swap_input(&swap_input(&win_input(&a, &b))), win_input(&a, &b));
    }
}
