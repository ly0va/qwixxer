//! Turn position of every board at an evaluation point (spec
//! 2026-10-12-pair-net-turn-and-threat-inputs-design.md, §2). A `BoardClock`
//! is a property of one board at one moment — independent of who evaluates —
//! so it travels with its board when opponent lists are reordered.

/// Most players the turn inputs support.
pub const MAX_PLAYERS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct BoardClock {
    /// Rolls before this board's player next rolls; 0 = rolls next.
    pub rolls_in: u8,
    /// This player is the active player with its colour phase outstanding.
    pub pending: bool,
}

/// Clocks for one evaluation: the evaluating player's board and each opponent
/// board, `opps[k]` aligned with the `opp_states[k]` evaluated alongside.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct EvalClock {
    pub us: BoardClock,
    pub opps: [BoardClock; MAX_PLAYERS - 1],
}

impl BoardClock {
    /// Clock of the board at seat `offset` (0 = the evaluating player) when
    /// the player at offset `next_roller` rolls next and the player at offset
    /// `pending` (if any) has its colour phase outstanding.
    pub fn at(n: usize, offset: usize, next_roller: usize, pending: Option<usize>) -> Self {
        debug_assert!(
            (1..=MAX_PLAYERS).contains(&n),
            "turn inputs support 1..={MAX_PLAYERS} players, got {n}"
        );
        debug_assert!(offset < n && next_roller < n);
        BoardClock {
            rolls_in: ((offset + n - next_roller) % n) as u8,
            pending: pending == Some(offset),
        }
    }
}

impl EvalClock {
    pub fn from_boards(us: BoardClock, opps: &[BoardClock]) -> Self {
        assert!(
            opps.len() < MAX_PLAYERS,
            "at most {} opponents, got {}",
            MAX_PLAYERS - 1,
            opps.len()
        );
        let mut a = [BoardClock::default(); MAX_PLAYERS - 1];
        a[..opps.len()].copy_from_slice(opps);
        EvalClock { us, opps: a }
    }

    /// Turn-ordered opponents (`opp_states[k]` at seat offset `k + 1`),
    /// nothing pending, the player at offset `next_roller` rolls next.
    pub fn turn_start(n: usize, next_roller: usize) -> Self {
        Self::seats(n, next_roller, None)
    }

    /// Turn-ordered; the player at offset `active` is mid-turn: its colour
    /// phase is outstanding and the seat after it rolls next.
    pub fn mid_turn(n: usize, active: usize) -> Self {
        Self::seats(n, (active + 1) % n, Some(active))
    }

    /// After the evaluating player's own complete turn (active phase-1 plans,
    /// phase-2 candidates): the next seat rolls next.
    pub fn after_own_turn(n: usize) -> Self {
        Self::turn_start(n, 1 % n)
    }

    fn seats(n: usize, next_roller: usize, pending: Option<usize>) -> Self {
        let mut opps = [BoardClock::default(); MAX_PLAYERS - 1];
        for o in 1..n {
            opps[o - 1] = BoardClock::at(n, o, next_roller, pending);
        }
        EvalClock {
            us: BoardClock::at(n, 0, next_roller, pending),
            opps,
        }
    }

    pub fn opp(&self, k: usize) -> BoardClock {
        self.opps[k]
    }

    /// The same moment seen by opponent `k`, whose opponent list is
    /// `[us, opps without k]` — the order `simulate_opp_phase1`, board-swapped
    /// training samples and distillation swaps build.
    pub fn reframe_to_opp(&self, k: usize, num_opps: usize) -> Self {
        let mut opps = [BoardClock::default(); MAX_PLAYERS - 1];
        opps[0] = self.us;
        let mut i = 1;
        for j in 0..num_opps {
            if j != k {
                opps[i] = self.opps[j];
                i += 1;
            }
        }
        EvalClock { us: self.opps[k], opps }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bc(rolls_in: u8, pending: bool) -> BoardClock {
        BoardClock { rolls_in, pending }
    }

    #[test]
    fn board_clock_counts_rolls_from_next_roller() {
        // 4 players, seat offset 2 rolls next: offsets 2,3,0,1 → rolls_in 0,1,2,3.
        assert_eq!(BoardClock::at(4, 2, 2, None), bc(0, false));
        assert_eq!(BoardClock::at(4, 3, 2, None), bc(1, false));
        assert_eq!(BoardClock::at(4, 0, 2, None), bc(2, false));
        assert_eq!(BoardClock::at(4, 1, 2, None), bc(3, false));
        assert_eq!(BoardClock::at(4, 1, 2, Some(1)), bc(3, true));
    }

    #[test]
    fn after_own_turn_and_turn_start() {
        // 1v1 after our turn: opponent rolls next, we roll in 1.
        let c = EvalClock::after_own_turn(2);
        assert_eq!((c.us, c.opp(0)), (bc(1, false), bc(0, false)));
        // 3 players, we (offset 0) roll next.
        let c = EvalClock::turn_start(3, 0);
        assert_eq!((c.us, c.opp(0), c.opp(1)), (bc(0, false), bc(1, false), bc(2, false)));
    }

    #[test]
    fn mid_turn_marks_active_pending_and_next_seat_rolling() {
        // 1v1 passive decision: opponent (offset 1) is active and pending; we roll next.
        let c = EvalClock::mid_turn(2, 1);
        assert_eq!((c.us, c.opp(0)), (bc(0, false), bc(1, true)));
        // 4 players, active at offset 2: offset 3 rolls next.
        let c = EvalClock::mid_turn(4, 2);
        assert_eq!(c.us, bc(1, false));
        assert_eq!(
            (c.opp(0), c.opp(1), c.opp(2)),
            (bc(2, false), bc(3, true), bc(0, false))
        );
    }

    #[test]
    fn reframe_moves_boards_with_their_clocks() {
        // Our frame, 4 players, we are mid-turn (pending), offset 1 rolls next.
        let ours = EvalClock::mid_turn(4, 0);
        // Opponent k = 1 sees [us, opp0, opp2] — the simulate_opp_phase1 order.
        let theirs = ours.reframe_to_opp(1, 3);
        assert_eq!(theirs.us, ours.opp(1));
        assert_eq!(theirs.opp(0), ours.us);
        assert_eq!(theirs.opp(1), ours.opp(0));
        assert_eq!(theirs.opp(2), ours.opp(2));
        assert!(theirs.opp(0).pending && !theirs.us.pending);
    }

    #[test]
    fn solo_clock_is_valid() {
        let c = EvalClock::turn_start(1, 0);
        assert_eq!(c.us, bc(0, false));
        assert_eq!(EvalClock::after_own_turn(1).us, bc(0, false));
    }

    #[test]
    #[should_panic(expected = "opponents")]
    fn from_boards_rejects_too_many_opponents() {
        EvalClock::from_boards(BoardClock::default(), &[BoardClock::default(); 4]);
    }
}
