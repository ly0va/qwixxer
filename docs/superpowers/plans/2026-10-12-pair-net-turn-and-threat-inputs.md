# Pair Net Turn-Position and Threat Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 7 per-board inputs (turn position, pending colour phase, P(forced strike), 4× per-row lock threat) to the pair value net, 45 → 59 inputs, with a lossless model migration and no behaviour change before retraining.

**Architecture:** A new `EvalClock` value (per-board `BoardClock { rolls_in, pending }`) is threaded through every `Bot` evaluation, so turn position is supplied by the caller that knows it; clocks travel with their boards, never derived from list order. The new board inputs are pure functions of (board, its clock, the other boards), computed from precomputed per-row tables. A migration tool widens the existing model's first layer with zero rows, so the migrated net is bit-identical and the whole change is proven inert by byte-identical benches before any training.

**Tech Stack:** Rust 2021, burn 0.20 (ndarray backend, `CompactRecorder`), rayon, clap.

**Spec:** `docs/superpowers/specs/2026-10-12-pair-net-turn-and-threat-inputs-design.md`

## Global Constraints

- Players supported for the new inputs: 2–4 (`MAX_PLAYERS = 4`); `n > 4` is a `debug_assert`, `rolls_in / 3` is clamped to ≤ 1.
- Feature layout (exact): board block 27 = old 0–19 unchanged, 20 `rolls_in/3`, 21 `pending`, 22 P(forced strike), 23–26 lock threat rows 0–3; full vector `[our 27 | paired 27 | pair-level 5]`; `BOARD_FEATURES = 27`, `PAIR_FEATURES = 59`, pair-level at 54–58.
- Old → new first-layer row map: 0–19 → 0–19, 20–39 → 27–46, 40–44 → 54–58; new rows zero.
- Every `Bot` evaluation takes an `EvalClock`; there is no default and no side channel.
- Production behaviour must be byte-identical (bench output, excluding the `Elapsed` line) after Task 3, and after Task 5 with the migrated model.
- Do not modify committed model artifacts except `pair_model/model.mpk` in Task 5 (the migrated, behaviour-identical model). Never touch `dqn_model/` or `champion.txt`.
- Format only touched files with `rustfmt --edition 2021 <files>`; never run `cargo fmt` on the whole crate (it reformats unrelated files).
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Non-turn-ordered opponent lists** (`simulate_opp_phase1`'s `[us, others…]`, board-swapped training samples, distillation swaps): each board must keep its own clock; covered by the reframe tests in Task 1, the pipeline audit in Task 3, and the swap-equivalence test in Task 4.
2. **Solo / empty opponent lists** (the pair bot's `default_opps` fallback; `SearchBot` never searches solo): evaluation must not index past the opponent array and must not panic; test in Task 4 (`evaluate_handles_empty_opponents` keeps passing with the new signature).
3. **Rows locked by another player** (`free = None` with any `total`, after lock propagation): lock threat for that row must be 0 and P(forced strike) must treat the row as offering nothing; tests in Task 2.
4. **4-player games** (3 opponents, the array maximum): the simulator, the rollout driver and feature building must all run at n = 4; the extended equivalence tests in Task 3 run n = 2, 3, 4.
5. **Game-ending distillation entries** (`ended == true`): emitting their features must use a valid clock and not panic; covered by the existing distillation emission tests run after Task 4.

---

### Task 0: Record pre-change baselines

**Files:** none in git (outputs under `target/ab/`, which is ignored).

- [ ] **Step 1: Build the current code**

Run: `cargo build --release`
Expected: finishes without errors.

- [ ] **Step 2: Record the benches every later guard compares against**

```bash
B=./target/release/qwixxer
mkdir -p target/ab
$B bench pair pair-search -n 2000 -s 3            | grep -v Elapsed > target/ab/inputs_base_1v1.txt
$B bench pair-search pair pair -n 600 -s 5        | grep -v Elapsed > target/ab/inputs_base_3p.txt
$B bench pair-search pair ga dqn -n 400 -s 9      | grep -v Elapsed > target/ab/inputs_base_4p.txt
$B bench ga pair -n 200000 -s 42                  > target/ab/inputs_base_static_speed.txt
QWIXXER_SEARCH= $B bench pair-search ga -n 2000 -s 11 > target/ab/inputs_base_search_speed.txt
```

Expected: five non-empty files. Note the games/s in the two `*_speed.txt` files (used in Task 5).

---

### Task 1: `EvalClock` — per-board turn position

**Files:**
- Create: `src/strategy/clock.rs`
- Modify: `src/strategy/mod.rs:1-4` (module declaration and re-export)

**Interfaces:**
- Produces: `qwixxer::strategy::clock::{MAX_PLAYERS, BoardClock, EvalClock}`, re-exported as `qwixxer::strategy::{BoardClock, EvalClock}`.
  - `BoardClock { pub rolls_in: u8, pub pending: bool }`, `BoardClock::at(n, offset, next_roller, pending: Option<usize>) -> BoardClock`
  - `EvalClock { pub us: BoardClock, pub opps: [BoardClock; 3] }`
  - `EvalClock::from_boards(us, opps: &[BoardClock])`, `turn_start(n, next_roller)`, `mid_turn(n, active)`, `after_own_turn(n)`, `opp(k)`, `reframe_to_opp(k, num_opps)`

- [ ] **Step 1: Write the failing tests** — create `src/strategy/clock.rs` with only the tests and empty stubs:

```rust
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
    pub fn at(n: usize, offset: usize, next_roller: usize, pending: Option<usize>) -> Self {
        todo!()
    }
}

impl EvalClock {
    pub fn from_boards(us: BoardClock, opps: &[BoardClock]) -> Self {
        todo!()
    }
    pub fn turn_start(n: usize, next_roller: usize) -> Self {
        todo!()
    }
    pub fn mid_turn(n: usize, active: usize) -> Self {
        todo!()
    }
    pub fn after_own_turn(n: usize) -> Self {
        todo!()
    }
    pub fn opp(&self, k: usize) -> BoardClock {
        todo!()
    }
    pub fn reframe_to_opp(&self, k: usize, num_opps: usize) -> Self {
        todo!()
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
        assert_eq!((c.opp(0), c.opp(1), c.opp(2)), (bc(2, false), bc(3, true), bc(0, false)));
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
```

Add to the top of `src/strategy/mod.rs` (after `pub mod bot_impl;`):

```rust
pub mod clock;
```

and next to the existing `pub(crate) use` line:

```rust
pub use clock::{BoardClock, EvalClock};
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib --release clock::`
Expected: 6 tests FAIL with `not yet implemented` (and the should_panic test fails on the wrong message).

- [ ] **Step 3: Implement** — replace the stub bodies:

```rust
impl BoardClock {
    /// Clock of the board at seat `offset` (0 = the evaluating player) when
    /// the player at offset `next_roller` rolls next and the player at offset
    /// `pending` (if any) has its colour phase outstanding.
    pub fn at(n: usize, offset: usize, next_roller: usize, pending: Option<usize>) -> Self {
        debug_assert!((1..=MAX_PLAYERS).contains(&n), "turn inputs support 1..={MAX_PLAYERS} players, got {n}");
        debug_assert!(offset < n && next_roller < n);
        BoardClock {
            rolls_in: ((offset + n - next_roller) % n) as u8,
            pending: pending == Some(offset),
        }
    }
}

impl EvalClock {
    pub fn from_boards(us: BoardClock, opps: &[BoardClock]) -> Self {
        assert!(opps.len() < MAX_PLAYERS, "at most {} opponents, got {}", MAX_PLAYERS - 1, opps.len());
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib --release clock::`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
rustfmt --edition 2021 src/strategy/clock.rs src/strategy/mod.rs
git add src/strategy/clock.rs src/strategy/mod.rs
git commit -m "feat(strategy): EvalClock — per-board turn position for evaluations

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Threat math — P(forced strike) and per-row lock threat

**Files:**
- Create: `src/threat.rs`
- Modify: `src/lib.rs` (add `pub mod threat;`)

**Interfaces:**
- Produces:
  - `qwixxer::threat::forced_strike_prob(s: &State) -> f32`
  - `qwixxer::threat::lock_threats(board: &State, others: &[&State]) -> [f32; 4]` — `others` = every other player's state (order irrelevant).

Both depend only on each row's `(direction, total, free)`, tabulated once in `OnceLock` tables, so they are cheap on search's hot path.

- [ ] **Step 1: Write the failing tests** — create `src/threat.rs`:

```rust
//! Dice-only threat quantities for the pair net's inputs (spec §1):
//! P(forced strike) and per-row P(locked by someone else within a round).
//! Every quantity depends only on each row's (direction, total, free), so
//! per-row tables are built once and lookups are cheap.

use crate::state::{Mark, State};

/// P(a full roll [W1, W2, R, Y, G, B] offers `s` no legal mark at all).
pub fn forced_strike_prob(s: &State) -> f32 {
    todo!()
}

/// Per row r: P(some player other than `board`'s gets an opportunity to lock
/// r during the next round — one roll by each player). `others` = all other
/// players' states, in any order. 0 for a locked row.
pub fn lock_threats(board: &State, others: &[&State]) -> [f32; 4] {
    todo!()
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
            assert!((forced_strike_prob(&s) - expected).abs() < 1e-6, "{s:?}: {} vs {expected}", forced_strike_prob(&s));
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
                    assert!((got[r] - want).abs() < 1e-5, "players {idxs:?} b {b} row {r}: {} vs {want}", got[r]);
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
```

Add `pub mod threat;` to `src/lib.rs` after `pub mod strategy;`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib --release threat::`
Expected: 4 tests FAIL with `not yet implemented`.

- [ ] **Step 3: Implement** — replace the two stubs and add the tables:

```rust
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
                            let none = (1..=6u8).filter(|&c| !s.can_mark(r, w1 + c) && !s.can_mark(r, w2 + c)).count();
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib --release threat::`
Expected: 4 passed (the brute-force test takes a few seconds in release).

- [ ] **Step 5: Commit**

```bash
rustfmt --edition 2021 src/threat.rs src/lib.rs
git add src/threat.rs src/lib.rs
git commit -m "feat(threat): exact P(forced strike) and per-row lock threat (table-based)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Thread `EvalClock` through every evaluation (no behaviour change)

**Files:**
- Modify: `src/strategy/mod.rs` (the `Bot` trait), `src/bot.rs:46-50` (GA impl), `src/dqn/mod.rs:421-…` (DQN impl), `src/dqn/pair.rs` (`Bot` + `WinProb` impls; the clock is accepted and ignored in this task), `src/strategy/bot_impl.rs`, `src/strategy/sim.rs`, `src/strategy/search.rs`, `src/dqn/pair_train.rs`, `examples/divergence.rs`, `examples/leaf_calibration.rs`
- Test: `src/strategy/bot_impl.rs` (new audit tests), `src/strategy/sim.rs` (extended equivalence tests)

**Interfaces:**
- Consumes: `EvalClock` (Task 1).
- Produces (exact new signatures):

```rust
pub trait Bot: std::fmt::Debug {
    fn evaluate(&self, our_state: &State, opp_states: &[State], clock: EvalClock) -> f32;
    fn evaluate_batch(&self, candidates: &[State], opp_states: &[State], clock: EvalClock) -> Vec<f32> {
        candidates.iter().map(|s| self.evaluate(s, opp_states, clock)).collect()
    }
    fn evaluate_batch_multi(&self, groups: &[(&[State], &[State], EvalClock)]) -> Vec<Vec<f32>> {
        groups.iter().map(|(c, o, k)| self.evaluate_batch(c, o, *k)).collect()
    }
}
// search.rs
pub trait WinProb: Bot {
    fn win_z_multi(&self, groups: &[(&State, &[State], EvalClock)]) -> Vec<f32>;
    fn win_prob_multi(&self, groups: &[(&State, &[State], EvalClock)]) -> Vec<f32> { … }
}
// bot_impl.rs
pub fn eval_decision(bot: &impl Bot, decision: Decision, eval_opps: &[State], clock: EvalClock) -> Option<Mark>;
pub(crate) fn passive_phase1_impl(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6], clock: EvalClock) -> Option<Mark>;
```

`active_phase1_choices`, `active_phase1_impl`, `active_phase2_impl`, `phase1_entry`, `phase2_entry`, `passive_entry` keep their signatures and compute clocks internally.

**The clock rule at every site** (n = `opp_states.len() + 1` in the evaluating player's turn-ordered frame):

| site | clock |
|---|---|
| plan / phase-2 candidate evaluation of the evaluating player (`active_phase1_impl`, `active_phase2_impl`, `SearchBot::active_phase1/2` values, `phase1_entry`'s phase-2 eval, `RecordingPair` active captures, lock ctx Ap1/Ap2) | `EvalClock::after_own_turn(n)` |
| passive decision from `Game::play` (`Strategy::passive_phase1`, `SearchBot::passive_phase1`, `RecordingPair::passive_phase1`, lock ctx Pp1) | `EvalClock::mid_turn(n, active_player + 1)` |
| opponent i's simulated passive decision in `simulate_opp_phase1` | `EvalClock::mid_turn(n, 0).reframe_to_opp(i, n - 1)` |
| passive decision of entry seat j in `passive_entry` / simulator (active at entry seat a; view = `SimGame::opp_view(&all, j)`) | `EvalClock::mid_turn(n, (a + n - j) % n)` |
| active seat's own plan / phase 2 in `passive_entry` / simulator (view from that seat) | `EvalClock::after_own_turn(n)` |
| search leaves (`search_pick`, player 0 view `opp_view(states, 0)`) | `EvalClock::turn_start(n, sim.active)` |

- [ ] **Step 1: Write the failing audit tests** — append to the `tests` module of `src/strategy/bot_impl.rs`:

```rust
    use crate::strategy::clock::{BoardClock, EvalClock};
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Bot that records every evaluation (candidates, opps, clock) and scores
    /// candidates by points (deterministic, no model needed).
    #[derive(Debug, Clone, Default)]
    struct RecordingBot {
        log: Rc<RefCell<Vec<(Vec<State>, Vec<State>, EvalClock)>>>,
    }

    impl Bot for RecordingBot {
        fn evaluate(&self, our: &State, opps: &[State], clock: EvalClock) -> f32 {
            self.evaluate_batch(&[*our], opps, clock)[0]
        }
        fn evaluate_batch(&self, cands: &[State], opps: &[State], clock: EvalClock) -> Vec<f32> {
            self.log.borrow_mut().push((cands.to_vec(), opps.to_vec(), clock));
            cands.iter().map(|s| s.count_points() as f32).collect()
        }
    }

    /// Distinct seat states (so a board's seat is identifiable by its state).
    fn seats(n: usize) -> Vec<State> {
        (0..n)
            .map(|i| State::from_parts(0, [(i as u8, Some(2 + i as u8)), (0, Some(2)), (0, Some(12)), (0, Some(12))]))
            .collect()
    }

    /// Truth: board at absolute seat `seat` when `next` rolls next and `pending` is outstanding.
    fn truth(n: usize, seat: usize, next: usize, pending: Option<usize>) -> BoardClock {
        BoardClock::at(n, seat, next, pending)
    }

    /// Seat of a (pre-move) board state among `seat_states`; the evaluating
    /// player's seat is the one missing from the evaluated opp list.
    /// (`State`'s `==` treats same-direction row swaps as equal; the `seats`
    /// boards differ only in red, so no two of them compare equal.)
    fn seat_of(seat_states: &[State], s: &State) -> usize {
        seat_states.iter().position(|x| x == s).expect("unknown board")
    }

    #[test]
    fn passive_decisions_get_mid_turn_clocks() {
        for n in 2..=4 {
            for active_abs in 1..n {
                // We sit at absolute seat 0; Game passes turn-ordered opps.
                let all = seats(n);
                let bot = RecordingBot::default();
                let mut b = bot.clone();
                let dice = [3, 4, 1, 1, 1, 1];
                crate::strategy::Strategy::passive_phase1(&mut b, &all[0], &all[1..], dice, active_abs - 1);
                assert!(!bot.log.borrow().is_empty(), "white sum 7 must leave a choice to evaluate");
                for (_, opps, clock) in bot.log.borrow().iter() {
                    let next = (active_abs + 1) % n;
                    assert_eq!(clock.us, truth(n, 0, next, Some(active_abs)), "n {n} active {active_abs}");
                    for (k, o) in opps.iter().enumerate() {
                        assert_eq!(clock.opp(k), truth(n, seat_of(&all, o), next, Some(active_abs)));
                    }
                }
            }
        }
    }

    #[test]
    fn simulated_opponent_passive_decisions_are_reframed() {
        for n in 3..=4 {
            let all = seats(n);
            let bot = RecordingBot::default();
            let dice = [3, 4, 2, 2, 2, 2];
            let _ = active_phase1_choices(&bot, &all[0], &all[1..], dice);
            let log = bot.log.borrow();
            assert!(!log.is_empty());
            // We (seat 0) are mid-turn: pending, seat 1 rolls next.
            for (cands, opps, clock) in log.iter() {
                // Evaluating seat = the one not in `opps`.
                let present: Vec<usize> = opps.iter().map(|o| seat_of(&all, o)).collect();
                let me = (0..n).find(|s| !present.contains(s)).unwrap();
                assert_ne!(me, 0, "only opponents' passive decisions are evaluated here");
                assert_eq!(clock.us, truth(n, me, 1, Some(0)), "cands {cands:?}");
                for (k, &s) in present.iter().enumerate() {
                    assert_eq!(clock.opp(k), truth(n, s, 1, Some(0)));
                }
            }
        }
    }

    #[test]
    fn own_turn_candidates_get_after_own_turn_clocks() {
        for n in 2..=4 {
            let all = seats(n);
            let bot = RecordingBot::default();
            let _ = active_phase2_impl(&bot, &all[0], &all[1..], [3, 4, 2, 2, 2, 2], true);
            assert!(!bot.log.borrow().is_empty(), "colour sums must leave a choice to evaluate");
            for (_, opps, clock) in bot.log.borrow().iter() {
                assert_eq!(clock.us, truth(n, 0, 1 % n, None));
                for (k, o) in opps.iter().enumerate() {
                    assert_eq!(clock.opp(k), truth(n, seat_of(&all, o), 1 % n, None));
                }
            }
        }
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test --lib --release bot_impl::`
Expected: compile errors (`evaluate_batch` takes 2 arguments; `EvalClock` unused) — the API does not exist yet.

- [ ] **Step 3: Change the `Bot` trait and its implementors**

In `src/strategy/mod.rs` replace the `Bot` trait with the version in **Interfaces** above (import `use clock::EvalClock;` is covered by the `pub use`).

`src/bot.rs` (GA):

```rust
impl Bot for DNA {
    fn evaluate(&self, our_state: &State, _opp_states: &[State], _clock: crate::strategy::EvalClock) -> f32 {
        self.instinct(our_state) as f32
    }
}
```

`src/dqn/mod.rs` (old DQN): add `_clock: crate::strategy::EvalClock` as the last parameter of `evaluate` and `evaluate_batch`, and pass it through where `evaluate` calls `evaluate_batch`.

`src/dqn/pair.rs`: change `Bot for PairStrategy` to

```rust
    fn evaluate(&self, our_state: &State, opp_states: &[State], clock: EvalClock) -> f32 {
        self.evaluate_batch(&[*our_state], opp_states, clock)[0]
    }

    fn evaluate_batch(&self, candidates: &[State], opp_states: &[State], clock: EvalClock) -> Vec<f32> {
        self.evaluate_batch_multi(&[(candidates, opp_states, clock)]).pop().unwrap()
    }

    fn evaluate_batch_multi(&self, groups: &[(&[State], &[State], EvalClock)]) -> Vec<Vec<f32>> {
```

and inside it iterate `for (candidates, opp_states, _clock) in groups` and `groups.iter().zip(leaders)` destructuring `((candidates, _, _), leader_points)`. Change `WinProb for PairStrategy::win_z_multi` to take `groups: &[(&State, &[State], EvalClock)]` and iterate `for (our, opps, _clock) in groups`. Add `use crate::strategy::EvalClock;` at the top. In `src/strategy/search.rs` change the `WinProb` trait to the signatures in **Interfaces** (`win_prob_multi` default body unchanged).

- [ ] **Step 4: Thread clocks through `bot_impl.rs`**

```rust
pub fn eval_decision(bot: &impl Bot, decision: Decision, eval_opps: &[State], clock: EvalClock) -> Option<Mark> {
    match decision {
        Decision::Forced(m) => m,
        Decision::Choices(cands) => {
            let states: Vec<State> = cands.iter().map(|(_, s)| *s).collect();
            let values = bot.evaluate_batch(&states, eval_opps, clock);
            cands[argmax(&values)].0
        }
    }
}

pub(crate) fn passive_phase1_impl(
    bot: &impl Bot,
    state: &State,
    opp_states: &[State],
    dice: [u8; 6],
    clock: EvalClock,
) -> Option<Mark> {
    eval_decision(bot, passive_phase1_choices(state, opp_states, dice), opp_states, clock)
}
```

`active_phase2_impl`: pass `EvalClock::after_own_turn(opp_states.len() + 1)` as the clock to `eval_decision`.

`simulate_opp_phase1`:

```rust
fn simulate_opp_phase1(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Vec<State> {
    // We are the active player, mid-turn; each opponent sees [us, others...].
    let ours = EvalClock::mid_turn(opp_states.len() + 1, 0);
    opp_states
        .iter()
        .enumerate()
        .map(|(i, opp)| {
            let their_opps: Vec<State> = std::iter::once(*state)
                .chain(opp_states.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, s)| *s))
                .collect();
            let clock = ours.reframe_to_opp(i, opp_states.len());
            match passive_phase1_impl(bot, opp, &their_opps, dice, clock) {
                Some(m) => {
                    let mut s = *opp;
                    s.apply_mark(m);
                    s
                }
                None => *opp,
            }
        })
        .collect()
}
```

`active_phase1_impl`: `eval_decision(bot, decision, &sim_opp, EvalClock::after_own_turn(opp_states.len() + 1))`.

Blanket `Strategy` impl `passive_phase1`: rename `_active_player` to `active_player` and call `passive_phase1_impl(self, state, opp_states, dice, EvalClock::mid_turn(opp_states.len() + 1, active_player + 1))`.

Add `use super::clock::EvalClock;` at the top of `bot_impl.rs`.

- [ ] **Step 5: Thread clocks through `sim.rs`**

In `play_sim_turn`, phase 1, compute per player `j`:

```rust
        let m = if j == sim.active {
            match fidelity {
                Fidelity::Full => active_phase1_impl(bot, &snapshot[j], &view, dice),
                Fidelity::Lite => eval_decision(
                    bot,
                    phase1_plan_choices(&snapshot[j], &view, dice),
                    &view,
                    EvalClock::after_own_turn(n),
                ),
            }
        } else {
            eval_decision(
                bot,
                passive_phase1_choices(&snapshot[j], &view, dice),
                &view,
                EvalClock::mid_turn(n, (sim.active + n - j) % n),
            )
        };
```

and in phase 2 pass `EvalClock::after_own_turn(n)` to `eval_decision`.

In `BatchedRollouts`: add `clock: EvalClock` to `PendingChoice`; in phase 1 set `clock: if j == sim.active { EvalClock::after_own_turn(n) } else { EvalClock::mid_turn(n, (sim.active + n - j) % n) }`; in phase 2 `clock: EvalClock::after_own_turn(sim.n())`; in `resolve` build `groups: Vec<(&[State], &[State], EvalClock)>` as `(p.states.as_slice(), p.view.as_slice(), p.clock)`. Add `use super::clock::EvalClock;`.

- [ ] **Step 6: Thread clocks through `search.rs`**

- `phase1_entry`: `match eval_decision(bot, d, &view, EvalClock::after_own_turn(all.len()))`.
- `passive_entry`, other passive players `j`: `passive_phase1_impl(bot, &all[j], &view, dice, EvalClock::mid_turn(n, (active_idx + n - j) % n))`; active player's phase 1 and phase 2: `eval_decision(bot, d, &view, EvalClock::after_own_turn(n))`.
- `search_pick` leaves: build `groups: Vec<(&State, &[State], EvalClock)>` with `EvalClock::turn_start(n, driver.sims[i].active)` per survivor; pass to `win_prob_multi` / `win_z_multi`.
- `adjudicate_lock`: add parameter `clock: EvalClock` and use it in `self.bot.evaluate_batch(&states, eval_opps, clock)`; callers pass `EvalClock::after_own_turn(n)` (ap1, ap2) and `EvalClock::mid_turn(n, active_player + 1)` (pp1).
- `SearchBot::active_phase1` values and `active_phase2` values: `EvalClock::after_own_turn(opp_states.len() + 1)`.
- `SearchBot::passive_phase1`: `let clock = EvalClock::mid_turn(opp_states.len() + 1, active_player + 1);` used for `passive_phase1_impl`, `eval_decision` and `evaluate_batch`.
- Search tests calling `passive_phase1_impl(&bot.bot, &state, &opps, dice)` pass `EvalClock::mid_turn(opps.len() + 1, 1)`; `win_prob_multi`/`win_z_multi` callers in tests pass `EvalClock::turn_start(2, 0)`.
- Add `use super::clock::EvalClock;`.

- [ ] **Step 7: Thread clocks through `pair_train.rs`**

- `RecordingPair::active_phase1` capture: `bot.evaluate_batch(&states, &sim_opp, EvalClock::after_own_turn(opp_states.len() + 1))` (both sites) and the ε-decline `eval_decision(bot, Decision::Choices(rf), &sim_opp, EvalClock::after_own_turn(opp_states.len() + 1))`.
- `RecordingPair::active_phase2`: same clock for both `evaluate_batch` sites and the decline `eval_decision`.
- `RecordingPair::passive_phase1`: `let clock = EvalClock::mid_turn(opp_states.len() + 1, active_player + 1);` for `passive_phase1_impl(self.policy.bot(), state, opp_states, dice, clock)`, both `evaluate_batch` sites and the decline `eval_decision`.
- `build_distill_samples`, lock ctx re-evaluation: `let clock = match lctx { LockCtx::Pp1 => EvalClock::mid_turn(opps.len() + 1, *active_player + 1), _ => EvalClock::after_own_turn(opps.len() + 1) };` and `boot.evaluate_batch(&states, eval_opps, clock)`.
- Test `epsilon_lock_declines_only_at_firings`: `eval_decision(&boot, rule_free, &opps, EvalClock::mid_turn(2, 1))`.
- Add `use crate::strategy::EvalClock;`.

- [ ] **Step 8: Update the examples**

`examples/leaf_calibration.rs` `play_game`: groups become `[(&st[0], &opp0[..], EvalClock::turn_start(2, sim.active)), (&st[1], &opp1[..], EvalClock::turn_start(2, (sim.active + 1) % 2))]` (player 1's frame: next roller offset = `(sim.active - 1) mod 2`).

`examples/divergence.rs` (mirrors production; apply the same rule table): `build_lock_cands` gains `clock: EvalClock` and uses it in both `evaluate_batch` calls; its callers pass `after_own_turn(n)` for `"ap1"`/`"ap2"` and `mid_turn(n, 1)` for `"pp1"` (the harness is 1v1 with the opponent active). `ShadowPair`, `LockShadowPair` and `VariantPair`: active phase 1/2 evaluations and `eval_decision`s use `after_own_turn(opp_states.len() + 1)`; passive ones use `mid_turn(opp_states.len() + 1, active_player + 1)` (rename `_active_player`). `phase1_entries`/`passive_entries` mirrors follow the `search.rs` rules of Step 6; `rollout_scores` leaf groups use `EvalClock::turn_start(n, sim.active)`; `relabel_event`/`adjudicate_event` re-evaluations use `after_own_turn(1 + opps.len())` for ap1/ap2 and `mid_turn(1 + opps.len(), 1)` for pp1. Add `use qwixxer::strategy::EvalClock;`.

- [ ] **Step 9: Extend the simulator equivalence tests to compare clocks**

In `src/strategy/sim.rs` tests, add a recording wrapper and assert identical evaluations (states and clocks) between `Game::play` (Full) and `play_sim_turn` (Full) for n = 2, 3, 4, and between the lockstep driver and sequential Lite sims. This is the only test that catches wrong clocks in the rollout driver: the byte-identity guards cannot, because until training the clocks carry zero weight.

```rust
    #[derive(Debug)]
    struct Rec {
        inner: PairStrategy,
        log: std::rc::Rc<std::cell::RefCell<Vec<(Vec<State>, Vec<State>, EvalClock)>>>,
    }
    impl Bot for Rec {
        fn evaluate(&self, o: &State, opps: &[State], c: EvalClock) -> f32 {
            self.evaluate_batch(&[*o], opps, c)[0]
        }
        fn evaluate_batch(&self, cands: &[State], opps: &[State], c: EvalClock) -> Vec<f32> {
            self.log.borrow_mut().push((cands.to_vec(), opps.to_vec(), c));
            self.inner.evaluate_batch(cands, opps, c)
        }
    }
```

Build each `Rec` with `PairStrategy::from_shared(model.clone(), device)` and a shared `log` (`PairStrategy` is not `Clone`). In `sim_turn_loop_matches_game_play`, give every `Game` player a `Rec` sharing one log, run the sim with a `Rec` on a second log, and add `assert_eq!(*game_log.borrow(), *sim_log.borrow(), "n={n} seed={seed}: evaluation clocks diverged")` — both loops call players in the same order, so the sequences must match exactly. In `lockstep_driver_matches_sequential_lite_sims`, wrap the bot in a `Rec` for both the sequential sims and the driver (the default `evaluate_batch_multi` logs each group through `evaluate_batch`); the driver batches calls across sims in a different order, so compare the logs as multisets:

```rust
        let canon = |log: &Vec<(Vec<State>, Vec<State>, EvalClock)>| {
            let mut v: Vec<String> = log.iter().map(|e| format!("{e:?}")).collect();
            v.sort();
            v
        };
        assert_eq!(canon(&seq_log.borrow()), canon(&drv_log.borrow()), "lockstep driver clocks diverged");
```

- [ ] **Step 10: Run all tests**

Run: `cargo test --lib --release`
Expected: all tests pass, including the 3 new audit tests and the extended sim tests.

Run: `cargo build --release --examples && (cd web && cargo check)`
Expected: no errors.

- [ ] **Step 11: Byte-identical guard** (the pair bot ignores clocks in this task, so output must not change)

```bash
cargo build --release
B=./target/release/qwixxer
$B bench pair pair-search -n 2000 -s 3       | grep -v Elapsed | diff - target/ab/inputs_base_1v1.txt
$B bench pair-search pair pair -n 600 -s 5   | grep -v Elapsed | diff - target/ab/inputs_base_3p.txt
$B bench pair-search pair ga dqn -n 400 -s 9 | grep -v Elapsed | diff - target/ab/inputs_base_4p.txt
```

Expected: all three `diff`s print nothing.

- [ ] **Step 12: Commit**

```bash
rustfmt --edition 2021 src/strategy/mod.rs src/bot.rs src/dqn/mod.rs src/dqn/pair.rs src/strategy/bot_impl.rs src/strategy/sim.rs src/strategy/search.rs src/dqn/pair_train.rs examples/leaf_calibration.rs
git add src/strategy/mod.rs src/bot.rs src/dqn/mod.rs src/dqn/pair.rs src/strategy/bot_impl.rs src/strategy/sim.rs src/strategy/search.rs src/dqn/pair_train.rs examples/divergence.rs examples/leaf_calibration.rs
git commit -m "refactor(bot): thread EvalClock through every evaluation (behaviour unchanged)

Bot::evaluate*, WinProb and the decision pipelines take an EvalClock; each
call site supplies the turn position it knows (mid-turn for passive
decisions, after-own-turn for plans and phase 2, reframed for simulated
opponents, turn-start for search leaves). Audited against seat geometry for
2-4 players and against Game::play; bench output byte-identical.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(`examples/divergence.rs` is not fmt-clean upstream; do not run rustfmt on it.)

---

### Task 4: The 59-input feature layout

**Files:**
- Modify: `src/dqn/pair.rs` (constants, `board_features`, `pair_features`, `PairModelConfig`, load guard, evaluation call sites, tests)
- Modify: `src/dqn/pair_train.rs` (`Snapshot`, recording, `build_pair_samples`, distillation features, `permute_colors`, model loading, tests)

**Interfaces:**
- Consumes: `forced_strike_prob`, `lock_threats` (Task 2); `BoardClock`, `EvalClock` (Task 1).
- Produces:

```rust
pub const BOARD_FEATURES: usize = 27;
pub const PAIR_FEATURES: usize = 59;
pub const PAIR_LEVEL: usize = 2 * BOARD_FEATURES; // 54
pub fn board_features(state: &State, clock: BoardClock, others: &[&State]) -> [f32; BOARD_FEATURES];
/// `paired = opps[paired_idx]`; `clock.opps` aligned with `opps`.
pub fn pair_features(our: &State, paired_idx: usize, opps: &[State], clock: EvalClock) -> [f32; PAIR_FEATURES];
#[derive(Config)] pub struct PairModelConfig { hidden1 = 128, hidden2 = 64, input = PAIR_FEATURES /* 59 */ }
pub fn check_input_dims(model: &PairModel<MyBackend>, dir: &str); // panics naming `pair-migrate`
type Snapshot = (State, Vec<State>, EvalClock); // pair_train.rs
```

- [ ] **Step 1: Write the failing tests** — in `src/dqn/pair.rs` tests:

```rust
    use crate::strategy::{BoardClock, EvalClock};

    #[test]
    fn new_board_inputs_are_appended() {
        let s = State::from_parts(1, [(5, Some(11)), (0, Some(2)), (0, Some(12)), (0, Some(12))]);
        let other = State::from_parts(0, [(6, Some(12)), (0, Some(2)), (0, Some(12)), (0, Some(12))]);
        let clock = BoardClock { rolls_in: 2, pending: true };
        let f = board_features(&s, clock, &[&other]);
        assert_eq!(f[20], 2.0 / 3.0);
        assert_eq!(f[21], 1.0);
        assert_eq!(f[22], crate::threat::forced_strike_prob(&s));
        assert_eq!(f[23..27], crate::threat::lock_threats(&s, &[&other]));
        assert!(f[23] > 0.0, "other player can lock red");
    }

    #[test]
    fn pair_features_are_invariant_to_opponent_order() {
        let a = State::from_parts(0, [(3, Some(7)), (0, Some(2)), (5, Some(4)), (0, Some(12))]);
        let b = State::from_parts(1, [(5, Some(12)), (2, Some(5)), (0, Some(12)), (1, Some(10))]);
        let c = State::from_parts(0, [(1, Some(4)), (6, Some(11)), (2, Some(9)), (0, Some(12))]);
        let clock = EvalClock::mid_turn(3, 2);
        let f1 = pair_features(&a, 0, &[b, c], clock);
        let swapped = EvalClock::from_boards(clock.us, &[clock.opp(1), clock.opp(0)]);
        let f2 = pair_features(&a, 1, &[c, b], swapped);
        assert_eq!(f1, f2);
    }

    #[test]
    #[should_panic(expected = "pair-migrate")]
    fn loading_an_old_45_input_model_names_the_migration() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let old = PairModelConfig::new().with_input(45).init::<MyBackend>(&device);
        check_input_dims(&old, "some_dir");
    }
```

In `src/dqn/pair_train.rs` tests, add the colour-symmetry and swap tests:

```rust
    /// Colour symmetry: swap rows of every board (ry: 0↔1, gb: 2↔3, pairs:
    /// 0↔2 and 1↔3 with free mirrored n → 14 − n); features must equal the
    /// permuted features of the original.
    fn permute_state(s: &State, ry: bool, gb: bool, pairs: bool) -> State {
        let mut parts: [(u8, Option<u8>); 4] =
            std::array::from_fn(|r| (s.row_totals()[r], s.row_free_values()[r]));
        if ry {
            parts.swap(0, 1);
        }
        if gb {
            parts.swap(2, 3);
        }
        if pairs {
            let mirror = |(t, f): (u8, Option<u8>)| (t, f.map(|x| 14 - x));
            parts = [mirror(parts[2]), mirror(parts[3]), mirror(parts[0]), mirror(parts[1])];
        }
        State::from_parts(s.strikes, parts)
    }

    #[test]
    fn colour_permutation_matches_the_game_symmetry() {
        let a = State::from_parts(1, [(5, Some(11)), (2, Some(6)), (4, Some(5)), (0, Some(12))]);
        let b = State::from_parts(0, [(6, Some(12)), (0, Some(2)), (5, Some(3)), (3, Some(8))]);
        let c = State::from_parts(2, [(1, Some(4)), (5, Some(12)), (0, Some(12)), (6, Some(2))]);
        let clock = EvalClock::mid_turn(3, 1);
        for (ry, gb, pairs) in [(true, false, false), (false, true, false), (false, false, true), (true, true, true)] {
            let mut expected = pair_features(&a, 0, &[b, c], clock);
            permute_colors(&mut expected, ry, gb, pairs);
            let p = |s: &State| permute_state(s, ry, gb, pairs);
            let got = pair_features(&p(&a), 0, &[p(&b), p(&c)], clock);
            for i in 0..PAIR_FEATURES {
                assert!((got[i] - expected[i]).abs() < 1e-6, "swap {ry} {gb} {pairs}, input {i}: {} vs {}", got[i], expected[i]);
            }
        }
    }

    #[test]
    fn swapped_samples_match_the_opponents_own_view() {
        // 3 players, we (seat 0) at a passive decision with seat 2 active.
        let us = State::from_parts(0, [(3, Some(7)), (0, Some(2)), (5, Some(4)), (0, Some(12))]);
        let o1 = State::from_parts(1, [(5, Some(12)), (2, Some(5)), (0, Some(12)), (1, Some(10))]);
        let o2 = State::from_parts(0, [(1, Some(4)), (6, Some(11)), (2, Some(9)), (0, Some(12))]);
        let clock = EvalClock::mid_turn(3, 2);
        // Swapped sample for k = 1 (o2): [us, o1] with the reframed clock.
        let swapped = pair_features(&o2, 0, &[us, o1], clock.reframe_to_opp(1, 2));
        // o2's own turn-ordered view: seat 0 (us) at offset 1, seat 1 (o1) at
        // offset 2; o2 itself is the active, pending player.
        let own = pair_features(&o2, 0, &[us, o1], EvalClock::mid_turn(3, 0));
        assert_eq!(swapped, own);
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test --lib --release pair`
Expected: compile errors (`board_features` takes 1 argument, `with_input` does not exist, `check_input_dims` missing).

- [ ] **Step 3: Implement the layout in `src/dqn/pair.rs`**

```rust
use crate::strategy::{BoardClock, EvalClock};

/// Per-board feature block size.
pub const BOARD_FEATURES: usize = 27;
/// Full pair input: two board blocks + 5 pair-level features.
pub const PAIR_FEATURES: usize = 59;
/// Index of the first pair-level feature.
pub const PAIR_LEVEL: usize = 2 * BOARD_FEATURES;

pub fn board_features(state: &State, clock: BoardClock, others: &[&State]) -> [f32; BOARD_FEATURES] {
    // ... existing body computing f[0..20] unchanged ...
    f[20] = (clock.rolls_in as f32 / 3.0).min(1.0);
    f[21] = if clock.pending { 1.0 } else { 0.0 };
    f[22] = crate::threat::forced_strike_prob(state);
    f[23..27].copy_from_slice(&crate::threat::lock_threats(state, others));
    f
}

pub fn pair_features(our: &State, paired_idx: usize, opps: &[State], clock: EvalClock) -> [f32; PAIR_FEATURES] {
    assert!(opps.len() < crate::strategy::clock::MAX_PLAYERS, "pair features support up to 4 players");
    let paired = &opps[paired_idx];
    let mut ours_others: [&State; 3] = [our; 3];
    for (i, o) in opps.iter().enumerate() {
        ours_others[i] = o;
    }
    let mut theirs_others: [&State; 3] = [our; 3];
    let mut m = 1;
    for (j, o) in opps.iter().enumerate() {
        if j != paired_idx {
            theirs_others[m] = o;
            m += 1;
        }
    }
    let mut f = [0.0f32; PAIR_FEATURES];
    f[..BOARD_FEATURES].copy_from_slice(&board_features(our, clock.us, &ours_others[..opps.len()]));
    f[BOARD_FEATURES..PAIR_LEVEL].copy_from_slice(&board_features(paired, clock.opp(paired_idx), &theirs_others[..m]));

    let cdiff = (our.count_points() - paired.count_points()) as f32;
    f[PAIR_LEVEL] = (cdiff / 100.0).clamp(-1.0, 1.0);
    f[PAIR_LEVEL + 1] = opps.len() as f32 / 4.0;
    f[PAIR_LEVEL + 2] = opps.iter().map(total_progress).fold(0.0, f32::max);
    f[PAIR_LEVEL + 3] = opps.iter().map(|s| s.strikes).max().unwrap_or(0) as f32 / 3.0;
    f[PAIR_LEVEL + 4] = opps.iter().map(lockable_rows).sum::<u8>() as f32 / 8.0;
    f
}
```

`PairModelConfig`: add

```rust
    #[config(default = 59)]
    pub input: usize,
```

and use `LinearConfig::new(self.input, self.hidden1)` in `init`. (The literal `59` must equal `PAIR_FEATURES`; add `const _: () = assert!(PAIR_FEATURES == 59);` next to the constants.)

Load guard:

```rust
/// Panic with the migration command if `model` was built for another input size.
pub fn check_input_dims(model: &PairModel<MyBackend>, dir: &str) {
    let d_in = model.layer1.weight.dims()[0];
    if d_in != PAIR_FEATURES {
        panic!(
            "pair model in {dir} has {d_in} inputs but this build expects {PAIR_FEATURES}; \
             convert it with `qwixxer pair-migrate --from {dir} --to <new dir>`"
        );
    }
}
```

Call it in `PairStrategy::load` right after `load_file(...)` and in `pair_train.rs` after each `load_file` of the training model (the warm-start load at the start of `self_play_train` and the checkpoint reload near the end).

Evaluation call sites in `pair.rs`:
- `evaluate_batch_multi`: find the leader by index (same tie-breaking as before — `max_by_key` returns the last maximum in both forms): `let leader = (0..opps.len()).max_by_key(|&i| opps[i].count_points()).unwrap();` then `feats.push(pair_features(c, leader, opps, *clock))` and `leaders.push(opps[leader].count_points())`. For the empty-opponent fallback, `opps = &default_opps` and the clock's `opps[0]` default is used.
- `win_z_multi`: same leader-by-index change; `feats.push(pair_features(our, leader, opps, *clock))`.
- Update the existing tests (`pair_features_fresh_1v1_golden`, `pair_features_swap_relationship`, `manual_forward_matches_burn_within_tolerance`, `wprob_*` tests) to the new signatures: `board_features(&s, BoardClock::default(), &[])` and `pair_features(&a, 0, &[b], EvalClock::after_own_turn(2))`; in the golden test keep the existing expected values for indices 0–19 of each block (now at 0–19 and 27–46) and pair-level values (now at 54–58), and assert indices 20–26 / 47–53 against `BoardClock`/`threat` values computed in the test.

- [ ] **Step 4: Training-side changes in `src/dqn/pair_train.rs`**

- `type Snapshot = (State, Vec<State>, EvalClock);`
- `RecordingPair::active_phase2`: every `self.recorded.borrow_mut().push((x, opp_states.to_vec()))` becomes `push((x, opp_states.to_vec(), EvalClock::after_own_turn(opp_states.len() + 1)))`.
- `RecordingPair::passive_phase1`: `push((post, opp_states.to_vec(), clock))` with the Task 3 `clock`.
- `build_pair_samples`:

```rust
    for k in 0..num_opps {
        let final_diff = our_final - opp_finals[k];
        let feats: Vec<[f32; PAIR_FEATURES]> = snapshots
            .iter()
            .map(|(our, opps, clock)| pair_features(our, k, opps, *clock))
            .collect();
        // ... cdiffs, mus, g unchanged ...
        for (t, (our, opps, clock)) in snapshots.iter().enumerate() {
            // forward sample unchanged (features: feats[t])
            let mut swapped_opps: Vec<State> = Vec::with_capacity(num_opps);
            swapped_opps.push(*our);
            swapped_opps.extend(opps.iter().enumerate().filter(|(j, _)| *j != k).map(|(_, s)| *s));
            samples.push(PairSample {
                features: pair_features(&opps[k], 0, &swapped_opps, clock.reframe_to_opp(k, num_opps)),
                value: -value,
                final_diff: -fdiff,
            });
        }
    }
```

- Distillation emission (`build_distill_samples`): entries are turn-complete with the rollout's `first_active` next; per entry `let clock = EvalClock::turn_start(entry.0.len(), first_active);` then `let feats = pair_features(&our, k, &e_opps, clock);` and `let swap_feats = pair_features(&e_opps[k], 0, &swapped_opps, clock.reframe_to_opp(k, num_opps));`.
- `permute_colors`: the per-row groups become `[0, 4, 8, 12, 23]` (ry: `base, base + 1`), `[2, 6, 10, 14, 25]` (gb), and `[0, 4, 8, 12, 23]` with `+2`/`+3` (pairs). Update its doc comment: lock threat at 23–26; pair-level 54–58 untouched.
- Update the existing `permute_colors_*` tests' index sets to include the new group, and every test that builds `Snapshot`s or calls `pair_features` to the new signatures.

- [ ] **Step 5: Run all tests**

Run: `cargo test --lib --release`
Expected: all pass, including the 5 new tests.

Run: `cargo build --release --examples && (cd web && cargo check)`
Expected: no errors (the examples only use the pair bot through `Bot`/`WinProb`).

Note: from this commit until Task 5, `pair_model/model.mpk` (45 inputs) does not load; benches with pair bots fail with the `pair-migrate` message. That is expected.

- [ ] **Step 6: Commit**

```bash
rustfmt --edition 2021 src/dqn/pair.rs src/dqn/pair_train.rs
git add src/dqn/pair.rs src/dqn/pair_train.rs
git commit -m "feat(pair): 59-input layout — turn position, pending colour phase, forced strike, lock threat

Board block 20 -> 27 (new inputs appended), pair-level moved to 54-58.
Training snapshots store the EvalClock used at the decision; board-swapped
and distillation samples reframe clocks instead of recomputing from list
order. Colour augmentation covers the per-row lock threat. Loading a
45-input model panics with the pair-migrate command.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `pair-migrate` and the migrated production model

**Files:**
- Modify: `src/dqn/pair.rs` (`migrate_input_index`, `migrate_model`, tests), `src/main.rs` (`PairMigrate` subcommand)
- Modify (artifact): `pair_model/model.mpk`

**Interfaces:**
- Consumes: `PairModelConfig::with_input`, `check_input_dims` (Task 4); `extract_linear` (existing, private in `pair.rs`).
- Produces: `pub fn migrate_input_index(old: usize) -> usize`, `pub fn migrate_model(old: &PairModel<MyBackend>, device: &NdArrayDevice) -> PairModel<MyBackend>`, `pub fn migrate_dir(from: &str, to: &str)`; CLI `qwixxer pair-migrate --from <dir> --to <dir>`.

- [ ] **Step 1: Write the failing tests** — in `src/dqn/pair.rs` tests:

```rust
    #[test]
    fn migration_index_map_follows_the_new_layout() {
        assert_eq!(migrate_input_index(0), 0);
        assert_eq!(migrate_input_index(19), 19);
        assert_eq!(migrate_input_index(20), 27);
        assert_eq!(migrate_input_index(39), 46);
        assert_eq!(migrate_input_index(40), 54);
        assert_eq!(migrate_input_index(44), 58);
    }

    #[test]
    fn migrated_model_ignores_new_inputs_and_keeps_old_outputs() {
        use rand::{Rng, SeedableRng};
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let old = PairModelConfig::new().with_input(45).init::<MyBackend>(&device);
        let new = migrate_model(&old, &device);
        check_input_dims(&new, "test");
        let mut rng = rand::rngs::SmallRng::seed_from_u64(5);
        for _ in 0..20 {
            let x45: Vec<f32> = (0..45).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let mut x59 = [0.0f32; PAIR_FEATURES];
            for (i, &v) in x45.iter().enumerate() {
                x59[migrate_input_index(i)] = v;
            }
            // Arbitrary values in the new slots must not matter.
            for i in (20..27).chain(47..54) {
                x59[i] = rng.gen_range(0.0..1.0);
            }
            let a = old.forward(Tensor::<MyBackend, 1>::from_floats(x45.as_slice(), &device).reshape([1, 45]));
            let b = new.forward(Tensor::<MyBackend, 1>::from_floats(x59.as_slice(), &device).reshape([1, PAIR_FEATURES]));
            let (a, b) = (a.into_data().to_vec::<f32>().unwrap(), b.into_data().to_vec::<f32>().unwrap());
            for (u, v) in a.iter().zip(&b) {
                assert!((u - v).abs() < 1e-5, "{a:?} vs {b:?}");
            }
        }
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test --lib --release migrat`
Expected: compile errors (`migrate_input_index`, `migrate_model` missing).

- [ ] **Step 3: Implement** in `src/dqn/pair.rs`:

```rust
/// Position of old (45-input layout) input `old` in the 59-input layout.
pub fn migrate_input_index(old: usize) -> usize {
    match old {
        0..=19 => old,
        20..=39 => old + 7,
        40..=44 => old + 14,
        _ => panic!("old pair layout has 45 inputs, got index {old}"),
    }
}

/// Widen a 45-input model to the 59-input layout: old first-layer rows move to
/// their new positions, the 14 new rows are zero, everything else is copied.
/// The new inputs therefore contribute exactly 0.0 — outputs are unchanged.
pub fn migrate_model(
    old: &PairModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
) -> PairModel<MyBackend> {
    let (w_old, _, d_in, d_out) = extract_linear(&old.layer1);
    assert_eq!(d_in, 45, "migrate_model expects a 45-input model");
    let mut w_new = vec![0.0f32; PAIR_FEATURES * d_out];
    for i in 0..45 {
        let j = migrate_input_index(i);
        w_new[j * d_out..(j + 1) * d_out].copy_from_slice(&w_old[i * d_out..(i + 1) * d_out]);
    }
    let mut new = PairModelConfig::new().init::<MyBackend>(device);
    new.layer1.weight = burn::module::Param::from_tensor(
        Tensor::<MyBackend, 1>::from_floats(w_new.as_slice(), device).reshape([PAIR_FEATURES, d_out]),
    );
    new.layer1.bias = old.layer1.bias.clone();
    new.layer2 = old.layer2.clone();
    new.output_mean = old.output_mean.clone();
    new.output_log_var = old.output_log_var.clone();
    new
}

/// Load the 45-input model in `from`, migrate it, save it to `to`.
pub fn migrate_dir(from: &str, to: &str) {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let old = PairModelConfig::new()
        .with_input(45)
        .init::<MyBackend>(&device)
        .load_file(format!("{from}/model"), &CompactRecorder::new(), &device)
        .expect("failed to load the 45-input pair model");
    let new = migrate_model(&old, &device);
    std::fs::create_dir_all(to).expect("cannot create the target directory");
    new.save_file(format!("{to}/model"), &CompactRecorder::new())
        .expect("failed to save the migrated model");
}
```

(`extract_linear` returns input-major weights `w[i * d_out..][..d_out]`, the layout burn stores; if the burn version stores `[d_out, d_in]`, the unit test above fails and the copy must transpose — do not skip that test.)

`src/main.rs`: add the subcommand

```rust
    /// Convert a 45-input pair model to the 59-input layout (lossless)
    PairMigrate {
        #[arg(long)]
        from: String,
        #[arg(long)]
        to: String,
    },
```

and dispatch `Some(Commands::PairMigrate { from, to }) => dqn::pair::migrate_dir(&from, &to),`.

- [ ] **Step 4: Run the tests**

Run: `cargo test --lib --release`
Expected: all pass.

- [ ] **Step 5: Migrate the production model and prove byte-identical play**

```bash
cargo build --release
B=./target/release/qwixxer
$B pair-migrate --from pair_model --to target/pair_model_mig
cp target/pair_model_mig/model.mpk pair_model/model.mpk
$B bench pair pair-search -n 2000 -s 3       | grep -v Elapsed | diff - target/ab/inputs_base_1v1.txt
$B bench pair-search pair pair -n 600 -s 5   | grep -v Elapsed | diff - target/ab/inputs_base_3p.txt
$B bench pair-search pair ga dqn -n 400 -s 9 | grep -v Elapsed | diff - target/ab/inputs_base_4p.txt
```

Expected: all three `diff`s print nothing. If any differs, stop: either the migration or the clock plumbing is wrong — do not commit the model.

- [ ] **Step 6: Measure the hot-path cost**

```bash
$B bench ga pair -n 200000 -s 42 | grep Elapsed
QWIXXER_SEARCH= $B bench pair-search ga -n 2000 -s 11 | grep Elapsed
```

Compare games/s with `target/ab/inputs_base_static_speed.txt` and `inputs_base_search_speed.txt`. Record both ratios in the commit message. If pair-search is more than 25% slower, report it before continuing (the tables in Task 2 are the intended mitigation; a slowdown beyond that means a regression to investigate, e.g. table lookups not inlined).

- [ ] **Step 7: Commit**

```bash
rustfmt --edition 2021 src/dqn/pair.rs src/main.rs
git add src/dqn/pair.rs src/main.rs pair_model/model.mpk
git commit -m "feat(pair): pair-migrate + migrated production model (59 inputs, behaviour-identical)

Old first-layer rows move to their new positions; the 14 new rows are zero,
so outputs are unchanged. Bench output byte-identical to pre-change
baselines (1v1 seed 3, 3p seed 5, 4p seed 9). Speed: static <X>x,
pair-search <Y>x of baseline.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(Replace `<X>`/`<Y>` with the measured ratios before committing.)

---

### Task 6: `pair-train --bench-ref` — head-to-head per-iteration bench

**Files:**
- Modify: `src/dqn/pair_train.rs` (`benchmark_vs_ref`, `self_play_train` parameter, test), `src/main.rs` (CLI flag)

**Interfaces:**
- Consumes: `PairStrategy::load`.
- Produces: `fn benchmark_vs_ref(artifact_dir: &str, ref_dir: &str, num_games: usize) -> f64` (tie-inclusive score of the current model, 0.5 = even); `self_play_train(..., bench_ref: Option<String>)` as the last parameter; CLI `--bench-ref <dir>`.

- [ ] **Step 1: Write the failing test** — in `pair_train.rs` tests:

```rust
    #[test]
    fn bench_vs_ref_is_exactly_even_against_itself() {
        let dir = std::env::temp_dir().join(format!("qwixxer-bench-ref-{}", std::process::id()));
        let dir = dir.to_str().unwrap().to_string();
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        std::fs::create_dir_all(&dir).unwrap();
        PairModelConfig::new()
            .init::<MyBackend>(&device)
            .save_file(format!("{dir}/model"), &CompactRecorder::new())
            .unwrap();
        // Paired dice + seat rotation: a model against itself mirrors exactly.
        assert_eq!(benchmark_vs_ref(&dir, &dir, 40), 0.5);
        std::fs::remove_dir_all(&dir).ok();
    }
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test --lib --release bench_vs_ref`
Expected: compile error (`benchmark_vs_ref` missing).

- [ ] **Step 3: Implement** next to `benchmark_vs_ga`:

```rust
/// Static head-to-head of the model in `artifact_dir` against the reference
/// model in `ref_dir`: paired dice, seat rotation, tie-inclusive score
/// (win 1, tie 0.5) of the current model.
fn benchmark_vs_ref(artifact_dir: &str, ref_dir: &str, num_games: usize) -> f64 {
    use crate::game::{Game, Player};

    let points: f64 = (0..num_games)
        .into_par_iter()
        .map_init(
            || (PairStrategy::load(artifact_dir), PairStrategy::load(ref_dir)),
            |templates, i| {
                let (cur, reference) = &*templates;
                let me = PairStrategy::from_shared(cur.model.clone(), cur.device.clone());
                let them = PairStrategy::from_shared(reference.model.clone(), reference.device.clone());
                let pair = (i / 2) as u64;
                let rotation = i % 2;
                let seat_dice = |seat: u64| Box::new(SmallRng::seed_from_u64(BENCH_SEED.wrapping_add(pair * 2 + seat)));
                let players: Vec<Player> = if rotation == 0 {
                    vec![Player::new(Box::new(me), seat_dice(0)), Player::new(Box::new(them), seat_dice(1))]
                } else {
                    vec![Player::new(Box::new(them), seat_dice(0)), Player::new(Box::new(me), seat_dice(1))]
                };
                let mut game = Game::new(players);
                game.play();
                let s: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
                match s[rotation].cmp(&s[1 - rotation]) {
                    std::cmp::Ordering::Greater => 1.0,
                    std::cmp::Ordering::Equal => 0.5,
                    std::cmp::Ordering::Less => 0.0,
                }
            },
        )
        .sum();
    points / num_games as f64
}
```

In `self_play_train`, add the final parameter `bench_ref: Option<String>`; where the per-iteration bench calls `benchmark_vs_ga(artifact_dir, &champion, bench_games)`, use

```rust
            let wr = match &bench_ref {
                Some(r) => benchmark_vs_ref(artifact_dir, r, bench_games),
                None => benchmark_vs_ga(artifact_dir, &champion, bench_games),
            };
```

and label its printout `"h2h vs {r}: score"` vs the existing `"vs GA"` text accordingly. Update the other `self_play_train` callers (only `main.rs`).

`src/main.rs`, `PairTrain` args:

```rust
        /// Per-iteration bench: static head-to-head against the pair model in
        /// this directory instead of vs GA
        #[arg(long)]
        bench_ref: Option<String>,
```

and pass `bench_ref` to `self_play_train`.

- [ ] **Step 4: Run tests**

Run: `cargo test --lib --release`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
rustfmt --edition 2021 src/dqn/pair_train.rs src/main.rs
git add src/dqn/pair_train.rs src/main.rs
git commit -m "feat(pair): pair-train --bench-ref — per-iteration static head-to-head vs a reference model

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Documentation and final verification

**Files:**
- Modify: `docs/EXPERIMENTS.md` (new Phase 24 stub), `docs/ARCHITECTURE.md` (pair net inputs, `EvalClock`, `pair-migrate`, `--bench-ref`)

- [ ] **Step 1: Append the Phase 24 stub** to `docs/EXPERIMENTS.md`:

```markdown

## Phase 24: Pair net inputs — turn position and threats

**Status: implemented; training run pending (pre-registered below).**

Spec: `docs/superpowers/specs/2026-10-12-pair-net-turn-and-threat-inputs-design.md`.
The pair net gains 7 inputs per board (45 → 59): turns until this player
rolls, colour phase pending, P(forced strike), and per-row P(locked by someone
else within a round). Turn position is passed to every evaluation as an
`EvalClock`; the production model was migrated losslessly (`pair-migrate`,
zero-weight new inputs) and plays byte-identically before training.

### Run recipe

    B=./target/release/qwixxer
    rm -f pair_model/iter-*.mpk
    cp -r pair_model pair_model_ref            # frozen reference: current model (already migrated)
    $B pair-train --distill --distill-passive --distill-m 2 --epsilon-lock 0.10 \
      -g 5000 -e 5 -c --start-iteration 90 -i 30 -b 500000 --bench-ref pair_model_ref

### Pre-registration

- **Primary (adoption):** static head-to-head, new vs `pair_model_ref`
  (`QWIXXER_ALT_MODEL=pair_model_ref bench pair pair-alt -n 200000 -s 42`):
  margin > 0 at z ≥ 2; and no regression in the Phase 22 suite (vs GA, DQN,
  Opportunist, Conservative; 3p and 4p mixes).
- **Headline, reported not required:** pair-search head-to-head
  (`bench pair-search pair-search-alt -n 50000 -s 42`).
- **Mechanism probes:** `leaf_calibration` side-to-move |b| < 0.03 (from
  ≈ 0.06); Phase 17 lock-ab suppression edge > +0.04pp. If static improves but
  neither moves, the gain is not attributed to these inputs.
- **Checkpoint selection:** best on the `--bench-ref` curve; confirm the top
  2–3 with the 200k primary bench.
```

- [ ] **Step 2: Update `docs/ARCHITECTURE.md`** — in the Pair Network section, replace "a joint two-board MLP (45 -> 128 -> 64 -> 2)" with "(59 -> 128 -> 64 -> 2)", replace "two 20-feature board blocks" with "two 27-feature board blocks (the original 20 plus turns-until-roll, colour-phase-pending, P(forced strike) and 4 per-row lock threats; `src/threat.rs`)", and add a sentence: "Every `Bot` evaluation takes an `EvalClock` (`src/strategy/clock.rs`) giving each board's turn position; callers supply it from the decision context. `qwixxer pair-migrate` converts 45-input models; `pair-train --bench-ref <dir>` benches each iteration head-to-head against a reference model."

- [ ] **Step 3: Final verification**

```bash
cargo test --lib --release
cargo build --release --examples
(cd web && cargo check)
git status --short
```

Expected: all tests pass; examples and the web crate build; only intended files modified.

- [ ] **Step 4: Commit**

```bash
git add docs/EXPERIMENTS.md docs/ARCHITECTURE.md
git commit -m "docs: Phase 24 stub — pair net turn/threat inputs, pre-registered recipe

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
