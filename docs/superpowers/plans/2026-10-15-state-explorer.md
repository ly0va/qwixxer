# State Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A second view of the web app (`#/explore`) where you set up any two-sheet Qwixx position, see your chance to win, and, with dice set, a ranked list of your legal moves tagged with each bot's pick.

**Architecture:** A pure-Rust `Explorer` (`web/src/explorer.rs`) evaluates positions with the pair net and runs the four web bots; `WasmExplorer` exposes it in the existing Web Worker next to `WasmGame`. On the TypeScript side a pure position model (`game/position.ts`), a link codec (`game/link.ts`) and an `ExplorerController` (`game/explorer.svelte.ts`) drive new Svelte views (`ExplorerDesktop`, `ExplorerPhone`) that reuse the game's sheet, chip, die and menu components. `App.svelte` routes between the game and the explorer by URL hash.

**Tech Stack:** Rust (wasm-bindgen, serde), Svelte 5 runes + TypeScript, Vite, Vitest.

**Spec:** `docs/superpowers/specs/2026-10-14-state-explorer-design.md` (mockups: the "Explorer" page of https://claude.ai/artifact/4eDmxXF5jEskvwxcFS26im).

## Global Constraints

- Never run `cargo fmt` on the whole crate; format touched Rust files with `rustfmt --edition 2021 <files>`.
- `cargo` inside `web/` may rewrite `web/Cargo.lock`; revert it (`git checkout web/Cargo.lock`) unless dependencies really changed (none do in this plan).
- Never commit `.superpowers/`, `AGENTS.md` or `web-ui.txt`; never edit `AGENTS.md`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Don't touch committed model artifacts (`champion.txt`, `pair_model/`, `dqn_model/`).
- No pushing (pushing `main` deploys). Work stays on branch `explorer`.
- No developer numbers (μ, σ, z, rollouts) in the UI; no pencil marks in the explorer: every X is ink.
- Same theme tokens, fonts and components as the game; light and dark both work.
- Copy, verbatim from the spec: "chance you win", "before the roll", "Best moves", "your chance to win after each", "Set the dice to rank the moves", "Tap a die to choose its face, or press Random.", "That link doesn't hold a valid position.", "Link copied", "Swap sides", "Clear sheets", "Your roll", "Opponent's roll", "Explore this position", "Explorer", "Back to game".
- Commands: Rust web tests `cargo test --release --manifest-path web/Cargo.toml` (from repo root); WASM build `./web/build.sh` (from repo root); TS tests `cd web && npx vitest run`; types `cd web && npx svelte-check --threshold error`; bundle `cd web && npx vite build`.

## Review Focus

1. Rapid edits while Master+ is thinking: only the latest position's answer may land; an older answer must never overwrite a newer one (pinned in Task 5's coalescing test).
2. A junk, truncated or old-version link: the message "That link doesn't hold a valid position." and the previous position kept (Task 4 codec tests, Task 5 `loadCode` test).
3. Positions the engine can't reach (gaps, end number marked with < 5 marks, every square marked, 3 strikes, every die face): no panic in any bot, and every bot's pick maps to a listed move (Task 1 fuzz test).
4. Keys in the explorer: Enter and Space must never confirm or roll the game hidden behind it (Task 8 `keyAction` test).
5. Garbage or missing local storage: the explorer starts from an empty position and works (Task 5 `init` test).

---

### Task 1: Rust explorer core and `WasmExplorer`

**Files:**
- Create: `web/src/explorer.rs`
- Modify: `web/src/bots.rs` (make `PAIR_MODEL` crate-visible, add `BotKind::ALL` and `BotKind::id`)
- Modify: `web/src/lib.rs` (module + `WasmExplorer`)

**Interfaces:**
- Consumes: `selection::{active_options, passive_options, union, n_locked}`, `session::{SheetSnap, Sq}` (both `pub` with `pub` fields), `bots::{make_bot, BotKind}`.
- Produces (JSON contract consumed by Task 2):
  - Input `Position`: `{ you: {marks: bool[4][11], strikes: u8}, opponent: {…}, locked: bool[4], dice: [u8;6] | null, roller: "you" | "opponent" }`. `locked` = rows closed by tapping their lock; rows whose end number is marked with ≥ 5 marks are closed by the engine's own rule and unioned in.
  - Output `Answer`: `{ over: null | {you: isize, opponent: isize}, eval: {p_win: f32, margin: f32, sigma: f32}, moves: null | [{marks: Sq[], strike: bool, pass: bool, p_win: f32, bots: ("master_plus"|"master"|"student"|"newbie")[]}] }`. `moves` is sorted by `p_win` descending (stable), `null` when no dice or the game is over.
  - WASM: `new WasmExplorer()`, `explorer.explore(position_json: string): string` (throws on bad JSON or invalid values).

- [ ] **Step 1: Expose what the explorer needs in `bots.rs`**

In `web/src/bots.rs` change `const PAIR_MODEL` to `pub(crate) const PAIR_MODEL` and add to `impl BotKind`:

```rust
    /// Every web bot, strongest first (the order of the explorer's tags).
    pub const ALL: [BotKind; 4] = [BotKind::MasterPlus, BotKind::Master, BotKind::Student, BotKind::Newbie];

    /// The id `parse` accepts (and the explorer reports).
    pub fn id(self) -> &'static str {
        match self {
            BotKind::MasterPlus => "master_plus",
            BotKind::Master => "master",
            BotKind::Student => "student",
            BotKind::Newbie => "newbie",
        }
    }
```

- [ ] **Step 2: Write the failing tests**

Create `web/src/explorer.rs` with only the test module for now (plus `use` lines it needs so it compiles once the code exists):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn sheet() -> SheetSnap {
        SheetSnap { marks: [[false; 11]; 4], strikes: 0 }
    }
    fn pos(dice: Option<[u8; 6]>, roller: Roller) -> Position {
        Position { you: sheet(), opponent: sheet(), locked: [false; 4], dice, roller }
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
        let a = Explorer::new().explore(&pos(Some([3, 4, 2, 5, 3, 1]), Roller::Opponent)).unwrap();
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
            let mut p = pos(None, if rng.gen_bool(0.5) { Roller::You } else { Roller::Opponent });
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
                    assert_eq!(moves.iter().filter(|m| m.bots.contains(&id)).count(), 1, "{id} in {p:?}");
                }
            }
        }
    }
}
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `cargo test --release --manifest-path web/Cargo.toml explorer`
Expected: FAIL to compile (`Explorer`, `Position`, … not found). (Add `pub mod explorer;` to `lib.rs` first so the module is built.)

- [ ] **Step 4: Implement `explorer.rs`**

Put this above the test module:

```rust
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
    let pass = Turn { marks: vec![], strike: false };
    match roller {
        Roller::You => {
            let o = active_options(you, opp, None, dice);
            let mut t: Vec<Turn> = o.selections.into_iter().map(|marks| Turn { marks, strike: false }).collect();
            if o.strike {
                t.push(Turn { marks: vec![], strike: true });
            }
            if o.pass {
                t.push(pass);
            }
            t
        }
        Roller::Opponent => {
            let mut t: Vec<Turn> =
                passive_options(you, dice).into_iter().map(|m| Turn { marks: vec![m], strike: false }).collect();
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
            // A pick that matches no listed move is an engine bug; the
            // random-position test pins that it never happens.
            if let Some(i) = keys.iter().position(|k| *k == pick) {
                out[i].bots.push(kind.id());
            }
        }
        out.sort_by(|a, b| b.p_win.partial_cmp(&a.p_win).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}
```

- [ ] **Step 5: Add `WasmExplorer` to `lib.rs`**

Add `pub mod explorer;` to the module list, update the module doc line to mention `WasmExplorer`, and append:

```rust
/// The state explorer: stateless calls, one explorer per worker.
#[wasm_bindgen]
pub struct WasmExplorer {
    inner: explorer::Explorer,
}

impl Default for WasmExplorer {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmExplorer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmExplorer {
        WasmExplorer { inner: explorer::Explorer::new() }
    }

    pub fn explore(&mut self, position: &str) -> Result<String, JsError> {
        let pos: explorer::Position = serde_json::from_str(position).map_err(js)?;
        let answer = self.inner.explore(&pos).map_err(js)?;
        Ok(serde_json::to_string(&answer).unwrap())
    }
}
```

- [ ] **Step 6: Run the tests to see them pass**

Run: `cargo test --release --manifest-path web/Cargo.toml` (all web crate tests)
Expected: PASS, including the 7 explorer tests. If the random test finds a panic or an unlisted pick, fix the cause in `explorer.rs` (never weaken the test); a bot panicking on an unreachable state is a finding to report.

Note on `opponents_roll_offers_white_sum_marks_and_pass`: on an empty sheet white 7 is legal in all four rows. If `generate_white_moves` dedups differently, adjust the expected count to what `passive_options` returns plus one, and say so in the report.

- [ ] **Step 7: Format, revert lockfile, commit**

```bash
rustfmt --edition 2021 web/src/explorer.rs web/src/lib.rs web/src/bots.rs
git checkout web/Cargo.lock 2>/dev/null; git status --short
git add web/src/explorer.rs web/src/lib.rs web/src/bots.rs
git commit -m "feat(web): explorer engine — pair-net evaluation, ranked legal moves, each bot's pick

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Worker RPC for `explore`

**Files:**
- Modify: `web/src/engine/protocol.ts`, `web/src/engine/serve.ts`, `web/src/engine/engine.worker.ts`, `web/src/engine/client.ts`
- Test: `web/src/engine/serve.test.ts`

**Interfaces:**
- Consumes: Task 1's JSON contract and `WasmExplorer`.
- Produces (TS, in `protocol.ts`):

```ts
export type Roller = 'you' | 'opponent';
export interface SheetSnap { marks: boolean[][]; strikes: number }
/** Mirrors explorer.rs `Position`. `locked` = rows closed by tapping their lock. */
export interface Position { you: SheetSnap; opponent: SheetSnap; locked: boolean[]; dice: number[] | null; roller: Roller }
export interface Eval { p_win: number; margin: number; sigma: number }
export interface MoveOut { marks: Sq[]; strike: boolean; pass: boolean; p_win: number; bots: BotKind[] }
export interface Answer { over: { you: number; opponent: number } | null; eval: Eval; moves: MoveOut[] | null }
// Engine gains:
explore(position: Position): Promise<Answer>;
```

- [ ] **Step 1: Write the failing test**

In `serve.test.ts`, add a fake explorer and include it in `setup`'s `wasm` object (`WasmGame: FakeGame, WasmExplorer: FakeExplorer`):

```ts
class FakeExplorer {
    static made = 0;
    constructor() { FakeExplorer.made++; }
    explore(json: string) {
        const p = JSON.parse(json);
        if (p.bad) throw new Error('bad position');
        return JSON.stringify({ over: null, eval: { p_win: 0.5, margin: 0, sigma: 10 }, moves: null, echo: p.roller });
    }
    free() {}
}
```

and the test:

```ts
    it('answers explore with one lazily created explorer, and reports bad positions as engine errors', async () => {
        const { send } = setup(0);
        FakeExplorer.made = 0;
        expect(await send(1, 'explore', [{ roller: 'you' }])).toMatchObject({ ok: true, result: { echo: 'you', eval: { p_win: 0.5 } } });
        expect(await send(2, 'explore', [{ roller: 'opponent' }])).toMatchObject({ ok: true, result: { echo: 'opponent' } });
        expect(FakeExplorer.made).toBe(1);
        expect(await send(3, 'explore', [{ bad: true }])).toMatchObject({ ok: false, kind: 'engine', error: 'bad position' });
    });
```

- [ ] **Step 2: Run it to see it fail**

Run: `cd web && npx vitest run src/engine/serve.test.ts`
Expected: FAIL ("unknown command explore").

- [ ] **Step 3: Implement**

`protocol.ts`: add the types above and `explore(position: Position): Promise<Answer>;` to `Engine` (after `submitActive`).

`serve.ts`:
- import type: `import type { WasmExplorer, WasmGame } from '../../pkg/qwixxer_web.js';`
- `Wasm` gains `WasmExplorer: { new (): WasmExplorer };`
- in `serve`, next to `let game`: `let explorer: WasmExplorer | null = null;`
- in `run`, before `default`:

```ts
            case 'explore':
                // Created on first use and kept: the game and the explorer share the worker.
                explorer ??= new wasm.WasmExplorer();
                return JSON.parse(explorer.explore(JSON.stringify(args[0])));
```

- update the doc comment: "…and runs one game and the explorer."

`engine.worker.ts`: `import init, { WasmExplorer, WasmGame } from '../../pkg/qwixxer_web.js';` and pass `WasmExplorer` in the object given to `serve`.

`client.ts`: import `Answer, Position` types and add `explore(position: Position) { return this.call<Answer>('explore', position); }`.

- [ ] **Step 4: Rebuild the WASM package and check**

Run (repo root): `./web/build.sh` then `cd web && npx vitest run && npx svelte-check --threshold error`
Expected: all tests PASS; 0 errors (svelte-check needs the rebuilt `pkg` to know `WasmExplorer`). Revert `web/Cargo.lock` if the build touched it.

- [ ] **Step 5: Commit**

```bash
git add web/src/engine
git commit -m "feat(web): explore RPC in the engine worker

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Position model

**Files:**
- Create: `web/src/game/position.ts`
- Test: `web/src/game/position.test.ts`
- Modify: `web/src/ui/Cell.svelte` (one attribute, see Step 3)

**Interfaces:**
- Consumes: `Position, Roller, SheetSnap, View, Answer` from `protocol.ts`; `numbers, NAME, ROWS, RowModel, CellModel, LockLook` from `game/rules.ts`.
- Produces (all edits are pure: they return a new `ExPosition`):

```ts
export type Who = 'you' | 'opponent';
export interface ExPosition { you: SheetSnap; opponent: SheetSnap; locked: boolean[]; dice: (number | null)[]; roller: Roller }
export function emptyPosition(): ExPosition;
export function closesRow(s: SheetSnap, r: number): boolean;
export function closedRows(p: ExPosition): boolean[];
export function toggleSquare(p: ExPosition, who: Who, row: number, i: number): ExPosition;
export function tapStrike(p: ExPosition, who: Who, k: number): ExPosition;
export function toggleLock(p: ExPosition, row: number): ExPosition;
export function clearSheet(p: ExPosition, who: Who): ExPosition;
export function clearSheets(p: ExPosition): ExPosition;
export function swapSides(p: ExPosition): ExPosition;
export function setRoller(p: ExPosition, r: Roller): ExPosition;
export function setDie(p: ExPosition, i: number, v: number | null): ExPosition;
export function randomDice(p: ExPosition, rand?: () => number): ExPosition;
export function clearDice(p: ExPosition): ExPosition;
export function nextUnsetDie(p: ExPosition, after: number): number | null;
export function toEngine(p: ExPosition): Position;
export function fromView(v: View): ExPosition;
export function rowPoints(s: SheetSnap): number[];
export function sheetTotal(s: SheetSnap): number;
export function exploreSheet(p: ExPosition, who: Who): RowModel[];
export function percent(p: number, digits: 0 | 1): string;
export function marginText(e: { margin: number; sigma: number }): string;
export function overText(o: { you: number; opponent: number }): string;
```

- [ ] **Step 1: Write the failing tests**

`web/src/game/position.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import type { View } from '../engine/protocol';
import {
    clearSheet, closedRows, emptyPosition, exploreSheet, fromView, marginText, nextUnsetDie, overText, percent, randomDice,
    setDie, sheetTotal, swapSides, tapStrike, toEngine, toggleLock, toggleSquare,
} from './position';

const markRed = (idx: number[]) => idx.reduce((p, i) => toggleSquare(p, 'you', 0, i), emptyPosition());

describe('position edits', () => {
    it('toggles squares without touching the original', () => {
        const p0 = emptyPosition();
        const p1 = toggleSquare(p0, 'you', 2, 3);
        expect(p1.you.marks[2][3]).toBe(true);
        expect(p0.you.marks[2][3]).toBe(false);
        expect(toggleSquare(p1, 'you', 2, 3).you.marks[2][3]).toBe(false);
    });
    it('sets strikes from the tapped box, and the last taken box removes itself', () => {
        let p = tapStrike(emptyPosition(), 'opponent', 2);
        expect(p.opponent.strikes).toBe(3);
        p = tapStrike(p, 'opponent', 2);
        expect(p.opponent.strikes).toBe(2);
        expect(tapStrike(p, 'opponent', 0).opponent.strikes).toBe(1);
    });
    it('closes a row by marking its end number with five marks, or by tapping its lock', () => {
        expect(closedRows(markRed([0, 1, 2, 10]))[0]).toBe(false); // four marks: not closed
        const byMark = markRed([0, 1, 2, 3, 10]);
        expect(closedRows(byMark)[0]).toBe(true);
        expect(toggleLock(byMark, 0)).toEqual(byMark); // closed by marking: the lock tap does nothing
        const byTap = toggleLock(emptyPosition(), 1);
        expect(closedRows(byTap)).toEqual([false, true, false, false]);
        expect(closedRows(toggleLock(byTap, 1))[1]).toBe(false);
    });
    it('clears a sheet and reopens rows closed by tapping', () => {
        const p = clearSheet(toggleLock(tapStrike(markRed([0, 1]), 'you', 0), 3), 'you');
        expect(p.you.marks.flat().some(Boolean)).toBe(false);
        expect(p.you.strikes).toBe(0);
        expect(p.locked).toEqual([false, false, false, false]);
    });
    it('swaps sides and whose roll; twice is the identity', () => {
        const p = tapStrike(markRed([4]), 'you', 1);
        const s = swapSides(p);
        expect(s.opponent.marks[0][4]).toBe(true);
        expect(s.opponent.strikes).toBe(2);
        expect(s.roller).toBe('opponent');
        expect(swapSides(s)).toEqual(p);
    });
    it('sends dice to the engine only when all six are set', () => {
        let p = setDie(emptyPosition(), 0, 3);
        expect(toEngine(p).dice).toBeNull();
        expect(nextUnsetDie(p, 0)).toBe(1);
        p = randomDice(p, () => 0.999);
        expect(toEngine(p).dice).toEqual([6, 6, 6, 6, 6, 6]);
        expect(nextUnsetDie(p, 0)).toBeNull();
    });
});

describe('from the game', () => {
    const empty = () => ({ marks: [0, 1, 2, 3].map(() => Array(11).fill(false)), strikes: 0, score: 0, row_points: [0, 0, 0, 0] });
    const view = (p: Partial<View>): View => ({ phase: 'bot_to_roll', dice: null, you: empty(), bot: empty(), locked: [false, false, false, false], bot_white: null, options: null, passive: [], over: null, ...p });
    it('copies the boards, and the dice and roller while a roll is resolved', () => {
        const v = view({ phase: 'passive', dice: [1, 2, 3, 4, 5, 6] });
        v.bot.marks[1][4] = true;
        v.you.strikes = 2;
        const p = fromView(v);
        expect(p.opponent.marks[1][4]).toBe(true);
        expect(p.you.strikes).toBe(2);
        expect(p.dice).toEqual([1, 2, 3, 4, 5, 6]);
        expect(p.roller).toBe('opponent');
        expect(fromView(view({ phase: 'active', dice: [1, 1, 1, 1, 1, 1] })).roller).toBe('you');
    });
    it('copies no dice between rolls', () => {
        const p = fromView(view({ phase: 'you_to_roll', dice: [1, 2, 3, 4, 5, 6] }));
        expect(p.dice).toEqual([null, null, null, null, null, null]);
        expect(p.roller).toBe('you');
    });
});

describe('sheet model and text', () => {
    it('inks marks, greys squares left of the last mark and outs closed rows', () => {
        const p = toggleLock(markRed([0, 3]), 2);
        const rows = exploreSheet(p, 'you');
        expect(rows[0].cells[0].mark).toBe('ink');
        expect(rows[0].cells[1].look).toBe('skipped');
        expect(rows[0].cells.every((c) => c.clickable)).toBe(true);
        expect(rows[0].lock).toBe('grey');
        expect(rows[2].band).toBe('locked');
        expect(rows[2].cells[0].look).toBe('out');
        expect(rows[2].lock).toBe('locked_other');
    });
    it('shows the lock filled on the sheet that closed the row by marking', () => {
        const p = markRed([0, 1, 2, 3, 10]);
        expect(exploreSheet(p, 'you')[0].lock).toBe('locked_by');
        expect(exploreSheet(p, 'opponent')[0].lock).toBe('locked_other');
    });
    it('scores like the engine: the lock counts as a mark, strikes cost 5', () => {
        const p = tapStrike(markRed([0, 1, 2, 3, 10]), 'you', 0);
        expect(sheetTotal(p.you)).toBe(21 - 5); // 5 marks + lock = 6 → 21
    });
    it('formats chances, margins and results', () => {
        expect(percent(0.3786, 0)).toBe('38%');
        expect(percent(0.41249, 1)).toBe('41.2%');
        expect(marginText({ margin: -8.6, sigma: 14.9 })).toBe('Opponent ahead by about 9 (± 15)');
        expect(marginText({ margin: 3.2, sigma: 10.4 })).toBe('You ahead by about 3 (± 10)');
        expect(marginText({ margin: 0.3, sigma: 12 })).toBe('Level (± 12)');
        expect(overText({ you: 41, opponent: 33 })).toBe('You win 41 – 33');
        expect(overText({ you: 30, opponent: 33 })).toBe('Opponent wins 33 – 30');
        expect(overText({ you: 20, opponent: 20 })).toBe('Draw 20 – 20');
    });
});
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && npx vitest run src/game/position.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `position.ts`**

```ts
import type { Position, Roller, SheetSnap, View } from '../engine/protocol';
import { NAME, numbers, ROWS, type CellLook, type CellModel, type LockLook, type RowModel } from './rules';

/** The explorer's two sides (spec "The position"). */
export type Who = 'you' | 'opponent';

/**
 * The explorer's position. `dice` per die (null = unset); `locked` = rows
 * closed by tapping their lock (rows closed by marking are derived).
 */
export interface ExPosition { you: SheetSnap; opponent: SheetSnap; locked: boolean[]; dice: (number | null)[]; roller: Roller }

const emptySheet = (): SheetSnap => ({ marks: [0, 1, 2, 3].map(() => Array(11).fill(false)), strikes: 0 });
export const emptyPosition = (): ExPosition => ({ you: emptySheet(), opponent: emptySheet(), locked: [false, false, false, false], dice: Array(6).fill(null), roller: 'you' });

// Positions may be Svelte state Proxies: copy through JSON, not structuredClone.
const copy = (p: ExPosition): ExPosition => JSON.parse(JSON.stringify(p));

/** The engine's lock rule: the row's end number marked, with five or more marks. */
export const closesRow = (s: SheetSnap, r: number) => s.marks[r][10] && s.marks[r].filter(Boolean).length >= 5;
export const closedRows = (p: ExPosition) => [0, 1, 2, 3].map((r) => p.locked[r] || closesRow(p.you, r) || closesRow(p.opponent, r));

export function toggleSquare(p: ExPosition, who: Who, row: number, i: number): ExPosition {
    const q = copy(p);
    q[who].marks[row][i] = !q[who].marks[row][i];
    return q;
}

/** Tap box k: strikes become k + 1, or k if box k was the last one taken. */
export function tapStrike(p: ExPosition, who: Who, k: number): ExPosition {
    const q = copy(p);
    q[who].strikes = q[who].strikes === k + 1 ? k : k + 1;
    return q;
}

/** A row closed by marking reopens only by unmarking its end number. */
export function toggleLock(p: ExPosition, row: number): ExPosition {
    if (closesRow(p.you, row) || closesRow(p.opponent, row)) return p;
    const q = copy(p);
    q.locked[row] = !q.locked[row];
    return q;
}

export function clearSheet(p: ExPosition, who: Who): ExPosition {
    const q = copy(p);
    q[who] = emptySheet();
    q.locked = [false, false, false, false];
    return q;
}
export const clearSheets = (p: ExPosition): ExPosition => clearSheet(clearSheet(p, 'you'), 'opponent');

export function swapSides(p: ExPosition): ExPosition {
    const q = copy(p);
    [q.you, q.opponent] = [q.opponent, q.you];
    q.roller = q.roller === 'you' ? 'opponent' : 'you';
    return q;
}

export function setRoller(p: ExPosition, r: Roller): ExPosition {
    const q = copy(p);
    q.roller = r;
    return q;
}

export function setDie(p: ExPosition, i: number, v: number | null): ExPosition {
    const q = copy(p);
    q.dice[i] = v;
    return q;
}
export function randomDice(p: ExPosition, rand: () => number = Math.random): ExPosition {
    const q = copy(p);
    q.dice = q.dice.map(() => 1 + Math.floor(rand() * 6));
    return q;
}
export function clearDice(p: ExPosition): ExPosition {
    const q = copy(p);
    q.dice = Array(6).fill(null);
    return q;
}
/** The next unset die after `after` (wrapping), or null when all are set. */
export function nextUnsetDie(p: ExPosition, after: number): number | null {
    for (let k = 1; k <= 6; k++) {
        const i = (after + k) % 6;
        if (p.dice[i] === null) return i;
    }
    return null;
}

export function toEngine(p: ExPosition): Position {
    const q = copy(p);
    const dice = q.dice.every((d) => d !== null) ? (q.dice as number[]) : null;
    return { you: q.you, opponent: q.opponent, locked: q.locked, dice, roller: q.roller };
}

/**
 * Spec "Explore this position": the game's boards (You → You, Bot →
 * Opponent) before the current roll's marks; while a roll is being
 * resolved, its dice and whose roll it is.
 */
export function fromView(v: View): ExPosition {
    const sheet = (s: View['you']): SheetSnap => ({ marks: s.marks.map((r) => [...r]), strikes: s.strikes });
    const you = sheet(v.you);
    const opponent = sheet(v.bot);
    const rolling = (v.phase === 'active' || v.phase === 'passive') && !!v.dice;
    return {
        you, opponent,
        locked: v.locked.map((l, r) => l && !closesRow(you, r) && !closesRow(opponent, r)),
        dice: rolling ? [...v.dice!] : Array(6).fill(null),
        roller: v.phase === 'passive' || v.phase === 'bot_to_roll' ? 'opponent' : 'you',
    };
}

/** Points per row as the engine counts them: the lock adds a mark. */
export function rowPoints(s: SheetSnap): number[] {
    return [0, 1, 2, 3].map((r) => {
        const n = s.marks[r].filter(Boolean).length + (closesRow(s, r) ? 1 : 0);
        return (n * (n + 1)) / 2;
    });
}
export const sheetTotal = (s: SheetSnap) => rowPoints(s).reduce((a, b) => a + b, 0) - 5 * s.strikes;

/** Rows for SheetRows / SheetColumns: every square tappable, all marks ink. */
export function exploreSheet(p: ExPosition, who: Who): RowModel[] {
    const s = p[who];
    const closed = closedRows(p);
    return ROWS.map((color, r) => {
        const marks = s.marks[r];
        const last = marks.lastIndexOf(true);
        const count = marks.filter(Boolean).length;
        const cells: CellModel[] = numbers(r).map((n, i) => {
            let look: CellLook = 'normal';
            if (closed[r] && !marks[i]) look = 'out';
            else if (i < last && !marks[i]) look = 'skipped';
            else if (i === 10 && !marks[i] && count < 5) look = 'skipped';
            return { n, sq: { row: r, n }, mark: marks[i] ? 'ink' : 'none', anim: '', look, clickable: true, label: `${NAME[r]} ${n}${marks[i] ? ', marked' : ''}` };
        });
        const lock: LockLook = closesRow(s, r) ? 'locked_by' : closed[r] ? 'locked_other' : count >= 5 ? 'open' : 'grey';
        return { row: r, color, band: closed[r] ? 'locked' : 'normal', cells, lock, label: `${NAME[r]} row${closed[r] ? ', locked' : ''}` };
    });
}

export const percent = (p: number, digits: 0 | 1) => `${(p * 100).toFixed(digits)}%`;

export function marginText(e: { margin: number; sigma: number }): string {
    const m = Math.round(e.margin);
    const s = Math.round(e.sigma);
    if (m > 0) return `You ahead by about ${m} (± ${s})`;
    if (m < 0) return `Opponent ahead by about ${-m} (± ${s})`;
    return `Level (± ${s})`;
}

export function overText(o: { you: number; opponent: number }): string {
    if (o.you > o.opponent) return `You win ${o.you} – ${o.opponent}`;
    if (o.you < o.opponent) return `Opponent wins ${o.opponent} – ${o.you}`;
    return `Draw ${o.you} – ${o.opponent}`;
}
```

In `web/src/ui/Cell.svelte` change `aria-pressed={cell.mark === 'pencil'}` to `aria-pressed={cell.mark === 'pencil' || (cell.clickable && cell.mark === 'ink')}` (in the game inked squares are never clickable, so its behaviour is unchanged; in the explorer marked squares read as pressed).

- [ ] **Step 4: Run the tests to see them pass**

Run: `cd web && npx vitest run && npx svelte-check --threshold error`
Expected: PASS; 0 errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/game/position.ts web/src/game/position.test.ts web/src/ui/Cell.svelte
git commit -m "feat(web): explorer position model — edits, closed rows, sheet model, text

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Link codec

**Files:**
- Create: `web/src/game/link.ts`
- Test: `web/src/game/link.test.ts`

**Interfaces:**
- Consumes: `ExPosition`, `emptyPosition` (Task 3).
- Produces: `encode(p: ExPosition): string` (21 characters: version `'1'` + 20 base64url) and `decode(code: string): ExPosition | null`.

Bit layout (MSB first, 117 bits, zero-padded to 15 bytes): you marks 44 (row 0 index 0 … row 3 index 10), you strikes 3, opponent marks 44, opponent strikes 3, tap-closed rows 4, dice 6 × 3 (0 = unset, 1–6), roller 1 (0 = you, 1 = opponent).

- [ ] **Step 1: Write the failing tests**

```ts
import { describe, expect, it } from 'vitest';
import { decode, encode } from './link';
import { emptyPosition, type ExPosition } from './position';

function randomPosition(rand: () => number): ExPosition {
    const p = emptyPosition();
    for (const s of [p.you, p.opponent]) {
        s.marks = s.marks.map((r) => r.map(() => rand() < 0.4));
        s.strikes = Math.floor(rand() * 5);
    }
    p.locked = p.locked.map(() => rand() < 0.2);
    p.dice = p.dice.map(() => (rand() < 0.2 ? null : 1 + Math.floor(rand() * 6)));
    p.roller = rand() < 0.5 ? 'you' : 'opponent';
    return p;
}

describe('link codec', () => {
    it('round-trips random positions in 21 URL-safe characters', () => {
        let seed = 7;
        const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
        for (let k = 0; k < 300; k++) {
            const p = randomPosition(rand);
            const code = encode(p);
            expect(code).toMatch(/^1[A-Za-z0-9_-]{20}$/);
            expect(decode(code)).toEqual(p);
        }
    });
    it('rejects malformed codes', () => {
        const good = encode(emptyPosition());
        for (const bad of ['', '1', good.slice(0, -1), good + 'A', '2' + good.slice(1), good.slice(0, -1) + '*', '1' + '/'.repeat(20)]) {
            expect(decode(bad)).toBeNull();
        }
    });
    it('rejects out-of-range strikes and dice', () => {
        // encode writes any 3-bit value; decode must reject strikes > 4 and face 7.
        const p = emptyPosition();
        p.opponent.strikes = 7;
        expect(decode(encode(p))).toBeNull();
        const q = emptyPosition();
        q.dice[2] = 7;
        expect(decode(encode(q))).toBeNull();
    });
});
```

(The last test relies on `encode` writing whatever 3-bit value it is given; `decode` must reject strikes > 4 and dice face 7.)

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && npx vitest run src/game/link.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `link.ts`**

```ts
import { emptyPosition, type ExPosition } from './position';

/**
 * Spec "Links and memory": a position as a short URL-safe code, a version
 * character then base64url of 117 bits (see the plan's layout): both
 * sheets' marks and strikes, tap-closed rows, dice (0 = unset), roller.
 */
const VERSION = '1';
const BYTES = 15;

class Bits {
    bytes = new Uint8Array(BYTES);
    pos = 0;
    put(v: number, n: number) {
        for (let k = n - 1; k >= 0; k--, this.pos++) if ((v >> k) & 1) this.bytes[this.pos >> 3] |= 0x80 >> (this.pos & 7);
    }
    get(n: number): number {
        let v = 0;
        for (let k = 0; k < n; k++, this.pos++) v = (v << 1) | ((this.bytes[this.pos >> 3] >> (7 - (this.pos & 7))) & 1);
        return v;
    }
}

export function encode(p: ExPosition): string {
    const b = new Bits();
    for (const s of [p.you, p.opponent]) {
        for (const row of s.marks) for (const m of row) b.put(m ? 1 : 0, 1);
        b.put(s.strikes, 3);
    }
    for (const l of p.locked) b.put(l ? 1 : 0, 1);
    for (const d of p.dice) b.put(d ?? 0, 3);
    b.put(p.roller === 'opponent' ? 1 : 0, 1);
    const b64 = btoa(String.fromCharCode(...b.bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    return VERSION + b64;
}

export function decode(code: string): ExPosition | null {
    if (!/^1[A-Za-z0-9_-]{20}$/.test(code)) return null;
    let raw: string;
    try {
        raw = atob(code.slice(1).replace(/-/g, '+').replace(/_/g, '/'));
    } catch {
        return null;
    }
    const b = new Bits();
    b.bytes = Uint8Array.from(raw, (c) => c.charCodeAt(0));
    const p = emptyPosition();
    for (const s of [p.you, p.opponent]) {
        s.marks = s.marks.map((row) => row.map(() => b.get(1) === 1));
        s.strikes = b.get(3);
        if (s.strikes > 4) return null;
    }
    p.locked = p.locked.map(() => b.get(1) === 1);
    for (let i = 0; i < 6; i++) {
        const d = b.get(3);
        if (d > 6) return null;
        p.dice[i] = d === 0 ? null : d;
    }
    p.roller = b.get(1) === 1 ? 'opponent' : 'you';
    return p;
}
```

- [ ] **Step 4: Run them to see them pass**

Run: `cd web && npx vitest run src/game/link.test.ts`
Expected: PASS. (`'1' + '/'.repeat(20)` fails the character check; a 20-char base64 of 15 bytes never needs padding.)

- [ ] **Step 5: Commit**

```bash
git add web/src/game/link.ts web/src/game/link.test.ts
git commit -m "feat(web): explorer link codec (21-character position codes)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `ExplorerController`

**Files:**
- Create: `web/src/game/explorer.svelte.ts`
- Test: `web/src/game/explorer.test.ts`

**Interfaces:**
- Consumes: `Engine.explore` (Task 2), position edits (Task 3), `encode/decode` (Task 4), `Store` from `game/controller.svelte.ts`.
- Produces:

```ts
export const BAD_LINK = "That link doesn't hold a valid position.";
export class ExplorerController {
    pos: ExPosition;            // $state
    answer: Answer | null;      // $state: the latest answer that matches pos, or the last one while pending
    pending: boolean;           // $state: pos changed and its answer has not arrived
    error: string;              // $state
    notice: string;             // $state: "Link copied", BAD_LINK, … (clears after 3 s)
    picker: number | null;      // $state: die index whose 1–6 picker is open
    showAll: boolean;           // $state
    constructor(engine: Engine, store: Store, clipboard?: (text: string) => Promise<void>, later?: (f: () => void, ms: number) => void);
    init(): void;
    loadCode(code: string): boolean;
    loadView(v: View): void;
    tapSquare(who: Who, row: number, i: number): void;
    tapStrike(who: Who, k: number): void;
    tapLock(row: number): void;
    clear(who: Who): void;
    clearAll(): void;
    swap(): void;
    setRoller(r: Roller): void;
    openPicker(i: number): void;
    pickFace(v: number): void;
    random(): void;
    clearDice(): void;
    linkFor(base: string): string;       // `${base}#/explore?p=${code}`
    copyLink(base: string): Promise<void>;
}
```

- [ ] **Step 1: Write the failing tests**

```ts
import { describe, expect, it, vi } from 'vitest';
import type { Answer, Engine, Position } from '../engine/protocol';
import type { Store } from './controller.svelte';
import { BAD_LINK, ExplorerController } from './explorer.svelte';
import { encode } from './link';
import { emptyPosition, toggleSquare } from './position';

const answer = (p_win: number): Answer => ({ over: null, eval: { p_win, margin: 0, sigma: 10 }, moves: null });
function memStore(init: Record<string, string> = {}): Store & { data: Record<string, string> } {
    const data = { ...init };
    return { data, get: (k) => data[k] ?? null, set: (k, v) => { data[k] = v; } };
}
/** An engine whose explore calls resolve only when the test says so. */
function manualEngine() {
    const calls: { pos: Position; resolve: (a: Answer) => void }[] = [];
    const engine = { explore: vi.fn((pos: Position) => new Promise<Answer>((resolve) => calls.push({ pos: structuredClone(pos), resolve }))) } as unknown as Engine;
    return { engine, calls };
}
const flush = () => new Promise((r) => setTimeout(r, 0));

describe('ExplorerController', () => {
    it('starts empty when storage is empty or garbage, and evaluates', async () => {
        for (const store of [memStore(), memStore({ 'qwixxer.explorer.v1': 'garbage' })]) {
            const { engine, calls } = manualEngine();
            const x = new ExplorerController(engine, store);
            x.init();
            expect(x.pos).toEqual(emptyPosition());
            expect(calls).toHaveLength(1);
            calls[0].resolve(answer(0.5));
            await flush();
            expect(x.answer?.eval.p_win).toBe(0.5);
            expect(x.pending).toBe(false);
        }
    });

    it('evaluates only the latest of rapid edits, and never lands an older answer', async () => {
        const { engine, calls } = manualEngine();
        const x = new ExplorerController(engine, memStore());
        x.init();
        x.tapSquare('you', 0, 0);
        x.tapSquare('you', 0, 1);
        x.tapSquare('you', 0, 2);
        expect(calls).toHaveLength(1); // the first call is still in flight
        calls[0].resolve(answer(0.1)); // answer for the empty position: stale
        await flush();
        expect(x.answer).toBeNull();
        expect(x.pending).toBe(true);
        expect(calls).toHaveLength(2);
        expect(calls[1].pos.you.marks[0].slice(0, 3)).toEqual([true, true, true]);
        calls[1].resolve(answer(0.6));
        await flush();
        expect(x.answer?.eval.p_win).toBe(0.6);
        expect(x.pending).toBe(false);
        expect(calls).toHaveLength(2);
    });

    it('remembers the position and restores it', async () => {
        const store = memStore();
        const { engine } = manualEngine();
        const x = new ExplorerController(engine, store);
        x.init();
        x.tapStrike('opponent', 1);
        const y = new ExplorerController(manualEngine().engine, store);
        y.init();
        expect(y.pos.opponent.strikes).toBe(2);
    });

    it('loads a link over the remembered position, and keeps it on a bad link', () => {
        const store = memStore();
        const x = new ExplorerController(manualEngine().engine, store, undefined, () => {});
        x.init();
        x.tapSquare('you', 3, 3);
        const linked = toggleSquare(emptyPosition(), 'opponent', 1, 1);
        expect(x.loadCode(encode(linked))).toBe(true);
        expect(x.pos).toEqual(linked);
        expect(x.loadCode('1nonsense')).toBe(false);
        expect(x.pos).toEqual(linked);
        expect(x.notice).toBe(BAD_LINK);
    });

    it('moves the die picker to the next unset die and closes it when all are set', () => {
        const x = new ExplorerController(manualEngine().engine, memStore());
        x.init();
        x.openPicker(4);
        x.pickFace(3);
        expect(x.pos.dice[4]).toBe(3);
        expect(x.picker).toBe(5);
        for (const v of [1, 2, 3, 4, 5]) x.pickFace(v);
        expect(x.picker).toBeNull();
        expect(x.pos.dice.every((d) => d !== null)).toBe(true);
    });

    it('copies a link and says so', async () => {
        const copied: string[] = [];
        const x = new ExplorerController(manualEngine().engine, memStore(), async (t) => { copied.push(t); }, () => {});
        x.init();
        await x.copyLink('https://e.x/qwixxer/');
        expect(copied[0]).toBe(`https://e.x/qwixxer/#/explore?p=${encode(emptyPosition())}`);
        expect(x.notice).toBe('Link copied');
    });

    it('shows an engine failure and recovers on the next edit', async () => {
        const engine = { explore: vi.fn().mockRejectedValueOnce(new Error('boom')).mockResolvedValue(answer(0.4)) } as unknown as Engine;
        const x = new ExplorerController(engine, memStore());
        x.init();
        await flush();
        expect(x.error).toBe('boom');
        expect(x.pending).toBe(false);
        x.swap();
        await flush();
        expect(x.error).toBe('');
        expect(x.answer?.eval.p_win).toBe(0.4);
    });
});
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && npx vitest run src/game/explorer.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `explorer.svelte.ts`**

```ts
import type { Answer, Engine, Roller, View } from '../engine/protocol';
import type { Store } from './controller.svelte';
import { decode, encode } from './link';
import * as P from './position';
import type { ExPosition, Who } from './position';

const SAVE = 'qwixxer.explorer.v1';
export const BAD_LINK = "That link doesn't hold a valid position.";

/** Spec "Web: state explorer": the position being explored and its answer. */
export class ExplorerController {
    pos = $state<ExPosition>(P.emptyPosition());
    answer = $state<Answer | null>(null);
    pending = $state(false);
    error = $state('');
    notice = $state('');
    picker = $state<number | null>(null);
    showAll = $state(false);

    private engine: Engine;
    private store: Store;
    private clipboard: (text: string) => Promise<void>;
    private later: (f: () => void, ms: number) => void;
    private inflight = false;
    private dirty = false;
    private noticeId = 0;

    constructor(
        engine: Engine,
        store: Store,
        clipboard: (text: string) => Promise<void> = (t) => navigator.clipboard.writeText(t),
        later: (f: () => void, ms: number) => void = (f, ms) => void setTimeout(f, ms),
    ) {
        this.engine = engine;
        this.store = store;
        this.clipboard = clipboard;
        this.later = later;
    }

    /** The remembered position (saved as a link code), else an empty one. */
    init() {
        let saved: ExPosition | null = null;
        try {
            const raw = this.store.get(SAVE);
            saved = raw ? decode(raw) : null;
        } catch {
            saved = null;
        }
        this.set(saved ?? P.emptyPosition());
    }

    loadCode(code: string): boolean {
        const p = decode(code);
        if (!p) {
            this.flash(BAD_LINK);
            return false;
        }
        this.set(p);
        return true;
    }

    loadView(v: View) { this.set(P.fromView(v)); }

    tapSquare(who: Who, row: number, i: number) { this.set(P.toggleSquare(this.pos, who, row, i)); }
    tapStrike(who: Who, k: number) { this.set(P.tapStrike(this.pos, who, k)); }
    tapLock(row: number) { this.set(P.toggleLock(this.pos, row)); }
    clear(who: Who) { this.set(P.clearSheet(this.pos, who)); }
    clearAll() { this.set(P.clearSheets(this.pos)); }
    swap() { this.set(P.swapSides(this.pos)); }
    setRoller(r: Roller) { this.set(P.setRoller(this.pos, r)); }
    openPicker(i: number) { this.picker = this.picker === i ? null : i; }
    pickFace(v: number) {
        if (this.picker === null) return;
        const p = P.setDie(this.pos, this.picker, v);
        this.picker = P.nextUnsetDie(p, this.picker);
        this.set(p);
    }
    random() { this.picker = null; this.set(P.randomDice(this.pos)); }
    clearDice() { this.picker = null; this.set(P.clearDice(this.pos)); }

    linkFor(base: string) { return `${base}#/explore?p=${encode(this.pos)}`; }

    async copyLink(base: string) {
        try {
            await this.clipboard(this.linkFor(base));
            this.flash('Link copied');
        } catch {
            this.flash('Couldn’t copy the link');
        }
    }

    private flash(msg: string) {
        const id = ++this.noticeId;
        this.notice = msg;
        this.later(() => { if (this.noticeId === id) this.notice = ''; }, 3000);
    }

    private set(p: ExPosition) {
        this.pos = p;
        this.showAll = false;
        try { this.store.set(SAVE, encode(p)); } catch { /* storage unavailable */ }
        void this.evaluate();
    }

    /**
     * Spec "Engine": one call in flight; edits made meanwhile coalesce into
     * one call for the latest position once it returns. An answer for an
     * outdated position never lands.
     */
    private async evaluate() {
        this.pending = true;
        if (this.inflight) {
            this.dirty = true;
            return;
        }
        this.inflight = true;
        try {
            do {
                this.dirty = false;
                const a = await this.engine.explore(P.toEngine(this.pos));
                if (!this.dirty) {
                    this.answer = a;
                    this.error = '';
                }
            } while (this.dirty);
        } catch (err) {
            this.error = err instanceof Error ? err.message : String(err);
        } finally {
            this.inflight = false;
            this.pending = false;
        }
    }
}
```

- [ ] **Step 4: Run them to see them pass**

Run: `cd web && npx vitest run && npx svelte-check --threshold error`
Expected: PASS; 0 errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/game/explorer.svelte.ts web/src/game/explorer.test.ts
git commit -m "feat(web): ExplorerController — edits, coalesced evaluation, memory, links

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Explorer UI pieces

**Files:**
- Modify: `web/src/ui/LockCell.svelte` (optional `onTap` → a button), `web/src/ui/SheetRows.svelte` and `web/src/ui/SheetColumns.svelte` (optional `onLock`; `SheetColumns` gains `compact`)
- Create: `web/src/ui/StrikeBoxes.svelte`, `web/src/ui/PositionCard.svelte`, `web/src/ui/RollCard.svelte`, `web/src/ui/MovesList.svelte`, `web/src/ui/Tabs.svelte`

**Interfaces:**
- Consumes: `ExplorerController` (Task 5), `exploreSheet`, `percent`, `marginText`, `overText`, `closedRows` (Task 3), `Chip` component + `chipsFor` from `rules.ts`, `BOT_NAMES` from `protocol.ts`, `Die` component.
- Produces components used by Task 7:
  - `<LockCell look color size label onTap?={() => void} />`
  - `<SheetRows rows size onTap? onLock?={(row: number) => void} />`, `<SheetColumns rows size onTap onLock? compact? />`
  - `<StrikeBoxes count size label onTap={(k: number) => void} />`
  - `<PositionCard x={ExplorerController} />`
  - `<RollCard x={ExplorerController} compact? />`
  - `<MovesList x={ExplorerController} compact? />`
  - `<Tabs current={'play' | 'explore'} onPlay onExplore />`

- [ ] **Step 1: `LockCell` as a button when tappable**

Replace the `<div class="lock …">…</div>` element with a branch (keep the SVG and the existing comment), add `onTap` to the props, and add `border: 0; padding: 0;` at the start of the `.lock` rule (later rules `.locking` / `.pencil` still set their borders):

```svelte
<script lang="ts">
    import type { LockLook } from '../game/rules';
    let { look, color, size, label, onTap }: { look: LockLook; color: string; size: number; label: string; onTap?: () => void } = $props();
    const stroke = $derived(/* unchanged */);
</script>

{#snippet icon()}
    <svg viewBox="0 0 24 24" width={size * 0.47} height={size * 0.47} fill="none" stroke={stroke} stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="5" y="11" width="14" height="10" rx="2"></rect><path d="M8 11V8a4 4 0 0 1 8 0v3"></path>
    </svg>
{/snippet}
{#if onTap}
    <button type="button" class="lock {look}" aria-label={label} aria-pressed={look === 'locked_by' || look === 'locked_other'} style="--s: {size}px; --ring: {size >= 40 ? 3 : 2}px; cursor: pointer" onclick={onTap}>{@render icon()}</button>
{:else}
    <div class="lock {look}" role="img" aria-label={label} style="--s: {size}px; --ring: {size >= 40 ? 3 : 2}px">{@render icon()}</div>
{/if}
```

- [ ] **Step 2: `onLock` and `compact` on the sheets**

`SheetRows.svelte`: props gain `onLock?: (row: number) => void`; the lock becomes `<LockCell look={row.lock} color={row.color} {size} label="{row.label} lock" onTap={onLock ? () => onLock(row.row) : undefined} />`.

`SheetColumns.svelte`: props gain `onLock?: (row: number) => void; compact?: boolean`; same `onTap` on `LockCell`; root `<div class="cols" class:compact>`; add styles:

```css
    /* Explorer phone: two sheets side by side, so tighter paper and bands. */
    .cols.compact { gap: 3px; padding: 4px; }
    .compact .band { padding: 3px; }
```

- [ ] **Step 3: `StrikeBoxes.svelte`**

```svelte
<script lang="ts">
    import MarkX from './MarkX.svelte';
    // Spec "Editing · Strike boxes": tap box k → k+1 strikes, or k if it was the last taken.
    let { count, size = 32, label, onTap }: { count: number; size?: number; label: string; onTap: (k: number) => void } = $props();
</script>

<div class="boxes" role="group" aria-label="{label} strikes">
    {#each [0, 1, 2, 3] as k}
        <button type="button" class="box" class:faded={k >= count} style="--s: {size}px" aria-label="{label}, strike {k + 1}{k < count ? ', taken' : ''}" aria-pressed={k < count} onclick={() => onTap(k)}>
            {#if k < count}<MarkX />{/if}
        </button>
    {/each}
</div>

<style>
    .boxes { display: flex; gap: 5px; }
    .box { position: relative; box-sizing: border-box; width: var(--s); height: var(--s); border-radius: 7px; padding: 0; border: 2px solid var(--ink); background: var(--paper); cursor: pointer; }
    .box.faded { opacity: 0.45; }
</style>
```

- [ ] **Step 4: `PositionCard.svelte`**

```svelte
<script lang="ts">
    import type { ExplorerController } from '../game/explorer.svelte';
    import { marginText, overText, percent } from '../game/position';
    let { x, compact = false }: { x: ExplorerController; compact?: boolean } = $props();
    const a = $derived(x.answer);
</script>

<section class="card" class:compact class:stale={x.pending} aria-label="Position" aria-busy={x.pending}>
    {#if a?.over}
        <h2>Game over</h2>
        <b class="big">{overText(a.over)}</b>
    {:else if a}
        {#if !compact}<div class="top"><h2>Position</h2><span class="muted">before the roll · Master+’s view</span></div>{/if}
        <div class="line"><b class="big">{percent(a.eval.p_win, 0)}</b><span class="what">chance you win</span>
            {#if !compact}<span class="margin">{marginText(a.eval)}</span>{/if}</div>
        <div class="bar" role="img" aria-label="You {percent(a.eval.p_win, 0)}"><span style="width: {a.eval.p_win * 100}%"></span></div>
        {#if compact}<span class="margin">{marginText(a.eval)} · before the roll</span>{/if}
    {:else}
        <span class="muted">{x.error ? `Something went wrong: ${x.error}` : 'Thinking…'}</span>
    {/if}
    {#if a && x.error}<span class="err">Something went wrong: {x.error}</span>{/if}
</section>

<style>
    .card { display: flex; flex-direction: column; gap: 8px; background: var(--paper); border-radius: 14px; padding: 12px 16px; box-shadow: 0 1px 0 var(--edge); transition: opacity 0.15s; }
    .card.compact { gap: 7px; padding: 10px 14px; }
    .stale { opacity: 0.6; }
    .top { display: flex; align-items: baseline; justify-content: space-between; }
    h2 { margin: 0; font-family: var(--cond); font-size: 20px; font-weight: 700; }
    .line { display: flex; align-items: baseline; gap: 10px; }
    .big { font-family: var(--cond); font-weight: 800; font-size: 40px; line-height: 1; }
    .compact .big { font-size: 36px; }
    .what { font-weight: 600; font-size: 16px; }
    .margin { margin-left: auto; font-size: 14px; color: var(--soft); }
    .compact .margin { margin-left: 0; font-size: 13px; }
    .muted { font-size: 13px; color: var(--muted); }
    .err { font-size: 13px; color: var(--strike-red); }
    .bar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--btn-muted-bg); }
    .compact .bar { height: 8px; }
    .bar span { background: var(--ink); }
</style>
```

- [ ] **Step 5: `RollCard.svelte`**

```svelte
<script lang="ts">
    import type { ExplorerController } from '../game/explorer.svelte';
    import { closedRows } from '../game/position';
    import Die from './Die.svelte';
    let { x, compact = false }: { x: ExplorerController; compact?: boolean } = $props();
    const FACES = [['var(--die-face)', 'var(--die-pip)', 'White'], ['var(--die-face)', 'var(--die-pip)', 'White'],
        ['var(--red-band)', 'var(--on-band)', 'Red'], ['var(--yellow-band)', 'var(--on-yellow)', 'Yellow'],
        ['var(--green-band)', 'var(--on-band)', 'Green'], ['var(--blue-band)', 'var(--on-band)', 'Blue']];
    const size = $derived(compact ? 40 : 52);
    const closed = $derived(closedRows(x.pos));
    // Colour dice are ignored on the opponent's roll and for closed rows: greyed.
    const out = (i: number) => i >= 2 && (x.pos.roller === 'opponent' || closed[i - 2]);
    const d = $derived(x.pos.dice);
    const white = $derived(d[0] !== null && d[1] !== null ? String(d[0] + d[1]) : '–');
</script>

<section class="card" class:compact aria-label="Roll">
    <div class="top">
        {#if !compact}<h2>Roll</h2>{/if}
        <div class="seg" role="radiogroup" aria-label="Whose roll">
            <button type="button" role="radio" aria-checked={x.pos.roller === 'you'} onclick={() => x.setRoller('you')}>Your roll</button>
            <button type="button" role="radio" aria-checked={x.pos.roller === 'opponent'} onclick={() => x.setRoller('opponent')}>{compact ? 'Opponent’s' : 'Opponent’s roll'}</button>
        </div>
        <button type="button" class="tool" aria-label="Random roll" onclick={() => x.random()}>
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="4"></rect><circle cx="8.5" cy="8.5" r="1.2" fill="currentColor"></circle><circle cx="15.5" cy="15.5" r="1.2" fill="currentColor"></circle><circle cx="15.5" cy="8.5" r="1.2" fill="currentColor"></circle><circle cx="8.5" cy="15.5" r="1.2" fill="currentColor"></circle></svg>
            {#if !compact}Random{/if}
        </button>
        <button type="button" class="tool muted" aria-label="Clear dice" onclick={() => x.clearDice()}>{compact ? '✕' : 'Clear'}</button>
    </div>
    <div class="dice">
        {#if compact}<div class="sum" role="status" aria-label="White sum {white}"><span>Sum</span><b>{white}</b></div>{/if}
        {#each FACES as f, i}
            <div class="slot">
                <button type="button" class="diebtn" class:open={x.picker === i} aria-expanded={x.picker === i}
                    aria-label="{f[2]} die, {d[i] ?? 'not set'}. Choose a face" onclick={() => x.openPicker(i)}>
                    {#if d[i] === null}
                        <span class="unset" style="--s: {size}px; --c: {i < 2 ? 'var(--faint)' : f[0]}">?</span>
                    {:else}
                        <Die value={d[i] ?? 0} face={f[0]} pip={f[1]} {size} out={out(i)} label="{f[2]} die, {d[i]}" />
                    {/if}
                </button>
                {#if x.picker === i}
                    <div class="picker" role="group" aria-label="Set the {f[2].toLowerCase()} die">
                        {#each [1, 2, 3, 4, 5, 6] as v}
                            <button type="button" aria-label="Face {v}" onclick={() => x.pickFace(v)}>{v}</button>
                        {/each}
                    </div>
                {/if}
            </div>
        {/each}
        {#if !compact}<div class="wsum"><span>White sum</span><b>{white}</b></div>{/if}
    </div>
</section>

<style>
    .card { display: flex; flex-direction: column; gap: 10px; background: var(--paper); border-radius: 14px; padding: 12px 16px; box-shadow: 0 1px 0 var(--edge); }
    .card.compact { padding: 10px 12px; }
    .top { display: flex; align-items: center; gap: 10px; }
    h2 { margin: 0; font-family: var(--cond); font-size: 20px; font-weight: 700; }
    .seg { display: flex; gap: 2px; padding: 3px; border-radius: 10px; background: var(--btn-muted-bg); }
    .seg button { height: 32px; padding: 0 12px; border-radius: 7px; border: none; background: transparent; color: var(--muted); font-weight: 600; font-size: 14px; cursor: pointer; }
    .seg button[aria-checked='true'] { background: var(--paper); color: var(--ink); font-weight: 700; box-shadow: 0 1px 0 var(--edge); }
    .tool { height: 36px; min-width: 40px; padding: 0 12px; border-radius: 9px; border: none; background: var(--ground); color: var(--ink); display: flex; align-items: center; justify-content: center; gap: 6px; font-weight: 600; font-size: 14px; cursor: pointer; }
    .top .tool:first-of-type { margin-left: auto; }
    .tool.muted { color: var(--muted); }
    .dice { display: flex; align-items: center; gap: 10px; }
    .compact .dice { gap: 6px; }
    .slot { position: relative; }
    .diebtn { display: block; padding: 0; border: none; background: none; border-radius: 12px; cursor: pointer; }
    .diebtn.open { outline: 3px solid var(--ink); outline-offset: 2px; }
    .unset { box-sizing: border-box; width: var(--s); height: var(--s); border-radius: calc(var(--s) * 0.21); border: 2px dashed var(--c); display: flex; align-items: center; justify-content: center; font-family: var(--cond); font-weight: 700; font-size: 22px; color: var(--muted); background: var(--paper); }
    .picker { position: absolute; z-index: 3; top: calc(100% + 8px); left: 0; display: flex; gap: 4px; padding: 6px; border-radius: 12px; background: var(--paper); box-shadow: 0 0 0 1px var(--edge), 0 10px 30px var(--shadow); }
    .slot:nth-last-child(-n + 4) .picker { left: auto; right: 0; }
    .picker button { width: 44px; height: 44px; border-radius: 9px; border: none; background: var(--ground); color: var(--ink); font-family: var(--cond); font-weight: 700; font-size: 22px; cursor: pointer; }
    .wsum, .sum { display: flex; flex-direction: column; gap: 2px; margin-left: 6px; }
    .wsum span, .sum span { font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .wsum b { font-family: var(--cond); font-weight: 800; font-size: 30px; line-height: 1; }
    .sum { min-height: 40px; min-width: 56px; margin: 0; flex-direction: row; align-items: center; justify-content: center; gap: 5px; border-radius: 10px; background: var(--ground); }
    .sum b { font-family: var(--cond); font-weight: 800; font-size: 24px; line-height: 1; }
</style>
```

- [ ] **Step 6: `MovesList.svelte`**

```svelte
<script lang="ts">
    import { BOT_NAMES } from '../engine/protocol';
    import type { ExplorerController } from '../game/explorer.svelte';
    import { percent } from '../game/position';
    import { chipsFor } from '../game/rules';
    import ChipView from './Chip.svelte';
    let { x, compact = false }: { x: ExplorerController; compact?: boolean } = $props();
    const moves = $derived(x.answer?.moves ?? null);
    const shown = $derived(moves ? (x.showAll ? moves : moves.slice(0, 6)) : []);
    const noDice = $derived(x.pos.dice.some((d) => d === null));
</script>

<section class="card" class:compact class:stale={x.pending} aria-label="Moves">
    <div class="top">
        <h2>Best moves</h2>
        {#if moves}<span class="muted">{moves.length} legal{compact ? '' : ' · your chance to win after each'}</span>{/if}
    </div>
    {#if noDice}
        <div class="empty">
            <b>Set the dice to rank the moves</b>
            <span>Tap a die to choose its face, or press Random.</span>
        </div>
    {:else if moves}
        <ol>
            {#each shown as m, i}
                <li class:best={i === 0}>
                    <span class="rank">{i + 1}</span>
                    <div class="what">
                        <div class="chips">
                            {#each chipsFor(m.marks, m.strike) as ch}<ChipView chip={ch} />{/each}
                            {#if m.strike}<span class="txt">Strike</span>{:else if m.pass}<span class="txt">Pass</span>{/if}
                            {#if !compact}{#each m.bots as b}<span class="tag" class:plus={b === 'master_plus'}>{BOT_NAMES[b]}</span>{/each}{/if}
                        </div>
                        {#if compact && m.bots.length}<div class="tags">{#each m.bots as b}<span class="tag" class:plus={b === 'master_plus'}>{BOT_NAMES[b]}</span>{/each}</div>{/if}
                    </div>
                    <div class="pct">
                        <b>{percent(m.p_win, 1)}</b>
                        <span class="bar"><span style="width: {m.p_win * 100}%"></span></span>
                    </div>
                </li>
            {/each}
        </ol>
        {#if moves.length > 6 && !x.showAll}
            <button type="button" class="all" onclick={() => (x.showAll = true)}>Show all {moves.length}</button>
        {/if}
    {/if}
</section>

<style>
    .card { display: flex; flex-direction: column; gap: 8px; background: var(--paper); border-radius: 14px; padding: 12px 16px; box-shadow: 0 1px 0 var(--edge); transition: opacity 0.15s; }
    .card.compact { padding: 10px 12px; gap: 7px; }
    .stale { opacity: 0.6; }
    .top { display: flex; align-items: center; gap: 10px; }
    h2 { margin: 0; font-family: var(--cond); font-size: 20px; font-weight: 700; }
    .muted { font-size: 13px; color: var(--muted); }
    .empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; padding: 24px 16px; text-align: center; color: var(--muted); font-size: 14px; }
    .empty b { color: var(--soft); font-size: 16px; }
    ol { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
    li { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: 10px; background: var(--log-bg); }
    li.best { background: var(--paper); box-shadow: inset 0 0 0 2px var(--ink); }
    .rank { width: 18px; font-family: var(--cond); font-weight: 700; font-size: 17px; color: var(--muted); }
    .what { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 5px; }
    .chips, .tags { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; }
    .txt { font-size: 14px; font-weight: 600; }
    .tag { height: 22px; padding: 0 7px; border-radius: 11px; display: inline-flex; align-items: center; font-size: 12px; font-weight: 700; box-shadow: inset 0 0 0 1.5px var(--muted); color: var(--soft); box-sizing: border-box; }
    .tag.plus { background: var(--ink); color: var(--ground); box-shadow: none; }
    .pct { width: 84px; display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }
    .pct b { font-family: var(--cond); font-weight: 800; font-size: 22px; line-height: 1; font-variant-numeric: tabular-nums; }
    .bar { width: 100%; height: 5px; border-radius: 3px; background: var(--btn-muted-bg); display: flex; overflow: hidden; }
    .bar span { background: var(--faint); }
    .best .bar span { background: var(--ink); }
    .all { align-self: flex-start; height: 36px; padding: 0 4px; border: none; background: none; color: var(--soft); font-weight: 600; font-size: 14px; text-decoration: underline; text-underline-offset: 3px; cursor: pointer; }
</style>
```

- [ ] **Step 7: `Tabs.svelte`**

```svelte
<script lang="ts">
    // Desktop Play / Explore switch (spec "Navigation"); the phone uses the ⋯ menu.
    let { current, onPlay, onExplore }: { current: 'play' | 'explore'; onPlay: () => void; onExplore: () => void } = $props();
</script>

<nav aria-label="Main" class="tabs">
    <a href="#/" aria-current={current === 'play' ? 'page' : undefined} onclick={(e) => { e.preventDefault(); onPlay(); }}>Play</a>
    <a href="#/explore" aria-current={current === 'explore' ? 'page' : undefined} onclick={(e) => { e.preventDefault(); onExplore(); }}>Explore</a>
</nav>

<style>
    .tabs { display: flex; gap: 2px; padding: 3px; border-radius: 12px; background: var(--btn-muted-bg); }
    a { min-height: 38px; padding: 0 18px; border-radius: 9px; display: flex; align-items: center; font-weight: 600; font-size: 15px; text-decoration: none; color: var(--muted); }
    a[aria-current='page'] { background: var(--paper); color: var(--ink); font-weight: 700; box-shadow: 0 1px 0 var(--edge); }
</style>
```

- [ ] **Step 8: Check and commit**

Run: `cd web && npx svelte-check --threshold error && npx vitest run && npx vite build`
Expected: 0 errors; tests PASS; build succeeds. (The new components are mounted in Task 7.)

```bash
git add web/src/ui
git commit -m "feat(web): explorer UI pieces — tappable locks, strike boxes, position, roll and moves cards, tabs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Explorer layouts

**Files:**
- Create: `web/src/ui/ExplorerDesktop.svelte`, `web/src/ui/ExplorerPhone.svelte`, `web/src/ui/ExSheet.svelte`
- Modify: `web/src/game/layout.ts` (+ `layout.test.ts`): `explorerPhoneSize`

**Interfaces:**
- Consumes: Task 6 components, `ExplorerController`, `GameController` (for the menu and theme), `Nav` (defined here, implemented in Task 8), `deskScale`, `DESK_W`, `DESK_H`.
- Produces:

```ts
// web/src/game/layout.ts
/** Square size of each explorer sheet on the phone (two side by side). */
export function explorerPhoneSize(innerWidth: number): number;
// web/src/ui/nav.ts
export interface Nav { play(): void; explorer(): void; exploreThis(): void; current: 'play' | 'explore' }
```
- `<ExplorerDesktop c x nav />`, `<ExplorerPhone c x nav />`.

- [ ] **Step 1: Write the failing layout test**

Append to `web/src/game/layout.test.ts`:

```ts
import { explorerPhoneSize } from './layout';

describe('explorerPhoneSize', () => {
    it('fits two compact sheets side by side, 24–40 px', () => {
        expect(explorerPhoneSize(390)).toBe(34); // 8s + 114 ≤ 390
        expect(explorerPhoneSize(320)).toBe(25);
        expect(explorerPhoneSize(1000)).toBe(39); // the 430 px phone column caps it
        expect(explorerPhoneSize(200)).toBe(24);
    });
});
```

(If `layout.test.ts` already imports from `./layout`, merge the import.)

- [ ] **Step 2: Run it to see it fail**

Run: `cd web && npx vitest run src/game/layout.test.ts`
Expected: FAIL (`explorerPhoneSize` is not exported).

- [ ] **Step 3: Implement `explorerPhoneSize`**

Append to `layout.ts`:

```ts
/**
 * Explorer phone: two compact SheetColumns side by side. Each is 4 s + 41 px
 * wide (paper padding 2×4, band padding 4×2×3, 3 gaps of 3), plus an 8 px gap
 * and 2×12 page padding: 8 s + 114 within the (≤ 430 px) column.
 */
export function explorerPhoneSize(innerWidth: number): number {
    const w = Math.min(innerWidth, 430);
    return Math.max(24, Math.min(40, Math.floor((w - 114) / 8)));
}
```

Run the test again. Expected: PASS.

- [ ] **Step 4: `nav.ts` and `ExSheet.svelte` (desktop sheet card)**

`web/src/ui/nav.ts`:

```ts
/** Navigation between the game and the explorer (implemented by App). */
export interface Nav {
    current: 'play' | 'explore';
    play(): void;
    explorer(): void;
    exploreThis(): void;
}
```

`web/src/ui/ExSheet.svelte`:

```svelte
<script lang="ts">
    import type { ExplorerController } from '../game/explorer.svelte';
    import { exploreSheet, rowPoints, sheetTotal, type Who } from '../game/position';
    import { cellIndex } from '../game/rules';
    import SheetRows from './SheetRows.svelte';
    import StrikeBoxes from './StrikeBoxes.svelte';
    let { x, who }: { x: ExplorerController; who: Who } = $props();
    const name = $derived(who === 'you' ? 'You' : 'Opponent');
    const s = $derived(x.pos[who]);
    const rows = $derived(exploreSheet(x.pos, who));
    const breakdown = $derived(rowPoints(s).join(' + ') + (s.strikes ? ` − ${5 * s.strikes}` : ''));
</script>

<section class="card" aria-label="{name} sheet">
    <div class="top">
        <h2>{name}</h2>
        <b class="total">{sheetTotal(s)}</b>
        <span class="muted">{breakdown}</span>
        <div class="right">
            <span class="label">Strikes</span>
            <StrikeBoxes count={s.strikes} label={name} onTap={(k) => x.tapStrike(who, k)} />
            <button type="button" class="clear" onclick={() => x.clear(who)}>Clear</button>
        </div>
    </div>
    <SheetRows {rows} size={42} onTap={(cell) => x.tapSquare(who, cell.sq.row, cellIndex(cell.sq.row, cell.sq.n))} onLock={(r) => x.tapLock(r)} />
</section>

<style>
    .card { display: flex; flex-direction: column; gap: 8px; background: var(--paper); border-radius: 14px; padding: 12px 14px; box-shadow: 0 1px 0 var(--edge), 0 8px 24px var(--shadow); }
    .top { display: flex; align-items: center; gap: 12px; }
    h2 { margin: 0; font-family: var(--cond); font-size: 22px; font-weight: 700; }
    .total { font-family: var(--cond); font-weight: 800; font-size: 22px; }
    .muted { font-size: 13px; color: var(--muted); }
    .right { margin-left: auto; display: flex; align-items: center; gap: 6px; }
    .label { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-right: 2px; }
    .clear { height: 34px; margin-left: 8px; padding: 0 12px; border-radius: 9px; border: none; background: var(--ground); color: var(--muted); font-weight: 600; font-size: 14px; cursor: pointer; }
</style>
```

(`cellIndex` maps a square to its index in the marks array.)

- [ ] **Step 5: `ExplorerDesktop.svelte`**

```svelte
<script lang="ts">
    import type { ExplorerController } from '../game/explorer.svelte';
    import type { GameController } from '../game/controller.svelte';
    import { DESK_H, DESK_W, deskScale } from '../game/layout';
    import ExSheet from './ExSheet.svelte';
    import Menu from './Menu.svelte';
    import MovesList from './MovesList.svelte';
    import type { Nav } from './nav';
    import PositionCard from './PositionCard.svelte';
    import RollCard from './RollCard.svelte';
    import Tabs from './Tabs.svelte';
    let { c, x, nav }: { c: GameController; x: ExplorerController; nav: Nav } = $props();
    let w = $state(window.innerWidth);
    let h = $state(window.innerHeight);
    const scale = $derived(deskScale(w, h));
    const base = () => location.origin + location.pathname;
</script>

<svelte:window bind:innerWidth={w} bind:innerHeight={h} />
<div class="fit" style="width: {Math.floor(DESK_W * scale)}px; height: {Math.floor(DESK_H * scale)}px">
<div class="desk" style="transform: scale({scale})">
    <header class="head">
        <div class="left"><div class="word">Qwixxer</div><Tabs current="explore" onPlay={nav.play} onExplore={nav.explorer} /></div>
        <div class="right">
            {#if x.notice}<span class="notice" role="status">{x.notice}</span>{/if}
            <button type="button" class="btn" onclick={() => x.copyLink(base())}>
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"></path><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"></path></svg>
                Copy link
            </button>
            <Menu {c} {nav} />
        </div>
    </header>
    <div class="body">
        <main>
            <ExSheet {x} who="you" />
            <div class="mid">
                <button type="button" class="btn small" onclick={() => x.swap()}>
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 4v16M7 20l-3-3M7 20l3-3M17 20V4M17 4l-3 3M17 4l3 3"></path></svg>
                    Swap sides
                </button>
                <span class="hint">Tap squares, locks and strike boxes to set up any position.</span>
            </div>
            <ExSheet {x} who="opponent" />
        </main>
        <aside>
            <PositionCard {x} />
            <RollCard {x} />
            <div class="moves"><MovesList {x} /></div>
        </aside>
    </div>
</div>
</div>

<style>
    .fit { margin: 0 auto; overflow: hidden; }
    .desk { width: 1280px; height: 800px; box-sizing: border-box; padding: 18px 32px 22px; display: flex; flex-direction: column; gap: 14px; overflow: hidden; transform-origin: top left; }
    .head { display: flex; align-items: center; justify-content: space-between; height: 44px; }
    .left, .right { display: flex; align-items: center; gap: 20px; }
    .right { gap: 10px; }
    .word { font-family: var(--cond); font-weight: 800; font-size: 30px; letter-spacing: 0.5px; }
    .notice { font-size: 14px; color: var(--soft); }
    .btn { height: 44px; padding: 0 14px; border-radius: 12px; border: none; background: var(--paper); box-shadow: 0 1px 0 var(--edge); color: var(--ink); display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 15px; cursor: pointer; }
    .btn.small { height: 34px; padding: 0 12px; border-radius: 9px; font-size: 14px; gap: 6px; }
    .body { display: flex; gap: 24px; flex: 1; min-height: 0; }
    main { width: 640px; display: flex; flex-direction: column; gap: 10px; }
    .mid { display: flex; align-items: center; gap: 10px; }
    .hint { font-size: 13px; color: var(--muted); }
    aside { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 12px; min-height: 0; }
    .moves { flex: 1; min-height: 0; overflow-y: auto; border-radius: 14px; }
</style>
```

- [ ] **Step 6: `ExplorerPhone.svelte`**

```svelte
<script lang="ts">
    import type { ExplorerController } from '../game/explorer.svelte';
    import type { GameController } from '../game/controller.svelte';
    import { explorerPhoneSize } from '../game/layout';
    import { exploreSheet, sheetTotal, type Who } from '../game/position';
    import { cellIndex } from '../game/rules';
    import Menu from './Menu.svelte';
    import MovesList from './MovesList.svelte';
    import type { Nav } from './nav';
    import PositionCard from './PositionCard.svelte';
    import RollCard from './RollCard.svelte';
    import SheetColumns from './SheetColumns.svelte';
    import StrikeBoxes from './StrikeBoxes.svelte';
    let { c, x, nav }: { c: GameController; x: ExplorerController; nav: Nav } = $props();
    let w = $state(window.innerWidth);
    const size = $derived(explorerPhoneSize(w));
    const base = () => location.origin + location.pathname;
    const sides: { who: Who; name: string }[] = [{ who: 'you', name: 'You' }, { who: 'opponent', name: 'Opponent' }];
</script>

<svelte:window bind:innerWidth={w} />
<div class="xphone">
    <header class="head">
        <div class="word">Explorer</div>
        <div class="right">
            <button type="button" class="icon" aria-label="Copy link to this position" onclick={() => x.copyLink(base())}>
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"></path><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"></path></svg>
            </button>
            <Menu {c} {nav} />
        </div>
    </header>
    {#if x.notice}<div class="notice" role="status">{x.notice}</div>{/if}
    <PositionCard {x} compact />
    <div class="sheets">
        {#each sides as sd (sd.who)}
            {@const rows = exploreSheet(x.pos, sd.who)}
            <section class="side" aria-label="{sd.name} sheet">
                <div class="name"><h2>{sd.name}</h2><b>{sheetTotal(x.pos[sd.who])}</b></div>
                <SheetColumns {rows} {size} compact onTap={(cell) => x.tapSquare(sd.who, cell.sq.row, cellIndex(cell.sq.row, cell.sq.n))} onLock={(r) => x.tapLock(r)} />
                <StrikeBoxes count={x.pos[sd.who].strikes} size={34} label={sd.name} onTap={(k) => x.tapStrike(sd.who, k)} />
            </section>
        {/each}
    </div>
    <div class="row">
        <button type="button" class="btn" onclick={() => x.swap()}>Swap sides</button>
        <button type="button" class="btn muted" onclick={() => x.clearAll()}>Clear sheets</button>
    </div>
    <RollCard {x} compact />
    <MovesList {x} compact />
</div>

<style>
    .xphone { max-width: 430px; margin: 0 auto; box-sizing: border-box; padding: 12px 12px 20px; display: flex; flex-direction: column; gap: 10px; }
    .head { display: flex; align-items: center; justify-content: space-between; height: 44px; }
    .word { font-family: var(--cond); font-weight: 800; font-size: 26px; letter-spacing: 0.5px; }
    .right { display: flex; align-items: center; gap: 8px; }
    .icon { width: 44px; height: 44px; border-radius: 12px; border: none; background: var(--paper); box-shadow: 0 1px 0 var(--edge); color: var(--ink); display: flex; align-items: center; justify-content: center; padding: 0; cursor: pointer; }
    .notice { font-size: 13px; color: var(--soft); }
    .sheets { display: flex; gap: 8px; justify-content: center; }
    .side { display: flex; flex-direction: column; align-items: center; gap: 6px; }
    .name { align-self: stretch; display: flex; align-items: baseline; justify-content: space-between; padding: 0 4px; }
    .name h2 { margin: 0; font-family: var(--cond); font-size: 19px; font-weight: 700; }
    .name b { font-family: var(--cond); font-weight: 800; font-size: 20px; }
    .row { display: flex; gap: 8px; }
    .btn { flex: 1; height: 40px; border-radius: 10px; border: none; background: var(--paper); box-shadow: 0 1px 0 var(--edge); color: var(--ink); font-weight: 600; font-size: 14px; cursor: pointer; }
    .btn.muted { color: var(--muted); }
</style>
```

- [ ] **Step 7: Check and commit**

`Menu` does not take `nav` until Task 8, so svelte-check will report the extra prop. Add `nav` to Menu's props now as optional (`nav?: Nav`), unused until Task 8, so this task type-checks on its own.

Run: `cd web && npx svelte-check --threshold error && npx vitest run && npx vite build`
Expected: 0 errors, tests PASS, build succeeds.

```bash
git add web/src/ui web/src/game/layout.ts web/src/game/layout.test.ts
git commit -m "feat(web): explorer desktop and phone layouts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Routing, menu and tabs

**Files:**
- Create: `web/src/game/route.ts`, `web/src/game/route.test.ts`
- Modify: `web/src/game/keys.ts` (+ `keys.test.ts`), `web/src/ui/App.svelte`, `web/src/ui/Menu.svelte`, `web/src/ui/Header.svelte`, `web/src/ui/Desktop.svelte`, `web/src/ui/Phone.svelte`

**Interfaces:**
- Consumes: `Nav` (Task 7), `ExplorerController.loadCode/loadView/init`, `Tabs` (Task 6).
- Produces: `parseRoute(hash: string): Route` where `type Route = { view: 'play' } | { view: 'explore'; code: string | null }`; `KeyTarget.explore?: boolean`.

- [ ] **Step 1: Write the failing tests**

`web/src/game/route.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { parseRoute } from './route';

describe('parseRoute', () => {
    it('reads the game, the explorer and an explorer link', () => {
        expect(parseRoute('')).toEqual({ view: 'play' });
        expect(parseRoute('#')).toEqual({ view: 'play' });
        expect(parseRoute('#/')).toEqual({ view: 'play' });
        expect(parseRoute('#/explore')).toEqual({ view: 'explore', code: null });
        expect(parseRoute('#/explore?p=1abc')).toEqual({ view: 'explore', code: '1abc' });
        expect(parseRoute('#/explore?x=1')).toEqual({ view: 'explore', code: null });
        expect(parseRoute('#/elsewhere')).toEqual({ view: 'play' });
    });
});
```

Append to `web/src/game/keys.test.ts` (inside the `describe`):

```ts
    it('never confirms or rolls the hidden game from the explorer', () => {
        const ex = { field: false, keyboardFocus: false, explore: true };
        expect(keyAction('Enter', ex, 'active')).toBe(null);
        expect(keyAction(' ', ex, 'you_to_roll')).toBe(null);
        expect(keyAction('Escape', ex, 'active')).toBe('close');
    });
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && npx vitest run src/game/route.test.ts src/game/keys.test.ts`
Expected: FAIL (route module missing; Enter returns 'confirm').

- [ ] **Step 3: Implement `route.ts` and the key rule**

`web/src/game/route.ts`:

```ts
/** Spec "Navigation": `#/explore` (optionally `?p=<code>`) is the explorer; anything else the game. */
export type Route = { view: 'play' } | { view: 'explore'; code: string | null };

export function parseRoute(hash: string): Route {
    const m = /^#\/explore(?:\?(.*))?$/.exec(hash);
    if (!m) return { view: 'play' };
    return { view: 'explore', code: new URLSearchParams(m[1] ?? '').get('p') };
}
```

`keys.ts`: add to `KeyTarget`:

```ts
    /** The explorer is showing: no game keys at all; Escape closes the menu or die picker. */
    explore?: boolean;
```

and at the top of `keyAction`:

```ts
    if (t.explore) return key === 'Escape' ? 'close' : null;
```

Run the tests again. Expected: PASS.

- [ ] **Step 4: Menu items per view**

`Menu.svelte`: props become `{ c, nav }: { c: GameController; nav: Nav }` (import `type { Nav } from './nav'`). Replace the item list with (keep the existing New game button, dark mode, Rules, Source code and notice markup as they are, reordered):

```svelte
            {#if nav.current === 'play'}
                <!-- existing New game button, unchanged -->
                <button type="button" role="menuitem" class="item" onclick={() => { c.menuOpen = false; nav.exploreThis(); }}>Explore this position</button>
                <button type="button" role="menuitem" class="item" onclick={() => { c.menuOpen = false; nav.explorer(); }}>Explorer</button>
            {:else}
                <button type="button" role="menuitem" class="item" bind:this={first} onclick={() => { c.menuOpen = false; nav.play(); }}>Back to game</button>
            {/if}
            <!-- existing Dark mode, Rules, Source code, legal notice -->
```

`bind:this={first}` stays on New game in the play branch (it is the first item); in the explorer branch it is on Back to game. Update the comment at the top of the script to list the items per view.

- [ ] **Step 5: Tabs in the game's desktop header; nav through the layouts**

`Header.svelte`: props `{ c, nav }: { c: GameController; nav: Nav }`; wrap the wordmark:

```svelte
    <div class="left"><div class="word">Qwixxer</div><Tabs current="play" onPlay={nav.play} onExplore={nav.explorer} /></div>
```

with `.left { display: flex; align-items: center; gap: 20px; }`, and pass `{nav}` to `<Menu>`.

`Desktop.svelte` and `Phone.svelte`: add `nav: Nav` to their props, pass `{nav}` to `<Header>` (desktop) and to `<Menu>` (phone).

- [ ] **Step 6: Route in `App.svelte`**

```svelte
<script lang="ts">
    import { onMount } from 'svelte';
    import { createEngine } from '../engine/client';
    import { GameController } from '../game/controller.svelte';
    import { ExplorerController } from '../game/explorer.svelte';
    import { keyAction } from '../game/keys';
    import { parseRoute, type Route } from '../game/route';
    import Desktop from './Desktop.svelte';
    import ExplorerDesktop from './ExplorerDesktop.svelte';
    import ExplorerPhone from './ExplorerPhone.svelte';
    import type { Nav } from './nav';
    import Phone from './Phone.svelte';

    const store = { /* unchanged */ };
    // One worker for both: the game and the explorer share the engine.
    const engine = createEngine();
    const c = new GameController(engine, store);
    const x = new ExplorerController(engine, store);
    // …mq / phone / drawer unchanged…
    let route = $state<Route>({ view: 'play' });

    // A link (`?p=`) loads once, then the address becomes plain #/explore so
    // a reload keeps your edits instead of reloading the link.
    function sync() {
        const r = parseRoute(location.hash);
        if (r.view === 'explore' && r.code !== null) {
            x.loadCode(r.code);
            history.replaceState(null, '', '#/explore');
            route = { view: 'explore', code: null };
        } else route = r;
        c.menuOpen = false;
        x.picker = null;
    }
    const nav: Nav = {
        get current() { return route.view; },
        play: () => { history.pushState(null, '', location.pathname + location.search); sync(); },
        explorer: () => { location.hash = '#/explore'; },
        exploreThis: () => { if (c.view) x.loadView(c.view); location.hash = '#/explore'; },
    };

    onMount(() => {
        c.init();
        x.init();
        sync();
        const f = () => { phone = mq.matches; if (!phone) drawer = false; };
        mq.addEventListener('change', f);
        window.addEventListener('hashchange', sync);
        window.addEventListener('popstate', sync);
        return () => { mq.removeEventListener('change', f); window.removeEventListener('hashchange', sync); window.removeEventListener('popstate', sync); };
    });
```

In `onKey`, pass `explore: route.view === 'explore'` in the `KeyTarget`, and handle `'close'` as: `if (c.menuOpen) c.menuOpen = false; else if (x.picker !== null) x.picker = null; else drawer = false;`.

Markup:

```svelte
{#if route.view === 'explore'}
    {#if phone}<ExplorerPhone {c} {x} {nav} />{:else}<ExplorerDesktop {c} {x} {nav} />{/if}
{:else if phone}<Phone {c} {nav} bind:drawer />{:else}<Desktop {c} {nav} />{/if}
```

- [ ] **Step 7: Check, build, try it**

Run: `cd web && npx svelte-check --threshold error && npx vitest run && npx vite build`
Expected: 0 errors, all tests PASS, build succeeds.

Then run the dev server (`cd web && npx vite --port 3000`) and check by hand, desktop width and a 390 px phone width, light and dark:
- The game header shows Play/Explore tabs on desktop; Explore opens the explorer; Play returns to the same game.
- "⋯ → Explore this position" during your turn copies the boards, dice and "Your roll"; during the bot's roll, "Opponent's roll".
- Tapping squares, locks (by tap and by marking 5 + end number), strike boxes, Swap sides, Clear, dice picker, Random and Clear dice all update the position and the answer.
- "Copy link", then opening the copied URL in a new tab shows the same position; a mangled `?p=` shows "That link doesn't hold a valid position."
- Enter and Space in the explorer do nothing to the game.

- [ ] **Step 8: Commit**

```bash
git add web/src
git commit -m "feat(web): explorer routing — #/explore, Play/Explore tabs, menu items, links

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Docs

**Files:**
- Modify: `README.md` ("Web UI" section), `web/README.md`, `docs/ARCHITECTURE.md` (the web section that says the explorer was removed and deferred), `docs/superpowers/specs/2026-10-14-state-explorer-design.md` (status line)

- [ ] **Step 1: Update the docs**

- `README.md` "Web UI": add one sentence: "An explorer (`#/explore`) evaluates any two-sheet position and, for a roll, ranks your moves and shows which one each bot would pick; positions can be shared as links."
- `web/README.md`: describe `#/explore`, `web/src/explorer.rs` / `WasmExplorer`, `game/position.ts`, `game/link.ts`, `game/explorer.svelte.ts`, and the explorer components; mention the `explore` worker command.
- `docs/ARCHITECTURE.md`: replace "`WebGame` … and the State Explorer page were removed; an explorer is deferred." with a short description of the new explorer (engine call, link codec, shared worker).
- Spec status line: "**Status:** implemented (branch `explorer`)."

- [ ] **Step 2: Final checks and commit**

Run: `cargo test --release --manifest-path web/Cargo.toml && cd web && npx vitest run && npx svelte-check --threshold error && npx vite build`
Expected: all PASS. Revert `web/Cargo.lock` if touched.

```bash
git add README.md web/README.md docs/ARCHITECTURE.md docs/superpowers/specs/2026-10-14-state-explorer-design.md
git commit -m "docs: state explorer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
