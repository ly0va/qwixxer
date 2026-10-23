# Game Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A chess.com-style review page for finished games against the bot: luck vs mistakes, a clickable win-chance graph, an eval bar, the move log, stepping, and a shareable game code.

**Architecture:** The game code and its replay live in Rust (`web/src/gamecode.rs`), so the rules check every code; the session records each roll's code value as it plays. A per-roll analysis (`web/src/review.rs`, `impl Explorer`) reuses the explorer's move values and luck tables, plus exact 1-ply for the roller's top candidates. The worker exposes `gameCode`, `reviewGame`, `reviewRoll`; a `ReviewController` runs the analysis roll by roll while the page is on screen; pure helpers in `game/review.ts` turn results into points, totals, sheets and log lines for the Svelte views.

**Tech Stack:** Rust (engine, wasm-bindgen), TypeScript, Svelte 5 (runes), Vitest.

**Spec:** `docs/superpowers/specs/2026-10-24-game-review-design.md`. Mock-ups: design canvas https://claude.ai/artifact/4eDmxXF5jEskvwxcFS26im, page "Review" (R1 desktop, R3 row B5 pencil, R4/R5 phone, R6/R7 dark, R8 analysing, R9 shared).

## Global Constraints

- Format only the Rust files you touch: `rustfmt --edition 2021 <file>`. Never `cargo fmt`.
- Don't push. Commit on branch `game-review`.
- Code: version `1`; header char bit 0 = bot rolls first, bits 1–2 = bot level as the index in `BotKind::ALL` (Master+, Master, Student, Newbie), bits 3–5 = 0; then exactly 4 chars per roll, 24 bits big-endian; alphabet `A–Z a–z 0–9 - _` (value = index).
- Roll value = `dice × 225 + action`; `dice` = base-6 digits (d − 1) in the order white 1, white 2, red, yellow, green, blue, first die most significant; `action = rollerWhite × 45 + rollerColour × 5 + otherWhite`; `rollerWhite`/`otherWhite` 0 = none, 1–4 = row 0–3 (red, yellow, green, blue); `rollerColour` 0 = none, else 1 + row × 2 + whiteDie.
- Units: P(win) as a fraction in the engine and TS; shown as points (× 100, one decimal) or percent.
- Names: own game "You / your", bot "Master"-style name from `BOT_NAMES`; a shared link shows "Player / Player's".
- Red dots on the graph: your mistakes costing ≥ 0.02 (2 pts). Candidates for 1-ply: the roller's static top 3 plus the played turn.
- Route `#/review?g=<code>`; last finished game in local storage key `qwixxer.lastGame`.
- Full CI before the final commit: root `cargo test --release`; `cd web && cargo test --release`; `npm test`; `npm run check`; `./web/build.sh`; `npx vite build`.

## Review Focus

1. A hostile or mangled `?g=` code (bad char, wrong length, illegal move, rolls after the game ended, non-canonical action) → an error message on the page, never a panic, hang or blank page. (Task 1 rejection tests; Task 6 `load` error test.)
2. Opening another game while one is analysing → the old game's results never land in the new one. (Task 6 stale test.)
3. An engine failure mid-analysis (the worker is reset by New game after an error) → the page shows the error with a retry, not "Analysing…" forever. (Task 6 error test.)
4. A game that ends in phase 1 of its last roll, or ends in a draw → replay and the last roll's analysis still work; after the move is exactly 0, ½ or 1. (Task 1 and Task 3 tests.)
5. A game saved before this change and resumed after it has no recorded rolls → no Analyse button for it, nothing breaks. (Task 2 test.)

---

### Task 1: Game code and replay (Rust)

**Files:**
- Create: `web/src/gamecode.rs`
- Modify: `web/src/lib.rs` (add `pub mod gamecode;`), `web/src/session.rs` (`SheetSnap::mark` → `pub(crate)`)

**Interfaces:**
- Consumes: `explorer::{Position, Roller, Turn, Over, states, legal_turns, after}`, `selection::passive_options`, `session::{SheetSnap, Sq, RollRecord, Who}`, `bots::BotKind`, `qwixx::state::{State, Mark}` (whatever path `explorer.rs` uses).
- Produces:
  - `pub struct Header { pub bot_first: bool, pub bot: BotKind }`
  - `pub fn pack(dice: [u8; 6], action: u32) -> u32` / `pub fn unpack(v: u32) -> Result<([u8; 6], u32), String>`
  - `pub fn turn_of(dice: [u8; 6], action: u32) -> (Vec<Mark>, Option<Mark>)` — the roller's marks (white first) and the other player's white-sum mark.
  - `pub fn canonical_action(roller: &State, other: &State, dice: [u8; 6], turn: &Turn, other_white: Option<Mark>) -> Option<u32>` — the lowest action whose `turn_of` gives the same marks (as a set) and the same strike/pass, and which is legal; `None` if no action does.
  - `pub fn encode(h: &Header, rolls: &[u32]) -> String`
  - `pub fn decode(code: &str) -> Result<(Header, Vec<u32>), String>` — syntax only.
  - `#[derive(Serialize)] pub struct ReviewRoll { pub before: Position, pub record: RollRecord, pub after: Position }` — `before` has the dice and the roller; `after` has `dice: None` and the next roller; `locked` always `[false; 4]` (rows close from marks).
  - `#[derive(Serialize)] pub struct ReviewGame { pub bot: &'static str /* BotKind::id */, pub bot_first: bool, pub rolls: Vec<ReviewRoll>, pub over: Option<Over> }`
  - `pub fn replay(code: &str) -> Result<ReviewGame, String>`

- [ ] **Step 1: Write the failing tests** in `gamecode.rs` `mod tests`:
  - `pack_unpack_round_trip_at_the_edges`: dice `[1;6]` action 0 → 0; dice `[6;6]` action 224 → 10_497_599; `unpack(10_497_600)` is `Err`.
  - `encode_decode_round_trip`: header `{bot_first: true, bot: BotKind::Student}` and rolls `[0, 10_497_599, 12345]` → `encode` starts with `"1"`, has length 2 + 12, and `decode` returns the same.
  - `decode_rejects_bad_codes`: `""`, `"2A"` (version), `"1"` (no header), `"1AAAA"` (3 chars of a roll), `"1A!AAA"` (bad char), a header with bit 3 set → each `Err`.
  - `turn_of_reads_the_fields`: dice `[3,4,2,5,6,1]`, action `2*45 + (1 + 1*2 + 1)*5 + 3` → roller marks `[yellow 7, yellow 9]`, other white `green 7`.
  - `equal_white_dice_take_white_die_0`: with dice `[4,4,…]`, `canonical_action` of a colour mark returns the whiteDie-0 action.
  - `replay_round_trips_self_play`: play 20 seeded games with `Session` (Master+ bot, human choices random legal: `you_options` / `passive_options` like `bench.rs`), build each code from `Session` history (Task 2 adds it; for this task, build the code in the test by calling `canonical_action` per roll before applying it), `replay` it, and assert: roll count, every `record` equals the session's `RollRecord`, the final `after` sheets equal the session snapshot's sheets, and `over` matches.
  - `replay_rejects_illegal_and_trailing_rolls`: flip one roll to an illegal action (a white mark in a row whose sum square is already crossed out) → `Err`; append a roll after the game is over → `Err`; a non-canonical action (whiteDie 1 with equal white dice) → `Err`.
  - `a_prefix_is_an_unfinished_game`: the first 5 rolls of a finished game's code replay with `over: None`.
  - `phase_one_end_skips_the_colour`: a hand-built code whose last roll closes a second row with the white sum (both rows ready) replays, `over` is `Some`, and a code that adds a colour mark to that roll is `Err`.

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && cargo test --release --lib gamecode`
Expected: compile errors (module/functions missing).

- [ ] **Step 3: Implement `gamecode.rs`.** Replay algorithm per roll `k` (roller = bot when `(k % 2 == 0) == bot_first`):
  1. `(you, bot, _) = states(&pos)`; frame `(r, o)` = roller first.
  2. `(marks, other_white) = turn_of(dice, action)`; turn = `Turn { marks, strike: false }`; if `marks` is empty it is the legal one of strike/pass in `legal_turns(&r, &o, dice, Roller::You, other_white)`.
  3. Reject unless that turn's `key()` is in `legal_turns(...)`, `other_white` is `None` or in `passive_options(&o, dice)`, and `canonical_action(...) == Some(action)`.
  4. Apply to the `SheetSnap`s (roller marks, strike, other mark), build the `RollRecord` (your marks in `you`: your turn on your roll, your white mark on the bot's; `locked` = rows locked after but not before), then check over: `strikes >= 4` or `n_locked(...) >= 2` → `Over { you, opponent }` points; any roll after that is `Err("roll after the game ended")`.
  Make `Turn::key` `pub(crate)` if needed.

- [ ] **Step 4: Run the tests**

Run: `cd web && cargo test --release --lib gamecode`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
rustfmt --edition 2021 web/src/gamecode.rs web/src/lib.rs web/src/session.rs web/src/explorer.rs
git add web/src/gamecode.rs web/src/lib.rs web/src/session.rs web/src/explorer.rs
git commit -m "feat(review): game code with replay validation"
```

---

### Task 2: The session records the game code

**Files:**
- Modify: `web/src/session.rs`, `web/src/lib.rs` (`WasmGame`)

**Interfaces:**
- Consumes: `gamecode::{canonical_action, pack, encode, Header}`.
- Produces:
  - `Snapshot.history: Option<Vec<u32>>` with `#[serde(default)]`: `Some(vec![])` for new games, `None` for snapshots saved before this change; one value pushed per resolved roll (in `submit_passive` and `submit_active`, computed from the states before the roll's marks).
  - `Session::code(&self, bot: BotKind) -> Result<String, String>` — `Err` when `history` is `None`. Header `bot_first: true` (every game starts with `Phase::BotToRoll`).
  - `WasmGame` keeps its `BotKind` and exposes `pub fn code(&self) -> Result<String, JsError>`.

- [ ] **Step 1: Write the failing tests** in `session.rs` tests:
  - `history_replays_to_the_same_game`: 10 seeded games played to the end as in Task 1's self-play test; `gamecode::replay(&s.code(BotKind::Master)?)` gives the same records and final sheets; `bot` is `"master"`.
  - `old_snapshots_have_no_code`: a snapshot JSON without `history` restores, plays on, and `code()` is `Err`.
  - `history_survives_a_snapshot_round_trip`: mid-game snapshot → JSON → restore → finish → code replays.
  - Change Task 1's `replay_round_trips_self_play` to use `Session::code`.
  - Add `#[test] #[ignore] fn sample_code()` printing a seeded finished game's code (for screenshots in Tasks 8–9).

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && cargo test --release --lib session`
Expected: compile errors (`history`, `code` missing).

- [ ] **Step 3: Implement** the field, the recording (bot roll: roller turn = `bot_white` + `bot_colour` or strike, other = your pick; your roll: roller turn = your selection or strike/pass, other = `bot_white`), `Session::code`, and `WasmGame::code`.

- [ ] **Step 4: Run the tests**

Run: `cd web && cargo test --release --lib`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
rustfmt --edition 2021 web/src/session.rs web/src/lib.rs web/src/gamecode.rs
git add web/src/session.rs web/src/lib.rs web/src/gamecode.rs
git commit -m "feat(review): the game records its code"
```

---

### Task 3: Per-roll analysis (Rust)

**Files:**
- Create: `web/src/review.rs` (`impl Explorer`)
- Modify: `web/src/explorer.rs` (make `net`, `move_values`, `turn_values`, `average_top`, `top`, `luck` helpers `pub(crate)` as needed), `web/src/lib.rs` (`pub mod review;`)

**Interfaces:**
- Consumes: `gamecode::{ReviewGame, ReviewRoll}`, explorer internals above.
- Produces:
  - `#[derive(Serialize)] pub struct BestMove { pub marks: Vec<Sq>, pub strike: bool, pub after: Position }`
  - `#[derive(Serialize)] pub struct RollAnalysis { pub start: Option<f32>, pub this_roll: f32, pub after_move: f32, pub you_cost: f32, pub you_best: BestMove, pub bot_cost: f32 }`
  - `impl Explorer { pub fn review_roll(&self, game: &ReviewGame, k: usize) -> Result<RollAnalysis, String> }`
  - All values in the player's ("you") P(win). `start` only for `k == 0`: E[f] of roll 1.
  - `fn one_ply(&self, after: &Position) -> f32`: game over → 1.0 / 0.0 / 0.5 by points; else `average_top(you, opp, after.roller, 0..21) as f32`.

Algorithm for roll `k` (`before`, `record`, `after` from `game.rolls[k]`):
- `this_roll` = `self.luck(&before, 0, 21)?.this_roll`.
- `after_move` = `one_ply(&after)` (the played turn's 1-ply value).
- Roller's candidates, in the roller's frame `(r, o)` with the other player's **actual** white mark `ow`: `legal_turns(&r, &o, dice, Roller::You, ow)` valued by `turn_values(..., Roller::You, ow, dice)`; take the top 3 by value plus the played turn (dedupe by `key()`); for each, the player-frame position after it (apply the turn and `ow`, `dice: None`, next roller) and its `one_ply` (reuse `after_move` for the played turn; candidates leaving identical sheets share one value). Roller = you: best = max, cost = best − played. Roller = bot: best = min, cost = played − best.
- Other player's white mark: your mark on the bot's roll: `move_values(&you, &bot, dice, Roller::Opponent)`, cost = top − value of the played key, best = top; the bot's mark on your roll: `move_values(&bot, &you, dice, Roller::Opponent)` (bot's frame), cost = top − played.
- `you_cost` / `bot_cost`: on your roll the roller cost / the bot's passive cost; on the bot's roll your passive cost / the roller cost.
- `you_best`: on your roll the best roller candidate; on the bot's roll your best passive mark; `after` = `before` with that move and everything else as it happened (on the bot's roll, the bot's actual marks, skipping any that the new lock makes illegal).

- [ ] **Step 1: Write the failing tests** in `review.rs` tests, on a seeded self-play game from `Session` (Master bot) replayed with `gamecode::replay`:
  - `after_move_is_the_next_rolls_luck_average`: for every `k < n−1`, `a.after_move == self.luck(&rolls[k+1].before, 0, 21).average as f32` (exact `==`).
  - `start_is_roll_ones_luck_average`: `a0.start == Some(luck(&rolls[0].before, 0, 21).average as f32)`; `start` is `None` for `k > 0`.
  - `this_roll_is_the_explorer_luck`: equals `luck(&before, 0, 21).this_roll`.
  - `one_ply_matches_brute_force`: for one mid-game candidate position, `one_ply` equals the weighted mean over `rolls()` of the next roller's `top` within 1e-5.
  - `costs_are_non_negative_and_best_is_a_candidate`: every `you_cost >= 0.0`, `bot_cost >= 0.0`; `you_best.after` contains `you_best.marks`.
  - `passive_cost_matches_the_explorer_list`: on a bot roll, `you_cost` equals the explorer's `move_values` top minus your played mark's value.
  - `the_last_roll_is_terminal`: `after_move` of the last roll is 0.0, 0.5 or 1.0.
  - `played_best_has_zero_cost`: in a game where the human always plays the 1-ply best (build it by playing `you_best` each roll), `you_cost == 0.0`.

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && cargo test --release --lib review`
Expected: compile errors.

- [ ] **Step 3: Implement `review.rs`** as above.

- [ ] **Step 4: Run the tests**

Run: `cd web && cargo test --release --lib`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
rustfmt --edition 2021 web/src/review.rs web/src/explorer.rs web/src/lib.rs
git add web/src/review.rs web/src/explorer.rs web/src/lib.rs
git commit -m "feat(review): per-roll analysis with 1-ply move costs"
```

---

### Task 4: Engine plumbing (WASM, worker, protocol)

**Files:**
- Modify: `web/src/lib.rs`, `web/src/engine/serve.ts`, `web/src/engine/protocol.ts`, `web/src/engine/client.ts`
- Test: `web/src/engine/serve.test.ts`, `web/src/engine/client.test.ts`

**Interfaces:**
- Produces (WASM): `WasmExplorer::review_game(code: &str) -> Result<String, JsError>` (JSON `ReviewGame`), `WasmExplorer::review_roll(code: &str, k: u32) -> Result<String, JsError>` (replays then `review_roll`), `WasmGame::code()` (Task 2).
- Produces (TS, `protocol.ts`):

```ts
export interface ReviewRoll { before: Position; record: RollRecord; after: Position }
export interface ReviewGame { bot: BotKind; bot_first: boolean; rolls: ReviewRoll[]; over: { you: number; opponent: number } | null }
export interface BestMove { marks: Sq[]; strike: boolean; after: Position }
export interface RollAnalysis { start: number | null; this_roll: number; after_move: number; you_cost: number; you_best: BestMove; bot_cost: number }
// Engine:
gameCode(): Promise<string>;
reviewGame(code: string): Promise<ReviewGame>;
reviewRoll(code: string, k: number): Promise<RollAnalysis>;
```

- Worker commands `gameCode`, `reviewGame`, `reviewRoll` (the explorer instance is created on first use, as for `explore`).

- [ ] **Step 1: Write the failing tests:** in `serve.test.ts`, the fake wasm's `WasmExplorer` gains `review_game`/`review_roll` and `WasmGame` gains `code`; assert each command forwards its arguments and parses the JSON. In `client.test.ts`, `reviewRoll('1A', 3)` posts `{cmd: 'reviewRoll', args: ['1A', 3]}`.

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && npx vitest run src/engine`
Expected: FAIL (unknown command / missing method).

- [ ] **Step 3: Implement** the three exports, the worker cases and the client methods.

- [ ] **Step 4: Run the tests and the build**

Run: `cd web && npx vitest run src/engine && ./build.sh && npm run check`
Expected: pass, build exit 0, 0 errors.

- [ ] **Step 5: Commit**

```bash
rustfmt --edition 2021 web/src/lib.rs
git add web/src/lib.rs web/src/engine
git commit -m "feat(review): engine calls for the game code and review"
```

---

### Task 5: Review model (pure TS)

**Files:**
- Create: `web/src/game/review.ts`, `web/src/game/review.test.ts`
- Modify: `web/src/ui/StrikeBoxes.svelte` (optional `pencil?: number` index drawn as `<MarkX pencil />`; `onTap` optional, read-only when absent)

**Interfaces:**
- Consumes: Task 4 types; `game/position.ts` (`exploreSheet`, `toEngine`-style conversions), `game/rules.ts` (`chipsFor`, `LogEntry`), `game/link.ts` (`encode`).
- Produces (point index `j`: 0 = start, `2k+1` = after roll k's dice, `2k+2` = after roll k's move, k 0-based; `n` rolls → `2n+1` points):

```ts
export type Side = 'played' | 'best';
export interface Names { you: string; You: string; your: string; Your: string }   // own: you/You/your/Your; shared: Player/Player/Player's/Player's
export function names(shared: boolean): Names;
export function pointValue(an: (RollAnalysis | undefined)[], j: number): number | null;
export function rollLuck(an: (RollAnalysis | undefined)[], k: number): number | null;  // this_roll − (k ? an[k−1].after_move : an[0].start)
export function totals(an: (RollAnalysis | undefined)[], n: number): { luck: number; you: number; bot: number; done: number };
export function verdict(t: ReturnType<typeof totals>, n: number, youWon: boolean | null, nm: Names): string;
export function positionAt(game: ReviewGame, an: (RollAnalysis | undefined)[], j: number, side: Side): { pos: ExPosition; prev: ExPosition };
export function reviewSheet(prev: ExPosition, pos: ExPosition, who: 'you' | 'opponent'): RowModel[];   // read-only; squares marked in pos but not prev → 'pencil'
export interface ReviewEntry extends LogEntry { k: number; luck: string | null; costs: (string | null)[] }   // costs per line, same order as lines
export function reviewLog(game: ReviewGame, an: (RollAnalysis | undefined)[], nm: Names): ReviewEntry[];
export function stepLabel(j: number, n: number): { title: string; sub: string };   // "Roll 23 of 36", "after the move" | "after the roll" | "before the first roll"
export function explorerHref(game: ReviewGame, k: number): string;   // '#/explore?p=' + encode(position before roll k's marks, with dice and roller)
export function gameHref(code: string): string;   // '#/review?g=' + code
```

Copy rules (spec): luck pill `luck +3.2` / `luck −4.8` (U+2212) / `luck …` when pending; cost `−4.6` for costs ≥ 0.005, else none; verdict per spec ("Mostly the dice" when |luck| > |bot − you|, else "Mostly {your} mistakes" after a loss, "Mostly {your} play" after a win; while analysing `Analysing… {done} of {n}`). Start point position: both sheets empty, roller = bot when `bot_first`.

- [ ] **Step 1: Write the failing tests** in `review.test.ts` with a hand-built 3-roll `ReviewGame` and analyses:
  - `pointValue` for j = 0..6 maps to start, this_roll, after_move in order; missing analysis → `null`.
  - `rollLuck(an, 0) === an[0].this_roll − an[0].start`; `rollLuck(an, 1) === an[1].this_roll − an[0].after_move`.
  - `totals` sums only analysed rolls and reports `done`.
  - `verdict` returns each of the four strings; `names(true).Your === 'Player’s'`.
  - `positionAt(…, 2, 'played')` is roll 0's `after`; `'best'` is `an[0].you_best.after`; `prev` is roll 0's `before` (without dice) in both.
  - `reviewSheet` marks exactly the new squares `'pencil'`, all cells `clickable: false`.
  - `reviewLog` puts the cost on your line, `luck …` on pending rolls, and the bot line's cost too.
  - `stepLabel(46, 36)` → `{ title: 'Roll 23 of 36', sub: 'after the move' }`.
  - `explorerHref` decodes back (via `link.ts` `decode`) to roll k's `before` with its dice and roller.

- [ ] **Step 2: Run to see them fail**

Run: `cd web && npx vitest run src/game/review.test.ts`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `review.ts`** and the `StrikeBoxes` props.

- [ ] **Step 4: Run the tests**

Run: `cd web && npx vitest run src/game && npm run check`
Expected: pass, 0 errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/game/review.ts web/src/game/review.test.ts web/src/ui/StrikeBoxes.svelte
git commit -m "feat(review): points, totals, sheets and log for the review page"
```

---

### Task 6: ReviewController

**Files:**
- Create: `web/src/game/review.svelte.ts`, `web/src/game/review.controller.test.ts`

**Interfaces:**
- Consumes: `Engine` (Task 4), `review.ts` (Task 5), a store `{ get(k): string | null; set(k, v): void }` as `App.svelte` builds it.
- Produces:

```ts
export class ReviewController {
    constructor(engine: Engine, store: Store);
    code: string | null; game: ReviewGame | null; an: (RollAnalysis | undefined)[];
    sel: number; side: Side; error: string | null; shared: boolean;
    load(code: string | null, shared: boolean): Promise<void>;   // null → the last game from 'qwixxer.lastGame'; same code → no-op
    setActive(on: boolean): void;   // analysis runs only while active; resumes at the first missing roll
    select(j: number): void; step(d: 1 | -1): void; first(): void; last(): void;   // clamp to 0..2n; select resets side to 'played'
    setSide(s: Side): void;
    retry(): void;   // after an error: clear it and resume
}
```

- [ ] **Step 1: Write the failing tests** with a fake `Engine` whose `reviewRoll` resolves on demand:
  - `load` replays the game, `sel = 0`, then (active) asks `reviewRoll(code, 0)`, `(code, 1)`… in order, one at a time.
  - Not active → no `reviewRoll`; `setActive(true)` resumes at the first missing roll.
  - Stale: start loading code A, resolve its roll 0 after `load(B)` → `an` of B is untouched and B starts at roll 0 (Review Focus 2).
  - Error: `reviewRoll` rejects → `error` set, loop stops; `retry()` resumes (Review Focus 3).
  - `reviewGame` rejects (bad code) → `error` set, `game` null (Review Focus 1).
  - `load(null)` reads `qwixxer.lastGame`; with nothing stored, `game` stays null and no engine call.
  - `step`, `first`, `last` clamp; `select` resets `side`.

- [ ] **Step 2: Run to see them fail**

Run: `cd web && npx vitest run src/game/review.controller.test.ts`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** the controller (Svelte 5 `$state` fields, like `ExplorerController`; a generation counter drops stale replies).

- [ ] **Step 4: Run the tests**

Run: `cd web && npx vitest run src/game && npm run check`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/game/review.svelte.ts web/src/game/review.controller.test.ts
git commit -m "feat(review): controller that analyses roll by roll"
```

---

### Task 7: App wiring (route, nav, keys, Analyse button)

**Files:**
- Modify: `web/src/game/route.ts` (+ test), `web/src/ui/nav.ts`, `web/src/ui/Tabs.svelte`, `web/src/ui/Menu.svelte`, `web/src/game/keys.ts` (+ test), `web/src/game/controller.svelte.ts` (+ test), `web/src/ui/ActionBar.svelte`, `web/src/ui/Phone.svelte`, `web/src/ui/App.svelte`

**Interfaces:**
- Produces:
  - `Route` gains `{ view: 'review'; code: string | null }`; `parseRoute('#/review?g=1Ab')` → that; `#/review` → `code: null`. The address keeps `?g=` (the code is immutable, so a reload shows the same game).
  - `Nav.current: 'play' | 'explore' | 'review'`; `Nav.review(): void` (→ `#/review`).
  - Tabs: Play, Explore, Review. Phone menu: a "Review last game" item when a last game exists.
  - `KeyTarget.review?: boolean`; `KeyAction` adds `'prev' | 'next' | 'first' | 'last'` for ArrowLeft / ArrowRight / Home / End on the review page (not in fields).
  - `GameController.lastCode: string | null`: when a step leaves the game `over`, `engine.gameCode()` once, store under `qwixxer.lastGame`; on `Err` (old save) stay `null`.
  - Game-over UI (desktop `ActionBar` result, phone result): an "Analyse game" button when `lastCode` is set → `location.hash = gameHref(lastCode)`.
  - `App.svelte`: a `ReviewController`; `sync()` calls `r.load(route.code, route.code !== null && route.code !== store.get('qwixxer.lastGame'))` for review routes (shared = a code that isn't your own last game); activity → `r.setActive(route.view === 'review' && visible)`; keys dispatch to `r.step/first/last`; renders `ReviewDesktop` / `ReviewPhone` (Tasks 8–9; until then a placeholder `<p>`).

- [ ] **Step 1: Write the failing tests:** `route.test.ts` (both review forms, explore still works); `keys.test.ts` (arrow/Home/End on review, ignored in a field, no game keys on review); `controller.test.ts` (game over → `gameCode` called once, stored; rejected `gameCode` → `lastCode` null).

- [ ] **Step 2: Run to see them fail**

Run: `cd web && npx vitest run src/game`
Expected: FAIL on the new cases.

- [ ] **Step 3: Implement** the wiring listed above.

- [ ] **Step 4: Run the tests**

Run: `cd web && npm test && npm run check`
Expected: pass, 0 errors.

- [ ] **Step 5: Commit**

```bash
git add web/src
git commit -m "feat(review): route, tabs, keys and the Analyse game button"
```

---

### Task 8: Desktop review page

**Files:**
- Create: `web/src/ui/ReviewDesktop.svelte`, `web/src/ui/EvalGraph.svelte`, `web/src/ui/EvalBar.svelte`, `web/src/ui/ReviewSummary.svelte`, `web/src/ui/SelectedRoll.svelte`, `web/src/ui/ReviewLog.svelte`
- Modify: `web/src/ui/App.svelte` (render it)

**Interfaces:**
- Consumes: `ReviewController` (Task 6), `review.ts` (Task 5), `SheetRows`, `StrikeBoxes`, `Chip`, `Die`, `Tabs`, `deskScale` (the 1280×800 desk scales like `Desktop.svelte`).
- Produces:
  - `EvalGraph.svelte` props `{ values: (number | null)[]; luck: (number | null)[]; dots: number[]; selected: number; pending: string | null; width: number; height: number; luckHeight: number; onSelect: (j: number) => void }` — SVG; your share filled from the bottom (`--ink`, 0.88 opacity) on `--btn-muted-bg`; dashed 50% line; red dots (`--unlucky-ink`) at `dots`; selection line and ring in `--lucky-ink` with a percentage tag; the running-luck line below (green tint above zero, red below, ±0.40 full scale), same selection line; unanalysed part hatched with `pending` text at its top; one hit button per point with an `aria-label`. Also used by the phone (Task 9).
  - `EvalBar.svelte` props `{ p: number | null; vertical: boolean }` — desktop: 30 px wide beside the sheets, number at its foot; phone: 6 px edge to edge.
  - `SelectedRoll.svelte` props `{ r: ReviewController; compact: boolean }` — roll title, whose roll, luck chip (the explorer's chip styles), dice, the Played / Best rows as a `radiogroup` (Played row shows the cost, Best row the chips or "working it out…"), and on desktop the step buttons and "Open in explorer".
- Layout and copy: as board R1 (light), R7 (dark), R8 (analysing), R9 (shared): header with tabs and "Share game" (copies `location.origin + location.pathname + gameHref(code)`; a shared game also shows the pill "Shared game · Player vs {BOT_NAMES[bot]}"); left the bar, both sheets (`reviewSheet` with pencil, read-only `StrikeBoxes` with `pencil`), the selected-roll strip; right summary, graphs, log. An error shows the message and a Retry button in the summary card; no game → "Finish a game or open a shared link to review it."; an unfinished game (`over: null`) shows "Game not finished" in place of the result.

- [ ] **Step 1: Build the components and the page.**

- [ ] **Step 2: Check types and build**

Run: `cd web && npm run check && npx vite build`
Expected: 0 errors; build succeeds.

- [ ] **Step 3: Compare with the mock-ups.** Get a code: `cd web && cargo test --release --lib sample_code -- --ignored --nocapture`. Run Vite (`npx vite --port 5199`) and screenshot `http://localhost:5199/qwixxer/#/review?g=<code>` at 1280×800 with headless Firefox (light, dark via the menu, mid-analysis, and with a code that isn't the stored last game for the shared view). Check against R1, R7, R8, R9: the pencil marks, the switch, the graph selection, the "…" states, "Player" naming. Fix differences.

- [ ] **Step 4: Commit**

```bash
git add web/src/ui
git commit -m "feat(review): desktop review page"
```

---

### Task 9: Phone review page

**Files:**
- Create: `web/src/ui/ReviewPhone.svelte`
- Modify: `web/src/ui/App.svelte`

**Interfaces:**
- Consumes: the Task 8 components with `compact`, `SheetColumns` (the phone sheets, side by side), `Menu`.
- Layout (R4, R5, R6): header logo, share button, ⋯ menu (no other text); summary with tiles in one row ("Dice luck", "Your errors" / "Player's errors", "Bot errors"); the graphs at the card's width; the selected-roll card ("Explore" link on the Best row); both sheets as columns; the log; pinned to the bottom: `EvalBar` (6 px) and the step bar (first, previous, title + "after the move · 12% you win", next, last; 48 px buttons). The page leaves room for the pinned bar.

- [ ] **Step 1: Build `ReviewPhone.svelte` and wire it.**

- [ ] **Step 2: Check types and build**

Run: `cd web && npm run check && npx vite build`
Expected: 0 errors; build succeeds.

- [ ] **Step 3: Compare with the mock-ups** at 390×844 and 360×800 (no horizontal scroll at 360), light and dark, top of the page and scrolled to the sheets: R4, R5, R6.

- [ ] **Step 4: Commit**

```bash
git add web/src/ui
git commit -m "feat(review): phone review page"
```

---

### Task 10: Timing and docs

**Files:**
- Create: `web/scripts/review-bench.mjs` (like `luck-bench.mjs`: load the built pkg, replay a code, time `review_roll` for every roll; print mean / max ms per roll and the total)
- Modify: `docs/ARCHITECTURE.md` (a "Game review" section: the code, the session history, `review_roll`, the controller, the page; the measured timing with date), the spec's Status line

- [ ] **Step 1: Write the bench and run it**

Run: `cd web && ./build.sh && node scripts/review-bench.mjs <sample code>`
Expected: per-roll and total times printed. If the total is far above the spec's 20–30 s for a typical game, stop and report the numbers instead of changing the method.

- [ ] **Step 2: Update the docs** with the measured numbers.

- [ ] **Step 3: Full CI**

Run: `cargo test --release && cd web && cargo test --release && npm test && npm run check && ./build.sh && npx vite build`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add web/scripts/review-bench.mjs docs/ARCHITECTURE.md docs/superpowers/specs/2026-10-24-game-review-design.md
git commit -m "docs: game review architecture and timing"
```
