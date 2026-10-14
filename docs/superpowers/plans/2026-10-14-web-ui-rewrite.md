# Web UI Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the web UI in `web/` with the one-confirm, paper-sheet game against Master+ / Master / Student bots described in the spec, built on a new Rust game session that runs in a Web Worker.

**Architecture:** The web crate (`web/src/*.rs`) gets a pure-Rust `Session` (turn flow of `src/game.rs`, human legal selections, snapshots) wrapped by a thin wasm-bindgen `WasmGame`. The Svelte app talks to it only through a Web Worker (`engine.worker.ts`), so bot thinking never blocks animation. All game-to-screen logic (which squares are legal, cell looks, log chips, confirm label) is pure TypeScript in `src/game/rules.ts`; a runes-based `GameController` sequences turns, pacing, tips and persistence; Svelte components only render.

**Tech Stack:** Rust 2021 + wasm-bindgen + wasm-pack (existing), burn ndarray (existing), Svelte 5 + TypeScript + Vite + Vitest, `@fontsource/barlow` and `@fontsource/barlow-condensed`.

**Spec:** `docs/superpowers/specs/2026-10-13-web-turn-flow-design.md` (read it before Task 1). The design canvas named there is the visual reference; the colour tokens and sizes are copied into this plan.

## Global Constraints

- Work on branch `web-rewrite` (already created from `distillation`, spec merged in). Do not merge or push without the user's go-ahead.
- Rust formatting: run `rustfmt --edition 2021 <changed .rs files>`; never `cargo fmt` on the whole crate (it reformats unrelated files).
- Root crate tests: `cargo test --lib --release` (CI runs this). Web crate tests: `cd web && cargo test --release`.
- `cargo` commands inside `web/` may rewrite `web/Cargo.lock` without real dependency changes; if `git diff web/Cargo.lock` shows only noise, `git checkout web/Cargo.lock` before committing.
- Never commit `.superpowers/`, `AGENTS.md` or `web-ui.txt` (the user's untracked files). Do not edit `AGENTS.md`.
- Do not change committed model artifacts (`champion.txt`, `dqn_model/`, `pair_model/`).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Bots, names and default (spec "Implementation decisions"): **Master+** = pair-search (default), **Master** = pair net, **Student** = GA champion. Engine ids: `master_plus`, `master`, `student`.
- The bot starts every game.
- Rules come from the engine only; TypeScript never re-derives legality (spec "Implementation notes · Rules source").
- Fonts are self-hosted (`@fontsource/*`), no Google Fonts requests.
- Accessibility now = semantic markup only: real `<button>`s for squares, strike boxes and actions; labels like "Blue 12, available"; `aria-pressed` / `aria-disabled`. Inactive buttons use `aria-disabled`, never `disabled`, so pressing them can show a tip.
- Light tokens: ground `#F3EDE1`, paper `#FFFDF8`, edge `#D9D0BF`, ink `#1D1B18`, muted `#6B6459`, soft text `#4A443B`, faint `#A39B8D`, log block `#F6F1E7`, pencil `#8B857C`, muted button `#E6DED0` / `#8A8273`, strike red `#C8323C`, locked band `#C9C1B2`; rows (band / hatch-dark / number ink): red `#C8323C` / `#9E2530` / `#C8323C`, yellow `#E9B824` / `#C99A12` / `#9A7400`, green `#2F8F4E` / `#23703C` / `#2F8F4E`, blue `#2C5FAF` / `#214B8C` / `#2C5FAF`.
- Dark tokens: ground `#171513`, paper `#24211D`, edge `#0E0D0B`, ink `#ECE5D7`, muted `#A39B8D`, soft text `#C9C1B2`, faint `#6E675C`, log block `#1C1A17`, pencil `#8E877C`, muted button `#2F2B26` / `#7D766B`, strike red `#E0575E`, locked band `#3A352E`; rows: red `#A8343B` / `#7C252B` / `#F08A8F`, yellow `#C49A22` / `#977414` / `#F0CD62`, green `#2D7E4A` / `#215E37` / `#74C994`, blue `#2F5E9F` / `#234776` / `#8DB0EE`.
- Type: Barlow Condensed (numbers, headings, wordmark), Barlow (text). Wordmark "Qwixxer".
- Pacing minimums (spec "Pacing"): tumble 600 ms (faces change every 75 ms), bot think 500 ms, bot colour-mark stagger 450 ms, hand-over to the bot's roll 900 ms.
- Desktop layout targets 1280×800 without scrolling; phone layout (portrait, below 1100 px wide) targets 390×844.

## Review Focus

1. **Reload mid-turn** (during the bot's turn, while choosing, or during the 900 ms hand-over) must resume the same turn: same dice, same revealed bot mark, no lost or duplicated rolls. Tested in Task 3 (snapshot round trip in every phase) and Task 7 (restore resumes a pending bot roll).
2. **Double clicks while the worker is busy** (Confirm, Roll, New game pressed twice quickly) must submit once. Tested in Task 7.
3. **Changing the bot mid-game** must not affect the running game; it applies to the next one and survives reload. Tested in Task 7.
4. **Engine/worker failure** (wasm fails to load, a command throws) must show a readable message and a working New game, not a frozen page. Tested in Task 7.
5. **A game ending at phase 1** (human's white sum or the bot's revealed mark closes the second row; a fourth strike) must stop the turn: no bot colour mark, no strike, result shown. Tested in Task 2 and Task 3.

---

## File structure

Rust (web crate):

- `web/src/selection.rs` — human legal selections on their own turn and on the bot's turn. Pure functions over `State`.
- `web/src/session.rs` — `Session`: one human-vs-bot game, phases, applying moves, lock propagation, game end, `View`, `RollRecord`, `Snapshot`.
- `web/src/bots.rs` — `BotKind`, `make_bot`, embedded model bytes, `MASTER_PLUS_SPEC`.
- `web/src/bench.rs` — `bench` wasm export: per-decision latency of a bot.
- `web/src/lib.rs` — wasm-bindgen `WasmGame` wrapper (JSON in/out) and module declarations. Old `WebGame` and `StateExplorer` are deleted.
- `src/dqn/pair.rs` (root crate) — adds `PairStrategy::load_from_bytes`.

Frontend (`web/`):

- `index.html`, `vite.config.ts`, `tsconfig.json`, `package.json`, `svelte.config.js`.
- `src/main.ts` — mounts `App`.
- `src/styles/theme.css` — tokens (light/dark) and shared keyframes.
- `src/engine/protocol.ts` — TS mirror of the Rust JSON types and the `Engine` interface.
- `src/engine/engine.worker.ts` — owns the `WasmGame` inside the worker.
- `src/engine/client.ts` — `WorkerEngine` (promise RPC to the worker).
- `src/game/rules.ts` — pure model: legality lookups, cell looks, confirm label, log chips.
- `src/game/controller.svelte.ts` — `GameController`: turn sequencing, pacing, tips, persistence.
- `src/ui/*.svelte` — components (Task 8 desktop, Task 9 phone).
- `scripts/bench.mjs` — runs the wasm `bench` in Node.
- Deleted: `src/ts/`, `explorer.html`, old `src/lib.rs` contents.

---

### Task 1: `PairStrategy::load_from_bytes` (root crate)

The web build cannot read files, so the pair net must load from embedded bytes, as `DqnStrategy::load_from_bytes` already does.

**Files:**
- Modify: `src/dqn/pair.rs` (impl `PairStrategy`, near `load` at line ~276; tests module at the end of the file)

**Interfaces:**
- Produces: `PairStrategy::load_from_bytes(model_bytes: &[u8]) -> PairStrategy`

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` in `src/dqn/pair.rs` (create the module if the file has none, with `use super::*;`):

```rust
    #[test]
    fn load_from_bytes_matches_file_load() {
        use crate::strategy::Bot;
        let from_file = PairStrategy::load("pair_model");
        let from_bytes = PairStrategy::load_from_bytes(include_bytes!("../../pair_model/model.mpk"));
        let mut a = State::default();
        a.apply_mark(crate::state::Mark { row: 0, number: 4 });
        let b = State::default();
        for (s, o) in [(a, b), (b, a)] {
            let x = from_file.evaluate(&s, &[o]);
            let y = from_bytes.evaluate(&s, &[o]);
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
    }
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test --lib --release load_from_bytes_matches_file_load`
Expected: compile error, `no function or associated item named load_from_bytes`.

- [ ] **Step 3: Implement**

In `impl PairStrategy`, after `load`:

```rust
    /// Load the model from embedded bytes (the web build has no file system).
    /// Same format as `load`: `CompactRecorder` = named MessagePack, half precision.
    pub fn load_from_bytes(model_bytes: &[u8]) -> Self {
        use burn::record::{HalfPrecisionSettings, NamedMpkBytesRecorder, Recorder};
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let record = NamedMpkBytesRecorder::<HalfPrecisionSettings>::new()
            .load(model_bytes.to_vec(), &device)
            .expect("Failed to load pair model from bytes");
        let model = PairModelConfig::new().init::<MyBackend>(&device).load_record(record);
        Self::from_model(model, device)
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test --lib --release load_from_bytes_matches_file_load`
Expected: PASS. Then `cargo test --lib --release` → all pass.

- [ ] **Step 5: Format and commit**

```bash
rustfmt --edition 2021 src/dqn/pair.rs
git add src/dqn/pair.rs
git commit -m "feat(pair): load the pair net from embedded bytes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Human legal selections (`web/src/selection.rs`)

Implements spec "Which squares are legal": a selection is a set of 0–2 squares; it is legal if some split into white sum (judged on the pre-roll board) plus colour mark (judged after both players' phase-1 marks and lock propagation) exists.

**Files:**
- Create: `web/src/selection.rs`
- Modify: `web/src/lib.rs` (add `pub mod selection;` at the top; leave the rest for Task 4)

**Interfaces:**
- Consumes: `qwixxer::state::{Mark, State}`: `generate_white_moves(u8) -> Vec<Mark>`, `generate_color_moves([u8; 6]) -> Vec<Mark>`, `apply_mark(Mark)`, `locked() -> [bool; 4]`, `lock([bool; 4])`.
- Produces:
  - `pub struct ActiveOptions { pub selections: Vec<Vec<Mark>>, pub strike: bool, pub pass: bool }` (each selection in application order: `[white]`, `[colour]` or `[white, colour]`; never empty)
  - `pub fn active_options(you: &State, bot: &State, bot_white: Option<Mark>, dice: [u8; 6]) -> ActiveOptions`
  - `pub fn passive_options(you: &State, dice: [u8; 6]) -> Vec<Mark>`
  - `pub fn union(a: [bool; 4], b: [bool; 4]) -> [bool; 4]` and `pub fn n_locked(l: [bool; 4]) -> usize`

- [ ] **Step 1: Write the failing tests**

Create `web/src/selection.rs` with only the tests first:

```rust
//! The human's legal selections (spec "Which squares are legal").

use qwixxer::state::{Mark, State};

#[cfg(test)]
mod tests {
    use super::*;

    fn m(row: usize, number: u8) -> Mark {
        Mark { row, number }
    }

    /// A state with `nums` marked in `row`, in order.
    fn with(mut s: State, row: usize, nums: &[u8]) -> State {
        for &n in nums {
            s.apply_mark(m(row, n));
        }
        s
    }

    fn has(opts: &ActiveOptions, sel: &[Mark]) -> bool {
        opts.selections.iter().any(|x| {
            x.len() == sel.len() && sel.iter().all(|s| x.contains(s))
        })
    }

    #[test]
    fn white_twelve_locking_second_row_ends_after_phase_one() {
        // Yellow is already locked; your white 12 closes red: no colour mark may follow.
        let mut you = with(State::default(), 0, &[2, 3, 4, 5, 6]);
        you.lock([false, true, false, false]);
        let mut bot = State::default();
        bot.lock([false, true, false, false]);
        let opts = active_options(&you, &bot, None, [6, 6, 1, 1, 1, 1]);
        assert!(has(&opts, &[m(0, 12)]));
        assert!(!opts.selections.iter().any(|x| x.len() == 2 && x.contains(&m(0, 12))));
        assert!(opts.strike);
        assert!(!opts.pass);
    }

    #[test]
    fn bot_reveal_locking_second_row_leaves_only_white_or_pass() {
        let mut you = with(State::default(), 0, &[2, 3]);
        you.lock([false, true, false, false]);
        let mut bot = with(State::default(), 0, &[2, 3, 4, 5, 6]);
        bot.lock([false, true, false, false]);
        let opts = active_options(&you, &bot, Some(m(0, 12)), [6, 6, 1, 1, 1, 1]);
        assert!(opts.pass);
        assert!(!opts.strike);
        assert!(!opts.selections.is_empty());
        assert!(opts.selections.iter().all(|x| x.len() == 1 && x[0].number == 12));
    }

    #[test]
    fn bot_locking_a_row_keeps_your_white_sum_but_not_colour_there() {
        let you = with(State::default(), 0, &[2, 3, 4, 5, 6]);
        let bot = with(State::default(), 0, &[2, 3, 4, 5, 6]);
        // White 6+6 = 12; red die 2 → red 8 (or 8 again).
        let opts = active_options(&you, &bot, Some(m(0, 12)), [6, 6, 2, 1, 1, 1]);
        assert!(has(&opts, &[m(0, 12)]));
        assert!(!opts.selections.iter().any(|x| x.contains(&m(0, 8))));
        assert!(opts.strike);
    }

    #[test]
    fn same_row_pair_needs_white_on_the_left() {
        // White 3+5 = 8; red colour sums 3+1 = 4 and 5+1 = 6.
        let opts = active_options(&State::default(), &State::default(), None, [3, 5, 1, 1, 1, 1]);
        assert!(has(&opts, &[m(0, 8)]));
        assert!(has(&opts, &[m(0, 4)]));
        // Red 4 + red 8 would need the colour mark (4) before the white sum (8).
        assert!(!has(&opts, &[m(0, 4), m(0, 8)]));
        // White red 8 with a colour mark in another row is fine.
        assert!(has(&opts, &[m(0, 8), m(1, 4)]));
    }

    #[test]
    fn white_mark_can_be_the_fifth_that_allows_a_colour_lock() {
        // Red has 4 marks; white 1+6 = 7 is the fifth, then colour 6+6 = 12 locks.
        let you = with(State::default(), 0, &[2, 3, 4, 5]);
        let opts = active_options(&you, &State::default(), None, [1, 6, 6, 1, 1, 1]);
        assert!(has(&opts, &[m(0, 7), m(0, 12)]));
        assert!(!has(&opts, &[m(0, 12)]));
    }

    #[test]
    fn square_that_is_both_white_and_colour_appears_once() {
        // White 3+4 = 7; red colour 4+3 = 7 too.
        let opts = active_options(&State::default(), &State::default(), None, [3, 4, 3, 1, 1, 1]);
        let n = opts.selections.iter().filter(|x| x.len() == 1 && x[0] == m(0, 7)).count();
        assert_eq!(n, 1);
    }

    #[test]
    fn no_legal_marks_leaves_only_the_strike() {
        // Every row sits on its terminal number with 4 marks: nothing is markable.
        let you = State::from_parts(0, [(4, Some(12)), (4, Some(12)), (4, Some(2)), (4, Some(2))]);
        let opts = active_options(&you, &State::default(), None, [3, 4, 2, 2, 2, 2]);
        assert!(opts.selections.is_empty());
        assert!(opts.strike);
        assert!(!opts.pass);
    }

    #[test]
    fn passive_options_are_the_white_sum_marks() {
        let opts = passive_options(&State::default(), [3, 4, 1, 1, 1, 1]);
        assert_eq!(opts, vec![m(0, 7), m(1, 7), m(2, 7), m(3, 7)]);
    }
}
```

Add `pub mod selection;` as the first line of `web/src/lib.rs`.

- [ ] **Step 2: Run to verify they fail**

Run: `cd web && cargo test --release selection`
Expected: compile errors, `cannot find function active_options` / `ActiveOptions`.

- [ ] **Step 3: Implement** (above the tests module)

```rust
/// What the human may confirm on their own turn, given the bot's revealed
/// white-sum mark.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveOptions {
    /// Legal selections in application order: `[white]`, `[colour]` or
    /// `[white, colour]`. Never empty; each set appears once.
    pub selections: Vec<Vec<Mark>>,
    /// The strike box may be confirmed (alone).
    pub strike: bool,
    /// Confirming nothing is legal ("Pass"): the game ends after phase 1.
    pub pass: bool,
}

pub fn union(a: [bool; 4], b: [bool; 4]) -> [bool; 4] {
    [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]
}

pub fn n_locked(l: [bool; 4]) -> usize {
    l.iter().filter(|&&x| x).count()
}

fn set_key(sel: &[Mark]) -> Vec<(usize, u8)> {
    let mut k: Vec<(usize, u8)> = sel.iter().map(|m| (m.row, m.number)).collect();
    k.sort_unstable();
    k
}

/// Spec rules 1–3: phase-1 marks are judged on the pre-roll boards (the
/// bot's simultaneous lock never blocks your white sum); phase 2 is judged
/// after both phase-1 marks and lock propagation; if two rows are locked
/// after phase 1 the turn ends there (no colour mark, no strike).
pub fn active_options(you: &State, bot: &State, bot_white: Option<Mark>, dice: [u8; 6]) -> ActiveOptions {
    let white_sum = dice[0] + dice[1];
    let mut bot_after = *bot;
    if let Some(w) = bot_white {
        bot_after.apply_mark(w);
    }
    let bot_locks = bot_after.locked();
    let base_locks = union(you.locked(), bot_locks);
    let ends_after_bot = n_locked(base_locks) >= 2;

    let mut raw: Vec<Vec<Mark>> = Vec::new();
    for w in you.generate_white_moves(white_sum) {
        raw.push(vec![w]);
        let mut after_white = *you;
        after_white.apply_mark(w);
        let locks = union(after_white.locked(), bot_locks);
        if n_locked(locks) >= 2 {
            continue;
        }
        after_white.lock(locks);
        for c in after_white.generate_color_moves(dice) {
            raw.push(vec![w, c]);
        }
    }
    if !ends_after_bot {
        let mut no_white = *you;
        no_white.lock(base_locks);
        for c in no_white.generate_color_moves(dice) {
            raw.push(vec![c]);
        }
    }

    let mut seen: Vec<Vec<(usize, u8)>> = Vec::new();
    let mut selections = Vec::new();
    for sel in raw {
        let k = set_key(&sel);
        if !seen.contains(&k) {
            seen.push(k);
            selections.push(sel);
        }
    }
    ActiveOptions {
        selections,
        strike: !ends_after_bot,
        pass: ends_after_bot,
    }
}

/// On the bot's turn the human may mark only the white sum, or pass.
pub fn passive_options(you: &State, dice: [u8; 6]) -> Vec<Mark> {
    you.generate_white_moves(dice[0] + dice[1])
}
```

- [ ] **Step 4: Run tests**

Run: `cd web && cargo test --release selection`
Expected: 8 passed.

- [ ] **Step 5: Format and commit**

```bash
rustfmt --edition 2021 web/src/selection.rs web/src/lib.rs
git add web/src/selection.rs web/src/lib.rs
git commit -m "feat(web): legal human selections per the turn-flow spec

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Game session (`web/src/session.rs`)

One human-vs-bot game following `src/game.rs`: phase-1 marks of both players are decided on the pre-roll boards and applied together, locks propagate, the game may end, then the active player's phase 2. Display marks (`[[bool; 11]; 4]`) are the source of truth; `State` is derived with `State::from_marks` plus the global locks.

**Files:**
- Create: `web/src/session.rs`
- Create: `web/src/bots.rs` (only `BotKind` and `make_bot`, needed by the tests; Task 4 adds the benchmark constant)
- Modify: `web/src/lib.rs` (add `pub mod session;` and `pub mod bots;`)

**Interfaces:**
- Consumes: Task 2 `active_options`, `passive_options`, `union`, `n_locked`; Task 1 `PairStrategy::load_from_bytes`; `qwixxer::strategy::Strategy` (`active_phase1(&mut self, &State, &[State], [u8; 6]) -> Option<Mark>`, `active_phase2(.., has_marked: bool)`, `passive_phase1(.., active_player: usize)`).
- Produces (all `serde` with `snake_case` enum values; these JSON shapes are mirrored in Task 5's `protocol.ts`):
  - `pub struct Sq { pub row: usize, pub n: u8 }`
  - `pub enum Phase { BotToRoll, Passive, YouToRoll, Active, Over }`
  - `pub enum Who { You, Bot }`, `pub enum OverReason { Locks, YouStrikes, BotStrikes }`
  - `pub struct RollRecord { pub roller: Who, pub you: Vec<Sq>, pub you_strike: bool, pub bot_white: Option<Sq>, pub bot_colour: Option<Sq>, pub bot_strike: bool, pub locked: Vec<usize> }`
  - `pub struct View { pub phase: Phase, pub dice: Option<[u8; 6]>, pub you: SheetView, pub bot: SheetView, pub locked: [bool; 4], pub bot_white: Option<Sq>, pub options: Option<OptionsView>, pub passive: Vec<Sq>, pub over: Option<OverReason> }`
  - `pub struct SheetView { pub marks: [[bool; 11]; 4], pub strikes: u8, pub score: isize, pub row_points: [isize; 4] }`
  - `pub struct OptionsView { pub selections: Vec<Vec<Sq>>, pub strike: bool, pub pass: bool }`
  - `pub struct Snapshot` (serde; opaque to the frontend)
  - `impl Session { pub fn new(strategy: Box<dyn Strategy>, seed: u64) -> Self; pub fn restore(snap: Snapshot, strategy: Box<dyn Strategy>, seed: u64) -> Self; pub fn snapshot(&self) -> Snapshot; pub fn view(&self) -> View; pub fn bot_roll(&mut self) -> Result<(), String>; pub fn submit_passive(&mut self, pick: Option<Sq>) -> Result<RollRecord, String>; pub fn you_roll(&mut self) -> Result<(), String>; pub fn submit_active(&mut self, picks: &[Sq], strike: bool) -> Result<RollRecord, String> }`
  - `pub enum BotKind { MasterPlus, Master, Student }` with `pub fn parse(s: &str) -> Result<BotKind, String>`; `pub fn make_bot(kind: BotKind) -> Box<dyn Strategy>`

- [ ] **Step 1: Create `bots.rs`**

```rust
//! The three web bots (spec "Implementation decisions").

use qwixxer::bot::{self, DNA};
use qwixxer::dqn::pair::PairStrategy;
use qwixxer::strategy::search::{SearchBot, SearchConfig};
use qwixxer::strategy::Strategy;
use std::sync::Arc;

const CHAMPION: &[u8] = include_bytes!("../../champion.txt");
const PAIR_MODEL: &[u8] = include_bytes!("../../pair_model/model.mpk");

/// Master+ search config: "" = native production (`SearchConfig::default()`).
/// Task 4's browser benchmark may set a lighter spec here.
pub const MASTER_PLUS_SPEC: &str = "";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotKind {
    MasterPlus,
    Master,
    Student,
}

impl BotKind {
    pub fn parse(s: &str) -> Result<BotKind, String> {
        match s {
            "master_plus" => Ok(BotKind::MasterPlus),
            "master" => Ok(BotKind::Master),
            "student" => Ok(BotKind::Student),
            _ => Err(format!("unknown bot {s:?} (master_plus, master, student)")),
        }
    }
}

pub fn make_bot(kind: BotKind) -> Box<dyn Strategy> {
    match kind {
        BotKind::MasterPlus => Box::new(SearchBot::with_config(
            PairStrategy::load_from_bytes(PAIR_MODEL),
            SearchConfig::parse(MASTER_PLUS_SPEC).expect("MASTER_PLUS_SPEC"),
        )),
        BotKind::Master => Box::new(PairStrategy::load_from_bytes(PAIR_MODEL)),
        BotKind::Student => Box::new(DNA::load_weights_from_bytes(CHAMPION, Arc::new(bot::default_genes()))),
    }
}
```

Add `pub mod bots;` and `pub mod session;` to the top of `web/src/lib.rs`.

- [ ] **Step 2: Write the failing session tests**

Create `web/src/session.rs` with the tests module first (the implementation goes above it in Step 4):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::{make_bot, BotKind};
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Bot that plays fixed marks and records the state it saw in phase 2.
    #[derive(Debug, Default)]
    struct Scripted {
        white: Option<Mark>,
        colour: Option<Mark>,
        passive: Option<Mark>,
        seen_phase2: Rc<RefCell<Option<State>>>,
    }

    impl Strategy for Scripted {
        fn active_phase1(&mut self, _: &State, _: &[State], _: [u8; 6]) -> Option<Mark> {
            self.white
        }
        fn active_phase2(&mut self, s: &State, _: &[State], _: [u8; 6], _: bool) -> Option<Mark> {
            *self.seen_phase2.borrow_mut() = Some(*s);
            self.colour
        }
        fn passive_phase1(&mut self, _: &State, _: &[State], _: [u8; 6], _: usize) -> Option<Mark> {
            self.passive
        }
    }

    fn sq(row: usize, n: u8) -> Sq {
        Sq { row, n }
    }

    fn sheet(rows: &[(usize, &[u8])], strikes: u8) -> SheetSnap {
        let mut marks = [[false; 11]; 4];
        for &(r, nums) in rows {
            for &n in nums {
                marks[r][cell_index(r, n)] = true;
            }
        }
        SheetSnap { marks, strikes }
    }

    fn snap(you: SheetSnap, bot: SheetSnap, locked: [bool; 4], phase: Phase, dice: [u8; 6], bot_white: Option<Sq>) -> Snapshot {
        Snapshot { you, bot, locked, phase, dice: Some(dice), bot_white, over: None }
    }

    #[test]
    fn bot_starts_and_wrong_phase_calls_fail() {
        let mut s = Session::new(make_bot(BotKind::Student), 1);
        assert_eq!(s.view().phase, Phase::BotToRoll);
        assert!(s.you_roll().is_err());
        assert!(s.submit_active(&[], true).is_err());
        s.bot_roll().unwrap();
        assert_eq!(s.view().phase, Phase::Passive);
        assert!(s.view().bot_white.is_none(), "the bot's own white sum stays hidden");
        assert!(s.bot_roll().is_err());
    }

    #[test]
    fn random_games_finish_with_consistent_scores() {
        use rand::seq::SliceRandom;
        let mut pick_rng = SmallRng::seed_from_u64(99);
        for seed in 0..60u64 {
            let mut s = Session::new(make_bot(BotKind::Student), seed);
            let mut rolls = 0;
            while s.view().phase != Phase::Over {
                rolls += 1;
                assert!(rolls < 400, "game {seed} did not end");
                match s.view().phase {
                    Phase::BotToRoll => s.bot_roll().unwrap(),
                    Phase::Passive => {
                        let opts = s.view().passive;
                        let pick = if pick_rng.gen_bool(0.5) { opts.choose(&mut pick_rng).copied() } else { None };
                        s.submit_passive(pick).unwrap();
                    }
                    Phase::YouToRoll => s.you_roll().unwrap(),
                    Phase::Active => {
                        let o = s.view().options.unwrap();
                        if let Some(sel) = o.selections.choose(&mut pick_rng) {
                            s.submit_active(sel, false).unwrap();
                        } else if o.strike {
                            s.submit_active(&[], true).unwrap();
                        } else {
                            s.submit_active(&[], false).unwrap();
                        }
                    }
                    Phase::Over => unreachable!(),
                }
                let v = s.view();
                for p in [&v.you, &v.bot] {
                    assert_eq!(p.score, p.row_points.iter().sum::<isize>() - 5 * p.strikes as isize);
                    assert!(p.strikes <= 4);
                }
            }
            let v = s.view();
            assert!(v.over.is_some());
            assert!(v.you.strikes == 4 || v.bot.strikes == 4 || v.locked.iter().filter(|&&l| l).count() >= 2);
        }
    }

    #[test]
    fn your_white_lock_on_the_bots_turn_removes_that_colour_for_its_phase_two() {
        let seen = Rc::new(RefCell::new(None));
        let bot = Scripted { seen_phase2: seen.clone(), ..Default::default() };
        let you = sheet(&[(0, &[2, 3, 4, 5, 6])], 0);
        let s0 = snap(you, sheet(&[], 0), [false; 4], Phase::Passive, [6, 6, 1, 1, 1, 1], None);
        let mut s = Session::restore(s0, Box::new(bot), 1);
        let rec = s.submit_passive(Some(sq(0, 12))).unwrap();
        assert_eq!(rec.locked, vec![0]);
        assert_eq!(seen.borrow().unwrap().locked()[0], true, "bot's phase 2 sees red locked");
        // The bot marked nothing in either phase: it takes a strike.
        assert!(rec.bot_strike);
        assert_eq!(s.view().bot.strikes, 1);
        assert_eq!(s.view().phase, Phase::YouToRoll);
    }

    #[test]
    fn game_ending_at_phase_one_on_the_bots_turn_skips_its_phase_two() {
        let seen = Rc::new(RefCell::new(None));
        let bot = Scripted { seen_phase2: seen.clone(), ..Default::default() };
        let you = sheet(&[(0, &[2, 3, 4, 5, 6])], 0);
        let s0 = snap(you, sheet(&[], 0), [false, true, false, false], Phase::Passive, [6, 6, 1, 1, 1, 1], None);
        let mut s = Session::restore(s0, Box::new(bot), 1);
        let rec = s.submit_passive(Some(sq(0, 12))).unwrap();
        assert!(seen.borrow().is_none(), "no phase 2 after the game ended");
        assert!(!rec.bot_strike);
        let v = s.view();
        assert_eq!(v.phase, Phase::Over);
        assert_eq!(v.over, Some(OverReason::Locks));
        assert_eq!(v.bot.strikes, 0);
    }

    #[test]
    fn bot_reveal_ending_the_game_allows_only_white_or_pass_and_no_strike() {
        let bot = Scripted { passive: Some(Mark { row: 0, number: 12 }), ..Default::default() };
        let bot_sheet = sheet(&[(0, &[2, 3, 4, 5, 6])], 0);
        let s0 = snap(sheet(&[], 0), bot_sheet, [false, true, false, false], Phase::YouToRoll, [6, 6, 1, 1, 1, 1], None);
        let mut s = Session::restore(s0, Box::new(bot), 1);
        s.you_roll().unwrap();
        let v = s.view();
        assert_eq!(v.bot_white.map(|w| (w.row, w.n)), Some((0, 12)));
        let o = v.options.unwrap();
        assert!(o.pass && !o.strike);
        assert!(s.submit_active(&[], true).is_err(), "no strike when the game ends after the white sum");
        let rec = s.submit_active(&[], false).unwrap();
        assert_eq!(rec.bot_white, Some(sq(0, 12)));
        assert_eq!(s.view().phase, Phase::Over);
        assert_eq!(s.view().you.strikes, 0);
    }

    #[test]
    fn illegal_selection_is_rejected_and_changes_nothing() {
        let mut s = Session::restore(
            snap(sheet(&[], 0), sheet(&[], 0), [false; 4], Phase::YouToRoll, [1, 1, 1, 1, 1, 1], None),
            Box::new(Scripted::default()),
            3,
        );
        s.you_roll().unwrap();
        let before = serde_json::to_string(&s.view()).unwrap();
        assert!(s.submit_active(&[sq(0, 11)], false).is_err());
        assert!(s.submit_active(&[], false).is_err(), "empty selection is not a pass here");
        assert_eq!(serde_json::to_string(&s.view()).unwrap(), before);
    }

    #[test]
    fn fourth_strike_ends_the_game() {
        let s0 = snap(sheet(&[], 3), sheet(&[], 0), [false; 4], Phase::YouToRoll, [1, 1, 1, 1, 1, 1], None);
        let mut s = Session::restore(s0, Box::new(Scripted::default()), 1);
        s.you_roll().unwrap();
        let rec = s.submit_active(&[], true).unwrap();
        assert!(rec.you_strike);
        assert_eq!(s.view().over, Some(OverReason::YouStrikes));
    }

    #[test]
    fn snapshot_round_trips_in_every_phase() {
        let mut s = Session::new(make_bot(BotKind::Student), 5);
        for _ in 0..12 {
            let json = serde_json::to_string(&s.snapshot()).unwrap();
            let back: Snapshot = serde_json::from_str(&json).unwrap();
            let r = Session::restore(back, make_bot(BotKind::Student), 77);
            assert_eq!(serde_json::to_string(&r.view()).unwrap(), serde_json::to_string(&s.view()).unwrap());
            match s.view().phase {
                Phase::BotToRoll => s.bot_roll().unwrap(),
                Phase::Passive => {
                    s.submit_passive(None).unwrap();
                }
                Phase::YouToRoll => s.you_roll().unwrap(),
                Phase::Active => {
                    let o = s.view().options.unwrap();
                    match o.selections.first() {
                        Some(sel) => s.submit_active(&sel.clone(), false).unwrap(),
                        None => s.submit_active(&[], o.strike).unwrap(),
                    };
                }
                Phase::Over => break,
            }
        }
    }
}
```

- [ ] **Step 3: Run to verify they fail**

Run: `cd web && cargo test --release session`
Expected: compile errors (`Session`, `Sq`, `Phase` not found).

- [ ] **Step 4: Implement** (top of `web/src/session.rs`, above the tests)

```rust
//! One human-vs-bot game: the turn flow of the web UI spec over the engine.
//! Follows `src/game.rs`: both phase-1 marks are decided on the pre-roll
//! boards and applied together, locks propagate, the game may end, then the
//! active player's phase 2. The human is player 0, the bot player 1; the bot
//! starts. Display marks are the source of truth; `State` is derived.

use crate::selection::{active_options, n_locked, passive_options, ActiveOptions};
use qwixxer::state::{Mark, State};
use qwixxer::strategy::Strategy;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Column of number `n` in row `row` (red/yellow 2→12, green/blue 12→2).
pub fn cell_index(row: usize, n: u8) -> usize {
    if row < 2 {
        (n - 2) as usize
    } else {
        (12 - n) as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sq {
    pub row: usize,
    pub n: u8,
}

impl From<Mark> for Sq {
    fn from(m: Mark) -> Self {
        Sq { row: m.row, n: m.number }
    }
}

impl From<Sq> for Mark {
    fn from(s: Sq) -> Self {
        Mark { row: s.row, number: s.n }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    BotToRoll,
    Passive,
    YouToRoll,
    Active,
    Over,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Who {
    You,
    Bot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverReason {
    Locks,
    YouStrikes,
    BotStrikes,
}

/// One resolved roll, for the log and for staging the bot's marks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RollRecord {
    pub roller: Who,
    pub you: Vec<Sq>,
    pub you_strike: bool,
    pub bot_white: Option<Sq>,
    pub bot_colour: Option<Sq>,
    pub bot_strike: bool,
    /// Rows that became locked during this roll.
    pub locked: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SheetSnap {
    pub marks: [[bool; 11]; 4],
    pub strikes: u8,
}

impl SheetSnap {
    fn empty() -> Self {
        SheetSnap { marks: [[false; 11]; 4], strikes: 0 }
    }

    fn mark(&mut self, s: Sq) {
        self.marks[s.row][cell_index(s.row, s.n)] = true;
    }

    fn state(&self, locked: [bool; 4]) -> State {
        let mut st = State::from_marks(&self.marks, self.strikes);
        st.lock(locked);
        st
    }

    fn view(&self, locked: [bool; 4]) -> SheetView {
        let row_points: [isize; 4] = core::array::from_fn(|r| {
            // The lock symbol counts as an extra mark for whoever locked the row.
            let n = self.marks[r].iter().filter(|&&m| m).count() as isize + self.marks[r][10] as isize;
            n * (n + 1) / 2
        });
        SheetView {
            marks: self.marks,
            strikes: self.strikes,
            score: self.state(locked).count_points(),
            row_points,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub you: SheetSnap,
    pub bot: SheetSnap,
    pub locked: [bool; 4],
    pub phase: Phase,
    pub dice: Option<[u8; 6]>,
    /// The bot's phase-1 mark for the current roll (hidden on its own turn).
    pub bot_white: Option<Sq>,
    pub over: Option<OverReason>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SheetView {
    pub marks: [[bool; 11]; 4],
    pub strikes: u8,
    pub score: isize,
    pub row_points: [isize; 4],
}

#[derive(Debug, Clone, Serialize)]
pub struct OptionsView {
    pub selections: Vec<Vec<Sq>>,
    pub strike: bool,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct View {
    pub phase: Phase,
    pub dice: Option<[u8; 6]>,
    pub you: SheetView,
    pub bot: SheetView,
    pub locked: [bool; 4],
    /// The bot's revealed white-sum mark (only during your turn).
    pub bot_white: Option<Sq>,
    /// Your legal selections (only during your turn).
    pub options: Option<OptionsView>,
    /// Your legal white-sum squares (only during the bot's turn).
    pub passive: Vec<Sq>,
    pub over: Option<OverReason>,
}

pub struct Session {
    g: Snapshot,
    strategy: Box<dyn Strategy>,
    rng: SmallRng,
}

impl Session {
    pub fn new(strategy: Box<dyn Strategy>, seed: u64) -> Self {
        let g = Snapshot {
            you: SheetSnap::empty(),
            bot: SheetSnap::empty(),
            locked: [false; 4],
            phase: Phase::BotToRoll,
            dice: None,
            bot_white: None,
            over: None,
        };
        Self::restore(g, strategy, seed)
    }

    pub fn restore(snap: Snapshot, strategy: Box<dyn Strategy>, seed: u64) -> Self {
        Session { g: snap, strategy, rng: SmallRng::seed_from_u64(seed) }
    }

    pub fn snapshot(&self) -> Snapshot {
        self.g.clone()
    }

    fn states(&self) -> (State, State) {
        (self.g.you.state(self.g.locked), self.g.bot.state(self.g.locked))
    }

    fn expect(&self, phase: Phase) -> Result<(), String> {
        if self.g.phase == phase {
            Ok(())
        } else {
            Err(format!("expected phase {phase:?}, game is in {:?}", self.g.phase))
        }
    }

    fn roll(&mut self) -> [u8; 6] {
        core::array::from_fn(|_| self.rng.gen_range(1..=6))
    }

    /// Lock every row whose terminal number got marked; returns the new ones.
    fn settle_locks(&mut self) -> Vec<usize> {
        let mut newly = Vec::new();
        for r in 0..4 {
            if !self.g.locked[r] && (self.g.you.marks[r][10] || self.g.bot.marks[r][10]) {
                self.g.locked[r] = true;
                newly.push(r);
            }
        }
        newly
    }

    /// Ends the game if a player has 4 strikes or two rows are locked.
    fn check_over(&mut self) -> bool {
        let reason = if self.g.you.strikes >= 4 {
            Some(OverReason::YouStrikes)
        } else if self.g.bot.strikes >= 4 {
            Some(OverReason::BotStrikes)
        } else if n_locked(self.g.locked) >= 2 {
            Some(OverReason::Locks)
        } else {
            None
        };
        if reason.is_some() {
            self.g.over = reason;
            self.g.phase = Phase::Over;
            self.g.bot_white = None;
        }
        reason.is_some()
    }

    fn active_opts(&self) -> Option<ActiveOptions> {
        if self.g.phase != Phase::Active {
            return None;
        }
        let (you, bot) = self.states();
        Some(active_options(&you, &bot, self.g.bot_white.map(Mark::from), self.g.dice?))
    }

    pub fn view(&self) -> View {
        let (you, _) = self.states();
        let passive = match (self.g.phase, self.g.dice) {
            (Phase::Passive, Some(d)) => passive_options(&you, d).into_iter().map(Sq::from).collect(),
            _ => Vec::new(),
        };
        View {
            phase: self.g.phase,
            dice: self.g.dice,
            you: self.g.you.view(self.g.locked),
            bot: self.g.bot.view(self.g.locked),
            locked: self.g.locked,
            bot_white: if self.g.phase == Phase::Active { self.g.bot_white } else { None },
            options: self.active_opts().map(|o| OptionsView {
                selections: o.selections.iter().map(|s| s.iter().map(|&m| Sq::from(m)).collect()).collect(),
                strike: o.strike,
                pass: o.pass,
            }),
            passive,
            over: self.g.over,
        }
    }

    /// Bot's turn: roll and decide its white sum (kept hidden until you confirm).
    pub fn bot_roll(&mut self) -> Result<(), String> {
        self.expect(Phase::BotToRoll)?;
        let dice = self.roll();
        let (you, bot) = self.states();
        self.g.bot_white = self.strategy.active_phase1(&bot, &[you], dice).map(Sq::from);
        self.g.dice = Some(dice);
        self.g.phase = Phase::Passive;
        Ok(())
    }

    /// Bot's turn: your white-sum mark (or pass), then the bot's phase 2.
    pub fn submit_passive(&mut self, pick: Option<Sq>) -> Result<RollRecord, String> {
        self.expect(Phase::Passive)?;
        let dice = self.g.dice.ok_or("no dice")?;
        let (you, _) = self.states();
        if let Some(p) = pick {
            if !passive_options(&you, dice).contains(&Mark::from(p)) {
                return Err(format!("{p:?} is not a legal white-sum mark"));
            }
        }
        let bot_white = self.g.bot_white.take();
        if let Some(p) = pick {
            self.g.you.mark(p);
        }
        if let Some(w) = bot_white {
            self.g.bot.mark(w);
        }
        let mut rec = RollRecord {
            roller: Who::Bot,
            you: pick.into_iter().collect(),
            you_strike: false,
            bot_white,
            bot_colour: None,
            bot_strike: false,
            locked: self.settle_locks(),
        };
        if self.check_over() {
            return Ok(rec);
        }
        let (you, bot) = self.states();
        match self.strategy.active_phase2(&bot, &[you], dice, bot_white.is_some()) {
            Some(m) => {
                self.g.bot.mark(m.into());
                rec.bot_colour = Some(m.into());
            }
            None if bot_white.is_none() => {
                self.g.bot.strikes += 1;
                rec.bot_strike = true;
            }
            None => {}
        }
        rec.locked.extend(self.settle_locks());
        if !self.check_over() {
            self.g.phase = Phase::YouToRoll;
        }
        Ok(rec)
    }

    /// Your turn: roll; the bot decides (and reveals) its white sum at once.
    pub fn you_roll(&mut self) -> Result<(), String> {
        self.expect(Phase::YouToRoll)?;
        let dice = self.roll();
        let (you, bot) = self.states();
        self.g.bot_white = self.strategy.passive_phase1(&bot, &[you], dice, 0).map(Sq::from);
        self.g.dice = Some(dice);
        self.g.phase = Phase::Active;
        Ok(())
    }

    /// Your turn: confirm a selection (as a set), the strike box, or a pass.
    pub fn submit_active(&mut self, picks: &[Sq], strike: bool) -> Result<RollRecord, String> {
        self.expect(Phase::Active)?;
        let opts = self.active_opts().ok_or("no options")?;
        let chosen: Vec<Mark> = if strike {
            if !picks.is_empty() || !opts.strike {
                return Err("a strike is not allowed here".into());
            }
            Vec::new()
        } else if picks.is_empty() {
            if !opts.pass {
                return Err("select a square, or the strike box".into());
            }
            Vec::new()
        } else {
            let want: Vec<Mark> = picks.iter().map(|&p| p.into()).collect();
            opts.selections
                .iter()
                .find(|s| s.len() == want.len() && want.iter().all(|w| s.contains(w)))
                .cloned()
                .ok_or_else(|| format!("{picks:?} is not a legal selection"))?
        };
        for &m in &chosen {
            self.g.you.mark(m.into());
        }
        if strike {
            self.g.you.strikes += 1;
        }
        let bot_white = self.g.bot_white.take();
        if let Some(w) = bot_white {
            self.g.bot.mark(w);
        }
        let rec = RollRecord {
            roller: Who::You,
            you: chosen.iter().map(|&m| m.into()).collect(),
            you_strike: strike,
            bot_white,
            bot_colour: None,
            bot_strike: false,
            locked: self.settle_locks(),
        };
        if !self.check_over() {
            self.g.phase = Phase::BotToRoll;
        }
        Ok(rec)
    }
}
```

- [ ] **Step 5: Run tests**

Run: `cd web && cargo test --release`
Expected: all selection and session tests pass (8 + 8).

- [ ] **Step 6: Format and commit**

```bash
rustfmt --edition 2021 web/src/session.rs web/src/bots.rs web/src/lib.rs
git add web/src/session.rs web/src/bots.rs web/src/lib.rs
git diff --quiet web/Cargo.lock || echo "check web/Cargo.lock diff before adding"
git commit -m "feat(web): game session with spec turn flow, snapshots and Master+/Master/Student bots

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: wasm bindings, benchmark, Master+ config

**Files:**
- Modify: `web/src/lib.rs` (replace everything below the module declarations)
- Create: `web/src/bench.rs`
- Create: `web/scripts/bench.mjs`
- Modify (maybe): `web/src/bots.rs` (`MASTER_PLUS_SPEC`), `web/Cargo.toml` (`opt-level`), `web/build.sh`

**Interfaces:**
- Consumes: Task 3 `Session`, `Sq`, `make_bot`, `BotKind`.
- Produces (JS, from `pkg/qwixxer_web.js`):
  - `class WasmGame { constructor(bot: string, seed: number); static restore(bot: string, snapshot: string, seed: number): WasmGame; view(): string; snapshot(): string; bot_roll(): void; submit_passive(pick_json: string): string; you_roll(): void; submit_active(picks_json: string, strike: boolean): string; free(): void }` — errors are thrown as JS `Error`s; `view()` returns `View` JSON, the two `submit_*` return `RollRecord` JSON; `pick_json` is `"null"` or `{"row":r,"n":n}`.
  - `function bench(bot: string, games: number, seed: number): string` → `{"decisions":N,"mean_ms":x,"p50_ms":x,"p95_ms":x,"max_ms":x}`

- [ ] **Step 1: Replace `web/src/lib.rs`**

```rust
//! WASM entry points for the web UI: `WasmGame` (one game, JSON in/out)
//! and `bench`. Game logic lives in `session.rs` / `selection.rs`.

pub mod bench;
pub mod bots;
pub mod selection;
pub mod session;

use bots::{make_bot, BotKind};
use session::{Session, Snapshot, Sq};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmGame {
    session: Session,
}

fn js(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen]
impl WasmGame {
    #[wasm_bindgen(constructor)]
    pub fn new(bot: &str, seed: u32) -> Result<WasmGame, JsError> {
        let kind = BotKind::parse(bot).map_err(js)?;
        Ok(WasmGame { session: Session::new(make_bot(kind), seed as u64) })
    }

    pub fn restore(bot: &str, snapshot: &str, seed: u32) -> Result<WasmGame, JsError> {
        let kind = BotKind::parse(bot).map_err(js)?;
        let snap: Snapshot = serde_json::from_str(snapshot).map_err(js)?;
        Ok(WasmGame { session: Session::restore(snap, make_bot(kind), seed as u64) })
    }

    pub fn view(&self) -> String {
        serde_json::to_string(&self.session.view()).unwrap()
    }

    pub fn snapshot(&self) -> String {
        serde_json::to_string(&self.session.snapshot()).unwrap()
    }

    pub fn bot_roll(&mut self) -> Result<(), JsError> {
        self.session.bot_roll().map_err(js)
    }

    pub fn submit_passive(&mut self, pick_json: &str) -> Result<String, JsError> {
        let pick: Option<Sq> = serde_json::from_str(pick_json).map_err(js)?;
        let rec = self.session.submit_passive(pick).map_err(js)?;
        Ok(serde_json::to_string(&rec).unwrap())
    }

    pub fn you_roll(&mut self) -> Result<(), JsError> {
        self.session.you_roll().map_err(js)
    }

    pub fn submit_active(&mut self, picks_json: &str, strike: bool) -> Result<String, JsError> {
        let picks: Vec<Sq> = serde_json::from_str(picks_json).map_err(js)?;
        let rec = self.session.submit_active(&picks, strike).map_err(js)?;
        Ok(serde_json::to_string(&rec).unwrap())
    }
}
```

- [ ] **Step 2: Write `web/src/bench.rs`**

```rust
//! Browser/Node latency benchmark: plays games with random human choices and
//! times every call into the bot.

use crate::bots::{make_bot, BotKind};
use crate::session::{Phase, Session};
use qwixxer::state::{Mark, State};
use qwixxer::strategy::Strategy;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
}

#[derive(Debug)]
struct Timed {
    inner: Box<dyn Strategy>,
    ms: Rc<RefCell<Vec<f64>>>,
}

impl Timed {
    fn time<T>(&mut self, f: impl FnOnce(&mut Box<dyn Strategy>) -> T) -> T {
        let t = now();
        let r = f(&mut self.inner);
        self.ms.borrow_mut().push(now() - t);
        r
    }
}

impl Strategy for Timed {
    fn active_phase1(&mut self, s: &State, o: &[State], d: [u8; 6]) -> Option<Mark> {
        self.time(|b| b.active_phase1(s, o, d))
    }
    fn active_phase2(&mut self, s: &State, o: &[State], d: [u8; 6], h: bool) -> Option<Mark> {
        self.time(|b| b.active_phase2(s, o, d, h))
    }
    fn passive_phase1(&mut self, s: &State, o: &[State], d: [u8; 6], a: usize) -> Option<Mark> {
        self.time(|b| b.passive_phase1(s, o, d, a))
    }
}

#[wasm_bindgen]
pub fn bench(bot: &str, games: u32, seed: u32) -> Result<String, JsError> {
    let kind = BotKind::parse(bot).map_err(|e| JsError::new(&e))?;
    let ms = Rc::new(RefCell::new(Vec::new()));
    let mut rng = SmallRng::seed_from_u64(seed as u64);
    for g in 0..games {
        let timed = Timed { inner: make_bot(kind), ms: ms.clone() };
        let mut s = Session::new(Box::new(timed), seed as u64 * 1000 + g as u64);
        loop {
            match s.view().phase {
                Phase::BotToRoll => s.bot_roll().unwrap(),
                Phase::Passive => {
                    let pick = s.view().passive.choose(&mut rng).copied();
                    s.submit_passive(pick).unwrap();
                }
                Phase::YouToRoll => s.you_roll().unwrap(),
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
    }
    let mut v = ms.borrow().clone();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    Ok(format!(
        "{{\"decisions\":{},\"mean_ms\":{:.2},\"p50_ms\":{:.2},\"p95_ms\":{:.2},\"max_ms\":{:.2}}}",
        v.len(),
        mean,
        q(0.5),
        q(0.95),
        q(1.0)
    ))
}
```

- [ ] **Step 3: Write `web/scripts/bench.mjs`**

```js
// Usage (repo root): ./web/build.sh && node web/scripts/bench.mjs [bot] [games]
import { readFile } from 'node:fs/promises';
import init, { bench } from '../pkg/qwixxer_web.js';

const wasm = await readFile(new URL('../pkg/qwixxer_web_bg.wasm', import.meta.url));
await init({ module_or_path: wasm });
const bot = process.argv[2] ?? 'master_plus';
const games = Number(process.argv[3] ?? 10);
console.log(bot, games, 'games:', bench(bot, games, 1));
```

- [ ] **Step 4: Build and run the native tests**

Run: `cd web && cargo test --release` → all pass (the `bench` module compiles natively; `now()` is only called in wasm).
Run: `./web/build.sh` (repo root) → prints `WASM built: …`.
Expected: both succeed. If the wasm build fails on `std::time::Instant` or threads in a newly reached code path, report the error and stop; do not stub engine code.

- [ ] **Step 5: Measure Master+ latency**

Run: `node web/scripts/bench.mjs master_plus 10` and `node web/scripts/bench.mjs master 10`.
Record the four numbers of each.

Decision rule (spec: "well under a second for typical decisions"):
1. Accept if Master+ `p95_ms ≤ 400` and `max_ms ≤ 2500`. Keep `MASTER_PLUS_SPEC = ""`.
2. Otherwise first try speed-optimised wasm: in `web/Cargo.toml` set `opt-level = 3` under `[profile.release]`, and in `web/build.sh` change `-Oz` to `-O3`; rebuild; re-run. Keep this if it passes (note the new `.wasm` size).
3. Otherwise set `MASTER_PLUS_SPEC = "k=128,noseq,cands=2,lock=128"` in `web/src/bots.rs`, rebuild, re-run; accept if it passes. If even this fails, stop and report the numbers to the user.

- [ ] **Step 6: Record and commit**

Add a short "Web bot latency" paragraph to `docs/ARCHITECTURE.md` (web section, Task 10 rewrites the rest) with the table of numbers, the chosen `MASTER_PLUS_SPEC` and opt-level, and the date.

```bash
rustfmt --edition 2021 web/src/lib.rs web/src/bench.rs web/src/bots.rs
git add web/src/lib.rs web/src/bench.rs web/src/bots.rs web/scripts/bench.mjs docs/ARCHITECTURE.md web/Cargo.toml web/build.sh
git commit -m "feat(web): WasmGame bindings and bot latency bench; Master+ config from the measurement

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Svelte scaffold, theme, worker engine

**Files:**
- Delete: `web/src/ts/` (whole directory), `web/explorer.html`
- Replace: `web/index.html`, `web/vite.config.ts`, `web/tsconfig.json`
- Create: `web/svelte.config.js`, `web/src/main.ts`, `web/src/vite-env.d.ts`, `web/src/styles/theme.css`, `web/src/engine/protocol.ts`, `web/src/engine/engine.worker.ts`, `web/src/engine/client.ts`, `web/src/ui/App.svelte` (temporary smoke version)
- Modify: `web/package.json`, `web/public/sw.js` (cache name), `web/public/manifest.json`

**Interfaces:**
- Consumes: Task 4 `WasmGame` JS API.
- Produces:
  - `protocol.ts` types `Sq`, `Phase`, `OverReason`, `SheetView`, `OptionsView`, `View`, `RollRecord`, `BotKind`, and `interface Engine { start(bot: BotKind, seed: number): Promise<View>; restore(bot: BotKind, snapshot: string, seed: number): Promise<View>; snapshot(): Promise<string>; botRoll(): Promise<View>; submitPassive(pick: Sq | null): Promise<Step>; youRoll(): Promise<View>; submitActive(picks: Sq[], strike: boolean): Promise<Step> }` with `type Step = { view: View; record: RollRecord }`
  - `client.ts`: `class WorkerEngine implements Engine` (constructor takes a `Worker`)
  - CSS custom properties named exactly as in `theme.css` below (components in Tasks 8–9 use them).

- [ ] **Step 1: Install dependencies**

```bash
cd web
npm install -D svelte@5 @sveltejs/vite-plugin-svelte vitest svelte-check @tsconfig/svelte
npm install @fontsource/barlow @fontsource/barlow-condensed
```

If npm reports a peer conflict between `@sveltejs/vite-plugin-svelte` and the installed `vite`, install the newest `vite` major the plugin supports (`npm install -D vite@<major>`), not `--force`.

Set `web/package.json` scripts:

```json
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "check": "svelte-check --tsconfig ./tsconfig.json",
    "test": "vitest run"
  },
```

- [ ] **Step 2: Config files**

`web/svelte.config.js`:

```js
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
export default { preprocess: vitePreprocess() };
```

`web/vite.config.ts`:

```ts
import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
    base: '/qwixxer/',
    plugins: [svelte()],
    worker: { format: 'es' },
    build: { outDir: 'dist', assetsInlineLimit: 0 },
    server: { fs: { allow: ['.'] } },
    test: { environment: 'node', include: ['src/**/*.test.ts'] },
});
```

`web/tsconfig.json`:

```json
{
    "extends": "@tsconfig/svelte/tsconfig.json",
    "compilerOptions": {
        "target": "ES2022",
        "module": "ESNext",
        "moduleResolution": "bundler",
        "strict": true,
        "skipLibCheck": true,
        "types": ["vite/client"]
    },
    "include": ["src/**/*.ts", "src/**/*.svelte"]
}
```

`web/src/vite-env.d.ts`:

```ts
/// <reference types="svelte" />
/// <reference types="vite/client" />
```

`web/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="theme-color" content="#F3EDE1" />
    <title>Qwixxer</title>
    <link rel="manifest" href="/qwixxer/manifest.json" />
    <link rel="icon" href="/qwixxer/icon-192.png" />
</head>
<body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
    <script>
        if ('serviceWorker' in navigator) navigator.serviceWorker.register('/qwixxer/sw.js');
    </script>
</body>
</html>
```

In `web/public/sw.js` change `const CACHE_NAME = 'qwixx-v1';` to `'qwixxer-v2'` (drops caches of the old UI). In `web/public/manifest.json` set `"name": "Qwixxer"`, `"short_name": "Qwixxer"`, `"background_color": "#F3EDE1"`, `"theme_color": "#F3EDE1"`.

- [ ] **Step 3: Theme tokens**

`web/src/styles/theme.css`:

```css
@import '@fontsource/barlow/400.css';
@import '@fontsource/barlow/500.css';
@import '@fontsource/barlow/600.css';
@import '@fontsource/barlow/700.css';
@import '@fontsource/barlow-condensed/500.css';
@import '@fontsource/barlow-condensed/600.css';
@import '@fontsource/barlow-condensed/700.css';
@import '@fontsource/barlow-condensed/800.css';

:root {
    --ground: #F3EDE1; --paper: #FFFDF8; --edge: #D9D0BF; --shadow: rgba(60, 40, 10, 0.08);
    --ink: #1D1B18; --muted: #6B6459; --soft: #4A443B; --faint: #A39B8D; --log-bg: #F6F1E7;
    --pencil: #8B857C; --btn-muted-bg: #E6DED0; --btn-muted-fg: #8A8273; --strike-red: #C8323C;
    --locked-band: #C9C1B2; --dot: #E9B824; --die-face: #FFFDF8; --die-pip: #1D1B18;
    --on-band: #FFFDF8; --on-yellow: #1D1B18; --score-yellow: #B88C0B; --soft-mix: 50%;
    --scrim: rgba(29, 27, 24, 0.45);
    --red-band: #C8323C; --red-dark: #9E2530; --red-ink: #C8323C;
    --yellow-band: #E9B824; --yellow-dark: #C99A12; --yellow-ink: #9A7400;
    --green-band: #2F8F4E; --green-dark: #23703C; --green-ink: #2F8F4E;
    --blue-band: #2C5FAF; --blue-dark: #214B8C; --blue-ink: #2C5FAF;
    --cond: 'Barlow Condensed', sans-serif; --text: 'Barlow', sans-serif;
}

:root[data-theme='dark'] {
    --ground: #171513; --paper: #24211D; --edge: #0E0D0B; --shadow: rgba(0, 0, 0, 0.35);
    --ink: #ECE5D7; --muted: #A39B8D; --soft: #C9C1B2; --faint: #6E675C; --log-bg: #1C1A17;
    --pencil: #8E877C; --btn-muted-bg: #2F2B26; --btn-muted-fg: #7D766B; --strike-red: #E0575E;
    --locked-band: #3A352E; --dot: #C49A22; --die-face: #ECE5D7; --die-pip: #171513;
    --on-band: #FFF8EC; --on-yellow: #1D1B18; --score-yellow: #C49A22; --soft-mix: 44%;
    --scrim: rgba(0, 0, 0, 0.6);
    --red-band: #A8343B; --red-dark: #7C252B; --red-ink: #F08A8F;
    --yellow-band: #C49A22; --yellow-dark: #977414; --yellow-ink: #F0CD62;
    --green-band: #2D7E4A; --green-dark: #215E37; --green-ink: #74C994;
    --blue-band: #2F5E9F; --blue-dark: #234776; --blue-ink: #8DB0EE;
}

html, body { margin: 0; background: var(--ground); color: var(--ink); font-family: var(--text); }
button, select { font: inherit; }

@keyframes qx-draw { from { stroke-dashoffset: 60; } to { stroke-dashoffset: 0; } }
@keyframes qx-ink { from { stroke-dashoffset: 60; } to { stroke-dashoffset: 0; } }
@keyframes qx-roll {
    0% { transform: rotate(0deg) scale(1); }
    45% { transform: rotate(200deg) scale(1.1); }
    100% { transform: rotate(360deg) scale(1); }
}
.anim-draw path { stroke-dasharray: 60; animation: qx-draw 0.6s ease-out both; }
.anim-ink path { stroke-dasharray: 60; animation: qx-ink 0.6s ease-out both; }
.qx-roll { animation: qx-roll 0.6s ease-out; }
@media (prefers-reduced-motion: reduce) {
    .anim-draw path, .anim-ink path, .qx-roll { animation: none; }
}
```

- [ ] **Step 4: Protocol, worker, client**

`web/src/engine/protocol.ts`:

```ts
// Mirrors web/src/session.rs (serde JSON). Keep in sync.
export interface Sq { row: number; n: number }
export type Phase = 'bot_to_roll' | 'passive' | 'you_to_roll' | 'active' | 'over';
export type OverReason = 'locks' | 'you_strikes' | 'bot_strikes';
export type BotKind = 'master_plus' | 'master' | 'student';
export interface SheetView { marks: boolean[][]; strikes: number; score: number; row_points: number[] }
export interface OptionsView { selections: Sq[][]; strike: boolean; pass: boolean }
export interface View {
    phase: Phase;
    dice: number[] | null;
    you: SheetView;
    bot: SheetView;
    locked: boolean[];
    bot_white: Sq | null;
    options: OptionsView | null;
    passive: Sq[];
    over: OverReason | null;
}
export interface RollRecord {
    roller: 'you' | 'bot';
    you: Sq[];
    you_strike: boolean;
    bot_white: Sq | null;
    bot_colour: Sq | null;
    bot_strike: boolean;
    locked: number[];
}
export interface Step { view: View; record: RollRecord }
export interface Engine {
    start(bot: BotKind, seed: number): Promise<View>;
    restore(bot: BotKind, snapshot: string, seed: number): Promise<View>;
    snapshot(): Promise<string>;
    botRoll(): Promise<View>;
    submitPassive(pick: Sq | null): Promise<Step>;
    youRoll(): Promise<View>;
    submitActive(picks: Sq[], strike: boolean): Promise<Step>;
}
export const BOT_NAMES: Record<BotKind, string> = { master_plus: 'Master+', master: 'Master', student: 'Student' };
```

`web/src/engine/engine.worker.ts`:

```ts
import init, { WasmGame } from '../../pkg/qwixxer_web.js';
import wasmUrl from '../../pkg/qwixxer_web_bg.wasm?url';

const ready = init({ module_or_path: wasmUrl });
let game: WasmGame | null = null;

function need(): WasmGame {
    if (!game) throw new Error('no game');
    return game;
}
const view = () => JSON.parse(need().view());

function run(cmd: string, args: any[]): unknown {
    switch (cmd) {
        case 'start':
            game?.free();
            game = new WasmGame(args[0], args[1]);
            return view();
        case 'restore':
            game?.free();
            game = WasmGame.restore(args[0], args[1], args[2]);
            return view();
        case 'snapshot':
            return need().snapshot();
        case 'botRoll':
            need().bot_roll();
            return view();
        case 'submitPassive': {
            const record = JSON.parse(need().submit_passive(JSON.stringify(args[0])));
            return { view: view(), record };
        }
        case 'youRoll':
            need().you_roll();
            return view();
        case 'submitActive': {
            const record = JSON.parse(need().submit_active(JSON.stringify(args[0]), args[1]));
            return { view: view(), record };
        }
        default:
            throw new Error(`unknown command ${cmd}`);
    }
}

self.onmessage = async (e: MessageEvent<{ id: number; cmd: string; args: any[] }>) => {
    const { id, cmd, args } = e.data;
    try {
        await ready;
        self.postMessage({ id, ok: true, result: run(cmd, args) });
    } catch (err) {
        self.postMessage({ id, ok: false, error: err instanceof Error ? err.message : String(err) });
    }
};
```

`web/src/engine/client.ts`:

```ts
import type { BotKind, Engine, Sq, Step, View } from './protocol';

type Pending = { resolve: (v: any) => void; reject: (e: Error) => void };

/** Promise RPC to engine.worker.ts. One worker owns the whole game. */
export class WorkerEngine implements Engine {
    private next = 1;
    private pending = new Map<number, Pending>();

    constructor(private worker: Worker) {
        worker.onmessage = (e: MessageEvent<{ id: number; ok: boolean; result?: any; error?: string }>) => {
            const p = this.pending.get(e.data.id);
            if (!p) return;
            this.pending.delete(e.data.id);
            if (e.data.ok) p.resolve(e.data.result);
            else p.reject(new Error(e.data.error));
        };
        worker.onerror = (e) => {
            for (const p of this.pending.values()) p.reject(new Error(e.message || 'engine failed to load'));
            this.pending.clear();
        };
    }

    private call<T>(cmd: string, ...args: unknown[]): Promise<T> {
        const id = this.next++;
        return new Promise<T>((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            this.worker.postMessage({ id, cmd, args });
        });
    }

    start(bot: BotKind, seed: number) { return this.call<View>('start', bot, seed); }
    restore(bot: BotKind, snapshot: string, seed: number) { return this.call<View>('restore', bot, snapshot, seed); }
    snapshot() { return this.call<string>('snapshot'); }
    botRoll() { return this.call<View>('botRoll'); }
    submitPassive(pick: Sq | null) { return this.call<Step>('submitPassive', pick); }
    youRoll() { return this.call<View>('youRoll'); }
    submitActive(picks: Sq[], strike: boolean) { return this.call<Step>('submitActive', picks, strike); }
}

export function createEngine(): Engine {
    const worker = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' });
    return new WorkerEngine(worker);
}
```

- [ ] **Step 5: Smoke app**

`web/src/main.ts`:

```ts
import './styles/theme.css';
import { mount } from 'svelte';
import App from './ui/App.svelte';

mount(App, { target: document.getElementById('app')! });
```

Temporary `web/src/ui/App.svelte` (Task 8 replaces it):

```svelte
<script lang="ts">
    import { createEngine } from '../engine/client';
    const engine = createEngine();
    let text = $state('loading engine…');
    engine
        .start('student', 1)
        .then((v) => engine.botRoll().then((b) => (text = `phase ${v.phase} → ${b.phase}, dice ${b.dice}`)))
        .catch((e) => (text = `error: ${e.message}`));
</script>

<p>{text}</p>
```

- [ ] **Step 6: Verify**

Run (repo root): `./web/build.sh`, then `cd web && npm run check && npm run build`.
Expected: `svelte-check` reports 0 errors; `vite build` succeeds and emits a worker chunk and a `.wasm` asset in `dist/assets`.
Run: `npm run dev`, open the printed URL + `/qwixxer/`.
Expected: the page shows `phase bot_to_roll → passive, dice …`, and the browser network panel shows no request to `fonts.googleapis.com`.

- [ ] **Step 7: Commit**

```bash
git rm -r web/src/ts web/explorer.html
git add web/index.html web/vite.config.ts web/tsconfig.json web/svelte.config.js web/package.json web/package-lock.json web/public/sw.js web/public/manifest.json web/src/main.ts web/src/vite-env.d.ts web/src/styles web/src/engine web/src/ui
git commit -m "feat(web): Svelte + Vite scaffold, theme tokens, self-hosted fonts, worker engine client

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Pure game model (`web/src/game/rules.ts`)

Everything the components need to know about a view, as pure functions with tests. No legality is re-derived: clickability is a lookup in `view.options.selections` / `view.passive`.

**Files:**
- Create: `web/src/game/rules.ts`, `web/src/game/rules.test.ts`

**Interfaces:**
- Consumes: Task 5 `protocol.ts` types.
- Produces:
  - `ROWS = ['red','yellow','green','blue'] as const`, `NAME = ['Red','Yellow','Green','Blue']`
  - `numbers(row: number): number[]`, `cellIndex(row: number, n: number): number`, `isTerminal(sq: Sq): boolean`
  - `sqKey(s: Sq): string`, `selKey(sel: Sq[]): string`, `has(list: Sq[], s: Sq): boolean`
  - `clickable(view: View, sel: Sq[], strikeSel: boolean, s: Sq): boolean`
  - `confirmState(view: View, sel: Sq[], strikeSel: boolean): { label: string; ok: boolean }`
  - `strikeOpen(view: View, sel: Sq[]): boolean`
  - `type CellLook = 'normal' | 'legal' | 'soft' | 'skipped' | 'out'`
  - `interface CellModel { n: number; sq: Sq; mark: 'none' | 'ink' | 'pencil'; anim: '' | 'draw' | 'ink'; look: CellLook; clickable: boolean; label: string }`
  - `type LockLook = 'grey' | 'open' | 'locking' | 'locked_by' | 'locked_other'`
  - `interface RowModel { row: number; color: (typeof ROWS)[number]; band: 'normal' | 'locking' | 'locked'; cells: CellModel[]; lock: LockLook; label: string }`
  - `interface SheetUi { sel: Sq[]; strikeSel: boolean; committing: boolean; fresh: Record<string, 'draw' | 'ink'>; hidden: string[] }`
  - `sheetModel(view: View, who: 'you' | 'bot', ui: SheetUi): RowModel[]`
  - `type Chip = { kind: 'num'; row: number; n: number } | { kind: 'strike' } | { kind: 'lock' }`
  - `interface LogLine { who: 'You' | 'Bot'; chips: Chip[]; empty: string }`, `interface LogEntry { n: number; lines: LogLine[] }`
  - `chipsFor(marks: Sq[], strike: boolean): Chip[]`, `logEntry(n: number, rec: RollRecord): LogEntry`
  - Fresh-key convention: `` `${who}:${sqKey(s)}` `` for squares, `` `${who}:strike:${index}` `` for strike boxes, `'bot:strike'` in `hidden` hides the bot's newest strike.

- [ ] **Step 1: Write the failing tests**

`web/src/game/rules.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import type { View } from '../engine/protocol';
import { clickable, confirmState, logEntry, sheetModel, strikeOpen, type SheetUi } from './rules';

const empty = () => ({ marks: [0, 1, 2, 3].map(() => Array(11).fill(false)), strikes: 0, score: 0, row_points: [0, 0, 0, 0] });
function view(p: Partial<View>): View {
    return {
        phase: 'active', dice: [3, 4, 1, 1, 1, 1], you: empty(), bot: empty(), locked: [false, false, false, false],
        bot_white: null, options: { selections: [], strike: true, pass: false }, passive: [], over: null, ...p,
    };
}
const ui = (p: Partial<SheetUi> = {}): SheetUi => ({ sel: [], strikeSel: false, committing: false, fresh: {}, hidden: [], ...p });
const mark = (v: View, who: 'you' | 'bot', row: number, idxs: number[]) => idxs.forEach((i) => (v[who].marks[row][i] = true));

describe('clickable / confirm', () => {
    const v = view({ options: { selections: [[{ row: 0, n: 7 }], [{ row: 0, n: 7 }, { row: 1, n: 5 }], [{ row: 2, n: 5 }]], strike: true, pass: false } });
    it('narrows to partners after the first pick', () => {
        expect(clickable(v, [], false, { row: 2, n: 5 })).toBe(true);
        expect(clickable(v, [{ row: 0, n: 7 }], false, { row: 1, n: 5 })).toBe(true);
        expect(clickable(v, [{ row: 0, n: 7 }], false, { row: 2, n: 5 })).toBe(false);
        expect(clickable(v, [], true, { row: 2, n: 5 })).toBe(false);
    });
    it('labels and validates the confirm button', () => {
        expect(confirmState(v, [], false)).toEqual({ label: 'Mark', ok: false });
        expect(confirmState(v, [{ row: 0, n: 7 }], false)).toEqual({ label: 'Mark 1', ok: true });
        expect(confirmState(v, [{ row: 1, n: 5 }, { row: 0, n: 7 }], false)).toEqual({ label: 'Mark 2', ok: true });
        expect(confirmState(v, [], true)).toEqual({ label: 'Take strike', ok: true });
        expect(confirmState(view({ options: { selections: [], strike: false, pass: true } }), [], false)).toEqual({ label: 'Pass', ok: true });
        expect(confirmState(view({ phase: 'passive' }), [], false)).toEqual({ label: 'Pass', ok: true });
        expect(confirmState(view({ phase: 'over' }), [], false)).toEqual({ label: 'New game', ok: true });
        expect(strikeOpen(v, [])).toBe(true);
        expect(strikeOpen(v, [{ row: 0, n: 7 }])).toBe(false);
    });
});

describe('sheetModel', () => {
    it('greys skipped squares and the terminal number until five marks, pencil included', () => {
        const v = view({ options: { selections: [[{ row: 0, n: 8 }]], strike: true, pass: false } });
        mark(v, 'you', 0, [0, 1, 2, 4]); // red 2,3,4,6 → 5 is skipped
        let rows = sheetModel(v, 'you', ui());
        expect(rows[0].cells[3].look).toBe('skipped');
        expect(rows[0].cells[10].look).toBe('skipped');
        expect(rows[0].lock).toBe('grey');
        rows = sheetModel(v, 'you', ui({ sel: [{ row: 0, n: 8 }] }));
        expect(rows[0].cells[6].mark).toBe('pencil');
        expect(rows[0].lock).toBe('open');
        expect(rows[0].cells[10].look).not.toBe('skipped');
    });
    it('marks legal squares heavy and other open ones soft while choosing', () => {
        const v = view({ options: { selections: [[{ row: 1, n: 7 }]], strike: true, pass: false } });
        const rows = sheetModel(v, 'you', ui());
        expect(rows[1].cells[5].look).toBe('legal');
        expect(rows[1].cells[5].clickable).toBe(true);
        expect(rows[1].cells[6].look).toBe('soft');
        expect(sheetModel(view({ phase: 'you_to_roll', options: null }), 'you', ui())[1].cells[6].look).toBe('normal');
    });
    it('shows a bot lock as pending on both sheets and greys locked rows', () => {
        const v = view({ bot_white: { row: 0, n: 12 } });
        mark(v, 'bot', 0, [0, 1, 2, 3, 4]);
        expect(sheetModel(v, 'you', ui())[0].band).toBe('locking');
        const bot = sheetModel(v, 'bot', ui());
        expect(bot[0].band).toBe('locking');
        expect(bot[0].cells[10].mark).toBe('pencil');
        const done = view({ phase: 'bot_to_roll', options: null, locked: [true, false, false, false] });
        mark(done, 'bot', 0, [0, 1, 2, 3, 4, 10]);
        expect(sheetModel(done, 'you', ui())[0].band).toBe('locked');
        expect(sheetModel(done, 'you', ui())[0].cells[5].look).toBe('out');
        expect(sheetModel(done, 'you', ui())[0].lock).toBe('locked_other');
        expect(sheetModel(done, 'bot', ui())[0].lock).toBe('locked_by');
    });
    it('hides staged bot marks and inks committing selections', () => {
        const v = view({ phase: 'passive', options: null, passive: [{ row: 0, n: 7 }] });
        mark(v, 'bot', 3, [0]);
        expect(sheetModel(v, 'bot', ui({ hidden: ['bot:3:12'] }))[3].cells[0].mark).toBe('none');
        const r = sheetModel(v, 'you', ui({ sel: [{ row: 0, n: 7 }], committing: true }));
        expect(r[0].cells[5]).toMatchObject({ mark: 'ink', anim: 'ink' });
    });
});

describe('log', () => {
    it('puts the roller first and adds lock and strike chips', () => {
        const e = logEntry(7, { roller: 'bot', you: [], you_strike: false, bot_white: { row: 0, n: 12 }, bot_colour: null, bot_strike: false, locked: [0] });
        expect(e.n).toBe(7);
        expect(e.lines[0].who).toBe('Bot');
        expect(e.lines[0].chips).toEqual([{ kind: 'num', row: 0, n: 12 }, { kind: 'lock' }]);
        expect(e.lines[1]).toMatchObject({ who: 'You', chips: [], empty: '–' });
        const s = logEntry(8, { roller: 'you', you: [], you_strike: true, bot_white: null, bot_colour: null, bot_strike: false, locked: [] });
        expect(s.lines[0].chips).toEqual([{ kind: 'strike' }]);
    });
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd web && npm test`
Expected: FAIL, cannot resolve `./rules`.

- [ ] **Step 3: Implement `web/src/game/rules.ts`**

```ts
import type { RollRecord, Sq, View } from '../engine/protocol';

export const ROWS = ['red', 'yellow', 'green', 'blue'] as const;
export const NAME = ['Red', 'Yellow', 'Green', 'Blue'];

export const numbers = (row: number): number[] => {
    const a = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    return row < 2 ? a : a.slice().reverse();
};
export const cellIndex = (row: number, n: number) => (row < 2 ? n - 2 : 12 - n);
export const isTerminal = (s: Sq) => cellIndex(s.row, s.n) === 10;
export const sqKey = (s: Sq) => `${s.row}:${s.n}`;
export const selKey = (sel: Sq[]) => sel.map(sqKey).sort().join('|');
export const has = (list: Sq[], s: Sq) => list.some((x) => x.row === s.row && x.n === s.n);

/** Can `s` still complete a legal selection, given the current one? (Lookup only.) */
export function clickable(view: View, sel: Sq[], strikeSel: boolean, s: Sq): boolean {
    if (view.phase === 'active' && view.options) {
        if (strikeSel || has(sel, s)) return false;
        return view.options.selections.some((x) => has(x, s) && sel.every((p) => has(x, p)));
    }
    if (view.phase === 'passive') return sel.length === 0 && has(view.passive, s);
    return false;
}

export function confirmState(view: View, sel: Sq[], strikeSel: boolean): { label: string; ok: boolean } {
    if (view.phase === 'over') return { label: 'New game', ok: true };
    if (view.phase === 'passive') return { label: sel.length ? 'Mark' : 'Pass', ok: true };
    if (view.phase === 'active' && view.options) {
        if (strikeSel) return { label: 'Take strike', ok: view.options.strike };
        const k = selKey(sel);
        const ok = sel.length > 0 ? view.options.selections.some((x) => selKey(x) === k) : view.options.pass;
        const label = sel.length === 2 ? 'Mark 2' : sel.length === 1 ? 'Mark 1' : view.options.pass ? 'Pass' : 'Mark';
        return { label, ok };
    }
    return { label: 'Mark', ok: false };
}

export const strikeOpen = (view: View, sel: Sq[]) => view.phase === 'active' && !!view.options?.strike && sel.length === 0;

export type CellLook = 'normal' | 'legal' | 'soft' | 'skipped' | 'out';
export interface CellModel { n: number; sq: Sq; mark: 'none' | 'ink' | 'pencil'; anim: '' | 'draw' | 'ink'; look: CellLook; clickable: boolean; label: string }
export type LockLook = 'grey' | 'open' | 'locking' | 'locked_by' | 'locked_other';
export interface RowModel { row: number; color: (typeof ROWS)[number]; band: 'normal' | 'locking' | 'locked'; cells: CellModel[]; lock: LockLook; label: string }
export interface SheetUi { sel: Sq[]; strikeSel: boolean; committing: boolean; fresh: Record<string, 'draw' | 'ink'>; hidden: string[] }

export function sheetModel(view: View, who: 'you' | 'bot', ui: SheetUi): RowModel[] {
    const sheet = view[who];
    const isYou = who === 'you';
    const choosing = isYou && (view.phase === 'active' || view.phase === 'passive');
    const botWhite = view.phase === 'active' ? view.bot_white : null;
    const lockingRow = botWhite && isTerminal(botWhite) ? botWhite.row : -1;
    const pencils: Sq[] = isYou ? ui.sel : botWhite ? [botWhite] : [];
    return ROWS.map((color, r) => {
        const ink = sheet.marks[r].map((m, i) => m && !ui.hidden.includes(`${who}:${sqKey({ row: r, n: numbers(r)[i] })}`));
        const last = ink.lastIndexOf(true);
        const pencilHere = pencils.filter((p) => p.row === r).length;
        const lockable = ink.filter(Boolean).length + pencilHere >= 5;
        const locked = view.locked[r];
        const cells: CellModel[] = numbers(r).map((n, i) => {
            const sq = { row: r, n };
            const isPencil = has(pencils, sq);
            const committed = isYou && ui.committing && isPencil;
            const mark: CellModel['mark'] = ink[i] || committed ? 'ink' : isPencil ? 'pencil' : 'none';
            const canClick = isYou && !ink[i] && clickable(view, ui.sel, ui.strikeSel, sq);
            let look: CellLook = 'normal';
            if (locked && !ink[i]) look = 'out';
            else if (i < last && !ink[i]) look = 'skipped';
            else if (i === 10 && mark === 'none' && !lockable) look = 'skipped';
            else if (choosing && mark === 'none') look = canClick ? 'legal' : 'soft';
            const anim = committed ? 'ink' : mark === 'pencil' ? 'draw' : mark === 'ink' ? (ui.fresh[`${who}:${sqKey(sq)}`] ?? '') : '';
            const state = mark === 'ink' ? ', marked' : mark === 'pencil' ? (isYou ? ', selected' : ', marked this roll') : canClick ? ', available' : '';
            return { n, sq, mark, anim, look, clickable: canClick, label: `${NAME[r]} ${n}${state}` };
        });
        const lock: LockLook = ink[10] ? 'locked_by' : locked ? 'locked_other' : r === lockingRow ? 'locking' : lockable ? 'open' : 'grey';
        const band = locked ? 'locked' : r === lockingRow ? 'locking' : 'normal';
        return { row: r, color, band, cells, lock, label: `${NAME[r]} row${locked ? ', locked' : r === lockingRow ? ', locking this roll' : ''}` };
    });
}

export type Chip = { kind: 'num'; row: number; n: number } | { kind: 'strike' } | { kind: 'lock' };
export interface LogLine { who: 'You' | 'Bot'; chips: Chip[]; empty: string }
export interface LogEntry { n: number; lines: LogLine[] }

export function chipsFor(marks: Sq[], strike: boolean): Chip[] {
    if (strike) return [{ kind: 'strike' }];
    return marks.flatMap((m): Chip[] => (isTerminal(m) ? [{ kind: 'num', row: m.row, n: m.n }, { kind: 'lock' }] : [{ kind: 'num', row: m.row, n: m.n }]));
}

export function logEntry(n: number, rec: RollRecord): LogEntry {
    const botMarks = [rec.bot_white, rec.bot_colour].filter((x): x is Sq => !!x);
    const you: LogLine = { who: 'You', chips: chipsFor(rec.you, rec.you_strike), empty: '–' };
    const bot: LogLine = { who: 'Bot', chips: chipsFor(botMarks, rec.bot_strike), empty: '–' };
    return { n, lines: rec.roller === 'you' ? [you, bot] : [bot, you] };
}
```

- [ ] **Step 4: Run tests**

Run: `cd web && npm test`
Expected: all `rules.test.ts` tests pass. Then `npm run check` → 0 errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/game/rules.ts web/src/game/rules.test.ts
git commit -m "feat(web): pure sheet/confirm/log model for the UI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Game controller (`web/src/game/controller.svelte.ts`)

Sequences turns with the spec's pacing, owns selection, tips, log, bot choice, theme, persistence and error state. Tested with a fake engine and instant sleeps.

**Files:**
- Create: `web/src/game/controller.svelte.ts`, `web/src/game/controller.test.ts`

**Interfaces:**
- Consumes: Task 5 `Engine`, `View`, `Step`, `BotKind`, `Sq`; Task 6 `clickable`, `confirmState`, `strikeOpen`, `has`, `isTerminal`, `logEntry`, `sqKey`, `NAME`, `type LogEntry`.
- Produces:
  - `export const PACE = { tumble: 600, tick: 75, think: 500, stagger: 450, handoff: 900 }`
  - `export interface Store { get(key: string): string | null; set(key: string, value: string): void }`
  - `export class GameController` with reactive fields `view: View | null`, `sel: Sq[]`, `strikeSel: boolean`, `tip: string`, `error: string`, `rolling: boolean`, `shownDice: number[] | null`, `thinking: boolean`, `busy: boolean`, `committing: boolean`, `log: LogEntry[]`, `fresh: Record<string, 'draw' | 'ink'>`, `hidden: string[]`, `rollNo: number`, `theme: 'light' | 'dark'`, `bot: BotKind`, `nextBot: BotKind`; and methods `init(): Promise<void>`, `newGame(): Promise<void>`, `roll(): Promise<void>`, `tapSquare(s: Sq): void`, `tapStrike(): void`, `confirm(): Promise<void>`, `clearSelection(): void`, `setBot(k: BotKind): void`, `toggleTheme(): void`. Constructor: `new GameController(engine: Engine, store: Store, sleep = (ms: number) => Promise<void>, seed = () => number)`.

- [ ] **Step 1: Write the failing tests**

`web/src/game/controller.test.ts`:

```ts
import { describe, expect, it, vi } from 'vitest';
import type { Engine, Step, View } from '../engine/protocol';
import { GameController, type Store } from './controller.svelte';

const empty = () => ({ marks: [0, 1, 2, 3].map(() => Array(11).fill(false)), strikes: 0, score: 0, row_points: [0, 0, 0, 0] });
const v = (p: Partial<View>): View => ({
    phase: 'bot_to_roll', dice: null, you: empty(), bot: empty(), locked: [false, false, false, false],
    bot_white: null, options: null, passive: [], over: null, ...p,
});
const rec = (p: Partial<Step['record']> = {}): Step['record'] => ({ roller: 'you', you: [], you_strike: false, bot_white: null, bot_colour: null, bot_strike: false, locked: [], ...p });

function fakeEngine(over: Partial<Record<keyof Engine, any>> = {}): Engine {
    return {
        start: vi.fn(async () => v({})),
        restore: vi.fn(async () => v({ phase: 'you_to_roll' })),
        snapshot: vi.fn(async () => '{"snap":1}'),
        botRoll: vi.fn(async () => v({ phase: 'passive', dice: [1, 2, 3, 4, 5, 6], passive: [{ row: 0, n: 3 }] })),
        submitPassive: vi.fn(async () => ({ view: v({ phase: 'you_to_roll' }), record: rec({ roller: 'bot' }) })),
        youRoll: vi.fn(async () => v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: { selections: [[{ row: 0, n: 3 }]], strike: true, pass: false } })),
        submitActive: vi.fn(async () => ({ view: v({ phase: 'bot_to_roll' }), record: rec() })),
        ...over,
    } as Engine;
}
function memStore(init: Record<string, string> = {}): Store & { data: Record<string, string> } {
    const data = { ...init };
    return { data, get: (k) => data[k] ?? null, set: (k, val) => { data[k] = val; } };
}
const instant = async () => {};

describe('GameController', () => {
    it('starts a new game with the bot rolling first', async () => {
        const e = fakeEngine();
        const c = new GameController(e, memStore(), instant, () => 1);
        await c.init();
        expect(e.start).toHaveBeenCalledWith('master_plus', 1);
        expect(e.botRoll).toHaveBeenCalledOnce();
        expect(c.view?.phase).toBe('passive');
        expect(c.rollNo).toBe(1);
    });

    it('submits once when confirm is pressed twice while busy', async () => {
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: { selections: [[{ row: 0, n: 3 }]], strike: true, pass: false } })) });
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 3, bot: 'master', nextBot: 'master' }) }), instant, () => 1);
        await c.init();
        c.tapSquare({ row: 0, n: 3 });
        await Promise.all([c.confirm(), c.confirm()]);
        expect(e.submitActive).toHaveBeenCalledOnce();
        expect(e.submitActive).toHaveBeenCalledWith([{ row: 0, n: 3 }], false);
    });

    it('shows a tip instead of submitting an empty selection', async () => {
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: { selections: [[{ row: 0, n: 3 }]], strike: true, pass: false } })) });
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 3, bot: 'master', nextBot: 'master' }) }), instant, () => 1);
        await c.init();
        await c.confirm();
        expect(e.submitActive).not.toHaveBeenCalled();
        expect(c.tip).toBe('Select a square to mark, or the strike box to take a strike.');
    });

    it('applies a bot change only to the next game, and persists it', async () => {
        const store = memStore();
        const e = fakeEngine();
        const c = new GameController(e, store, instant, () => 1);
        await c.init();
        c.setBot('student');
        expect(c.bot).toBe('master_plus');
        expect(JSON.parse(store.data['qwixxer.prefs.v1']).nextBot).toBe('student');
        await c.newGame();
        expect(e.start).toHaveBeenLastCalledWith('student', 1);
        expect(c.bot).toBe('student');
    });

    it('restores a saved game and resumes a pending bot roll', async () => {
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'bot_to_roll' })) });
        const saved = { snapshot: '{"s":1}', log: [{ n: 4, lines: [] }], rollNo: 4, bot: 'master', nextBot: 'master' };
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify(saved) }), instant, () => 9);
        await c.init();
        expect(e.restore).toHaveBeenCalledWith('master', '{"s":1}', 9);
        expect(e.start).not.toHaveBeenCalled();
        expect(e.botRoll).toHaveBeenCalledOnce();
        expect(c.log[c.log.length - 1].n).toBe(4);
        expect(c.bot).toBe('master');
    });

    it('surfaces engine failures and stays usable', async () => {
        const e = fakeEngine({ start: vi.fn(async () => { throw new Error('wasm failed'); }) });
        const c = new GameController(e, memStore(), instant, () => 1);
        await c.init();
        expect(c.error).toContain('wasm failed');
        expect(c.busy).toBe(false);
    });

    it('clears the selection on Escape', async () => {
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: { selections: [[{ row: 0, n: 3 }]], strike: true, pass: false } })) });
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 1, bot: 'master', nextBot: 'master' }) }), instant, () => 1);
        await c.init();
        c.tapSquare({ row: 0, n: 3 });
        c.clearSelection();
        expect(c.sel).toEqual([]);
        expect(c.strikeSel).toBe(false);
    });

    it('explains a blocked square in a locking row', async () => {
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'active', dice: [6, 6, 2, 1, 1, 1], bot_white: { row: 0, n: 12 }, options: { selections: [[{ row: 0, n: 12 }]], strike: true, pass: false } })) });
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 1, bot: 'master', nextBot: 'master' }) }), instant, () => 1);
        await c.init();
        c.tapSquare({ row: 0, n: 8 });
        expect(c.tip).toContain('Red is locking');
    });
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd web && npm test`
Expected: FAIL, cannot resolve `./controller.svelte`.

- [ ] **Step 3: Implement `web/src/game/controller.svelte.ts`**

```ts
import type { BotKind, Engine, Sq, Step, View } from '../engine/protocol';
import { clickable, confirmState, has, isTerminal, logEntry, NAME, sqKey, strikeOpen, type LogEntry } from './rules';

/** Spec "Pacing": minimum durations in ms. */
export const PACE = { tumble: 600, tick: 75, think: 500, stagger: 450, handoff: 900 };

export interface Store { get(key: string): string | null; set(key: string, value: string): void }

const SAVE = 'qwixxer.save.v1';
const PREFS = 'qwixxer.prefs.v1';

interface Saved { snapshot: string; log: LogEntry[]; rollNo: number; bot: BotKind; nextBot: BotKind }

const randomDice = () => Array.from({ length: 6 }, () => 1 + Math.floor(Math.random() * 6));

export class GameController {
    view = $state<View | null>(null);
    sel = $state<Sq[]>([]);
    strikeSel = $state(false);
    tip = $state('');
    error = $state('');
    rolling = $state(false);
    shownDice = $state<number[] | null>(null);
    thinking = $state(false);
    busy = $state(false);
    committing = $state(false);
    log = $state<LogEntry[]>([]);
    fresh = $state<Record<string, 'draw' | 'ink'>>({});
    hidden = $state<string[]>([]);
    rollNo = $state(0);
    theme = $state<'light' | 'dark'>('light');
    bot = $state<BotKind>('master_plus');
    nextBot = $state<BotKind>('master_plus');

    // Plain fields assigned in the constructor: `.svelte.ts` files do not
    // support TypeScript parameter properties.
    private engine: Engine;
    private store: Store;
    private sleep: (ms: number) => Promise<void>;
    private seed: () => number;

    constructor(
        engine: Engine,
        store: Store,
        sleep: (ms: number) => Promise<void> = (ms) => new Promise((r) => setTimeout(r, ms)),
        seed: () => number = () => Math.floor(Math.random() * 2 ** 32),
    ) {
        this.engine = engine;
        this.store = store;
        this.sleep = sleep;
        this.seed = seed;
    }

    /** Escape key: drop the current selection and strike. */
    clearSelection() {
        if (this.busy) return;
        this.sel = [];
        this.strikeSel = false;
        this.tip = '';
    }

    async init() {
        const prefs = this.read<{ theme: 'light' | 'dark'; nextBot: BotKind }>(PREFS);
        if (prefs) {
            this.theme = prefs.theme ?? 'light';
            this.nextBot = prefs.nextBot ?? 'master_plus';
        }
        const saved = this.read<Saved>(SAVE);
        if (!saved) return this.newGame();
        await this.guard(async () => {
            this.bot = saved.bot;
            this.nextBot = saved.nextBot;
            this.log = saved.log;
            this.rollNo = saved.rollNo;
            this.view = await this.engine.restore(saved.bot, saved.snapshot, this.seed());
            if (this.view.phase === 'bot_to_roll') await this.botTurn();
        });
    }

    async newGame() {
        if (this.busy) return;
        await this.guard(async () => {
            this.bot = this.nextBot;
            this.log = [];
            this.rollNo = 0;
            this.sel = [];
            this.strikeSel = false;
            this.tip = '';
            this.fresh = {};
            this.hidden = [];
            this.view = await this.engine.start(this.bot, this.seed());
            await this.botTurn();
        });
    }

    /** Your Roll button. */
    async roll() {
        if (this.busy || this.view?.phase !== 'you_to_roll') return;
        await this.guard(async () => {
            this.fresh = {};
            const result = this.engine.youRoll();
            const [v] = await Promise.all([result, this.tumble().then(() => this.think())]);
            this.thinking = false;
            this.view = v;
            this.rollNo += 1;
            this.save();
        });
    }

    tapSquare(s: Sq) {
        const v = this.view;
        if (!v) return;
        if (v.phase !== 'active' && v.phase !== 'passive') {
            this.tip = v.phase === 'you_to_roll' ? 'Press Roll to start your turn.' : '';
            return;
        }
        if (this.busy) return;
        if (has(this.sel, s)) {
            this.sel = this.sel.filter((x) => x.row !== s.row || x.n !== s.n);
            this.tip = '';
            return;
        }
        if (clickable(v, this.sel, this.strikeSel, s)) {
            this.sel = [...this.sel, s];
            this.strikeSel = false;
            this.tip = '';
            return;
        }
        const name = NAME[s.row].toLowerCase();
        if (this.strikeSel) this.tip = 'Clear the strike first to mark a square.';
        else if (v.phase === 'active' && v.bot_white && isTerminal(v.bot_white) && v.bot_white.row === s.row)
            this.tip = `${NAME[s.row]} is locking: the bot marked ${name} ${v.bot_white.n}. Your white sum may still go in ${name}; a colour mark can’t.`;
        else if (v.phase === 'passive')
            this.tip = `On the bot’s turn you may only mark the white sum (${(v.dice?.[0] ?? 0) + (v.dice?.[1] ?? 0)}), or pass.`;
        else if (this.sel.length === 2) this.tip = 'You already picked two squares. Tap one to unselect it.';
        else this.tip = 'That square isn’t available with this roll.';
    }

    tapStrike() {
        const v = this.view;
        if (!v || this.busy) return;
        if (v.phase !== 'active') {
            this.tip = v.phase === 'passive' ? 'You can’t take a strike on the bot’s turn.' : '';
            return;
        }
        if (this.sel.length > 0) this.tip = 'Unselect your squares first to take a strike.';
        else if (!strikeOpen(v, this.sel)) this.tip = 'The game ends after the white sum, so no strike this turn.';
        else {
            this.strikeSel = !this.strikeSel;
            this.tip = '';
        }
    }

    async confirm() {
        const v = this.view;
        if (!v || this.busy) return;
        if (v.phase === 'over') return this.newGame();
        if (v.phase === 'passive') return this.resolvePassive();
        if (v.phase !== 'active') {
            this.tip = v.phase === 'you_to_roll' ? 'Press Roll to start your turn.' : '';
            return;
        }
        const c = confirmState(v, this.sel, this.strikeSel);
        if (!c.ok) {
            this.tip = this.sel.length === 0 ? 'Select a square to mark, or the strike box to take a strike.' : 'Those squares don’t work together with this roll.';
            return;
        }
        await this.guard(async () => {
            const picks = this.sel;
            const strike = this.strikeSel;
            this.committing = true;
            const step = await this.engine.submitActive(picks, strike);
            const fresh: Record<string, 'draw' | 'ink'> = {};
            step.record.you.forEach((s) => (fresh[`you:${sqKey(s)}`] = 'ink'));
            if (step.record.you_strike) fresh[`you:strike:${step.view.you.strikes - 1}`] = 'ink';
            if (step.record.bot_white) fresh[`bot:${sqKey(step.record.bot_white)}`] = 'ink';
            this.apply(step, fresh);
            if (step.view.phase === 'over') return;
            await this.sleep(PACE.handoff);
            await this.botTurn();
        });
    }

    setBot(k: BotKind) {
        this.nextBot = k;
        this.savePrefs();
        this.save();
    }

    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        this.savePrefs();
    }

    // ---- internals ----

    private async botTurn() {
        this.fresh = {};
        const result = this.engine.botRoll();
        const [v] = await Promise.all([result, this.tumble()]);
        this.view = v;
        this.rollNo += 1;
        this.save();
    }

    /** Your white-sum mark inks at once; the bot thinks, draws its white sum, then its colour mark. */
    private async resolvePassive() {
        await this.guard(async () => {
            const pick = this.sel[0] ?? null;
            this.committing = true;
            this.thinking = true;
            const [step] = await Promise.all([this.engine.submitPassive(pick), this.sleep(PACE.think)]);
            this.thinking = false;
            const r = step.record;
            const fresh: Record<string, 'draw' | 'ink'> = {};
            r.you.forEach((s) => (fresh[`you:${sqKey(s)}`] = 'ink'));
            if (r.bot_white) fresh[`bot:${sqKey(r.bot_white)}`] = 'draw';
            const staged = [...(r.bot_colour ? [`bot:${sqKey(r.bot_colour)}`] : []), ...(r.bot_strike ? ['bot:strike'] : [])];
            this.hidden = staged;
            this.apply(step, fresh);
            if (staged.length) {
                await this.sleep(PACE.stagger);
                this.fresh = r.bot_colour ? { [`bot:${sqKey(r.bot_colour)}`]: 'draw' } : {};
                this.hidden = [];
            }
        });
    }

    private apply(step: Step, fresh: Record<string, 'draw' | 'ink'>) {
        this.view = step.view;
        this.sel = [];
        this.strikeSel = false;
        this.committing = false;
        this.tip = '';
        this.fresh = fresh;
        this.log = [logEntry(this.rollNo, step.record), ...this.log];
        this.save();
    }

    private async tumble() {
        this.rolling = true;
        this.shownDice = randomDice();
        const t = setInterval(() => (this.shownDice = randomDice()), PACE.tick);
        try {
            await this.sleep(PACE.tumble);
        } finally {
            clearInterval(t);
            this.rolling = false;
            this.shownDice = null;
        }
    }

    private async think() {
        this.thinking = true;
        await this.sleep(PACE.think);
    }

    private async guard(work: () => Promise<void>) {
        if (this.busy) return;
        this.busy = true;
        this.error = '';
        try {
            await work();
        } catch (e) {
            this.error = e instanceof Error ? e.message : String(e);
            this.thinking = false;
            this.rolling = false;
            this.committing = false;
        } finally {
            this.busy = false;
        }
    }

    private save() {
        if (!this.view) return;
        const meta = { log: this.log, rollNo: this.rollNo, bot: this.bot, nextBot: this.nextBot };
        this.engine
            .snapshot()
            .then((snapshot) => this.store.set(SAVE, JSON.stringify({ snapshot, ...meta })))
            .catch(() => {});
    }

    private savePrefs() {
        this.store.set(PREFS, JSON.stringify({ theme: this.theme, nextBot: this.nextBot }));
    }

    private read<T>(key: string): T | null {
        try {
            const raw = this.store.get(key);
            return raw ? (JSON.parse(raw) as T) : null;
        } catch {
            return null;
        }
    }
}
```

Note on `newGame` from `confirm()` while `over`: `confirm` returns `this.newGame()` before `guard`, and `newGame` checks `busy` itself, so a double click starts one game.

- [ ] **Step 4: Run tests**

Run: `cd web && npm test`
Expected: all controller and rules tests pass. If the "restores and keeps log" test fails because `botTurn` prepends nothing, check that `log` was restored before `botTurn` (it is) — the assertion reads the last entry.

- [ ] **Step 5: Commit**

```bash
git add web/src/game/controller.svelte.ts web/src/game/controller.test.ts
git commit -m "feat(web): game controller with pacing, tips, persistence and error handling

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Desktop UI

Renders the controller with the paper-sheet design (canvas artboards A1–A3 and the prototype). 1280×800, no scrolling.

**Files:**
- Create: `web/src/ui/MarkX.svelte`, `web/src/ui/Cell.svelte`, `web/src/ui/LockCell.svelte`, `web/src/ui/SheetRows.svelte`, `web/src/ui/Die.svelte`, `web/src/ui/DiceTray.svelte`, `web/src/ui/Chip.svelte`, `web/src/ui/Log.svelte`, `web/src/ui/BotPanel.svelte`, `web/src/ui/ActionBar.svelte`, `web/src/ui/Header.svelte`, `web/src/ui/Desktop.svelte`
- Replace: `web/src/ui/App.svelte`

**Interfaces:**
- Consumes: Task 7 `GameController` (all fields and methods), Task 6 `sheetModel`, `confirmState`, `strikeOpen`, `RowModel`, `CellModel`, `Chip`, `LogEntry`, `ROWS`, `NAME`; Task 5 `BOT_NAMES`, `createEngine`.
- Produces: `App.svelte` (mounts layout, theme effect, `controller.init()`); presentational components reused by Task 9 (`MarkX`, `Cell`, `LockCell`, `Die`, `Chip`, `Log`, `BotPanel`, `SheetRows`).

- [ ] **Step 1: Leaf components**

`web/src/ui/MarkX.svelte`:

```svelte
<script lang="ts">
    let { pencil = false, anim = '' }: { pencil?: boolean; anim?: string } = $props();
</script>

<svg viewBox="0 0 40 40" class="x {anim ? `anim-${anim}` : ''}" aria-hidden="true">
    <path d="M10 9 C 17 16, 24 24, 31 32 M31 9 C 24 16, 17 24, 9 31" stroke={pencil ? 'var(--pencil)' : 'var(--ink)'} stroke-width={pencil ? 3 : 4} stroke-linecap="round" fill="none" />
    {#if pencil}
        <path d="M11 11 C 18 17, 23 24, 30 30 M29 11 C 23 17, 18 23, 11 29" stroke="var(--pencil)" stroke-width="1.2" stroke-linecap="round" fill="none" opacity="0.6" />
    {/if}
</svg>

<style>
    .x { position: absolute; inset: 3px; width: calc(100% - 6px); height: calc(100% - 6px); }
</style>
```

`web/src/ui/Cell.svelte`:

```svelte
<script lang="ts">
    import type { CellModel } from '../game/rules';
    import MarkX from './MarkX.svelte';
    let { cell, color, size, onTap }: { cell: CellModel; color: string; size: number; onTap?: (c: CellModel) => void } = $props();
</script>

{#if onTap}
    <button type="button" class="cell look-{cell.look}" style="--c: var(--{color}-ink); --s: {size}px"
        aria-label={cell.label} aria-pressed={cell.mark === 'pencil'} aria-disabled={!cell.clickable && cell.mark !== 'pencil'}
        onclick={() => onTap(cell)}>
        <span>{cell.n}</span>
        {#if cell.mark !== 'none'}<MarkX pencil={cell.mark === 'pencil'} anim={cell.anim} />{/if}
    </button>
{:else}
    <div class="cell look-{cell.look}" role="img" aria-label={cell.label} style="--c: var(--{color}-ink); --s: {size}px">
        <span>{cell.n}</span>
        {#if cell.mark !== 'none'}<MarkX pencil={cell.mark === 'pencil'} anim={cell.anim} />{/if}
    </div>
{/if}

<style>
    .cell {
        position: relative; box-sizing: border-box; width: var(--s); height: var(--s); border: none; padding: 0;
        border-radius: calc(var(--s) * 0.15); display: flex; align-items: center; justify-content: center;
        background: var(--paper); color: var(--c); font-family: var(--cond); font-weight: 700; font-size: calc(var(--s) * 0.5);
    }
    button.cell { cursor: default; }
    .look-legal { font-weight: 800; cursor: pointer !important; }
    .look-soft { font-weight: 500; color: color-mix(in srgb, var(--c) var(--soft-mix), transparent); }
    .look-skipped { opacity: 0.35; }
    .look-out { opacity: 0.3; color: var(--muted); }
    [aria-pressed='true'] { cursor: pointer !important; }
</style>
```

`web/src/ui/LockCell.svelte`:

```svelte
<script lang="ts">
    import type { LockLook } from '../game/rules';
    let { look, color, size, label }: { look: LockLook; color: string; size: number; label: string } = $props();
    const stroke = $derived(look === 'locked_by' ? 'var(--paper)' : look === 'grey' || look === 'open' ? `var(--${color}-ink)` : 'var(--ink)');
</script>

<div class="lock {look}" role="img" aria-label={label} style="--s: {size}px">
    <svg viewBox="0 0 24 24" width={size * 0.47} height={size * 0.47} fill="none" stroke={stroke} stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="5" y="11" width="14" height="10" rx="2"></rect><path d="M8 11V8a4 4 0 0 1 8 0v3"></path>
    </svg>
</div>

<style>
    .lock { box-sizing: border-box; width: var(--s); height: var(--s); border-radius: 50%; display: flex; align-items: center; justify-content: center; background: var(--paper); flex-shrink: 0; }
    .grey { opacity: 0.35; }
    .locking { box-shadow: 0 0 0 3px var(--ink); }
    .locked_by { background: var(--ink); }
    .locked_other { box-shadow: inset 0 0 0 2px var(--ink); }
</style>
```

`web/src/ui/SheetRows.svelte` (horizontal sheet; desktop "You" at 58 px, bot at 25 px and in the phone drawer):

```svelte
<script lang="ts">
    import type { CellModel, RowModel } from '../game/rules';
    import Cell from './Cell.svelte';
    import LockCell from './LockCell.svelte';
    let { rows, size, onTap }: { rows: RowModel[]; size: number; onTap?: (c: CellModel) => void } = $props();
    const gap = $derived(size >= 40 ? 6 : 3);
</script>

<div class="rows" style="gap: {gap}px">
    {#each rows as row (row.row)}
        <div class="band {row.band}" aria-label={row.label} style="--band: var(--{row.color}-band); --dark: var(--{row.color}-dark); gap: {gap}px; padding: {gap}px; border-radius: {size >= 40 ? 10 : 7}px">
            {#each row.cells as cell (cell.n)}
                <Cell {cell} color={row.color} {size} {onTap} />
            {/each}
            <span style="width: {size >= 40 ? 4 : 2}px"></span>
            <LockCell look={row.lock} color={row.color} {size} label="{row.label} lock" />
        </div>
    {/each}
</div>

<style>
    .rows { display: flex; flex-direction: column; }
    .band { display: flex; align-items: center; width: fit-content; background: var(--band); }
    .locking { background: repeating-linear-gradient(135deg, var(--band) 0 10px, var(--dark) 10px 20px); }
    .locked { background: var(--locked-band); }
</style>
```

`web/src/ui/Die.svelte`:

```svelte
<script lang="ts">
    let { value, face, pip, out = false, idle = false, rolling = false, size = 56, label }: { value: number; face: string; pip: string; out?: boolean; idle?: boolean; rolling?: boolean; size?: number; label: string } = $props();
    const MAP: Record<number, number[]> = { 1: [4], 2: [0, 8], 3: [0, 4, 8], 4: [0, 2, 6, 8], 5: [0, 2, 4, 6, 8], 6: [0, 2, 3, 5, 6, 8] };
    const on = $derived(out || idle || !value ? [] : MAP[value]);
</script>

<div class="die" class:qx-roll={rolling && !out} class:out class:idle role="img" aria-label={label}
    style="--s: {size}px; background: {out ? 'var(--locked-band)' : face}">
    {#each [0, 1, 2, 3, 4, 5, 6, 7, 8] as i}
        <span style="background: {on.includes(i) ? pip : 'transparent'}"></span>
    {/each}
</div>

<style>
    .die {
        box-sizing: border-box; width: var(--s); height: var(--s); border-radius: calc(var(--s) * 0.21); padding: calc(var(--s) * 0.14);
        display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 3px;
        box-shadow: inset 0 -3px 0 rgba(0, 0, 0, 0.18), 0 2px 6px var(--shadow);
    }
    .die span { width: calc(var(--s) * 0.18); height: calc(var(--s) * 0.18); border-radius: 50%; place-self: center; }
    .out { opacity: 0.5; box-shadow: none; }
    .idle { opacity: 0.4; }
</style>
```

`web/src/ui/Chip.svelte`:

```svelte
<script lang="ts">
    import type { Chip } from '../game/rules';
    import { NAME, ROWS } from '../game/rules';
    let { chip, size = 26 }: { chip: Chip; size?: number } = $props();
</script>

{#if chip.kind === 'num'}
    <span class="chip" role="img" aria-label="{NAME[chip.row]} {chip.n}" style="--s: {size}px; background: var(--{ROWS[chip.row]}-band); color: {chip.row === 1 ? 'var(--on-yellow)' : 'var(--on-band)'}">{chip.n}</span>
{:else if chip.kind === 'strike'}
    <span class="chip strike" role="img" aria-label="strike" style="--s: {size}px">
        <svg viewBox="0 0 24 24" width={size * 0.54} height={size * 0.54} fill="none" stroke="var(--strike-red)" stroke-width="3" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"></path></svg>
    </span>
{:else}
    <span class="chip lock" role="img" aria-label="locked the row" style="--s: {size}px">
        <svg viewBox="0 0 24 24" width={size * 0.54} height={size * 0.54} fill="none" stroke="var(--paper)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="5" y="11" width="14" height="10" rx="2"></rect><path d="M8 11V8a4 4 0 0 1 8 0v3"></path></svg>
    </span>
{/if}

<style>
    .chip { box-sizing: border-box; min-width: var(--s); height: var(--s); padding: 0 4px; border-radius: calc(var(--s) * 0.27); display: inline-flex; align-items: center; justify-content: center; font-family: var(--cond); font-weight: 700; font-size: calc(var(--s) * 0.6); }
    .strike { background: var(--paper); box-shadow: inset 0 0 0 2px var(--strike-red); }
    .lock { background: var(--ink); }
</style>
```

- [ ] **Step 2: Panels**

`web/src/ui/Log.svelte`:

```svelte
<script lang="ts">
    import type { LogEntry } from '../game/rules';
    import ChipView from './Chip.svelte';
    // Spec "Implementation notes · Log": the log scrolls instead of truncating.
    let { entries, current = null, chip = 26 }: { entries: LogEntry[]; current?: LogEntry | null; chip?: number } = $props();
    const shown = $derived([...(current ? [current] : []), ...entries]);
</script>

<section aria-label="Turn log" class="log">
    <h2>Log</h2>
    {#if shown.length === 0}<p class="empty">Nothing yet.</p>{/if}
    {#each shown as e, i (i + ':' + e.n)}
        <div class="roll">
            <span class="n">{e.n}</span>
            <div class="lines">
                {#each e.lines as line}
                    <div class="line">
                        <span class="who" class:you={line.who === 'You'}>{line.who}</span>
                        {#each line.chips as c}<ChipView chip={c} size={chip} />{/each}
                        {#if line.chips.length === 0}<span class="empty">{line.empty}</span>{/if}
                    </div>
                {/each}
            </div>
        </div>
    {/each}
</section>

<style>
    .log { display: flex; flex-direction: column; gap: 8px; min-height: 0; max-height: 100%; overflow-y: auto; }
    h2 { margin: 0; font-family: var(--cond); font-size: 20px; font-weight: 700; }
    .roll { display: flex; align-items: flex-start; gap: 6px; padding: 6px 8px; border-radius: 10px; background: var(--log-bg); }
    .n { width: 26px; font-family: var(--cond); font-weight: 700; font-size: 16px; color: var(--muted); }
    .lines { display: flex; flex-direction: column; gap: 4px; flex: 1; }
    .line { display: flex; align-items: center; gap: 5px; min-height: 26px; }
    .who { width: 30px; font-size: 13px; font-weight: 700; color: var(--muted); }
    .who.you { color: var(--ink); }
    .empty { color: var(--faint); font-size: 14px; }
</style>
```

`web/src/ui/BotPanel.svelte`:

```svelte
<script lang="ts">
    import { BOT_NAMES, type BotKind } from '../engine/protocol';
    import type { GameController } from '../game/controller.svelte';
    import { sheetModel } from '../game/rules';
    import SheetRows from './SheetRows.svelte';
    let { c, size = 25 }: { c: GameController; size?: number } = $props();
    const rows = $derived(c.view ? sheetModel(c.view, 'bot', { sel: [], strikeSel: false, committing: false, fresh: c.fresh, hidden: c.hidden }) : []);
    const strikes = $derived((c.view?.bot.strikes ?? 0) - (c.hidden.includes('bot:strike') ? 1 : 0));
</script>

<section aria-label="Bot's sheet" class="panel">
    <div class="head">
        <div class="pick">
            <label for="bot-picker">Bot</label>
            <select id="bot-picker" value={c.nextBot} onchange={(e) => c.setBot((e.currentTarget as HTMLSelectElement).value as BotKind)}>
                {#each Object.entries(BOT_NAMES) as [id, name]}<option value={id}>{name}</option>{/each}
            </select>
        </div>
        <span class="meta">
            {#if c.thinking}<i>thinking…</i>{/if}
            {c.view?.bot.score ?? 0} pts · {strikes} strikes
        </span>
    </div>
    {#if c.nextBot !== c.bot}<p class="note">{BOT_NAMES[c.nextBot]} plays from the next game.</p>{/if}
    <SheetRows {rows} {size} />
</section>

<style>
    .panel { display: flex; flex-direction: column; gap: 6px; background: var(--paper); border-radius: 14px; padding: 12px; box-shadow: 0 1px 0 var(--edge); }
    .head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .pick { display: flex; align-items: center; gap: 8px; }
    label { font-family: var(--cond); font-size: 20px; font-weight: 700; }
    select { height: 32px; border-radius: 8px; border: 2px solid var(--ink); background: var(--paper); color: var(--ink); font-weight: 600; font-size: 14px; padding: 0 8px; }
    .meta { font-size: 13px; color: var(--muted); }
    .meta i { margin-right: 6px; }
    .note { margin: 0; font-size: 12px; color: var(--muted); }
</style>
```

`web/src/ui/DiceTray.svelte`:

```svelte
<script lang="ts">
    import type { GameController } from '../game/controller.svelte';
    import { ROWS } from '../game/rules';
    import Die from './Die.svelte';
    let { c, size = 56, compact = false }: { c: GameController; size?: number; compact?: boolean } = $props();
    const v = $derived(c.view);
    const idle = $derived(v?.phase === 'you_to_roll' && !c.rolling);
    const values = $derived(c.rolling ? c.shownDice : v?.dice);
    const faces = [['var(--die-face)', 'var(--die-pip)', 'White'], ['var(--die-face)', 'var(--die-pip)', 'White'],
        ['var(--red-band)', 'var(--on-band)', 'Red'], ['var(--yellow-band)', 'var(--on-yellow)', 'Yellow'],
        ['var(--green-band)', 'var(--on-band)', 'Green'], ['var(--blue-band)', 'var(--on-band)', 'Blue']];
    const canRoll = $derived(v?.phase === 'you_to_roll' && !c.busy);
    const rollLabel = $derived(c.rolling ? 'Rolling…' : canRoll ? 'Roll' : v?.phase === 'passive' || v?.phase === 'bot_to_roll' ? 'Bot rolled' : 'Rolled');
    const white = $derived(v?.dice && !idle && !c.rolling ? String(v.dice[0] + v.dice[1]) : '–');
</script>

<section aria-label="Dice" class="tray">
    {#if !compact || canRoll}
        <button type="button" class="roll" class:primary={canRoll} aria-disabled={!canRoll} onclick={() => c.roll()} style="width: {compact ? 110 : 148}px">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="4"></rect><circle cx="8.5" cy="8.5" r="1.2" fill="currentColor"></circle><circle cx="15.5" cy="15.5" r="1.2" fill="currentColor"></circle><circle cx="15.5" cy="8.5" r="1.2" fill="currentColor"></circle><circle cx="8.5" cy="15.5" r="1.2" fill="currentColor"></circle></svg>
            <span>{compact ? 'Roll' : rollLabel}</span>
        </button>
    {:else}
        <div class="white-badge" role="status" aria-label="White sum {white}"><span>White</span><b>{white}</b></div>
    {/if}
    {#each faces as f, i}
        <Die value={values?.[i] ?? 0} face={f[0]} pip={f[1]} {size} rolling={c.rolling} idle={idle}
            out={i >= 2 && !!v?.locked[i - 2]} label={i >= 2 && v?.locked[i - 2] ? `${f[2]} die, out of the game` : values?.[i] ? `${f[2]} die, ${values[i]}` : `${f[2]} die, not rolled`} />
    {/each}
    {#if !compact}
        <div class="sum"><span>White sum</span><b>{white}</b></div>
    {/if}
</section>

<style>
    .tray { display: flex; align-items: center; gap: 14px; }
    .roll { min-height: 48px; padding: 0; border: none; border-radius: 12px; display: inline-flex; align-items: center; justify-content: center; gap: 8px; font-weight: 700; font-size: 17px; flex-shrink: 0; cursor: not-allowed; background: var(--btn-muted-bg); color: var(--btn-muted-fg); }
    .roll.primary { cursor: pointer; background: var(--ink); color: var(--ground); box-shadow: 0 0 0 4px var(--ground), 0 0 0 6px var(--ink); }
    .sum { margin-left: 8px; display: flex; flex-direction: column; gap: 2px; }
    .sum span, .white-badge span { font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .sum b { font-family: var(--cond); font-weight: 800; font-size: 34px; line-height: 1; }
    .white-badge { min-height: 44px; width: 110px; box-sizing: border-box; border-radius: 12px; background: var(--paper); box-shadow: 0 1px 0 var(--edge); display: flex; align-items: center; justify-content: center; gap: 6px; flex-shrink: 0; }
    .white-badge b { font-family: var(--cond); font-weight: 800; font-size: 26px; line-height: 1; }
</style>
```

`web/src/ui/ActionBar.svelte`:

```svelte
<script lang="ts">
    import type { GameController } from '../game/controller.svelte';
    import { confirmState, strikeOpen } from '../game/rules';
    import MarkX from './MarkX.svelte';
    let { c }: { c: GameController } = $props();
    const v = $derived(c.view);
    const conf = $derived(v ? confirmState(v, c.sel, c.strikeSel) : { label: 'Mark', ok: false });
    const open = $derived(v ? strikeOpen(v, c.sel) : false);
    const strikes = $derived(v?.you.strikes ?? 0);
    const result = $derived(v?.over ? { title: v.you.score > v.bot.score ? 'You win' : v.you.score < v.bot.score ? 'Bot wins' : 'Draw',
        reason: v.over === 'locks' ? 'Two rows are locked.' : v.over === 'you_strikes' ? 'You took a fourth strike.' : 'The bot took a fourth strike.' } : null);
</script>

<section aria-label="Your move" class="bar">
    <div class="strikes">
        <span class="label">Strikes</span>
        {#each [0, 1, 2, 3] as i}
            {@const taken = i < strikes}
            {@const pencil = i === strikes && c.strikeSel}
            <button type="button" class="box" class:faded={!taken && !pencil && !(i === strikes && open)}
                aria-label={taken ? 'Strike taken' : i === strikes ? 'Strike box' : 'Strike box, unavailable'}
                aria-pressed={pencil} aria-disabled={!(i === strikes && open)} onclick={() => i === strikes && c.tapStrike()}>
                {#if taken || pencil}<MarkX pencil={pencil} anim={pencil ? 'draw' : (c.fresh[`you:strike:${i}`] ?? '')} />{/if}
            </button>
        {/each}
    </div>
    <div class="msg">
        {#if c.error}
            <div class="tip" role="alert">Something went wrong: {c.error}</div>
        {:else if result}
            <div class="result" role="status"><b>{result.title}</b><span class="score">{v?.you.score} – {v?.bot.score}</span><span class="reason">{result.reason}</span></div>
        {:else if c.tip}
            <div class="tip" role="alert">{c.tip}</div>
        {/if}
    </div>
    <button type="button" class="confirm" class:primary={conf.ok || !!c.error} aria-disabled={!conf.ok && !c.error}
        onclick={() => (c.error ? c.newGame() : c.confirm())}>{c.error ? 'New game' : conf.label}</button>
</section>

<style>
    .bar { display: flex; align-items: center; gap: 20px; background: var(--paper); border-radius: 14px; padding: 0 16px; height: 72px; box-sizing: border-box; flex-shrink: 0; box-shadow: 0 1px 0 var(--edge); }
    .strikes { display: flex; align-items: center; gap: 8px; }
    .label { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
    .box { position: relative; box-sizing: border-box; width: 40px; height: 40px; border-radius: 8px; padding: 0; border: 2px solid var(--ink); background: var(--paper); cursor: pointer; }
    .box.faded { opacity: 0.35; cursor: default; }
    .msg { flex: 1; display: flex; justify-content: flex-end; }
    .tip { display: flex; align-items: center; gap: 8px; background: var(--paper); border-radius: 10px; padding: 6px 12px; font-size: 13px; line-height: 1.3; max-height: 52px; overflow: hidden; box-shadow: inset 0 0 0 2px var(--ink); max-width: 460px; }
    .result { display: flex; align-items: baseline; gap: 12px; }
    .result b { font-family: var(--cond); font-weight: 800; font-size: 28px; }
    .score { font-family: var(--cond); font-weight: 700; font-size: 22px; }
    .reason { font-size: 14px; color: var(--muted); }
    .confirm { min-height: 48px; padding: 0 28px; border: none; border-radius: 12px; font-weight: 700; font-size: 17px; cursor: not-allowed; background: var(--btn-muted-bg); color: var(--btn-muted-fg); }
    .confirm.primary { cursor: pointer; background: var(--ink); color: var(--ground); }
</style>
```

`web/src/ui/Header.svelte`:

```svelte
<script lang="ts">
    import type { GameController } from '../game/controller.svelte';
    let { c }: { c: GameController } = $props();
    const p = $derived(c.view?.phase);
    const pill = $derived(p === 'over' ? 'Game over' : p === 'you_to_roll' || p === 'active' ? 'Your turn' : 'Bot’s turn');
</script>

<header class="head">
    <div class="word">Qwixxer</div>
    <div class="pill"><span class="dot"></span>{pill}</div>
    <div class="right">
        <span>You <b>{c.view?.you.score ?? 0}</b></span>
        <span class="muted">Bot <b>{c.view?.bot.score ?? 0}</b></span>
        <button type="button" class="theme" aria-label="Dark mode" aria-pressed={c.theme === 'dark'} onclick={() => c.toggleTheme()}>
            {#if c.theme === 'dark'}
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="var(--ink)" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path></svg>
            {:else}
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="var(--ink)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"></path></svg>
            {/if}
        </button>
    </div>
</header>

<style>
    .head { display: flex; align-items: center; justify-content: space-between; }
    .word { font-family: var(--cond); font-weight: 800; font-size: 30px; letter-spacing: 0.5px; }
    .pill { display: flex; align-items: center; gap: 10px; background: var(--ink); color: var(--ground); padding: 8px 16px; border-radius: 999px; font-weight: 600; font-size: 15px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--dot); }
    .right { display: flex; align-items: center; gap: 18px; font-weight: 600; font-size: 15px; }
    .right b { font-family: var(--cond); font-size: 22px; color: var(--ink); }
    .muted { color: var(--muted); }
    .theme { width: 44px; height: 44px; border-radius: 12px; border: none; background: var(--paper); box-shadow: 0 1px 0 var(--edge); display: flex; align-items: center; justify-content: center; cursor: pointer; padding: 0; }
</style>
```

- [ ] **Step 3: Desktop layout and App**

`web/src/ui/Desktop.svelte`:

```svelte
<script lang="ts">
    import type { GameController } from '../game/controller.svelte';
    import { chipsFor, ROWS, sheetModel, type LogEntry } from '../game/rules';
    import ActionBar from './ActionBar.svelte';
    import BotPanel from './BotPanel.svelte';
    import DiceTray from './DiceTray.svelte';
    import Header from './Header.svelte';
    import Log from './Log.svelte';
    import SheetRows from './SheetRows.svelte';
    let { c }: { c: GameController } = $props();
    const rows = $derived(c.view ? sheetModel(c.view, 'you', { sel: c.sel, strikeSel: c.strikeSel, committing: c.committing, fresh: c.fresh, hidden: [] }) : []);
    const current = $derived<LogEntry | null>(c.view?.phase === 'active'
        ? { n: c.rollNo, lines: [{ who: 'You', chips: [], empty: 'choosing…' }, { who: 'Bot', chips: chipsFor(c.view.bot_white ? [c.view.bot_white] : [], false), empty: '–' }] }
        : c.view?.phase === 'passive' ? { n: c.rollNo, lines: [{ who: 'Bot', chips: [], empty: '…' }, { who: 'You', chips: [], empty: 'choosing…' }] } : null);
    const scoreParts = $derived(c.view ? ROWS.map((r, i) => ({ r, text: `${['Red', 'Yellow', 'Green', 'Blue'][i]} ${c.view!.you.row_points[i]}` })) : []);
</script>

<div class="desk">
    <Header {c} />
    <div class="body">
        <main>
            <DiceTray {c} />
            <section aria-label="Your sheet" class="you">
                <h2>You</h2>
                <div class="sheet"><SheetRows {rows} size={58} onTap={(cell) => c.tapSquare(cell.sq)} /></div>
                <div class="score">
                    {#each scoreParts as p}<span class="part" style="background: {p.r === 'yellow' ? 'var(--score-yellow)' : `var(--${p.r}-band)`}; color: {p.r === 'yellow' ? 'var(--on-yellow)' : 'var(--on-band)'}">{p.text}</span>{/each}
                    <span class="part strikes">−{5 * (c.view?.you.strikes ?? 0)} strikes</span>
                    <b>= {c.view?.you.score ?? 0}</b>
                </div>
            </section>
            <ActionBar {c} />
        </main>
        <aside>
            <BotPanel {c} />
            <div class="logbox"><Log entries={c.log} {current} /></div>
        </aside>
    </div>
</div>

<style>
    .desk { width: 1280px; height: 800px; box-sizing: border-box; padding: 22px 32px; display: flex; flex-direction: column; gap: 16px; overflow: hidden; margin: 0 auto; }
    .body { display: flex; gap: 24px; flex: 1; min-height: 0; }
    main { flex: 1; display: flex; flex-direction: column; gap: 14px; min-width: 0; }
    .you { display: flex; flex-direction: column; gap: 8px; }
    h2 { margin: 0; font-family: var(--cond); font-size: 22px; font-weight: 700; }
    .sheet { background: var(--paper); border-radius: 14px; padding: 10px; width: fit-content; box-shadow: 0 1px 0 var(--edge), 0 8px 24px var(--shadow); }
    .score { display: flex; gap: 8px; font-size: 14px; color: var(--soft); align-items: center; }
    .part { padding: 2px 8px; border-radius: 6px; font-weight: 600; }
    .part.strikes { background: var(--btn-muted-bg); color: var(--ink); }
    main :global(.bar) { margin-top: auto; }
    aside { width: 392px; display: flex; flex-direction: column; gap: 14px; min-height: 0; }
    .logbox { flex: 1; min-height: 0; background: var(--paper); border-radius: 14px; padding: 12px 14px; box-shadow: 0 1px 0 var(--edge); overflow: hidden; }
</style>
```

Replace `web/src/ui/App.svelte`:

```svelte
<script lang="ts">
    import { onMount } from 'svelte';
    import { createEngine } from '../engine/client';
    import { GameController } from '../game/controller.svelte';
    import Desktop from './Desktop.svelte';

    const store = { get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
        set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } } };
    const c = new GameController(createEngine(), store);
    onMount(() => { c.init(); });
    $effect(() => { document.documentElement.dataset.theme = c.theme; });

    // Spec "Implementation notes · Keyboard": Enter confirms, Space rolls,
    // Escape clears. A focused button keeps its native Enter/Space click.
    function onKey(e: KeyboardEvent) {
        const t = e.target as HTMLElement | null;
        if (t && (t.closest('button') || t.closest('select') || t.closest('input'))) return;
        if (e.key === 'Enter') { e.preventDefault(); c.confirm(); }
        else if (e.key === ' ' && c.view?.phase === 'you_to_roll') { e.preventDefault(); c.roll(); }
        else if (e.key === 'Escape') c.clearSelection();
    }
</script>

<svelte:window onkeydown={onKey} />
<Desktop {c} />
```

- [ ] **Step 4: Verify**

Run: `cd web && npm run check && npm test && npm run build` → 0 errors, tests pass, build succeeds.
Run: `npm run dev`, open `/qwixxer/` at 1280×800 and play one full game. Check against the canvas prototype:
1. The bot rolls first (dice tumble), the pill reads "Bot’s turn", your legal white-sum squares are heavy; Pass/Mark works.
2. Your Roll: tumble, "thinking…" next to Bot, then its pencil X draws in; only legal squares are heavy; second pick narrows; strike box only with nothing selected.
3. Confirm inks pencil Xs; the bot's turn follows after a beat; its colour mark lands ~450 ms after its white sum.
4. A locked row turns grey, its die greys out; a lock symbol goes regular at five marks including a pencil fifth.
5. Theme toggle switches every colour; reload restores the game, theme and bot choice; switching the bot shows "plays from the next game".
6. No vertical scroll at 1280×800.

- [ ] **Step 5: Commit**

```bash
git add web/src/ui
git commit -m "feat(web): desktop paper-sheet UI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Phone layout and responsive switch

Portrait phone design (canvas P0–P2): sheet turned sideways (one column per row, 48 px squares, lock at the bottom), right column with score, 2×2 strikes and a tall confirm, header Bot button with the bot's latest chips that opens a drawer (bot sheet, picker, log). No status strip. Tip/result line in the spare space at the bottom. Below 1100 px wide or in portrait, the phone layout is used.

**Files:**
- Create: `web/src/ui/SheetColumns.svelte`, `web/src/ui/Phone.svelte`
- Modify: `web/src/ui/App.svelte`

**Interfaces:**
- Consumes: Task 8 `Cell`, `LockCell`, `MarkX`, `Chip`, `Log`, `BotPanel`, `DiceTray` (with `compact`), Task 6 `sheetModel`, `chipsFor`, `confirmState`, `strikeOpen`, `ROWS`; Task 7 `GameController`.
- Produces: `Phone.svelte`; `App.svelte` chooses `Desktop` or `Phone` via `matchMedia('(max-width: 1099px), (orientation: portrait)')`.

- [ ] **Step 1: `web/src/ui/SheetColumns.svelte`**

```svelte
<script lang="ts">
    import type { CellModel, RowModel } from '../game/rules';
    import Cell from './Cell.svelte';
    import LockCell from './LockCell.svelte';
    let { rows, size = 48, onTap }: { rows: RowModel[]; size?: number; onTap: (c: CellModel) => void } = $props();
</script>

<div class="cols">
    {#each rows as row (row.row)}
        <div class="band {row.band}" aria-label={row.label} style="--band: var(--{row.color}-band); --dark: var(--{row.color}-dark)">
            {#each row.cells as cell (cell.n)}<Cell {cell} color={row.color} {size} {onTap} />{/each}
            <span style="height: 3px"></span>
            <LockCell look={row.lock} color={row.color} {size} label="{row.label} lock" />
        </div>
    {/each}
</div>

<style>
    .cols { display: flex; gap: 6px; background: var(--paper); border-radius: 14px; padding: 6px; box-shadow: 0 1px 0 var(--edge), 0 8px 24px var(--shadow); }
    .band { display: flex; flex-direction: column; gap: 3px; padding: 4px; border-radius: 10px; background: var(--band); }
    .locking { background: repeating-linear-gradient(135deg, var(--band) 0 10px, var(--dark) 10px 20px); }
    .locked { background: var(--locked-band); }
</style>
```

- [ ] **Step 2: `web/src/ui/Phone.svelte`**

```svelte
<script lang="ts">
    import type { GameController } from '../game/controller.svelte';
    import { chipsFor, confirmState, ROWS, sheetModel, strikeOpen, type Chip, type LogEntry } from '../game/rules';
    import BotPanel from './BotPanel.svelte';
    import ChipView from './Chip.svelte';
    import DiceTray from './DiceTray.svelte';
    import Log from './Log.svelte';
    import MarkX from './MarkX.svelte';
    import SheetColumns from './SheetColumns.svelte';
    let { c }: { c: GameController } = $props();
    let drawer = $state(false);
    const v = $derived(c.view);
    // Square size: 48 px on a 390×844 phone, down to 36 px on short screens.
    let h = $state(844);
    $effect(() => { const f = () => (h = window.innerHeight); f(); window.addEventListener('resize', f); return () => window.removeEventListener('resize', f); });
    const size = $derived(Math.max(36, Math.min(48, Math.floor((h - 12 - 16 - 44 - 44 - 20 - 12 - 8 - 33 - 3 - 50) / 12))));
    const rows = $derived(v ? sheetModel(v, 'you', { sel: c.sel, strikeSel: c.strikeSel, committing: c.committing, fresh: c.fresh, hidden: [] }) : []);
    const conf = $derived(v ? confirmState(v, c.sel, c.strikeSel) : { label: 'Mark', ok: false });
    const open = $derived(v ? strikeOpen(v, c.sel) : false);
    // The bot's most recent marks ride on the header button (its revealed white sum on your turn).
    const botChips = $derived<Chip[]>(v?.phase === 'active' && v.bot_white ? chipsFor([v.bot_white], false)
        : (c.log[0]?.lines.find((l) => l.who === 'Bot')?.chips ?? []));
    const current = $derived<LogEntry | null>(v?.phase === 'active'
        ? { n: c.rollNo, lines: [{ who: 'You', chips: [], empty: 'choosing…' }, { who: 'Bot', chips: chipsFor(v.bot_white ? [v.bot_white] : [], false), empty: '–' }] } : null);
    const result = $derived(v?.over ? `${v.you.score > v.bot.score ? 'You win' : v.you.score < v.bot.score ? 'Bot wins' : 'Draw'} ${v.you.score} – ${v.bot.score}` : '');
</script>

<div class="phone">
    <header class="head">
        <div class="word">Qwixxer</div>
        <div class="right">
            <span>You <b>{v?.you.score ?? 0}</b></span>
            <button type="button" class="botbtn" aria-label="Bot, {v?.bot.score ?? 0} points. Open bot sheet and log" onclick={() => (drawer = true)}>
                Bot <b>{v?.bot.score ?? 0}</b>
                {#each botChips as ch}<ChipView chip={ch} size={20} />{/each}
            </button>
            <button type="button" class="theme" aria-label="Dark mode" aria-pressed={c.theme === 'dark'} onclick={() => c.toggleTheme()}>
                {#if c.theme === 'dark'}
                    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="var(--ink)" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path></svg>
                {:else}
                    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="var(--ink)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"></path></svg>
                {/if}
            </button>
        </div>
    </header>
    <DiceTray {c} size={38} compact />
    <div class="body">
        <SheetColumns {rows} {size} onTap={(cell) => c.tapSquare(cell.sq)} />
        <aside class="side">
            <div class="card">
                <span class="label">Score</span>
                {#each ROWS as r, i}<span class="part" style="background: {r === 'yellow' ? 'var(--score-yellow)' : `var(--${r}-band)`}; color: {r === 'yellow' ? 'var(--on-yellow)' : 'var(--on-band)'}">{['Red', 'Yellow', 'Green', 'Blue'][i]} {v?.you.row_points[i] ?? 0}</span>{/each}
                <span class="part strikes">Strikes −{5 * (v?.you.strikes ?? 0)}</span>
                <b class="total">= {v?.you.score ?? 0}</b>
            </div>
            <div class="card">
                <span class="label">Strikes</span>
                <div class="grid">
                    {#each [0, 1, 2, 3] as i}
                        {@const strikes = v?.you.strikes ?? 0}
                        {@const pencil = i === strikes && c.strikeSel}
                        <button type="button" class="box" class:faded={!(i < strikes) && !pencil && !(i === strikes && open)}
                            aria-label={i < strikes ? 'Strike taken' : 'Strike box'} aria-pressed={pencil} aria-disabled={!(i === strikes && open)}
                            onclick={() => i === strikes && c.tapStrike()}>
                            {#if i < strikes || pencil}<MarkX pencil={pencil} anim={pencil ? 'draw' : (c.fresh[`you:strike:${i}`] ?? '')} />{/if}
                        </button>
                    {/each}
                </div>
            </div>
            <button type="button" class="confirm" class:primary={conf.ok || !!c.error} aria-disabled={!conf.ok && !c.error}
                onclick={() => (c.error ? c.newGame() : c.confirm())}>{c.error ? 'New game' : conf.label}</button>
        </aside>
    </div>
    <div class="msg" role="status">{c.error ? `Something went wrong: ${c.error}` : result || c.tip}</div>

    {#if drawer}
        <button type="button" class="scrim" aria-label="Close" onclick={() => (drawer = false)}></button>
        <section class="drawer" role="dialog" aria-label="Bot and log">
            <div class="handle"></div>
            <BotPanel {c} size={24} />
            <Log entries={c.log} {current} chip={24} />
            <button type="button" class="close" onclick={() => (drawer = false)}>Close</button>
        </section>
    {/if}
</div>

<style>
    .phone { position: relative; max-width: 430px; height: 100dvh; margin: 0 auto; box-sizing: border-box; padding: 12px 12px 16px; display: flex; flex-direction: column; gap: 10px; overflow: hidden; }
    .head { display: flex; align-items: center; justify-content: space-between; height: 44px; }
    .word { font-family: var(--cond); font-weight: 800; font-size: 26px; }
    .right { display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 14px; }
    .right b { font-family: var(--cond); font-size: 20px; }
    .botbtn, .theme { height: 44px; border-radius: 12px; border: none; background: var(--paper); box-shadow: 0 1px 0 var(--edge); color: var(--muted); display: flex; align-items: center; gap: 5px; padding: 0 10px; cursor: pointer; }
    .theme { width: 44px; justify-content: center; color: var(--ink); font-size: 18px; }
    .body { display: flex; gap: 10px; align-items: stretch; }
    .side { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 10px; }
    .card { display: flex; flex-direction: column; gap: 5px; background: var(--paper); border-radius: 14px; padding: 8px 6px; box-shadow: 0 1px 0 var(--edge); }
    .label { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; padding-left: 4px; }
    .part { padding: 2px 6px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    .part.strikes { background: var(--btn-muted-bg); color: var(--ink); }
    .total { font-family: var(--cond); font-size: 22px; text-align: right; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; justify-items: center; }
    .box { position: relative; box-sizing: border-box; width: 40px; height: 40px; border-radius: 8px; padding: 0; border: 2px solid var(--ink); background: var(--paper); }
    .box.faded { opacity: 0.35; }
    .confirm { margin-top: auto; min-height: 64px; border: none; border-radius: 14px; font-weight: 700; font-size: 18px; background: var(--btn-muted-bg); color: var(--btn-muted-fg); }
    .confirm.primary { background: var(--ink); color: var(--ground); }
    .msg { font-size: 13px; line-height: 1.3; color: var(--soft); min-height: 34px; }
    .scrim { position: absolute; inset: 0; border: none; background: var(--scrim); }
    .drawer { position: absolute; left: 0; right: 0; bottom: 0; max-height: 75%; box-sizing: border-box; background: var(--paper); border-radius: 20px 20px 0 0; padding: 8px 14px 16px; display: flex; flex-direction: column; gap: 12px; overflow: hidden; }
    .handle { align-self: center; width: 40px; height: 5px; border-radius: 3px; background: var(--edge); }
    .close { align-self: flex-end; min-height: 44px; padding: 0 16px; border-radius: 12px; border: none; background: var(--ground); color: var(--ink); font-weight: 600; }
</style>
```

- [ ] **Step 3: Responsive switch in `App.svelte`**

Replace the markup and add the media query:

```svelte
<script lang="ts">
    import { onMount } from 'svelte';
    import { createEngine } from '../engine/client';
    import { GameController } from '../game/controller.svelte';
    import Desktop from './Desktop.svelte';
    import Phone from './Phone.svelte';

    const store = { get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
        set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } } };
    const c = new GameController(createEngine(), store);
    const mq = window.matchMedia('(max-width: 1099px), (orientation: portrait)');
    let phone = $state(mq.matches);
    onMount(() => {
        c.init();
        const f = () => (phone = mq.matches);
        mq.addEventListener('change', f);
        return () => mq.removeEventListener('change', f);
    });
    $effect(() => { document.documentElement.dataset.theme = c.theme; });

    function onKey(e: KeyboardEvent) {
        const t = e.target as HTMLElement | null;
        if (t && (t.closest('button') || t.closest('select') || t.closest('input'))) return;
        if (e.key === 'Enter') { e.preventDefault(); c.confirm(); }
        else if (e.key === ' ' && c.view?.phase === 'you_to_roll') { e.preventDefault(); c.roll(); }
        else if (e.key === 'Escape') c.clearSelection();
    }
</script>

<svelte:window onkeydown={onKey} />
{#if phone}<Phone {c} />{:else}<Desktop {c} />{/if}
```

- [ ] **Step 4: Verify**

Run: `cd web && npm run check && npm test && npm run build`.
Run `npm run dev`; in the browser's device toolbar choose 390×844, then 375×667:
1. Sheet columns: red/yellow 2→12, green/blue 12→2, locks at the bottom; squares 48 px on 390×844 and no smaller than 36 px on 375×667; no page scroll.
2. Before your roll the Roll button sits in the dice row; after rolling its slot shows "White N".
3. The Bot header button shows the bot's latest chips; tapping opens the drawer with picker and log; Close returns.
4. Tips and the game result appear in the line under the sheet.
5. Rotating to landscape on a wide screen switches to the desktop layout.

- [ ] **Step 5: Commit**

```bash
git add web/src/ui
git commit -m "feat(web): phone layout and responsive switch

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Docs and final checks

**Files:**
- Modify: `docs/ARCHITECTURE.md` (web section: replace the `WebGame` / `StateExplorer` description, keep Task 4's latency paragraph), `web/README.md`

- [ ] **Step 1: Update `docs/ARCHITECTURE.md`**

Replace the "WASM Crate (`web/src/lib.rs`)" subsection and the `web/` tree lines with: the module list from this plan's File structure (Rust and frontend), the worker architecture (one `WasmGame` per worker; `WorkerEngine` RPC), the `Session` phases (`bot_to_roll → passive → you_to_roll → active → …`, `over`), the bots (Master+ / Master / Student and `MASTER_PLUS_SPEC`), persistence keys (`qwixxer.save.v1`, `qwixxer.prefs.v1`), and a pointer to the spec. Remove statements about `StateExplorer` and the explorer page, noting the explorer is deferred.

- [ ] **Step 2: Update `web/README.md`**

Rewrite the intro ("Browser Qwixx against three bots: Master+, Master and Student") and the commands: `./web/build.sh` (wasm), `cd web && npm install`, `npm run dev`, `npm test`, `npm run check`, `npm run build`, and `node web/scripts/bench.mjs master_plus 10` for bot latency. Remove explorer references; keep the prerequisites list; note that `champion.txt` and `pair_model/model.mpk` are embedded at build time.

- [ ] **Step 3: Full verification**

Run, from the repo root:

```bash
cargo test --lib --release
(cd web && cargo test --release)
./web/build.sh
(cd web && npm run check && npm test && npm run build)
git status --short
```

Expected: all Rust and TS tests pass, svelte-check 0 errors, build succeeds, `git status` shows only the intended files plus the user's untracked `.superpowers/`, `AGENTS.md`, `web-ui.txt` (and `web/Cargo.lock` only if dependencies changed).

- [ ] **Step 4: Commit**

```bash
git add docs/ARCHITECTURE.md web/README.md
git commit -m "docs: web UI rewrite architecture and README

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
