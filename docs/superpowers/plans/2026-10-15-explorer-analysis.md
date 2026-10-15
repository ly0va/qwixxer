# Explorer Position Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The explorer's Position card shows a play-out win chance that keeps refining while the explorer is visible; the move list shows the net's estimate as a score out of 1000.

**Architecture:** A stateless Rust call `Explorer::analyze(position, start, count)` plays games `start..start+count` (Master vs Master, batched net) with dice seeded per game number, so batches compose exactly. The worker exposes it as an `analyze` command (one request, one reply). `ExplorerController` loops batches (~100 ms each, adaptive) while active, accumulating totals keyed by the position without its dice; the Svelte card renders them.

**Tech Stack:** Rust (root crate `qwixxer`, web crate `qwixxer-web`, wasm-bindgen), TypeScript, Svelte 5 runes, Vitest.

**Spec:** `docs/superpowers/specs/2026-10-15-explorer-analysis-design.md` (builds on `docs/superpowers/specs/2026-10-14-state-explorer-design.md`).

## Global Constraints

- Branch `explorer-analysis`. Never commit `.superpowers/`, `AGENTS.md`, `web-ui.txt`; never edit `AGENTS.md`.
- Format Rust with `rustfmt --edition 2021 <files>` only, never `cargo fmt` on the whole crate.
- After any `cargo`/`wasm-pack` run in `web/`, revert `web/Cargo.lock` with `git checkout web/Cargo.lock` (no dependency changes in this plan).
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not touch `champion.txt`, `pair_model/`, `dqn_model/`. Do not push.
- Rebuild WASM with `./web/build.sh` (repo root) after Rust changes that the browser must see.
- Copy, verbatim from the spec: "chance you win"; "± 1.2 · 3,200 games"; "*N* legal · score after each, out of 1000"; "…" on a cold open; margin text unchanged ("Opponent ahead by about 9 (± 15)", "You ahead by…", "Level (± 15)").
- Win chance p = (wins + ties / 2) / games; ± = 1.96 · √(p(1 − p) / games); one decimal once ± < 0.5 points.
- Batch target 100 ms; restart on marks/strikes/closed rows/roller change, not on dice-only change.
- Keep runs short: no long benches; throughput measurement uses at most ~2 s of games.

## Review Focus

1. Worker reset or engine error during analysis: the loop stops without an error banner on the card, and the next edit or re-entering the explorer starts it again.
2. Game-over position: no `analyze` call is ever sent; editing back to a live position restarts analysis.
3. Rapid edits: never more than one `analyze` call in flight; a reply for an older position is discarded.
4. Leaving the explorer (Play tab) or hiding the tab while a batch is in flight: no further batches are sent; returning resumes from the kept totals.
5. Swap sides / setting "Opponent's roll": the totals restart (the key includes sheets and roller).

Each is pinned by a Vitest case in Task 4.

## Rulings made while planning

- Play-outs use `BatchedRollouts` (Lite fidelity, as Master+'s own rollouts use): active phase 1 is judged against current opponent states, without Master's nested opponent prediction. Cost if wrong: play-out players are a slightly simplified Master.
- `Answer.eval` stays in the engine reply (unused by the card now); removing it is out of scope.
- The batch size adapts per controller, not per position; "the first batch uses the lower bound" means the controller's first batch.

---

### Task 1: Rust `analyze` engine call

**Files:**
- Modify: `web/src/explorer.rs` (new `Analysis` type, `Explorer::analyze`, tests in the existing `mod tests`)
- Modify: `web/src/lib.rs` (`WasmExplorer::analyze`)

**Interfaces:**
- Produces (Rust): `pub struct Analysis { pub games: u32, pub wins: u32, pub ties: u32, pub diff_sum: f64, pub diff_sq_sum: f64 }` (Serialize, PartialEq, Debug, Clone); `Explorer::analyze(&self, pos: &Position, start: u32, count: u32) -> Result<Analysis, String>`.
- Produces (WASM): `WasmExplorer.analyze(position: string, start: number, count: number): string` returning the JSON of `Analysis`.

- [ ] **Step 1: Write the failing tests** (append inside `mod tests` in `web/src/explorer.rs`, which already has `sheet()` and `pos(dice, roller)` helpers)

```rust
    #[test]
    fn analysis_batches_compose_and_replay() {
        let ex = Explorer::new();
        let p = pos(None, Roller::You);
        let a = ex.analyze(&p, 0, 40).unwrap();
        let b = ex.analyze(&p, 40, 60).unwrap();
        let all = ex.analyze(&p, 0, 100).unwrap();
        assert_eq!(all.games, 100);
        assert_eq!(
            (a.games + b.games, a.wins + b.wins, a.ties + b.ties),
            (all.games, all.wins, all.ties)
        );
        // Diffs are integers, so f64 sums are exact.
        assert_eq!(a.diff_sum + b.diff_sum, all.diff_sum);
        assert_eq!(a.diff_sq_sum + b.diff_sq_sum, all.diff_sq_sum);
        assert_eq!(ex.analyze(&p, 0, 100).unwrap(), all);
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
        let p = (a.wins as f64 + a.ties as f64 / 2.0) / a.games as f64;
        assert!((0.35..0.65).contains(&p), "p {p}");
        assert!(a.diff_sq_sum > 0.0);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd web && cargo test --release analysis 2>&1 | tail -5; cd .. && git checkout web/Cargo.lock`
Expected: compile error, `no method named analyze` / `cannot find type Analysis`.

- [ ] **Step 3: Implement**

In `web/src/explorer.rs`, extend the imports:

```rust
use qwixxer::strategy::search::{context_seed, phi, sample_player_seed, SearchBot, WinProb};
use qwixxer::strategy::sim::{BatchedRollouts, SimGame};
use rand::rngs::SmallRng;
use rand::SeedableRng;
```

Add the type next to `Answer`:

```rust
/// Spec "Engine" (`2026-10-15-explorer-analysis-design.md`): totals of
/// play-out games `start .. start + count` of a position. Batches add up.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Analysis {
    pub games: u32,
    /// Games you win (higher final score) and games tied.
    pub wins: u32,
    pub ties: u32,
    /// Sum, and sum of squares, of your final score minus the opponent's.
    pub diff_sum: f64,
    pub diff_sq_sum: f64,
}
```

Add to `impl Explorer` (after `explore`):

```rust
    /// Plays games `start .. start + count` from the position before the
    /// roll (its dice ignored), the roller first, both seats Master (the
    /// pair net, batched across games). Game k's dice depend only on the
    /// position and k, so batches compose and replay exactly.
    pub fn analyze(&self, pos: &Position, start: u32, count: u32) -> Result<Analysis, String> {
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
        let mut turns = 0;
        while !driver.all_over() {
            driver.step_turn();
            turns += 1;
            assert!(turns <= 200, "play-out exceeded 200 turns");
        }
        let mut a = Analysis {
            games: 0,
            wins: 0,
            ties: 0,
            diff_sum: 0.0,
            diff_sq_sum: 0.0,
        };
        for sim in &driver.sims {
            let d = (sim.states[0].count_points() - sim.states[1].count_points()) as f64;
            a.games += 1;
            a.wins += (d > 0.0) as u32;
            a.ties += (d == 0.0) as u32;
            a.diff_sum += d;
            a.diff_sq_sum += d * d;
        }
        Ok(a)
    }
```

If `BatchedRollouts`, `SimGame` fields, `context_seed` or `sample_player_seed` are not `pub` from the root crate, make the minimal visibility change in `src/strategy/sim.rs` / `src/strategy/search.rs` (they are `pub` at the time of writing).

In `web/src/lib.rs`, inside `#[wasm_bindgen] impl WasmExplorer`, after `explore`:

```rust
    /// Play-out totals for games `start .. start + count` (spec
    /// `2026-10-15-explorer-analysis-design.md`), as JSON.
    pub fn analyze(&self, position: &str, start: u32, count: u32) -> Result<String, JsError> {
        let pos: explorer::Position = serde_json::from_str(position).map_err(js)?;
        let a = self.inner.analyze(&pos, start, count).map_err(js)?;
        Ok(serde_json::to_string(&a).unwrap())
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `rustfmt --edition 2021 web/src/explorer.rs web/src/lib.rs && cd web && cargo test --release 2>&1 | grep -E "test result|FAILED|panicked"; cd .. && git checkout web/Cargo.lock`
Expected: `test result: ok. 31 passed` (27 existing + 4 new), no failures. If `analysis_batches_compose_and_replay` fails only on the f64 sums or on totals, the cause is batch-size-dependent float results in the net: stop and report (do not loosen the test).

- [ ] **Step 5: Commit**

```bash
git add web/src/explorer.rs web/src/lib.rs
git commit -m "feat(web): explorer analyze call plays a position out in composable batches

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Worker `analyze` command and client

**Files:**
- Modify: `web/src/engine/protocol.ts` (type `Analysis`, `Engine.analyze`)
- Modify: `web/src/engine/serve.ts` (`case 'analyze'`)
- Modify: `web/src/engine/client.ts` (`WorkerEngine.analyze`)
- Test: `web/src/engine/serve.test.ts`

**Interfaces:**
- Consumes: `WasmExplorer.analyze(position: string, start: number, count: number): string` (Task 1).
- Produces: `export interface Analysis { games: number; wins: number; ties: number; diff_sum: number; diff_sq_sum: number }` in `protocol.ts`; `Engine.analyze(position: Position, start: number, count: number): Promise<Analysis>`.

- [ ] **Step 1: Write the failing test**

In `web/src/engine/serve.test.ts`, add to `FakeExplorer`:

```ts
    analyze(json: string, start: number, count: number) {
        const p = JSON.parse(json);
        if (p.over) throw new Error('the game is over');
        return JSON.stringify({ games: count, wins: start, ties: 0, diff_sum: 0, diff_sq_sum: 0 });
    }
```

and add a test inside the existing `describe`:

```ts
    it('analyze shares the explorer and passes the batch range', async () => {
        const { send } = setup(0);
        const made = FakeExplorer.made;
        const r = await send(1, 'analyze', [{ roller: 'you' }, 40, 25]);
        expect(r).toEqual({ id: 1, ok: true, result: { games: 25, wins: 40, ties: 0, diff_sum: 0, diff_sq_sum: 0 } });
        await send(2, 'explore', [{ roller: 'you' }]);
        expect(FakeExplorer.made).toBe(made + 1);
        const bad = await send(3, 'analyze', [{ over: true }, 0, 10]);
        expect(bad).toMatchObject({ id: 3, ok: false, error: 'the game is over', kind: 'engine' });
    });
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd web && npx vitest run src/engine/serve.test.ts`
Expected: FAIL, the reply is `{ ok: false, error: 'unknown command analyze' }`.

- [ ] **Step 3: Implement**

`web/src/engine/serve.ts`, in `run`, after `case 'explore'`:

```ts
            case 'analyze':
                explorer ??= new wasm.WasmExplorer();
                return JSON.parse(explorer.analyze(JSON.stringify(args[0]), args[1], args[2]));
```

`web/src/engine/protocol.ts`, after `Answer`:

```ts
/** Mirrors explorer.rs `Analysis`: play-out totals of one batch (or a running sum). */
export interface Analysis { games: number; wins: number; ties: number; diff_sum: number; diff_sq_sum: number }
```

and in `interface Engine`, after `explore`:

```ts
    /** Play-out games `start .. start + count` of the position (its dice ignored). */
    analyze(position: Position, start: number, count: number): Promise<Analysis>;
```

`web/src/engine/client.ts`: import `type Analysis` and add after `explore(...)`:

```ts
    analyze(position: Position, start: number, count: number) { return this.call<Analysis>('analyze', position, start, count); }
```

`serve.ts`'s `Wasm` type takes `WasmExplorer` from `../../pkg/qwixxer_web.js`; the generated `.d.ts` gains `analyze` once Task 1's WASM is built. Run `./web/build.sh` (repo root) first if `npm run check` reports `analyze` missing on `WasmExplorer`.

- [ ] **Step 4: Run tests and type check**

Run: `./web/build.sh >/dev/null && git checkout web/Cargo.lock; cd web && npx vitest run && npm run check 2>&1 | tail -3`
Expected: all Vitest tests pass; svelte-check `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add web/src/engine/protocol.ts web/src/engine/serve.ts web/src/engine/client.ts web/src/engine/serve.test.ts
git commit -m "feat(web): worker analyze command

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Analysis numbers and score in the position model

**Files:**
- Modify: `web/src/game/position.ts`
- Test: `web/src/game/position.test.ts`

**Interfaces:**
- Consumes: `Analysis` from `../engine/protocol` (Task 2).
- Produces (all exported from `position.ts`):
  - `analysisKey(p: ExPosition): string` — equal for positions differing only in dice.
  - `isOver(p: ExPosition): boolean` — 4 strikes on a sheet or ≥ 2 closed rows.
  - `addAnalysis(a: Analysis, b: Analysis): Analysis`
  - `ZERO_ANALYSIS: Analysis`
  - `winChance(t: Analysis): { p: number; pm: number }`
  - `analysisMargin(t: Analysis): { margin: number; sigma: number }`
  - `chanceText(w: { p: number; pm: number }): string`
  - `gamesText(w: { p: number; pm: number }, games: number): string`
  - `score(p: number): number`

- [ ] **Step 1: Write the failing tests** (append to `web/src/game/position.test.ts`; add the new names to its import from `./position`)

```ts
describe('analysis numbers', () => {
    const t = (games: number, wins: number, ties: number, diff_sum = 0, diff_sq_sum = 0) => ({ games, wins, ties, diff_sum, diff_sq_sum });

    it('analysisKey ignores dice only', () => {
        const p = emptyPosition();
        const withDice = { ...p, dice: [1, 2, 3, 4, 5, 6] };
        expect(analysisKey(withDice)).toBe(analysisKey(p));
        expect(analysisKey(toggleSquare(p, 'you', 0, 0))).not.toBe(analysisKey(p));
        expect(analysisKey({ ...p, roller: 'opponent' })).not.toBe(analysisKey(p));
        expect(analysisKey({ ...p, locked: [true, false, false, false] })).not.toBe(analysisKey(p));
        expect(analysisKey(swapSides(toggleSquare(p, 'you', 0, 0)))).not.toBe(analysisKey(toggleSquare(p, 'you', 0, 0)));
    });

    it('isOver: four strikes or two closed rows', () => {
        const p = emptyPosition();
        expect(isOver(p)).toBe(false);
        expect(isOver({ ...p, opponent: { ...p.opponent, strikes: 4 } })).toBe(true);
        expect(isOver({ ...p, locked: [true, false, false, false] })).toBe(false);
        expect(isOver({ ...p, locked: [true, false, true, false] })).toBe(true);
    });

    it('adds batches and derives the win chance, ± and margin', () => {
        const s = addAnalysis(addAnalysis(ZERO_ANALYSIS, t(100, 50, 10, 200, 10000)), t(300, 170, 20, 600, 30000));
        expect(s).toEqual(t(400, 220, 30, 800, 40000));
        const w = winChance(s);
        expect(w.p).toBeCloseTo((220 + 15) / 400);
        expect(w.pm).toBeCloseTo(1.96 * Math.sqrt(w.p * (1 - w.p) / 400));
        const m = analysisMargin(s);
        expect(m.margin).toBe(2);
        expect(m.sigma).toBeCloseTo(Math.sqrt(100 - 4));
    });

    it('formats: whole percent until ± < 0.5 points, then one decimal', () => {
        expect(chanceText({ p: 0.5432, pm: 0.012 })).toBe('54%');
        expect(chanceText({ p: 0.5432, pm: 0.0049 })).toBe('54.3%');
        expect(gamesText({ p: 0.54, pm: 0.012 }, 3200)).toBe('± 1.2 · 3,200 games');
        expect(gamesText({ p: 0.54, pm: 0.3 }, 1)).toBe('± 30.0 · 1 game');
    });

    it('score is P(win) × 1000, rounded', () => {
        expect(score(0.5464)).toBe(546);
        expect(score(0)).toBe(0);
        expect(score(1)).toBe(1000);
    });
});
```

(If `swapSides` / `toggleSquare` / `emptyPosition` / `describe` are not already imported in the test file, add them.)

- [ ] **Step 2: Run to verify failure**

Run: `cd web && npx vitest run src/game/position.test.ts`
Expected: FAIL, `analysisKey is not a function` (or import errors).

- [ ] **Step 3: Implement** (append to `web/src/game/position.ts`; import `type Analysis` from `'../engine/protocol'` alongside the existing protocol imports)

```ts
/** Spec "Controller": analysis restarts when this changes (dice don't count). */
export const analysisKey = (p: ExPosition) => JSON.stringify([p.you, p.opponent, p.locked, p.roller]);

/** Four strikes on a sheet or two closed rows: nothing to play out. */
export const isOver = (p: ExPosition) =>
    p.you.strikes >= 4 || p.opponent.strikes >= 4 || closedRows(p).filter(Boolean).length >= 2;

export const ZERO_ANALYSIS: Analysis = { games: 0, wins: 0, ties: 0, diff_sum: 0, diff_sq_sum: 0 };

export const addAnalysis = (a: Analysis, b: Analysis): Analysis => ({
    games: a.games + b.games,
    wins: a.wins + b.wins,
    ties: a.ties + b.ties,
    diff_sum: a.diff_sum + b.diff_sum,
    diff_sq_sum: a.diff_sq_sum + b.diff_sq_sum,
});

/** Ties count half; ± is 1.96 standard errors. */
export function winChance(t: Analysis): { p: number; pm: number } {
    const p = (t.wins + t.ties / 2) / t.games;
    return { p, pm: 1.96 * Math.sqrt((p * (1 - p)) / t.games) };
}

/** Mean final score difference and its spread, for `marginText`. */
export function analysisMargin(t: Analysis): { margin: number; sigma: number } {
    const mean = t.diff_sum / t.games;
    return { margin: mean, sigma: Math.sqrt(Math.max(0, t.diff_sq_sum / t.games - mean * mean)) };
}

/** Whole percent until ± < 0.5 points, then one decimal. */
export const chanceText = (w: { p: number; pm: number }) => percent(w.p, w.pm < 0.005 ? 1 : 0);

export const gamesText = (w: { p: number; pm: number }, games: number) =>
    `± ${(w.pm * 100).toFixed(1)} · ${games.toLocaleString('en-US')} ${games === 1 ? 'game' : 'games'}`;

/** The move list's number: the net's P(win) as a score out of 1000. */
export const score = (p: number) => Math.round(p * 1000);
```

- [ ] **Step 4: Run to verify pass**

Run: `cd web && npx vitest run src/game/position.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/game/position.ts web/src/game/position.test.ts
git commit -m "feat(web): analysis totals, win chance, margin and score helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Controller analysis loop

**Files:**
- Modify: `web/src/game/explorer.svelte.ts`
- Test: `web/src/game/explorer.test.ts`

**Interfaces:**
- Consumes: `Engine.analyze` (Task 2); `analysisKey`, `isOver`, `addAnalysis`, `ZERO_ANALYSIS` (Task 3).
- Produces on `ExplorerController`:
  - `analysis = $state<{ key: string; totals: Analysis } | null>(null)` — totals and the `analysisKey` they belong to.
  - `analysing = $state(false)` — a batch loop is running.
  - `setActive(on: boolean): void` — the explorer is the visible view (route and tab).
  - Constructor gains a fifth optional parameter `now: () => number = () => performance.now()`.
  - Exported constants `BATCH_MS = 100`, `MIN_BATCH = 8`, `MAX_BATCH = 2048`.

- [ ] **Step 1: Write the failing tests** (append to `web/src/game/explorer.test.ts`)

```ts
describe('ExplorerController analysis', () => {
    type Call = { kind: 'explore' | 'analyze'; pos: Position; start?: number; count?: number; resolve: (v: any) => void; reject: (e: Error) => void };
    function loopEngine() {
        const calls: Call[] = [];
        const push = (kind: Call['kind'], pos: Position, start?: number, count?: number) =>
            new Promise((resolve, reject) => calls.push({ kind, pos: structuredClone(pos), start, count, resolve, reject }));
        const engine = {
            explore: vi.fn((pos: Position) => push('explore', pos)),
            analyze: vi.fn((pos: Position, start: number, count: number) => push('analyze', pos, start, count)),
        } as unknown as Engine;
        const reply = (games: number, wins: number) => {
            const c = calls.filter((c) => c.kind === 'analyze').at(-1)!;
            c.resolve({ games, wins, ties: 0, diff_sum: games, diff_sq_sum: games * 4 });
        };
        return { engine, calls, reply, analyses: () => calls.filter((c) => c.kind === 'analyze') };
    }
    const setup = (t = { now: 0 }) => {
        const e = loopEngine();
        const x = new ExplorerController(e.engine, memStore(), undefined, undefined, () => t.now);
        x.init();
        return { ...e, x, t };
    };

    it('does nothing until active, then accumulates batches from the running game count', async () => {
        const { x, analyses, reply } = setup();
        x.tapSquare('you', 0, 0);
        expect(analyses()).toHaveLength(0);
        x.setActive(true);
        expect(analyses()).toHaveLength(1);
        expect(analyses()[0]).toMatchObject({ start: 0, count: MIN_BATCH });
        expect(analyses()[0].pos.dice).toBeNull();
        reply(MIN_BATCH, 5);
        await flush();
        expect(x.analysis?.totals.games).toBe(MIN_BATCH);
        expect(x.analysing).toBe(true);
        expect(analyses()[1].start).toBe(MIN_BATCH);
        reply(analyses()[1].count!, 1);
        await flush();
        expect(x.analysis?.totals.wins).toBe(6);
    });

    it('a mark edit restarts at zero and discards the stale reply; a dice edit does not restart', async () => {
        const { x, analyses, reply } = setup();
        x.setActive(true);
        reply(MIN_BATCH, 3);
        await flush();
        x.openPicker(0);
        x.pickFace(4);
        reply(analyses()[1].count!, 2);
        await flush();
        expect(x.analysis?.totals.wins).toBe(5); // dice-only: kept accumulating
        x.tapSquare('you', 0, 0);
        reply(analyses()[2].count!, 7); // started before the edit: stale
        await flush();
        expect(x.analysis?.totals.wins).toBe(5);
        expect(x.analysis?.key).not.toBe(analysisKey(x.pos)); // old numbers stay (faded) until fresh ones arrive
        expect(analyses()[3].start).toBe(0);
        reply(analyses()[3].count!, 1);
        await flush();
        expect(x.analysis).toMatchObject({ key: analysisKey(x.pos), totals: { wins: 1 } });
    });

    it('never has two batches in flight, and an explore call goes before the next batch', async () => {
        const { x, calls, analyses, reply } = setup();
        x.setActive(true);
        x.tapSquare('you', 0, 0);
        x.tapSquare('you', 0, 1);
        x.setActive(true);
        expect(analyses()).toHaveLength(1);
        reply(MIN_BATCH, 0);
        await flush();
        const kinds = calls.map((c) => c.kind);
        expect(kinds.indexOf('explore')).toBeLessThan(kinds.lastIndexOf('analyze'));
        expect(analyses()).toHaveLength(2);
    });

    it('pauses when inactive and resumes with its totals', async () => {
        const { x, analyses, reply } = setup();
        x.setActive(true);
        x.setActive(false);
        reply(MIN_BATCH, 2);
        await flush();
        expect(x.analysis?.totals.games).toBe(MIN_BATCH); // the in-flight batch still counts
        expect(analyses()).toHaveLength(1);
        expect(x.analysing).toBe(false);
        x.setActive(true);
        expect(analyses()).toHaveLength(2);
        expect(analyses()[1].start).toBe(MIN_BATCH);
    });

    it('sends nothing for a finished game, and restarts when edited back to a live one', async () => {
        const { x, analyses, reply } = setup();
        x.tapStrike('opponent', 3); // 4 strikes
        x.setActive(true);
        expect(analyses()).toHaveLength(0);
        x.tapStrike('opponent', 3); // back to 3
        expect(analyses()).toHaveLength(1);
        reply(MIN_BATCH, 1);
        await flush();
        expect(x.analysis?.totals.games).toBe(MIN_BATCH);
    });

    it('stops quietly on an engine error and starts again on the next edit', async () => {
        const { x, analyses } = setup();
        x.setActive(true);
        analyses()[0].reject(new Error('engine was reset'));
        await flush();
        expect(x.analysing).toBe(false);
        expect(x.error).toBe('');
        expect(analyses()).toHaveLength(1);
        x.tapSquare('you', 0, 0);
        expect(analyses()).toHaveLength(2);
    });

    it('adapts the batch size toward 100 ms within bounds', async () => {
        const t = { now: 0 };
        const { x, analyses, reply } = setup(t);
        x.setActive(true);
        t.now += 10; // 8 games in 10 ms → 80 next
        reply(MIN_BATCH, 0);
        await flush();
        expect(analyses()[1].count).toBe(80);
        t.now += 1000; // 80 games in 1 s → 8 next
        reply(80, 0);
        await flush();
        expect(analyses()[2].count).toBe(MIN_BATCH);
        t.now += 1; // 8 games in 1 ms → 800
        reply(MIN_BATCH, 0);
        await flush();
        expect(analyses()[3].count).toBe(Math.min(800, MAX_BATCH));
        t.now += 1; // that many games in 1 ms → far over the cap
        reply(analyses()[3].count!, 0);
        await flush();
        expect(analyses()[4].count).toBe(MAX_BATCH);
    });
});
```

Add `analysisKey` to the `./position` import and `MAX_BATCH, MIN_BATCH` to the `./explorer.svelte` import at the top of the file.

- [ ] **Step 2: Run to verify failure**

Run: `cd web && npx vitest run src/game/explorer.test.ts`
Expected: FAIL (`setActive is not a function`, missing exports).

- [ ] **Step 3: Implement** in `web/src/game/explorer.svelte.ts`

Imports: `import type { Analysis, Answer, Engine, Roller, View } from '../engine/protocol';`

Constants after `BAD_LINK`:

```ts
/** Spec "Controller": each analysis batch aims at about this long, within these sizes. */
export const BATCH_MS = 100;
export const MIN_BATCH = 8;
export const MAX_BATCH = 2048;
```

Fields (after `showAll`):

```ts
    /** Play-out totals and the position (dice aside) they belong to. */
    analysis = $state<{ key: string; totals: Analysis } | null>(null);
    /** A batch loop is running. */
    analysing = $state(false);
```

Private fields (after `noticeId`):

```ts
    private now: () => number;
    private active = false;
    private looping = false;
    private batch = MIN_BATCH;
```

Constructor: add the parameter `now: () => number = () => performance.now(),` after `later`, and `this.now = now;` in the body.

New public method (after `refresh`):

```ts
    /** The explorer is the visible view (route and tab): analysis runs only then. */
    setActive(on: boolean) {
        this.active = on;
        void this.analyse();
    }
```

In `set(p)`, after `void this.evaluate();` add `void this.analyse();`.

New private method (after `evaluate`):

```ts
    /**
     * Spec "Controller": one batch in flight while active; each reply adds
     * to the totals of the position it was asked for, and is dropped if
     * the position (dice aside) changed meanwhile. Errors stop the loop
     * quietly; the next edit or activation starts it again.
     */
    private async analyse() {
        if (this.looping || !this.active || P.isOver(this.pos)) return;
        this.looping = true;
        this.analysing = true;
        try {
            while (this.active && !P.isOver(this.pos)) {
                const key = P.analysisKey(this.pos);
                const prev = this.analysis?.key === key ? this.analysis.totals : P.ZERO_ANALYSIS;
                const count = this.batch;
                const t0 = this.now();
                const r = await this.engine.analyze(P.toEngine(this.pos), prev.games, count);
                const ms = this.now() - t0;
                this.batch = Math.min(MAX_BATCH, Math.max(MIN_BATCH, Math.round((count * BATCH_MS) / Math.max(ms, 1))));
                const cur = this.analysis?.key === key ? this.analysis.totals : P.ZERO_ANALYSIS;
                if (P.analysisKey(this.pos) === key && cur.games === prev.games) {
                    this.analysis = { key, totals: P.addAnalysis(cur, r) };
                }
            }
        } catch {
            // Stop quietly: the move list shows engine errors.
        } finally {
            this.looping = false;
            this.analysing = false;
        }
    }
```

Note on the "pauses" test: the in-flight batch's reply still lands (`setActive(false)` only stops the next send).

The adapt test's numbers follow this formula: 8 games / 10 ms → round(8·100/10) = 80; 80 / 1000 ms → 8; 8 / 1 ms → 800; then capped at `MAX_BATCH`.

- [ ] **Step 4: Run to verify pass**

Run: `cd web && npx vitest run && npm run check 2>&1 | tail -3`
Expected: all Vitest tests pass (existing explorer tests included: they never call `setActive`, so no `analyze` is sent); svelte-check `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add web/src/game/explorer.svelte.ts web/src/game/explorer.test.ts
git commit -m "feat(web): explorer controller runs play-out batches while visible

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Card, move scores, wiring, throughput and docs

**Files:**
- Modify: `web/src/ui/PositionCard.svelte`, `web/src/ui/MovesList.svelte`, `web/src/ui/App.svelte`
- Create: `web/scripts/analyze-bench.mjs`
- Modify: `docs/ARCHITECTURE.md` (explorer section), `web/README.md` (State explorer bullets), `docs/superpowers/specs/2026-10-14-state-explorer-design.md` (move-list wording), `docs/superpowers/specs/2026-10-15-explorer-analysis-design.md` (Status), possibly `web/src/game/explorer.svelte.ts` (`MIN_BATCH`/`MAX_BATCH`)

**Interfaces:**
- Consumes: `x.analysis`, `x.analysing`, `x.setActive` (Task 4); `analysisKey`, `winChance`, `analysisMargin`, `chanceText`, `gamesText`, `score`, `marginText`, `overText` (Task 3 / existing).

- [ ] **Step 1: Measure WASM throughput**

Create `web/scripts/analyze-bench.mjs`:

```js
// Play-out throughput of the explorer's analyze call in WASM.
// Usage (repo root): ./web/build.sh && node web/scripts/analyze-bench.mjs
import { readFile } from 'node:fs/promises';
import init, { WasmExplorer } from '../pkg/qwixxer_web.js';

await init({ module_or_path: await readFile(new URL('../pkg/qwixxer_web_bg.wasm', import.meta.url)) });
const empty = { marks: Array.from({ length: 4 }, () => Array(11).fill(false)), strikes: 0 };
const pos = JSON.stringify({ you: empty, opponent: empty, locked: [false, false, false, false], dice: null, roller: 'you' });
const ex = new WasmExplorer();
ex.analyze(pos, 0, 8); // warm up
for (const n of [8, 64, 256]) {
    const t0 = performance.now();
    ex.analyze(pos, 0, n);
    const ms = performance.now() - t0;
    console.log(`${n} games: ${ms.toFixed(0)} ms, ${((n / ms) * 1000).toFixed(0)} games/s`);
}
```

Run: `./web/build.sh >/dev/null && git checkout web/Cargo.lock; node web/scripts/analyze-bench.mjs`
Expected: three lines with games/s (empty position = longest games, the slow case).
Then set the bounds from the 256-game rate G (games/s): keep `MIN_BATCH = 8` unless 8 games take over 100 ms (then `MIN_BATCH = 1`); set `MAX_BATCH` to the power of two nearest G (one second of work at most), not below 64. If either constant changes, update the adapt test in `explorer.test.ts` to the new values and rerun `npx vitest run`.

- [ ] **Step 2: Position card** — replace the `<script>` and the markup of `web/src/ui/PositionCard.svelte` (keep the existing `<style>` rules and add the new ones below):

```svelte
<script lang="ts">
    import type { ExplorerController } from '../game/explorer.svelte';
    import { analysisKey, analysisMargin, chanceText, gamesText, marginText, overText, winChance } from '../game/position';
    let { x, compact = false }: { x: ExplorerController; compact?: boolean } = $props();
    const a = $derived(x.answer);
    const t = $derived(x.analysis?.totals ?? null);
    // Numbers for an older position stay, faded, until the first batch for this one.
    const fresh = $derived(x.analysis?.key === analysisKey(x.pos));
    const w = $derived(t ? winChance(t) : null);
    const m = $derived(t ? marginText(analysisMargin(t)) : '');
</script>

<section class="card" class:compact class:stale={!fresh} aria-label="Position" aria-busy={!fresh}>
    {#if a?.over}
        <h2>Game over</h2>
        <b class="big">{overText(a.over)}</b>
    {:else if t && w}
        {#if !compact}<div class="top"><h2>Position</h2><span class="muted">before the roll</span></div>{/if}
        <div class="line"><b class="big">{chanceText(w)}</b><span class="what">chance you win</span>
            {#if !compact}<span class="margin">{m}</span>{/if}</div>
        <div class="bar" role="img" aria-label="You {chanceText(w)}"><span style="width: {w.p * 100}%"></span></div>
        <span class="games">{#if x.analysing && fresh}<span class="dot" aria-hidden="true"></span>{/if}{gamesText(w, t.games)}</span>
        {#if compact}<span class="margin">{m} · before the roll</span>{/if}
    {:else}
        <span class="muted">{x.error ? `Something went wrong: ${x.error}` : '…'}</span>
    {/if}
    {#if (t || a) && x.error}<span class="err">Something went wrong: {x.error}</span>{/if}
</section>
```

Add to `<style>`:

```css
    .games { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ink); animation: pulse 1.2s ease-in-out infinite; }
    @keyframes pulse { 50% { opacity: 0.25; } }
    @media (prefers-reduced-motion: reduce) { .dot { animation: none; } }
```

- [ ] **Step 3: Move scores** — in `web/src/ui/MovesList.svelte`:
  - header: `{moves.length} legal · your chance to win after each` → `{moves.length} legal · score after each, out of 1000`
  - row number: `<b>{percent(m.p_win, 1)}</b>` → `<b>{score(m.p_win)}</b>`
  - import `score` from `'../game/position'`; drop `percent` from that import if nothing else in the file uses it.
  - The bar keeps `width: {m.p_win * 100}%` (a score of 546 fills 54.6%, i.e. scaled to 1000).

- [ ] **Step 4: Wire activity** — in `web/src/ui/App.svelte`:
  - Add after `sync`'s definition:

```ts
    // Spec "Controller · Pause": analysis runs only while the explorer is on screen.
    const activity = () => x.setActive(route.view === 'explore' && document.visibilityState === 'visible');
```

  - At the end of `sync()` add `activity();`.
  - In `onMount`, add `document.addEventListener('visibilitychange', activity);` next to the other listeners and `document.removeEventListener('visibilitychange', activity);` in the returned cleanup.

- [ ] **Step 5: Check, test, build**

Run: `cd web && npx vitest run && npm run check 2>&1 | tail -3 && npm run build 2>&1 | tail -3`
Expected: tests pass, `0 errors`, build succeeds.

- [ ] **Step 6: Docs**
  - `docs/ARCHITECTURE.md`, explorer section: one paragraph — the Position card's win chance is a play-out estimate (`Explorer::analyze`, Master vs Master through `BatchedRollouts`, dice seeded per game so batches compose), driven by `ExplorerController` in ~100 ms batches while the explorer is visible, restarting on non-dice edits; the move list shows the net's P(win) × 1000 as a score; and the measured WASM throughput from Step 1 (games/s on the empty position, with the date 2026-10-15 and `node web/scripts/analyze-bench.mjs`).
  - `web/README.md`, "State explorer" first bullet: replace "the pair net's P(win) and margin for the position" with "a play-out win chance and margin for the position that keeps refining while the explorer is open (`analyze`)", and "ranked by the pair net's P(win) after it" with "ranked by the pair net's score (P(win) × 1000) after it"; add `scripts/analyze-bench.mjs` to the "Bot latency" section as the analysis throughput check.
  - `docs/superpowers/specs/2026-10-14-state-explorer-design.md`: in "Moves (dice set)", change `"*N* legal · your chance to win after each"` to `"*N* legal · score after each, out of 1000"` and "**The chance to win** is the pair net's P(win)" to "**The score** is the pair net's P(win) × 1000"; in "Evaluation (always shown)" add one line: "Superseded by `2026-10-15-explorer-analysis-design.md`: the position's numbers come from play-outs."
  - `docs/superpowers/specs/2026-10-15-explorer-analysis-design.md`: `**Status:** designed.` → `**Status:** implemented (branch `explorer-analysis`).`

- [ ] **Step 7: Commit**

```bash
git add web/src/ui/PositionCard.svelte web/src/ui/MovesList.svelte web/src/ui/App.svelte web/scripts/analyze-bench.mjs web/src/game/explorer.svelte.ts web/src/game/explorer.test.ts docs/ARCHITECTURE.md web/README.md docs/superpowers/specs/2026-10-14-state-explorer-design.md docs/superpowers/specs/2026-10-15-explorer-analysis-design.md
git commit -m "feat(web): position card shows refining play-out win chance; moves show a score

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
