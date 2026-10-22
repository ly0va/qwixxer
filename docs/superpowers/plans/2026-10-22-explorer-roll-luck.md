# Explorer Roll Luck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show on the explorer's Roll card how lucky the set roll is for you: the move list's top P(win) for this roll minus its exact average over all rolls.

**Architecture:** The move list's valuation is split out of `Explorer::moves` into a function that values every legal turn, so luck's f(roll) is that function's max by construction. `Explorer::luck` computes f for the set dice and E[f] over the 27,216 distinct rolls. A brute-force reference proves it exact; per-white-pair tables make it fast. A new worker command `luck` feeds `ExplorerController`, and the Roll card shows one line.

**Tech Stack:** Rust (`web/src/explorer.rs`, wasm-bindgen), TypeScript + Svelte 5 (`web/src`), vitest.

**Spec:** `docs/superpowers/specs/2026-10-22-explorer-roll-luck-design.md`

## Global Constraints

- Luck = f(this roll) − E[f], always your P(win), in percentage points; f = the top `p_win` of `explore`'s move list for that roll.
- E[f] over all 6⁶ rolls, each die uniform: 21 unordered white pairs (weight 1/36 equal, 2/36 unequal) × 1,296 colour outcomes.
- `luck` errors on no dice or a game-over position.
- Explorer only. The move list never waits on luck.
- Copy: "Lucky roll +3.2 pts" / "Unlucky roll −2.7 pts" (U+2212 minus) / "Average roll" when |luck| < 0.05 pts; detail "this roll 55.3% · average 52.1%"; one decimal. Lucky text green, unlucky red (theme tokens). Previous value faded while recomputing, "…" with none.
- Speed target: about 0.5 s in WASM on a busy midgame position. Not reachable exactly → stop and report the measured time; never approximate.
- `rustfmt --edition 2021 <files>` on touched Rust files only, never `cargo fmt`.

## Review Focus

- A roll where the opponent's white-sum mark closes a row and ends the game after phase 1 (your roll): f must still equal the move list's top. Task 2's exactness positions include a near-end position.
- Closed rows: their colour dice must still be enumerated (or grouped with weight 6), never dropped from the weights. Task 2 includes a position with a closed row.
- A position whose only legal turn is a pass or a strike: tables must still give f. Covered by the late position in Task 2.
- A stale `luck` reply after a dice edit must not overwrite the newer one. Task 4 tests it.
- Clearing the dice must clear the luck line, not leave the old one. Task 4 tests it.

---

### Task 1: Move valuation split out, `luck` by brute force

**Files:**
- Modify: `web/src/explorer.rs` (`moves`, new `move_values`, `Luck`, `luck`, `luck_reference`)
- Test: `web/src/explorer.rs` (tests module)

**Interfaces:**
- Produces: `fn move_values(&self, you: &State, opp: &State, dice: [u8; 6], roller: Roller) -> (Vec<Turn>, Vec<f32>)`: each legal turn and its `p_win` (as `moves` computes them, the opponent's predicted white mark included on your roll). `moves` uses it.
- Produces: `#[derive(Debug, Clone, PartialEq, Serialize)] pub struct Luck { pub this_roll: f32, pub average: f32 }` and `pub fn luck(&self, pos: &Position) -> Result<Luck, String>`.
- Produces: `#[cfg(test)] fn luck_reference(&self, you: &State, opp: &State, roller: Roller) -> f64`, the weighted mean of `move_values`' max over the 27,216 distinct rolls. Also `fn rolls() -> impl Iterator<Item = ([u8; 6], f64)>` (each distinct roll with its probability; sums to 1) for reuse in Task 2.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn luck_this_roll_is_the_move_lists_top() {
    // empty, midgame and late positions (reuse the sheets the random-position test builds), both rollers
    for p in luck_positions() {
        let a = Explorer::new().explore(&p).unwrap();
        let l = Explorer::new().luck(&p).unwrap();
        assert_eq!(l.this_roll, a.moves.unwrap()[0].p_win);
    }
}
#[test]
fn the_rolls_cover_every_roll_once() {
    let v: Vec<_> = rolls().collect();
    assert_eq!(v.len(), 21 * 1296);
    assert!((v.iter().map(|r| r.1).sum::<f64>() - 1.0).abs() < 1e-12);
}
#[test]
fn luck_refuses_no_dice_and_a_finished_game() { /* dice None → Err; opponent.strikes = 4 → Err; two locked rows → Err */ }
```

`luck_positions()` (test helper): the empty position, a midgame, a late position with one closed row and a near-end one (a row one mark from closing for the opponent), each with fixed dice and with both rollers.

- [ ] **Step 2: Run them; expect compile errors (`luck`, `rolls` missing)**

Run: `cd web && cargo test --release luck`

- [ ] **Step 3: Implement `move_values`, `rolls`, `luck` (this roll from `move_values`, average from the brute-force mean for now) and `luck_reference`; make `moves` call `move_values`**

- [ ] **Step 4: Run the whole web suite; expect all pass**

Run: `cd web && cargo test --release`

- [ ] **Step 5: Commit** (`feat(explorer): roll luck, by brute force`)

### Task 2: Exact and fast E[f]

**Files:**
- Modify: `web/src/explorer.rs` (new `fn average_top(&self, you: &State, opp: &State, roller: Roller) -> f64`, used by `luck`)
- Create: `web/scripts/luck-bench.mjs` (like `analyze-bench.mjs`: times `luck` on the empty position and a busy midgame, both rollers)

**Interfaces:**
- Consumes: `move_values`, `rolls`, `luck_reference` (Task 1).
- Produces: `luck` returns `average: average_top(...) as f32`.

- [ ] **Step 1: Write the failing exactness test**

```rust
#[test]
fn the_average_is_exact() {
    let ex = Explorer::new();
    for p in luck_positions() {
        let (you, opp, _) = states(&p);
        let want = ex.luck_reference(&you, &opp, p.roller);
        let got = ex.luck(&p).unwrap().average as f64;
        assert!((got - want).abs() < 1e-5, "{p:?}: {got} vs {want}");
    }
}
```

It passes against Task 1's brute force; make it fail first by swapping `luck` to the stub `average_top` that returns 0.0, then run it.

Run: `cd web && cargo test --release the_average_is_exact`. Expected: FAIL.

- [ ] **Step 2: Your roll.** Per white pair: the opponent's predicted white mark (`predict_opp_white`, check it ignores the colour dice). Value every distinct legal turn once in one batch, keyed by the one colour die it uses (or none). Then f for each of the 1,296 colour outcomes is the max over the turns that outcome makes legal (table lookups). A turn's legality must be judged with that colour outcome's dice. Run the test until the `Roller::You` positions pass.

- [ ] **Step 3: The opponent's roll.** Try per-die tables for the opponent's predicted phase-1 and colour marks (`passive_turn_predictions`). If they can't be made exact against Master's decision rules, fall back to `move_values` per roll, with exact shortcuts: a closed row's colour die doesn't matter (one face, weight × 6), nor does a face no player can mark in that row. Run the test until every position passes.

- [ ] **Step 4: Build WASM and time it**

Run (repo root): `./web/build.sh && node web/scripts/luck-bench.mjs`
Expected: the busy midgame at most about 0.5 s for each roller. **If either is well over, stop: report the times to the user before going on.**

- [ ] **Step 5: Run the whole web suite, rustfmt the file, commit** (`perf(explorer): exact roll luck from per-white-pair tables`)

### Task 3: WASM and worker command

**Files:**
- Modify: `web/src/lib.rs` (`WasmExplorer::luck(&self, position: &str) -> Result<String, JsError>`, JSON like `analyze`)
- Modify: `web/src/engine/serve.ts` (case `'luck'`), `web/src/engine/protocol.ts` (`export interface Luck { this_roll: number; average: number }`; `Engine.luck(position: Position): Promise<Luck>`), `web/src/engine/client.ts` (`luck(position)`)
- Test: `web/src/engine/serve.test.ts`, `web/src/engine/client.test.ts`

**Interfaces:**
- Consumes: `Explorer::luck` (Task 1).
- Produces: `engine.luck(position): Promise<Luck>`.

- [ ] **Step 1: Failing tests** mirroring the existing `analyze` cases: serve routes `luck` to `WasmExplorer.luck` with the position JSON and returns the parsed reply; client sends `('luck', position)`.
- [ ] **Step 2: Run; expect FAIL.** `cd web && npx vitest run src/engine`
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run; expect PASS.** Then `npm run check`.
- [ ] **Step 5: Commit** (`feat(web): luck engine command`)

### Task 4: Controller and Roll card

**Files:**
- Modify: `web/src/game/explorer.svelte.ts` (state `luck = $state<{ key: string; value: Luck } | null>(null)`; requested after each `explore` reply whose position has dice and isn't over, for the latest position only; cleared when the dice are cleared)
- Modify: `web/src/game/position.ts` (`luckKey(p: ExPosition): string`: the whole position including dice and roller; `luckText(l: Luck): { label: string; value: string; tone: 'lucky' | 'unlucky' | 'average'; detail: string }`)
- Modify: `web/src/ui/RollCard.svelte` (the line under the dice, both layouts; faded when `x.luck.key !== luckKey(x.pos)`; "…" with none)
- Test: `web/src/game/explorer.test.ts`, `web/src/game/position.test.ts`

**Interfaces:**
- Consumes: `engine.luck`, `Luck` (Task 3).

- [ ] **Step 1: Failing tests**

```ts
it('luck text', () => {
    expect(luckText({ this_roll: 0.553, average: 0.521 })).toEqual({ label: 'Lucky roll', value: '+3.2 pts', tone: 'lucky', detail: 'this roll 55.3% · average 52.1%' });
    expect(luckText({ this_roll: 0.494, average: 0.521 }).value).toBe('−2.7 pts');
    expect(luckText({ this_roll: 0.5212, average: 0.5210 })).toMatchObject({ label: 'Average roll', value: '', tone: 'average' });
});
```

Controller (with the existing fake engines): after an `explore` reply with dice, `luck` is called once for that position; a reply for an older position is dropped; clearing the dice sets `luck` to null and calls nothing; an over position calls nothing; `luck` goes out after `explore`, before the next `analyze` batch.

- [ ] **Step 2: Run; expect FAIL.** `cd web && npx vitest run src/game`
- [ ] **Step 3: Implement.** Luck calls share the one-in-flight discipline `evaluate` uses (coalesce to the latest position).
- [ ] **Step 4: Run all web tests and the check; expect PASS.** `cd web && npm test && npm run check`
- [ ] **Step 5: Commit** (`feat(web): roll luck on the explorer's Roll card`)

### Task 5: Docs

**Files:**
- Modify: `docs/ARCHITECTURE.md` (explorer section: the `luck` call, its definition, and the WASM timing from Task 2)
- Modify: `docs/superpowers/specs/2026-10-22-explorer-roll-luck-design.md` (status: implemented, and which opponent-roll route was taken)

- [ ] **Step 1: Write both; run the full CI set** (`cargo test --release` at the root, `cd web && cargo test --release && npm test && npm run check`)
- [ ] **Step 2: Commit** (`docs: explorer roll luck`)
