# Web: explorer position analysis

**Status:** implemented (branch `explorer-analysis`). Builds on the state explorer
(`2026-10-14-state-explorer-design.md`); where the two differ, this spec
wins.

**Revision (2026-10-22): truncated play-outs.** Each play-out now stops
after 4 turns (`PLAY_OUT_TURNS`) and takes the win net's P(win) and its
predicted final score difference (aux head) there; a game that ends sooner
counts its result. Phase 28 measured no systematic error against full
play-outs at 4 turns, and ±1 point now takes about 60–70× less time. The
totals become sums of those values, ± comes from their spread, and the
margin line drops its spread (the samples never reach a final score). The
sections below are updated to match.

## Goal

The explorer's "chance you win" for the position is the pair net's static
P(win), which is skewed. Replace it with a play-out estimate that keeps
refining while you look at the position, like a chess engine's infinite
analysis: more games, smaller ±. The move list stays instant and cheap;
its numbers become a "score" so they are not read as the same kind of
probability.

## Scope

- Only the position (the Position card) is played out. Moves are not.
- Analysis runs while the explorer is the visible view, without limit, and
  pauses otherwise.
- Out of scope: playing out moves, Master+ inside play-outs, a stop
  button, persisting analysis across reloads, a second worker.

## Engine

New call in `web/src/explorer.rs`, exposed on `WasmExplorer`:

```
analyze(position_json, start: u32, count: u32) -> {
  games: u32, win_sum: f64, win_sq_sum: f64, diff_sum: f64
}
```

- Plays games `start .. start + count` from the position **before the
  roll**: the position's sheets and closed rows, the roller acting first.
  The position's dice are ignored.
- Both seats are Master: the pair net through the shared decision
  pipeline, run with the existing batched rollout driver in
  `src/strategy/search.rs` (`BatchedRollouts`, as `rollout_final_states`
  uses it), so the net is evaluated in batches across games.
- Game *k*'s dice streams are seeded from a hash of the position (dice
  excluded) and *k* alone. Consequences: batches compose exactly
  (`[0, a)` + `[a, b)` = `[0, b)`), and the same position always yields
  the same totals.
- Each play-out runs at most 4 turns. Its win value is the result if the
  game ended (1 win, ½ tie, 0 loss), else the win net's P(you win) for
  whoever rolls next; its `diff` is your final score minus the opponent's,
  or the net's predicted one (aux head). `win_sum` / `win_sq_sum` sum the
  win values and their squares, `diff_sum` the diffs.
- A game-over position (4 strikes on a sheet, two closed rows) is an
  error; the caller never asks.

## Worker

A new `analyze` command in the existing worker (`web/src/engine/`), one
request and one reply per batch, like `explore`. No streaming messages.

## Controller

`ExplorerController` (`web/src/game/explorer.svelte.ts`) drives the loop:

- Keeps running totals for the current position: games, win sum, win²
  sum, diff sum.
- Sends one `analyze` batch at a time; on reply, adds it to the totals
  and sends the next, starting at the current game count.
- **Batch size** adapts so a batch takes about 100 ms: next count =
  previous count × (100 ms / measured time), clamped to bounds fixed after
  measuring WASM throughput (see Testing). The first batch uses the lower
  bound.
- **Restart:** a change to marks, strikes, closed rows or roller resets
  the totals to zero. A dice-only change does not. A reply that belongs
  to an older position is discarded.
- **Priority:** an `explore` call (move ranking after an edit) is sent
  before the next batch; the worker handles calls in order, so the move
  list waits at most one batch.
- **Pause:** no batch is sent while the explorer is not the active route
  or `document.visibilityState` is `hidden`. Returning resumes with the
  totals kept.
- **Throttle:** once ± is below 2 points, the loop rests after each batch
  as long as the batch took (half a core). Precision for this test uses the
  plus-four estimate (two extra wins and two extra losses added to the
  totals), so a small
  one-sided sample (plain ± = 0) never counts as precise. An edit or a pause
  ends a rest at once; a new position starts at full speed.
- **Stop:** at 50,000 play-outs (`MAX_GAMES`, ± about 0.2 points) the
  loop stops for that position; the last batch asks only for the games
  left. Leaving and returning doesn't restart it; an edit other than the
  dice does.
- **Game over:** no analysis.
- Totals live in memory only.

Derived numbers:

- Win chance p = win sum / games; ± = 1.96 · √(v / games), v = win² sum /
  games − p².
- Margin = mean diff.

## UI

Position card (desktop and phone compact form):

- "**54%** chance you win" with the bar, from the play-outs. Shown to a
  whole percent; once ± < 0.5, to one decimal ("54.3%").
- New line under the bar: a small pulsing dot while analysis runs, then
  "± 1.2 · 3,200 games". The dot is absent when paused or over.
- Margin text "Opponent ahead by ~9 points", "You ahead by ~1 point",
  "Level", from the play-outs' mean diff.
- Until the first batch for a position arrives, the previous position's
  numbers stay, faded (as edits do now); on a cold open with none, "…".
- Game over: the result, as now; no dot, no game count.

Move list:

- Each row's number is the net's P(win) for the move as a **percentage
  with one decimal** ("54.6%"); the bar shows the same. (Was a score out
  of 1000 while the pair net's P(win) was poorly calibrated; the win net's
  is not.)
- Header: "*N* legal".
- Ranking, ties, bot tags and (i) notes are unchanged.

The explorer spec's wording "your chance to win after each move" for the
list becomes "score".

## Testing

Rust (`web/src/explorer.rs`):

- Batches compose: `[0,100)` + `[100,250)` totals equal `[0,250)`.
- Determinism: the same position twice gives identical totals.
- Dice-blind: the same position with different dice gives identical
  totals.
- Game-over position: error.
- Sanity: the empty position over a few hundred games has a win chance
  within a loose band around 50%.

Vitest (controller, fake client):

- Totals accumulate across batches.
- A mark edit restarts; a dice-only edit does not.
- A stale reply is discarded.
- Hidden tab or leaving the explorer pauses; returning resumes with
  totals kept.
- An `explore` call goes before the next batch.
- Batch size adapts toward 100 ms within its bounds.

Throughput: measure play-out games per second in WASM (Node, like
`web/scripts/bench.mjs`) and set the batch bounds from it; record the
figure in `docs/ARCHITECTURE.md`.
