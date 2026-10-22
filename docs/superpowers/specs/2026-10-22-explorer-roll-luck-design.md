# Web: explorer roll luck

**Status:** implemented (branch `explorer-luck`). Opponent's roll: the move
list's code per roll with a cached net, about 1 s in WASM (accepted in place
of the 0.5 s target); the line reserves its height while computing. Builds on the state explorer
(`2026-10-14-state-explorer-design.md`) and its analysis
(`2026-10-15-explorer-analysis-design.md`).

## Intent

When the explorer's position has dice, show one number that says whether
the roll was lucky or unlucky for you, and by how much. Asked for by the
user: "some kind of quantity from which i can tell if the roll was lucky,
unlucky, and by how much".

## Definition

- f(roll) = the **top P(win) in the move list** for that roll: the best of
  your legal turns, valued exactly as `Explorer::moves` values them (on
  your roll: after your turn, against the opponent's predicted white-sum
  mark; on theirs: at the end of their predicted turn, Master's
  lookahead). Always your P(win), whoever rolls.
- **Luck = f(this roll) − E[f]**, the expectation over all 6⁶ rolls, each
  die uniform and independent. In percentage points of win chance, the
  move list's units. Positive is lucky for you; on the opponent's roll a
  roll that helps them shows negative.
- The best turn, not the one any bot or you would play: luck stays apart
  from skill. Master's fixed rules (forced locks, dominance pruning) do
  not enter f; the move list's top row ignores them too.
- f and E[f] use the same valuation, so the net's own error cancels. Never
  subtract the Position card's play-out estimate from a move's P(win).

## Goals

- Exact: E[f] equals the plain average of f over every roll, to float
  rounding, not a sample.
- Explorer only. Not in games against the bot, the log, or anywhere else.
- Doesn't slow the move list: the move list appears as it does today.

## Non-goals

- Luck of past rolls in a game, cumulative luck, luck-adjusted results.
- Skill or error measures (best move vs played move).
- Luck from Master+'s point of view (its searched values).

## Engine

New call in `web/src/explorer.rs`, exposed on `WasmExplorer` and the
worker like `analyze`:

```
luck(position_json) -> { this_roll: f32, average: f32 }
```

- Requires dice and a game that isn't over; otherwise an error (the
  caller never asks).
- `this_roll` equals the top `p_win` of `explore`'s move list for the
  same position, bit for bit. `average` is E[f] over all rolls; the dice
  in the position matter only for `this_roll`.
- Distinct rolls: the two white dice unordered (21 outcomes, weights 1/36
  or 2/36) times the four colour dice (1,296). Every roll is equally
  likely, so E[f] is the weighted mean over those 27,216.

**Speed.** Running the move-list valuation on all 27,216 rolls is the
reference, and too slow (order of seconds natively, more in WASM). Target
(my call): about **0.5 s in WASM** on a busy midgame position, measured with a bench
script like `web/scripts/analyze-bench.mjs`. Suggested route, per white
pair:

- **Your roll.** The opponent's predicted white-sum mark should depend on
  the white dice only (`predict_opp_white`, its plain passive decision;
  verify, the exactness test catches it if not). Each of your legal turns
  (pass, strike, white only, a colour mark, white then colour) uses at
  most one colour die, so its value depends on the white dice and that
  one die face. Value every distinct turn once, in one batch, then
  enumerate the 1,296 colour outcomes with table lookups: f is the max
  over the turns that roll makes legal.
- **The opponent's roll.** Your candidates (pass or the white sum in a
  row) depend on the white dice only, but their values depend on the
  opponent's predicted turn, which sees all four colour dice. The same
  idea applies if the predicted phase-1 mark and colour mark can be
  reproduced from per-die tables. If that can't be made exact against
  Master's decision rules, fall back to the reference with shortcuts that
  stay exact: a closed row's colour die doesn't matter (group its six
  faces), nor does a face that gives no legal mark to either player in
  that row.

Builder's call: the route, as long as the exactness test below passes.
If the target can't be reached exactly, stop and bring the measured time
back rather than approximating.

## Worker and controller

- `luck` is a worker command like `explore`. `ExplorerController` calls
  it after each `explore` whose position has dice, for the latest
  position only (stale replies dropped, as for `explore`), and stores
  `luck: { key, this_roll, average } | null`.
- It runs after the move list's `explore` and before the next analysis
  batch, so the move list never waits on it.
- A dice edit recomputes it; the analysis keeps running as today.

## UI

On the Roll card, under the dice (both layouts), once dice are set and
the game isn't over. Wording and placement are my calls:

- "Lucky roll **+3.2 pts**" / "Unlucky roll **−2.7 pts**" / "Average
  roll" (|luck| < 0.05 pts), one decimal.
- A small muted second part: "this roll 55.3% · average 52.1%".
- While computing, or when stale after an edit: the previous value
  faded, or "…" with none, as the Position card does.
- My call (not discussed): green/red tint for lucky/unlucky, using the
  existing theme tokens.

## Testing

- **Exactness (native, `web/src/explorer.rs`):** on a few positions (the
  empty position, a midgame, a late one with a closed row, each with both
  rollers), `average` equals the brute-force mean of `explore`'s top
  `p_win` over all 27,216 distinct rolls (weights as above), within 1e-5;
  `this_roll` equals the top `p_win` exactly. Mark the brute-force test
  slow-but-run if needed; CI runs the full suite.
- Luck averages to 0 over all rolls (follows from the above; one cheap
  assertion).
- Refuses no-dice and game-over positions.
- TS: controller calls `luck` after `explore` with dice, drops stale
  replies, clears it without dice; text formatting (sign, one decimal,
  "Average roll").
- WASM timing recorded in ARCHITECTURE with the bench.

## Left to the builder

- The exact fast route, and how the tables are shaped.
- Whether `luck` shares work with `explore` (e.g. the opponent's
  predicted white-sum mark).
