# Web: game review

**Status:** design, agreed 2026-10-24. Mock-ups: the design canvas
(https://claude.ai/artifact/4eDmxXF5jEskvwxcFS26im), page "Review",
boards R1 (desktop), R3 row B5 (pencil marks), R4–R5 (phone), R6–R7
(dark), R8 (still analysing), R9 (opened from a shared link). Builds on the
explorer (`2026-10-14-state-explorer-design.md`) and its roll luck
(`2026-10-22-explorer-roll-luck-design.md`).

## Intent

After a game against the bot, answer one question: **did I win or lose
because of luck or because of skill?** In the user's words: "i want to
check if i lost/won mainly because of luck, or because of skill." Then let
them walk through the game point by point, the way chess.com's game
review does: a clickable eval graph, an eval bar, the move log, luck.

For each mistake the user wants "by how much it set me back and what
would've been the best move instead".

Games come from two places: one just played against the bot, or a URL
someone shared. Every reviewed game can be shared the same way.

## Goals

- One page, desktop and phone, light and dark, matching the mock-ups.
- Luck and skill in the same unit (points of win chance), so they can be
  compared directly.
- A game fits in a short, URL-safe code; the link alone reproduces the
  review anywhere.
- The page is usable while the analysis is still running.

## Non-goals (for now)

- Move-quality labels ("best / good / mistake / blunder"). Wanted later;
  leave room for a label next to each cost in the log and the Played row.
- Reviewing games played outside the app, or entering a game by hand.
- Cumulative stats across games.
- Showing the bot's valuation drift (see "The numbers": ignored by the
  user's decision).

## Anti-goals

- Luck that mixes in anyone's play, or mistakes that mix in the dice.
  Both are defined below so they stay apart.
- Numbers that disagree with the explorer for the same position, apart
  from the one agreed exception (see "Mistake": the roller's turn is priced
  against the other player's actual white-sum mark). "Open in explorer"
  from any roll shows the same luck and, when the actual mark matches the
  explorer's prediction, the same move values.

## The game code

Agreed in conversation; layout details marked *my call* are mine.

- **Store the dice and the choices, not the sheets.** Sheets, strikes,
  closed rows and the end of the game come from replaying the turns
  through the rules. A code that doesn't replay is rejected.
- **Characters** (base64url alphabet `A–Z a–z 0–9 - _`, value 0–63):
  1. Version: `1`.
  2. Header, one character (*my call*): bit 0 = who rolls first (0 = the
     player, 1 = the bot); bits 1–2 = the bot level (0 Master+, 1 Master,
     2 Student, 3 Newbie); bits 3–5 = 0, reserved.
  3. Then **exactly 4 characters per roll** (24 bits, big-endian), in
     game order. The string's length gives the number of rolls.
- **One roll** = `dice × 225 + action`, which is below 46,656 × 225 =
  10,497,600 < 2²⁴:
  - `dice` = the six dice as base-6 digits, d − 1 each, first die most
    significant, in the order white 1, white 2, red, yellow, green, blue.
    All six are kept, even for closed rows: luck needs them.
  - `action = rollerWhite × 45 + rollerColour × 5 + otherWhite`:
    - `rollerWhite` (0–4): 0 = none, 1–4 = red, yellow, green, blue: the
      roller marks the white sum in that row (the number comes from the
      dice).
    - `rollerColour` (0–8): 0 = none, else 1 + row × 2 + whiteDie (row
      0–3 as above, whiteDie 0 or 1): white die + that row's colour die.
      When both white dice show the same value, always whiteDie 0.
    - `otherWhite` (0–4): the other player's white-sum mark, as
      `rollerWhite`.
  - Whoever rolls, their marks go in `rollerWhite` / `rollerColour`, and
    the non-roller's white-sum mark in `otherWhite`: on the player's roll
    the bot's passive mark, on the bot's roll the player's.
  - A roller who marks nothing takes a strike; no bit for it. Rows close
    by the rules; no bits for that either.
- **Why fixed fields, not "index into the legal moves":** the index would
  save ~3 bits per roll but tie old links to the move generator's order.
- A code cut after any whole roll is a valid earlier point of the game.
  A typical game is about 160 characters.
- *My call:* decoding accepts an unfinished game (the page shows "Game not
  finished" instead of the result); the app itself only offers review for
  finished games.

**What the game must keep:** today's turn log (`RollRecord`) has the marks
but not the dice. The game has to record each roll's six dice (and who
started, and the bot level) so it can build the code when the game ends.
The bot level is fixed for a game (a level change applies to the next game,
`nextBot`). Local storage keeps the last finished game as its code string.
The game has one rule set, so the code carries no rule options (the
explorer's `force_lock` is a bot habit, not a game rule).

## The numbers

All values are "your" (the player's) P(win) from the win net, in points
(percentage points of win chance). Luck is valued exactly as the explorer
values it; move costs look one roll further (1-ply, below).

- **Points on the graph: two per roll.**
  - *After the roll:* f(roll) as defined for roll luck: your P(win) after
    the roller's best turn.
  - *After the move:* the played move's 1-ply value: E[f] of the next
    roll, from the real position after this roll (both players' actual
    marks applied). It is the same number as the next roll's luck average,
    computed once and used for both. After the last roll it is exactly 0%
    or 100%.
  - Plus a start point before roll 1: E[f] of roll 1 (the empty position,
    the first roller).
- **Luck** of a roll: f(this roll) − E[f], exactly as in the explorer
  (exact expectation, not "the previous point"). Agreed after measuring:
  using the previous point instead is close (mean gap 0.44 pts vs a mean
  |luck| of 4.5 over 307 rolls) but would disagree with the explorer's
  chip.
- **1-ply value** of a turn: E[f] of the next roll from the position the
  turn leaves (the average, over every next roll, of the next roller's
  best turn, valued statically: the luck machinery). A turn that ends the
  game takes its exact 0% or 100%; turns leaving identical sheets share
  one average.
- **Mistake** of a choice, from the chooser's side, ≥ 0:
  - The roller's turn: the best 1-ply value minus the played turn's 1-ply
    value. The candidates are the static top 3 of the explorer's move list
    plus the played turn; "Best" is the highest of them by 1-ply. Agreed
    after a probe (16 self-play games, 380 rolls, 2026-10-24): 1-ply
    changed the best move on 29 rolls, and on 18 of those Master+'s search
    picked the 1-ply move (7 the static one); costs moved 0.64 pts on
    average, 3.7% of them by more than 2 pts. Exact, so no sampling noise.
  - Turns are valued **against the other player's actual white-sum mark**
    (the user's choice, option A: the roller sees that mark before the
    colour decision, and bots other than Master often mark differently).
    On the bot's roll, with the sheets swapped, as for luck. This is what
    makes the played turn's 1-ply value equal the next roll's luck average.
  - The other player's white-sum mark: the top of their passive options
    minus the one they chose (the explorer's opponent-roll list for your
    mark; sheets swapped for the bot's). *My call:* valued as the explorer
    does, against the roller's predicted turn, since that mark is made
    before the roller's colour decision.
  - Your mistakes count against you; the bot's count for you.
- **The gap is ignored.** Each luck step on the graph is exact (after the
  move → after the next roll is f − E[f]), but the step from "after the
  roll" (static) to "after the move" (1-ply) is the mistake plus the
  difference between the static and 1-ply values of the best move. The
  user decided to ignore such gaps and not show them.
- **Totals:** the sum of luck over all rolls; the sum of your mistakes;
  the sum of the bot's mistakes.
- **Verdict** (*my call*): compare |luck total| with |bot's mistakes −
  your mistakes|. If luck is larger: "Mostly the dice". Otherwise "Mostly
  your mistakes" after a loss, "Mostly your play" after a win.

## The page

Route `#/review?g=<code>` (*my call*). The main nav gets a third tab,
"Review", after Play and Explore. With no code it shows the last finished
game, kept in local storage; with none of those, an empty state saying to
finish a game or open a shared link (*my call*).

Entry points: an "Analyse game" button on the game-over screen; opening a
shared link.

**Desktop (R1):**

- Header: the nav, and "Share game" (copies the link). Nothing else for
  your own game.
- Left: a vertical eval bar (your win chance at the selected point, number
  at its foot), then both sheets as they were at the selected point, then
  the selected-roll strip:
  - "Roll 23", whose roll, the luck chip (as in the explorer:
    "Lucky +3.2" / "Unlucky −4.8" / "Average").
  - The six dice.
  - **The Played / Best switch:** two rows, always about the human's own
    decision for that roll: on your roll your turn, on the bot's roll
    your white-sum mark (*my call*; the bot's mistakes show only as costs
    in the log). "Played" shows the played move's chips and its cost
    ("−4.6 pts"); "Best" the best move's chips. The selected row decides
    which outcome both sheets show: the chosen move of yours with
    everything else as it really happened.
  - First / previous / next / last buttons; "Open in explorer": the
    position before the roll's marks, with the dice and the roller, for
    either point of the roll (*my call*); the explorer represents both
    rollers already.
- Right:
  - Summary: the result ("Master won 41–58"), the verdict, and three
    tiles: Dice luck, Your mistakes, Master's mistakes.
  - "Your win chance" graph: the points above, your share filled from
    the bottom, a dashed 50% line, red dots on your mistakes costing 2+
    pts (*my call*: the threshold), the selected point as a vertical line
    and ring with its percentage. Every point is clickable.
  - "Dice luck": the running total of luck, one line around zero (above
    green, below red), aligned with the graph above it, same selection
    line.
  - "Moves": the game log as on the game page, one entry per roll, each
    with a luck pill; your mistakes show their cost. Clicking an entry
    selects it; the selected entry is outlined and scrolled into view.
- **This roll's marks are drawn in pencil** (the game's pencil mark,
  `MarkX` with `pencil`), on whichever outcome the switch shows. No
  coloured frames. (Chosen from the R2/R3 options.)

**Phone (R4, R5):** one scrolling page: summary (tiles in one row, short
labels), the two graphs, the selected-roll card (dice, the switch rows,
"Explore"), both sheets side by side as columns (as in the explorer's
phone layout), the log. Pinned to the bottom of the screen: a 6 px eval
bar running edge to edge, and the step bar (first, previous, "Roll 23 of
36 · after the move · 12% you win", next, last). No header text besides
the logo, share and menu.

**Stepping:** the selected point moves through start → roll 1 after the
roll → roll 1 after the move → roll 2 … Arrow keys step, Home/End jump
(*my call*, as chess.com). At an "after the roll" point the sheets show
the position before the move, with the dice.

**Shared link (R9):** the same page with "You / your" replaced by
"Player / Player's" everywhere (the user chose "Player"), and a header
pill "Shared game · Player vs Master" (the bot level from the code).

**Dark mode (R6, R7):** the app's dark tokens; nothing new.

## While analysing (R8)

- The page appears at once: replaying the code gives the sheets, the log
  and the result without any analysis.
- Analysis runs in the worker roll by roll from the start: a move list,
  the luck average (shared with the played move's 1-ply value), and one
  average per other candidate (usually 2, at most 3), about 0.5–0.8 s per
  roll in WASM, so 20–30 s for a typical game on a laptop, more on a
  phone. The user accepted the wait.
- The graph and luck line grow left to right; the rest of the graph is
  hatched with "Analysing… 14 of 36 rolls". Tiles show totals "so far";
  the verdict reads "Analysing… 14 of 36". Rolls not yet analysed show
  "…" for luck, cost and best move ("working it out…").
- Selecting a roll that isn't analysed yet is allowed. *My call:* the
  analysis doesn't reorder to jump to it.
- *My call:* the analysis runs only while the review page is on screen
  (like the explorer's `setActive`), resumes where it stopped, and is
  dropped when a different game is opened. It must not disturb the
  explorer's own state if both share the engine worker.

## Testing

- Code: encode → decode round trip on recorded games; every field's edge
  values; equal white dice canonicalised; codes that don't replay (an
  illegal mark, a wrong length, a bad character, an unknown version) are
  rejected; a code cut at any roll boundary decodes.
- Numbers: on a recorded game, each roll's luck, and the passive-mark
  mistakes, equal what the explorer returns for that position (same
  position, same dice, bit for bit); the static values that pick the
  roller's top 3 candidates equal the explorer's move list whenever the
  actual white-sum mark matches its prediction.
- 1-ply: the played turn's 1-ply value equals the next roll's luck average
  bit for bit; a candidate's 1-ply value equals the brute-force mean of
  the next roller's best turn over all 27,216 rolls; game-ending
  candidates take 0% or 100%.
- Luck is zero-sum: the bot's luck is minus yours (as in the explorer).
- UI: stepping order; the switch shows played vs best on the sheets;
  pending states while analysing; "Player" naming on a shared link.

## Left to the builder

- Where the analysis code lives (a new worker call reusing the explorer's
  move list and luck, or calls to the existing ones).
- How the graph is drawn (SVG is what the mock-ups use).
- Caching analysed games (e.g. by code) so revisiting doesn't recompute.
