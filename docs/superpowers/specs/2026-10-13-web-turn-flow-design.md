# Web UI: one-confirm turns vs a bot

**Status:** design agreed, not scheduled. Input for the planned from-scratch
rewrite of the web UI; independent of the current implementation.

## Goal

Playing against a bot should take one decision per turn. Today the user picks
a white-sum square (or skips), clicks to advance to the colour phase, and picks
again. Instead the user selects everything for the turn on the board and
confirms once. The two rule phases still exist; the UI shows them only as the
order of animations.

Success: every legal turn can be entered as one selection plus one confirm; no
illegal turn can be confirmed; no strike is ever taken by accident; the user
always sees what the bot just did.

## Scope

- 1 human vs 1 bot. Multiplayer notes are in "Later".
- Turn input, legality, strike handling, and making bot moves visible.
- Out of scope: visual style, framework choice, the state explorer, MCTS or
  search in the browser, game setup and settings screens.

## Rules the flow must respect

These follow the engine (`src/game.rs`), which is the source of truth:

1. **Phase 1 (white sum) is simultaneous.** Every player may mark the white
   sum. All phase-1 marks are judged against the boards as they were *before*
   the roll, then applied together, then locks are propagated to everyone.
2. **Game end after phase 1.** If two rows are locked after phase 1, the game
   ends there: the active player gets no phase 2 and takes no strike.
3. **Phase 2 (white + colour) is the active player's only.** It is judged on
   the boards *after* phase 1 and lock propagation. A row locked in phase 1 (by
   anyone) is unavailable.
4. **Strike.** The active player takes a strike if they mark nothing in either
   phase. The active player may also choose to take a strike while a mark is
   available. A passive player never takes a strike.
5. **Ordering inside one row.** Marks go left to right, so if both of the
   active player's marks are in the same row, the white-sum mark must be the
   left one. A phase-1 mark can be the fifth mark that makes a phase-2 lock
   legal.

## The user's turn (user active)

1. **Roll.** Dice appear.
2. **Bot's white-sum mark is revealed immediately.** The bot's phase-1 choice
   depends only on the dice and the pre-roll boards (rule 1), never on the
   user's choice, so revealing it first is fair and gives the user full
   information. It animates onto the bot's board and is shown as pending until
   the turn resolves. If it locks a row, that row is shown as closing.
3. **Select.** The user clicks squares on their own board:
   - up to 2 number squares, **or**
   - the strike box (see "Strike protection").
   Clicking a selected square deselects it.
4. **Confirm.** One button whose label states the consequence: "Mark 1",
   "Mark 2", "Take strike", or "Pass" (only when rule 2 applies and nothing is
   selected). It is disabled whenever the selection is not a legal turn. Enter
   confirms.
5. **Resolve.** Phase-1 marks (both players) are applied, locks propagate, then
   the user's phase-2 mark, then end-of-game check. Animated in that order.

### Which squares are legal

The selection is a set of squares, not a sequence. The user never says which
square is the white sum: two marked squares give the same final board whichever
one was white, so it is enough that **some** legal split exists.

A selection is legal if one of these holds:

- **Empty:** only when the game ends after phase 1 (rule 2). Label "Pass".
- **One square q:** q is a legal white-sum mark on the pre-roll board, or q is
  a legal colour mark in phase 2 with no own phase-1 mark (phase 2 judged after
  the bot's revealed phase-1 mark and lock propagation; not available if the
  game ends after phase 1).
- **Two squares {a, b}:** for some assignment (w, c) of the two squares, w is a
  legal white-sum mark on the pre-roll board; after applying w and the bot's
  phase-1 mark and propagating locks the game is not over; and c is a legal
  colour mark on the resulting board.
- **Strike box alone:** active player, and the game does not end after phase 1.

**Highlighting.** Only squares that can still complete a legal selection,
given the current selection, are clickable and highlighted. Concretely: square
s is highlighted if some legal selection contains the current selection plus s.
After the first pick the highlights shrink to its possible partners. If none
remain, a short hint explains why, e.g. "Game ends after the white sum" or "Red
is locked".

**Engine support.** The WASM layer exposes, per roll, the list of legal
selections for the human (each a set of 0–2 squares, plus the strike option).
There are at most a few dozen, so the UI tests membership and derives the
highlights from the list instead of re-implementing the rules.
Optionally each greyed square carries a reason string for a hover tooltip
("needs 5 marks to lock", "row locked by the bot this roll").

## The bot's turn (user passive)

1. **Roll.** Dice appear.
2. **Select.** The user may click one white-sum square on their board, or
   nothing. The strike box is greyed out (rule 4).
3. **Confirm.** "Mark" or "Pass".
4. **Resolve.** The bot's phase-1 mark and the user's mark are applied
   together, locks propagate, then the bot's phase 2 (or its strike), then the
   end-of-game check. The user must confirm first because a lock from the
   user's white-sum mark can remove a colour from the bot's phase 2 (rule 3).
   The bot's phase-1 choice was already fixed independently (rule 1).

## Strike protection

- The strike box is a selectable target, like a square. A strike happens only
  when the user selects it and confirms.
- An empty selection on the user's turn never confirms (except "Pass" under
  rule 2), so a stray click on Confirm does nothing.
- The strike box is greyed out whenever anything else is selected, and number
  squares are greyed out while the strike box is selected. Switching means
  deselecting first.
- The strike box is greyed out on the bot's turn and when the game ends after
  phase 1.

## Seeing what the bot did

- **Last-move highlight.** The bot's marks (and strike) from its most recent
  turn, or its revealed white-sum mark on the user's turn, stay outlined until
  the user's next confirm.
- **Animation.** Bot marks appear in rule order, one at a time: phase 1, lock
  closing, phase 2 or strike. Short (about 300–500 ms per step); clicking
  anywhere or pressing a key skips to the end state.
- **Turn log.** A compact list, newest on top, one line per player per turn,
  e.g. "Bot: white 7 → red 7, blue 9 (locks blue)", "You: yellow 5 + strike".
  Includes locks and game-end reasons.

## Edge cases to test

- White sum 2 or 12 that locks a second row: no second square selectable,
  confirm reads "Mark 1", resolve skips phase 2, no strike.
- Bot's revealed phase-1 mark locks the second row: game ends after phase 1;
  user may still mark the white sum, or pass; no strike option.
- Bot locks row X in phase 1: the user's white-sum mark in X stays legal, a
  colour mark in X does not.
- Both marks in one row: legal only with the white-sum square on the left.
- The white-sum mark is the fifth in a row and makes the colour lock legal.
- The same square is both a legal white sum and a legal colour mark: one
  click selects it; it counts once.
- No legal marks at all on the user's turn: only the strike box is
  highlighted.

## Later

- **More than one bot.** Reveal every bot's white-sum mark on the user's turn.
  On a bot's turn the user still confirms first; the other bots' phase-1 marks
  are fixed independently, as above.
- Hover tooltips for greyed squares, if the reason strings turn out cheap.
- Animation speed setting.
