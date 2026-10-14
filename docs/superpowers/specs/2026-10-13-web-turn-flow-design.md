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

1. **Roll.** The user presses a physical **Roll** button in the dice tray.
   It is redundant (the turn could roll by itself) but deliberate: it gives the
   player agency. Before the roll the dice are blank, the white sum reads "–"
   and the sheet shows no legal-move cues.
2. **Bot's white-sum mark is revealed immediately.** The bot's phase-1 choice
   depends only on the dice and the pre-roll boards (rule 1), never on the
   user's choice, so revealing it first is fair and gives the user full
   information. It draws in once on the bot's sheet as a pencil X (pending)
   and is inked when the turn resolves. If it locks a row, see "Pending lock" below.
3. **Select.** The user clicks squares on their own board:
   - up to 2 number squares, **or**
   - the strike box (see "Strike protection").
   Clicking a selected square deselects it.
4. **Confirm.** One button whose label states the consequence: "Mark 1",
   "Mark 2", "Take strike", or "Pass" (only when rule 2 applies and nothing is
   selected). It is inactive whenever the selection is not a legal turn, but
   stays clickable (`aria-disabled`, not `disabled`): pressing it then shows a
   short tip in the action bar, e.g. "Select a square to mark, or the strike
   box to take a strike." The action bar shows nothing else; no running
   commentary on the selection. Enter confirms.
5. **Resolve.** Phase-1 marks (both players) are applied, locks propagate, then
   the user's phase-2 mark, then end-of-game check. Animated in that order;
   the user's pencil Xs are re-stroked in ink.

### Pending lock

If the bot's revealed white-sum mark locks a row, that row is marked as
**about to be locked** on the user's board as well as the bot's, whether or not
the lock ends the game. The user must be able to tell at a glance that:

- the row closes when the turn resolves;
- their own white-sum mark in that row is still allowed this turn (rule 1),
  and only a colour mark there is not (rule 3);
- if it is the second locked row, the game ends after phase 1 (rule 2).

Agreed look: the row's colour band is hatched and its lock symbol ringed, on
both sheets, until the turn resolves; then it turns into the normal locked
state. No banner. The explanation is a tip, shown when the user taps a
blocked square in that row ("Red is locking: the bot marked red 12. Your
white sum may still go in red; a colour mark can't."), using the same tip
area as the inactive confirm button.

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
given the current selection, are clickable. Concretely: square s is clickable
if some legal selection contains the current selection plus s. After the first
pick the set shrinks to its possible partners. If none remain, a short hint
explains why, e.g. "Game ends after the white sum" or "Red is locked".

The cue is **weight and colour contrast**, not outlines or motion:

- legal squares keep their number heavy (extra-bold) in full row colour;
- the other open squares to the right of the row's last mark turn light
  weight in a softened row colour;
- squares left of the row's last mark can never be marked and are always
  greyed out (both sheets, all the time);
- before the roll, all open numbers look normal.

Nothing moves while the user is deciding; motion is reserved for events.

**Engine support.** The WASM layer exposes, per roll, the list of legal
selections for the human (each a set of 0–2 squares, plus the strike option).
There are at most a few dozen, so the UI tests membership and derives the
highlights from the list instead of re-implementing the rules.
Optionally each greyed square carries a reason string for a hover tooltip
("needs 5 marks to lock", "row locked by the bot this roll").

## The bot's turn (user passive)

1. **Roll.** The bot rolls (animated dice; no button for the user).
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

- **Draw-in, no highlight.** Each bot mark draws in once, like a pen stroke:
  its revealed white sum on the user's turn in pencil (pending), its own
  turn's marks in ink. No outlines, pulsing or persistent highlight; the log
  is the record if the user looked away.
- **Animation.** Bot marks appear in rule order, one at a time: phase 1, lock
  closing, phase 2 or strike. Short (about 300–500 ms per step); clicking
  anywhere or pressing a key skips to the end state.
- **Turn log.** No sentences. One block per roll, newest on top, labelled
  with the roll number; the roller's line first, then the other player's.
  Each mark is a rounded square in its row colour with the number, a strike is
  a red cross chip, a lock is a dark lock chip after the mark that caused it,
  "–" means a pass. The current roll is not highlighted; the user's line reads
  "choosing…" until they confirm.

## Game over

No dialog or overlay: the final sheets stay fully visible. The result sits in
the action bar where tips appear ("You win 45 – 38 · Two rows are locked."),
the turn pill reads "Game over", and the confirm button becomes **New game**.

## Pencil and ink

Pending marks are **pencil**: a thin, doubled graphite-grey X (the user's
selected squares, the bot's revealed white sum). Final marks are **ink**: a
bold hand-drawn X. Confirming re-strokes the user's pencil Xs in ink; the
bot's pencil mark is inked when the turn resolves. Strikes are ink Xs in the
strike boxes. This separates pending from done without motion, so it also
works with reduced motion.

## Header and chrome

- Wordmark "Qwixxer", a turn pill ("Your turn" or "Bot's turn", nothing
  more), both scores, and a
  **dark-mode toggle** (moon button). The dark palette is still to be
  designed.
- **Bot picker:** a dropdown in the bot sheet's header (pair search, pair
  net, DQN, GA champion, Opportunist, Conservative). Open question: a change
  mid-game should most likely apply from the next game, with a note saying
  so.

## Visual design

Settled on the canvas "Qwixxer UI directions" (claude.ai artifact
4eDmxXF5jEskvwxcFS26im, private to the owner), direction A, artboards A1–A3:
the paper-sheet look, desktop 1280×800, no scrolling.

- **Layout.** Left: dice tray with the Roll button and white sum; the user's
  sheet (four coloured row bands, number squares, lock symbol, then the
  score strip); the action bar (strike boxes, tip area, confirm). Right
  column: the bot's sheet (smaller) with the picker, then the log.
- **Sheet.** Rows keep the physical sheet's structure: red and yellow 2→12,
  green and blue 12→2, lock symbol last, four strike boxes. No Qwixx logo or
  box-art trade dress.
- **Tokens.** Ground #F3EDE1, paper #FFFDF8, ink #1D1B18, muted #6B6459,
  pencil #8B857C; rows red #C8323C, yellow #E9B824 (number ink #9A7400),
  green #2F8F4E, blue #2C5FAF. Type: Barlow Condensed (numbers, headings,
  wordmark), Barlow (text).
- **Dark tokens** (the same sheet at night; artboards "A3 · Dark mode" and
  "P1 · Phone, dark mode"): ground #171513, paper #24211D, log blocks
  #1C1A17, ink (chalk) #ECE5D7, muted #A39B8D, pencil #8E877C; bands deeper
  and numbers lighter tints of the row colour: red #A8343B / #F08A8F,
  yellow #C49A22 / #F0CD62, green #2D7E4A / #74C994, blue #2F5E9F / #8DB0EE.
  Primary buttons and the turn pill invert (chalk on charcoal); greyed and
  soft squares dim a little more than in light mode.
- **Dice** have pips. Rolling plays a short tumble (about 0.6 s): the dice
  spin while their faces shuffle, then settle; skipped with reduced motion.
  A locked row's die is out of the game: greyed, blank, never rolls.
- **Lock symbol** at the end of each row: greyed while that sheet has fewer
  than 5 marks in the row (it cannot be locked yet), regular from 5 marks,
  filled in ink for the player who locked it. Pencil Xs count toward the
  five, since a pending fifth mark can be followed by the lock in the same
  turn. The row-ending number (12 for red/yellow, 2 for green/blue) greys out
  by the same rule.
- **Locked row:** the colour band turns a neutral grey (light #C9C1B2, dark
  #3A352E) and every remaining square greys out; marks stay readable.
- **Reduced motion:** draw-ins are skipped; everything else is already
  static.
- **Phone (portrait, artboards P0–P2, 390×844).** The sheet is turned
  sideways: one column per row (red and yellow 2→12 top to bottom, green and
  blue 12→2, lock at the bottom), with square 48 px cells. A narrow right
  column holds the score breakdown, the strike boxes (2×2) and a tall confirm
  button at thumb height. Above the sheet: header (wordmark, your score, a Bot
  button that opens the drawer, dark-mode toggle), the dice tray with Roll,
  whose Roll button's slot shows the white sum after rolling ("White 12").
  There is no status strip or banner: the header's Bot button carries the
  bot's most recent marks as small log chips (its revealed white sum on the
  user's turn, its previous turn's marks before the user rolls), and locking
  shows on the sheet itself. The bot's sheet (compact, horizontal, view-only), the bot picker and the
  log live in a bottom drawer. Short phones shrink the cells, down to about
  36 px; no scrolling. Portrait only.
- Clickable prototype: the "Prototype · play a game (desktop)" artboard plays
  full games against a simple greedy stand-in bot (rules re-implemented in
  JavaScript for the prototype only).

## Edge cases to test

- White sum 2 or 12 that locks a second row: no second square selectable,
  confirm reads "Mark 1", resolve skips phase 2, no strike.
- Bot's revealed phase-1 mark locks the second row: game ends after phase 1;
  user may still mark the white sum, or pass; no strike option.
- Bot locks row X in phase 1 (game continues, and separately when it is the
  second lock): row X shows as about to be locked on both boards; the user's
  white-sum mark in X stays legal, a colour mark in X does not.
- Both marks in one row: legal only with the white-sum square on the left.
- The white-sum mark is the fifth in a row and makes the colour lock legal.
- The same square is both a legal white sum and a legal colour mark: one
  click selects it; it counts once.
- No legal marks at all on the user's turn: only the strike box is
  selectable.

## Later

- **More than one bot.** Reveal every bot's white-sum mark on the user's turn.
  On a bot's turn the user still confirms first; the other bots' phase-1 marks
  are fixed independently, as above.
- Hover tooltips for greyed squares, if the reason strings turn out cheap.
- Animation speed setting.
