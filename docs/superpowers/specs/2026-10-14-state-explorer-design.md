# Web: state explorer

**Status:** implemented (branch `explorer`). Mockups: the "Explorer" page of the design canvas,
https://claude.ai/artifact/4eDmxXF5jEskvwxcFS26im, artboards E1–E5.

## Goal

Set up any Qwixx position, two players, and ask the bots about it: how likely
you are to win from here and, for a given roll, which move is best and which
move each bot would make. It serves players learning the game ("was that a
good move?") and the developer studying the bots, through one plain view.

Success: any position you can draw gets an answer; a position from a running
game opens in the explorer in one tap; a position can be sent to someone as
a link and opens identically.

## Scope

- Two sheets, "You" and "Opponent", both editable. 1 v 1 only, like the game.
- Position evaluation always; with dice set, a ranked list of your legal
  moves and each bot's pick.
- Entry: by hand, from the game ("Explore this position"), from a link.
- Out of scope: developer numbers (μ, σ, z, rollout counts). They were in an
  earlier mockup and were dropped. Ranking more than two players, move trees
  ("what if the opponent then…"), playing on from a position.

## Navigation

- The explorer is a second view of the same app at `#/explore`, sharing the
  worker, the WASM build, the theme and the sheet and dice components.
  `#/explore?p=<code>` opens a position (see "Links").
- **Desktop** (the game's desktop layout breakpoint): Play / Explore tabs next
  to the "Qwixxer" wordmark, on both views.
- **Phone:** no tabs. The game header keeps "Qwixxer"; the explorer header
  shows "Explorer" in the same place, then a copy-link button and "⋯".
- The "⋯" menu, game: New game · Explore this position · Explorer · Dark mode
  · Rules · Source code (+ notice). Explorer: Back to game · Dark mode · Rules
  · Source code (+ notice). On desktop the same items appear; the tabs are an
  extra route, not a replacement.
- "Explore this position" copies the game; "Explorer" opens the explorer's
  last position. Neither changes the game in progress, and "Back to game" (or
  the Play tab) returns to it exactly as it was. If the bot is mid-think when
  you switch, the game finishes that step in the background as usual.

## The position

A position is: both sheets' marks (44 squares each) and strikes (0–4), which
rows are closed, the six dice (each 1–6, or unset), and whose roll it is
("Your roll" or "Opponent's roll").

### Editing

- **Squares:** tap toggles the ink X. Any combination is allowed, including
  gaps and squares left of other marks; the explorer does not check that a
  position is reachable. Squares left of a row's rightmost mark render greyed
  as in the game, but stay tappable.
- **Strike boxes:** tap box *k* sets that sheet's strikes to *k*+1, or to *k*
  if it was already *k*+1 (so tapping the last taken box removes it).
- **Locks:** a row is closed when either sheet has its row-ending number (red
  and yellow 12, green and blue 2) marked with 5 or more marks in that row
  (the engine's rule), or when the row's lock was tapped. The lock circle is
  ink-filled on the sheet that closed it by marking, greyed on the other, and
  greyed on both when it was closed by a tap. Tapping a lock toggles the tap
  flag; a row closed by marking reopens only by unmarking its end number.
- **Clear** (per sheet on desktop; "Clear sheets" for both on phone) empties
  marks and strikes; tap-closed rows reopen.
- **Swap sides** exchanges the two sheets; dice stay, "whose roll" flips.
- **Dice:** tap a die to open a 1–6 picker under it; choosing a face sets it
  and moves the picker to the next unset die. "Random" sets all six; "Clear"
  unsets all six. Closed rows' colour dice render greyed and are ignored,
  as in the game. On "Opponent's roll" only the white dice matter and the
  colour dice render greyed.

Every edit re-evaluates (see "Engine"); the last answer stays on screen,
faded, until the new one arrives.

## Evaluation (always shown)

- "**38%** chance you win" with a bar, and "Opponent ahead by about 9 (± 15)"
  (or "You ahead by…", "Level (± 15)"). Under it, "before the roll".
- Numbers come from the pair net (Master's and Master+'s evaluator): P(win)
  is its Gaussian win probability for You against Opponent; the margin is
  current score difference + μ, rounded, with σ rounded as the ±.
- P(win) is shown to the nearest whole percent here and to one decimal in
  the move list.
- **Game over** positions (4 strikes on a sheet, or two closed rows) show
  the result instead ("You win 41 – 33", "Draw…") and no move list.

## Moves (dice set)

- Header "Best moves", "*N* legal · your chance to win after each".
- **Your roll:** every legal full turn from the game's rules (the same set
  the game offers: white only, colour only, white then colour), plus
  "Strike" (or "Pass" when the game's rules make a pass legal). The
  opponent marks the white sum at the same time; the explorer predicts
  that mark the way Master does and uses it everywhere: for legality (a
  lock it makes can end the game after the white sum), for the chances
  below, and for every bot's colour-mark decision. The Roll card shows it:
  "Opponent likely marks [chip]" or "Opponent likely passes".
- **Opponent's roll:** each legal white-sum mark, plus "Pass".
- Each row: rank, the marks as row-coloured chips (strike chip / "Pass"
  text), the bot tags, and your chance to win after the move with a bar.
- **The chance to win** is the pair net's P(win) for your sheet after the
  move, judged as Master judges it: against the opponent's sheet with its
  predicted white-sum mark on your roll, against their sheet on theirs.
  Moves are listed highest to lowest; on a tie Master+'s move, then
  Master's, comes first.
- **Why a bot plays a lower move:** Master plays the top move unless a
  fixed rule forces another (a safe lock, a game-winning end); Master+ can
  also prefer another move after playing the roll out. When Master+'s or
  Master's move is not first, its row has an (i) icon whose tooltip (hover,
  focus or tap) says which.
- No row is highlighted. The first 6 rows show; "Show all *N*" expands the
  list in place.
- **Bot tags:** Master+, Master, Student, Newbie, each on the move that bot
  would make in your seat with this roll. Master+ is ink-filled, the others
  outlined. Several tags may share a row.
- No dice: the list area says "Set the dice to rank the moves" and "Tap a
  die to choose its face, or press Random."

## Links and memory

- "Copy link" (desktop button, phone icon) copies `…/#/explore?p=<code>`
  and briefly shows "Link copied".
- `<code>`: base64url of a bit-packed position with a leading version
  character: per sheet 44 marks + 3 bits strikes, 4 tap-closed flags, 6
  dice × 3 bits (0 = unset), 1 bit whose roll. About 20 characters.
- A malformed or unknown-version code opens the last position with the
  message "That link doesn't hold a valid position."
- The explorer stores its current position in local storage (wrapped in
  try/catch like the game's save) and restores it when opened via
  "Explorer" or the tab. Opening a link or "Explore this position" replaces
  it.

## Explore this position (from the game)

Copies the game's current sheets (You → You, Bot → Opponent), closed rows,
and, while a roll is being resolved, its dice and whose roll it is. The
bot's revealed white-sum mark for that roll is not copied: the boards are
the ones before the roll's marks. Between rolls the dice are copied unset.

## Engine

A new WASM object `WasmExplorer` in the existing worker (`web/src/explorer.rs`),
holding the pair net and one instance of each bot. One call:

```
explore(position_json) -> {
  over: null | { you: i32, opponent: i32 },
  eval: { p_win: f32, margin: f32, sigma: f32 },
  moves: null | [ { marks: Sq[], strike: bool, pass: bool, p_win: f32,
                    bots: BotKind[] } ]
}
```

- Legal turns come from `selection.rs` (`active_options` with no opponent
  mark; `passive_options`), so the explorer and the game agree.
- Bot picks run each bot's own two-phase decision on the position (as
  `web/examples/wiring_check.rs` does) and map the result to a listed move.
  A pick that matches no listed move is an engine bug: tag nothing and log
  it in development builds.
- The worker answers explorer and game calls in order. Repeated edits
  coalesce: the client sends only the latest position once the previous
  call returns.
- Expected cost: Master+ dominates (game measurements: p95 ≈ 120 ms, max ≈
  300 ms in WASM); the rest is a few ms.

## Layout

- **Desktop:** sheets stacked on the left (You, then Swap sides and a one-line
  hint, then Opponent), each with name, total, score breakdown, strike boxes
  and Clear. Right column: Position card, Roll card (whose roll, Random,
  Clear, dice, white sum), Best moves.
- **Phone:** one scrolling page: header, Position card, both sheets side by
  side as columns (33 px squares; strike boxes under each), Swap sides /
  Clear sheets, Roll card, Best moves.
- Same tokens, fonts, themes, chips and lock looks as the game. No pencil
  marks in the explorer; every X is ink.

## Edge cases to test

- Arbitrary marks: gaps, marks left of others, end number marked with fewer
  than 5 marks (not closed), all 11 marked.
- Both sheets mark the same row's end number (closed; both show the lock).
- Two closed rows or 4 strikes: game-over result, no moves, dice ignored.
- A closed row's colour die set: ignored, no moves for that row.
- "Opponent's roll" with every white-sum square blocked: only "Pass".
- Link codes: round trip of random positions; truncated, padded, wrong
  version, non-base64 input.
- Swap sides twice returns the original position.
- Bot tags: every pick maps to a listed move over many random positions
  (a native test like `wiring_check`).
- Explore this position at each game phase, including during the bot's
  think; the game continues unchanged afterwards.

## Later

- Tap a move to preview it on your sheet, or to apply it and continue.
- Developer details (μ, σ, z, search rollouts), if wanted again.
- More than two players.
