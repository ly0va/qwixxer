# Explorer: expected marks and strikes, and Play

Two additions to the state explorer (`2026-10-14-state-explorer-design.md`).
Mockups: the "Qwixxer UI directions" canvas, Explorer page, E8–E14.

## Expected marks and strikes

The win net's aux heads predict each side's final marks per row and final
strikes. The explorer shows what each side is expected to *add*, in pencil,
so it never reads as points:

- **Rows:** "+N ✕" with N = round(expected − marks now), at the end of each
  row on desktop and under each column on the phone. None for a closed row or
  when N < 1.
- **Strikes:** "+x.x ✕" to one decimal (strike counts are small), after the
  strike boxes on desktop and on its own line under them on the phone. None
  below 0.1 or with four strikes.
- **Explanation:** a tooltip on each badge (hover, focus or tap), e.g. "About
  3 more yellow marks by the end of the game (the net's estimate)". No
  heading or legend. On the phone the tooltip spans the sheet.
- The badges dim while a new answer is pending and disappear once the game
  is over. The desktop sheet column is 666 px so a row and its badge fit.

## Play

Each move in the list has a "→ Play" button (filled for the top move). Play:

- applies your move with the opponent's predicted part, exactly as the
  explorer predicts it: on your roll its simultaneous white-sum mark; on
  theirs its whole turn as Master's lookahead predicts it after your mark;
- hands the roll to the other side and clears the dice.

The engine does this (`Explorer::play`), rejecting an illegal move, a
position without dice and a finished game.

A line at the top of the Roll card shows the last play: "Played [chips] ·
opponent marked [chips]" (or "passed" / "took a strike") with **Undo**. Undo
steps back through every play. Editing the sheets (marks, strikes, locks,
Clear, Swap, loading a position) clears the plays; setting the dice and
choosing whose roll it is keep them, so a line can be played out move by
move.
