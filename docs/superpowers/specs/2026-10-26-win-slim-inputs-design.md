# Win net: slim inputs

Status: implemented, not yet fitted.

## The question

The thermometer encoding
(`2026-10-19-win-thermo-aux-design.md`) kept the plain scalars beside the
bits, so a plain net could be widened with zero weights. Three of those
scalars are exact linear sums of their bits:

- a row half's progress (cells passed / 10 = the 10 progress bits / 10);
- a row half's mark count (marks / 11 = the 12 mark bits / 11);
- a player's strikes (strikes / 3 = the 3 strike bits / 3; four strikes is
  a finished game, valued exactly, never by the net).

The first layer is linear, so these add nothing the net can represent:
any weight on a scalar can be spread over its bits. They can still change
how training goes, by handing it a ready-made direction. Today's champion
is fitted from scratch, so the warm-start reason for them is gone. Does a
net without them fit and play as well, or better?

What stays: the locked flag (the only thing telling a closed row from an
open one at its end number), the next-mark weight (a nonlinear function of
progress and marks, a hand-made feature), and the cross-row globals
(blanks, total next-mark weight, lockable rows, score difference), which
the head cannot rebuild from its inputs in one linear step.

## Slim nets

A slim net reads [`slim`](../../../src/dqn/win.rs) inputs: the thermometer
input with those 18 scalars set to zero (the first two inputs of each of
the 8 row halves, and each player's strikes in the globals). A zero input
adds nothing and its weights get no gradient, so this is the same net as
one without the inputs, while every layout (tokens, swap, colour
relabellings, the row lookup table, the aux and tempo heads) stays as it
is. Weight decay shrinks the dead weights toward zero; they never matter.

- `config.json` gains `"slim": true`; `WinStrategy::load` (and the web
  build's loader) reads it, so a slim dir plays anywhere a win dir does.
- `win-arm --slim` fits one. It needs the thermometer inputs.

Speed is unchanged: the row encoder is a lookup table at inference, and
the head's first layer skips zero inputs.

## `win-arm` defaults

Thermometer inputs and the aux and tempo heads are now `win-arm`'s
defaults (the champion's recipe), with `--no-thermo`, `--no-aux` and
`--no-tempo` to turn them off. `--legacy-fit` implies all three off.

## The experiment

On the current dataset (r2), the champion's recipe and schedule, the same
seed, one control and one slim arm:

    $Q win-arm --data win_data_r2 --out arm_c --lr 1e-3 --epochs 10
    $Q win-arm --data win_data_r2 --out arm_s --lr 1e-3 --epochs 10 --slim

1. Compare the `fit.csv` curves: held-out excess and candidate-difference
   RMSE per epoch, best epoch.
2. Gate slim against the control, then against the champion:

       $Q win-gate --a arm_s --b arm_c
       $Q win-gate --a arm_s --b win_model

Slim replaces the recipe if it fits at least as well and the gate against
the control is level or better (z > −1); the champion changes only on the
usual z > 2 against it.
