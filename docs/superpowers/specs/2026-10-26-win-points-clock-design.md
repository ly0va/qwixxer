# Win net: row points, and a game-end clock head

Status: implemented, not yet fitted.

## Why

`win-eval`'s breakdown on r4 (champion) puts the own error in the late
game: 0.0204 at 40+ marks on both boards against 0.0088–0.0113 earlier,
and 0.0159 / 0.0148 at strike and lock decisions. The pick rate is close
to its ceiling (a second 64-rollout labelling agrees 66.2%, the champion
74.3%, a perfect predictor about 75%), so these arms are judged on own
error and games, not picks.

The slim experiment showed that scalars the first layer could rebuild
from the bits still help it fit. Two more of that kind, aimed at the
endgame:

## Row points (`win-arm --points`)

Each row half gets one more input after its bits: the row's points,
marks·(marks + 1)/2 / 78 with the lock bonus counted
([`points_input`](../../../src/dqn/win.rs)). It is a fixed weighted sum
of the mark bits (the j-th mark scores j), so it adds no information,
only a ready-made direction; and with the head's sum over rows it gives
the net each player's total score, which today it sees only as the
difference.

- A points net's row tokens are 54 wide (`P_TOKEN`; thermometer 52), the
  globals unchanged. The net's [`Encoding`](../../../src/dqn/win.rs)
  (plain, thermo, points) is told by its row encoder's width, as the
  thermometer net is today.
- `config.json` gains `"points": true`; both loaders build from the
  config, so a points dir plays anywhere. The row lookup table covers it
  (the points depend only on the half's mark count).
- `--points` needs `--thermo`; `--init` must match (a points init with
  `--points`, a non-points one without).

## Clock head (`win-arm --clock`)

An auxiliary head predicting the turns (either player's) left until the
game ends: a linear map of [h(x) | h(swap x)] → turns left / 50, MSE loss
with weight `--clock-weight` (0.5). Like the other aux heads it only
shapes the trunk; unlike them it is not saved. Its held-out RMSE, in
turns, is the last `fit.csv` column (`heldout_clock`, empty without the
head) and goes on the epoch line.

The label needs new data: every rollout now counts the turns it took
(`rollout_finals_and_turns`), and `win-data` stores the mean per sample.
That is `data.bin` version 3; version 2 files still load (without the
label, so `--clock` refuses them) but cannot be resumed.

## The experiments

On this branch `win-arm`'s heads and inputs are opt-in (the new defaults
are on `win-slim-inputs`), so the recipe is spelled out. Arm 1 compares
with `arm_t2` (same recipe, seed and data):

    $Q win-arm --data win_data_r4 --out arm_p --thermo --aux --tempo --points --lr 1e-3 --epochs 10
    $Q win-eval --model arm_p --data win_data_r4
    $Q win-gate --a arm_p --b arm_t2

Arm 2 needs a dataset with the label (`win-data` from `win_model`, r4's
settings), then a control and a clock arm on it:

    $Q win-data --player win_model --out win_data_r5 ...
    $Q win-arm --data win_data_r5 --out arm_c5 --thermo --aux --tempo --lr 1e-3 --epochs 10
    $Q win-arm --data win_data_r5 --out arm_k5 --thermo --aux --tempo --clock --lr 1e-3 --epochs 10
    $Q win-gate --a arm_k5 --b arm_c5

For each: held-out excess and diff RMSE by epoch against the control, the
40+ slice's own error in `win-eval`, then the gate (z > 2 against the
champion to replace it).
