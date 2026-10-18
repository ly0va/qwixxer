# Win Net: Thermometer Inputs, Auxiliary Heads, Held-Out Fitting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A stored, held-out-split dataset with auxiliary labels (`win-data`), a fit of one experiment arm with thermometer inputs, auxiliary heads, cosine decay, weight averaging and per-epoch held-out measurement (`win-arm`), and a head-to-head between two win model dirs (`win-gate`).

**Architecture:** Thermometer encoding and thermometer-capable nets live in `src/dqn/win.rs` (config option `thermo: Option<bool>`, `WinModel::trunk`, `ManualWinNet::logit_states`, `widen_to_thermo`). Labelled samples, generation, the split and the file format live in a new `src/dqn/win_data.rs`. The auxiliary heads (`AuxHeads`) live in `src/dqn/win.rs` so the web build can use them later; the metrics, schedule, weight averaging and the fit loop live in a new `src/dqn/win_arm.rs`. A fit with heads saves them to `aux.mpk` beside `model.mpk` (the model file itself is unchanged), and `WinStrategy::load` picks them up for a hand-written forward pass (`ManualWinNet::aux_states`) that play never calls. `win-gate` is a small function in `src/dqn/win_pi.rs`. Three CLI subcommands in `src/main.rs`.

**Tech Stack:** Rust 2021, burn 0.20.1 (ndarray backend, autodiff), rayon, clap 4.

**Spec:** `docs/superpowers/specs/2026-10-19-win-thermo-aux-design.md`

## Global Constraints

- 2-player only. Trunk shape unchanged (W from the init net, default 64 for a from-scratch net).
- Existing model dirs (no `thermo` in `config.json`) must load and play exactly as before.
- Not touched: `win-pi`'s loop and fit, `win-search`, `pair-train`, the web app, model artifacts (`win_model/`, `pair_model/`, `champion.txt`, `dqn_model/`).
- `src/dqn/win.rs` is compiled without the `dqn` feature (web build): after any change to it, `cargo check --lib --no-default-features` must pass with only its two existing warnings (unused `active_phase1_impl…` imports, unused `gates`).
- No new dependencies.
- Formatting: `rustfmt --edition 2021 <file>` on touched files only (repo `rustfmt.toml`: `max_width = 120`). Never `cargo fmt`. Never rustfmt `src/dqn/mod.rs`, `src/lib.rs`, `src/dqn/pair.rs`, or any file with an out-of-line `mod x;` declaration (check with `grep -n "^\s*\(pub \)\?mod [a-z_]*;" <file>` — must print nothing). Edit `src/dqn/mod.rs` by hand.
- Never commit `.superpowers/`, `AGENTS.md`, `web-ui.txt`, `win_training.csv`, `win_model/`, `win_pi/`, `win_pi_cand*/`. Never edit `AGENTS.md`. `git add` named files only.
- If `web/Cargo.lock` changes, revert it (`git checkout web/Cargo.lock`).
- Commit messages end with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Read the test output before committing; never chain a commit after a test command in one shell line.
- Test command for the whole suite: `cargo test --lib --release` (164 tests pass before Task 1).

## Review Focus

1. **A thermometer net where plain inputs are assumed** (`win-pi --init`, `win-gap`, `win-fit --player`, `win-search`, `bench win`): it must either play correctly (everything that goes through `WinStrategy` / `logit_states`) or be refused with a clear message (`win-pi --init`, `ManualWinNet::logit`). Tests: Task 2 `a_thermometer_net_refuses_plain_inputs`, `win_strategy_plays_a_thermometer_net`, `win_pi_refuses_a_thermometer_init`.
2. **Passive candidates** (stored from the opponent's side): every auxiliary label must swap with the players, not only `win`. Test: Task 3 `full_labels_match_the_win_only_labels` (swapped decision checked against `from_finals(...).swapped()`).
3. **Held-out games without decisions, or no held-out games at all:** early stopping falls back to the held-out win excess; a dataset with no held-out games is refused before fitting. Tests: Task 6 `held_out_games_without_decisions_stop_on_the_win_excess`, `validate_refuses_bad_settings`.
4. **Labels at exactly 0 or 1** (ended positions, unanimous rollouts): entropy, BCE and the noise floor stay finite and zero where they should. Test: Task 5 `metrics_are_finite_at_certain_labels`.
5. **A truncated or foreign dataset file:** `win_data::load` returns an error, never panics or returns garbage. Test: Task 4 `a_damaged_file_is_an_error`.

---

### Task 1: State bytes and thermometer encoding

**Files:**
- Modify: `src/state.rs` (add `State::to_bytes`, `State::from_bytes` after `row_free_values`; test at the end of `mod tests`)
- Modify: `src/dqn/win.rs` (constants and two functions after `swap_input`; tests in `mod tests`)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `State::to_bytes(&self) -> [u8; 13]`, `State::from_bytes(b: [u8; 13]) -> State`
  - `pub const T_HALF: usize = 26; pub const T_TOKEN: usize = 52; pub const T_GLOBALS: usize = 15; pub const WIN_INPUTS_T: usize = 223;`
  - `pub fn thermo_input(me: &State, opp: &State) -> [f32; WIN_INPUTS_T]`
  - `pub fn swap_thermo(x: &[f32; WIN_INPUTS_T]) -> [f32; WIN_INPUTS_T]`
  - Layout: row token r = `[me 4 scalars | me 10 progress bits | me 12 mark bits | opp same]`; globals = `[today's 9 | me strike bits ≥1,≥2,≥3 | opp strike bits]`.

- [ ] **Step 1: Write the failing tests**

In `src/state.rs`, at the end of `mod tests` (before its closing `}`):

```rust
    #[test]
    fn bytes_round_trip_keeps_every_row() {
        let mut s = State::default();
        for n in [2, 3, 4, 5, 6, 12] {
            s.apply_mark(Mark { row: 0, number: n });
        }
        s.apply_mark(Mark { row: 3, number: 9 });
        s.apply_strike();
        s.apply_strike();
        let b = s.to_bytes();
        let t = State::from_bytes(b);
        assert_eq!(t.to_bytes(), b);
        assert_eq!(t.row_totals(), s.row_totals());
        assert_eq!(t.row_free_values(), s.row_free_values());
        assert_eq!((t.strikes, t.count_points()), (s.strikes, s.count_points()));
        assert_eq!(b[3], 0, "a locked row stores free 0");
        let fresh = State::default().to_bytes();
        assert_eq!(State::from_bytes(fresh).to_bytes(), fresh);
    }
```

In `src/dqn/win.rs`, inside `mod tests` after `swapped_input_equals_input_with_players_exchanged`:

```rust
    #[test]
    fn thermometer_input_keeps_the_scalars_and_adds_level_bits() {
        let (a, b) = boards();
        let (p, x) = (win_input(&a, &b), thermo_input(&a, &b));
        for r in 0..4 {
            for side in 0..2 {
                let half = &x[T_TOKEN * r + T_HALF * side..][..T_HALF];
                assert_eq!(&half[..4], &p[TOKEN * r + 4 * side..][..4], "row {r} side {side} scalars");
            }
        }
        assert_eq!(&x[4 * T_TOKEN..4 * T_TOKEN + GLOBALS], &p[4 * TOKEN..]);
        // Me, red: marks 2 and 5, next free 6 → 4 cells passed, 2 marks.
        let red = &x[4..T_HALF];
        assert_eq!(&red[..10], &[1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]);
        assert_eq!(&red[10..], &[1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]);
        // Me, green (descending): mark 12, next free 11 → 1 cell, 1 mark.
        let green = &x[2 * T_TOKEN + 4..2 * T_TOKEN + T_HALF];
        assert_eq!(green[..10].iter().sum::<f32>(), 1.0);
        assert_eq!(green[10..].iter().sum::<f32>(), 1.0);
        // Opp, blue (descending): marks 9 and 7, next free 6 → 6 cells, 2 marks.
        let blue = &x[3 * T_TOKEN + T_HALF + 4..4 * T_TOKEN];
        assert_eq!(blue[..10].iter().sum::<f32>(), 6.0);
        assert_eq!(blue[10..].iter().sum::<f32>(), 2.0);
        // Strike bits: me one strike, opp none.
        assert_eq!(&x[4 * T_TOKEN + GLOBALS..], &[1., 0., 0., 0., 0., 0.]);
    }

    #[test]
    fn a_locked_row_sets_every_progress_bit() {
        let mut a = State::default();
        for n in [2, 3, 4, 5, 6, 12] {
            a.apply_mark(Mark { row: 0, number: n });
        }
        for _ in 0..3 {
            a.apply_strike();
        }
        let x = thermo_input(&a, &State::default());
        assert_eq!(&x[4..14], &[1.0; 10]);
        assert_eq!(x[14..T_HALF].iter().sum::<f32>(), 7.0, "6 marks and the lock bonus");
        assert_eq!(&x[4 * T_TOKEN + GLOBALS..], &[1., 1., 1., 0., 0., 0.]);
        // Fresh boards: no bits at all.
        let f = thermo_input(&State::default(), &State::default());
        for r in 0..4 {
            for side in 0..2 {
                assert!(f[T_TOKEN * r + T_HALF * side + 4..][..22].iter().all(|&v| v == 0.0));
            }
        }
    }

    #[test]
    fn swapped_thermometer_input_equals_input_with_players_exchanged() {
        let (a, b) = boards();
        assert_eq!(swap_thermo(&thermo_input(&a, &b)), thermo_input(&b, &a));
        assert_eq!(swap_thermo(&swap_thermo(&thermo_input(&a, &b))), thermo_input(&a, &b));
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib --release -- bytes_round_trip thermometer_input a_locked_row swapped_thermometer`
Expected: compile errors — `no method named 'to_bytes'`, `cannot find function 'thermo_input'`, `cannot find value 'T_TOKEN'`.

- [ ] **Step 3: Implement**

In `src/state.rs`, inside `impl State` right after `row_free_values`:

```rust
    /// Raw form for storage: strikes, then per row (ascending, total, next
    /// free number or 0 when locked).
    pub fn to_bytes(&self) -> [u8; 13] {
        let mut b = [0u8; 13];
        b[0] = self.strikes;
        for (i, r) in self.rows.iter().enumerate() {
            b[1 + 3 * i] = r.ascending as u8;
            b[2 + 3 * i] = r.total;
            b[3 + 3 * i] = r.free.unwrap_or(0);
        }
        b
    }

    /// The inverse of [`State::to_bytes`].
    pub fn from_bytes(b: [u8; 13]) -> State {
        let mut s = State {
            strikes: b[0],
            rows: [Row::default(); 4],
        };
        for (i, r) in s.rows.iter_mut().enumerate() {
            *r = Row {
                ascending: b[1 + 3 * i] != 0,
                total: b[2 + 3 * i],
                free: (b[3 + 3 * i] != 0).then_some(b[3 + 3 * i]),
            };
        }
        s
    }
```

In `src/dqn/win.rs`, after `swap_input`:

```rust
/// Thermometer row half: a player's 4 row scalars, 10 progress bits
/// (cells passed ≥ j), 12 mark bits (row total ≥ j).
pub const T_HALF: usize = 26;
/// Thermometer row token: my half, then the opponent's.
pub const T_TOKEN: usize = 2 * T_HALF;
/// Thermometer globals: today's 9, then 3 strike bits per player (≥ 1, 2, 3).
pub const T_GLOBALS: usize = GLOBALS + 6;
/// Full thermometer input.
pub const WIN_INPUTS_T: usize = 4 * T_TOKEN + T_GLOBALS;

/// One player's row bits: `progress` is the row's `row_progress` (cells
/// passed / 10; a locked row is 1), `total` its mark count.
fn row_bits(out: &mut [f32], progress: f32, total: u8) {
    let cells = (progress * 10.0).round() as usize;
    for j in 1..=10 {
        out[j - 1] = (cells >= j) as u8 as f32;
    }
    for j in 1..=12 {
        out[9 + j] = (total as usize >= j) as u8 as f32;
    }
}

/// [`win_input`] with each row's progress and mark count also as
/// thermometer bits, and each player's strikes as 3 bits
/// (docs/superpowers/specs/2026-10-19-win-thermo-aux-design.md). The bits
/// are direction-free like the scalars, so colour relabellings still only
/// permute the tokens.
pub fn thermo_input(me: &State, opp: &State) -> [f32; WIN_INPUTS_T] {
    let plain = win_input(me, opp);
    let totals = [me.row_totals(), opp.row_totals()];
    let mut x = [0.0f32; WIN_INPUTS_T];
    for r in 0..4 {
        for side in 0..2 {
            let half = &mut x[T_TOKEN * r + T_HALF * side..][..T_HALF];
            half[..4].copy_from_slice(&plain[TOKEN * r + 4 * side..][..4]);
            row_bits(&mut half[4..], plain[TOKEN * r + 4 * side], totals[side][r]);
        }
    }
    let g = 4 * T_TOKEN;
    x[g..g + GLOBALS].copy_from_slice(&plain[4 * TOKEN..]);
    for (side, strikes) in [me.strikes, opp.strikes].into_iter().enumerate() {
        for j in 1..=3 {
            x[g + GLOBALS + 3 * side + j - 1] = (strikes as usize >= j) as u8 as f32;
        }
    }
    x
}

/// [`swap_input`] for [`thermo_input`]: swap the halves of every token, the
/// per-player globals and the strike bits; negate the difference.
pub fn swap_thermo(x: &[f32; WIN_INPUTS_T]) -> [f32; WIN_INPUTS_T] {
    let mut s = *x;
    for r in 0..4 {
        let t = T_TOKEN * r;
        s[t..t + T_HALF].copy_from_slice(&x[t + T_HALF..t + T_TOKEN]);
        s[t + T_HALF..t + T_TOKEN].copy_from_slice(&x[t..t + T_HALF]);
    }
    let g = 4 * T_TOKEN;
    s[g..g + 4].copy_from_slice(&x[g + 4..g + 8]);
    s[g + 4..g + 8].copy_from_slice(&x[g..g + 4]);
    s[g + 8] = -x[g + 8];
    s[g + 9..g + 12].copy_from_slice(&x[g + 12..g + 15]);
    s[g + 12..g + 15].copy_from_slice(&x[g + 9..g + 12]);
    s
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib --release -- bytes_round_trip thermometer_input a_locked_row swapped_thermometer`
Expected: `test result: ok. 4 passed`.

Run: `cargo check --lib --no-default-features 2>&1 | grep -c "^warning"`
Expected: `3` (the two existing warnings plus the summary line), nothing new.

- [ ] **Step 5: Format, full suite, commit**

```bash
rustfmt --edition 2021 src/state.rs src/dqn/win.rs
cargo test --lib --release 2>&1 | tail -3
```
Expected: `test result: ok. 168 passed`. Then:

```bash
git add src/state.rs src/dqn/win.rs
git commit -m "feat(win): thermometer inputs and a byte form for State

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Thermometer nets

**Files:**
- Modify: `src/dqn/win.rs` (`WinModelConfig`, `WinModel::{is_thermo, trunk, g}`, `ManualWinNet`, `WinStrategy`, new `widen_to_thermo`, `flat_tensor`, `model_logits_t`; tests)
- Modify: `src/dqn/win_train.rs:876` (`value_gap` prediction)
- Modify: `src/dqn/win_pi.rs` (`load_win_ad` becomes `pub(crate)`; `WinPiCfg::validate` refuses a thermometer `--init`; test)

**Interfaces:**
- Consumes (Task 1): `thermo_input`, `swap_thermo`, `T_HALF`, `T_TOKEN`, `T_GLOBALS`, `WIN_INPUTS_T`.
- Produces:
  - `WinModelConfig { width: usize, thermo: Option<bool> }`, `WinModelConfig::is_thermo(&self) -> bool`; `init` builds a `T_TOKEN`-wide row encoder and a `W + T_GLOBALS`-wide head when thermo.
  - `WinModel::is_thermo(&self) -> bool`; `WinModel::trunk(&self, x: Tensor<B, 2>) -> (Tensor<B, 3>, Tensor<B, 2>)` returning (row encodings `[n, 4, W]`, last hidden `[n, W]`); `WinModel::g` = `out(trunk(x).1)`.
  - `ManualWinNet::is_thermo(&self) -> bool`; `ManualWinNet::logit_states(&self, me: &State, opp: &State) -> f32`; `ManualWinNet::logit` panics with "a thermometer net takes logit_states" for a thermometer net.
  - `pub fn widen_to_thermo<B: Backend>(model: &WinModel<B>, device: &B::Device) -> WinModel<B>`
  - `pub fn flat_tensor<B: Backend>(flat: &[f32], d: usize, device: &B::Device) -> Tensor<B, 2>`
  - `pub fn model_logits_t(model: &WinModel<MyBackend>, rows: &[[f32; WIN_INPUTS_T]]) -> Vec<f32>`
  - `pub(crate) fn load_win_ad(dir: &str) -> (WinModelConfig, WinModel<Ad>)` in `win_pi.rs`.

- [ ] **Step 1: Write the failing tests**

In `src/dqn/win.rs` `mod tests`, after `manual_net_matches_the_burn_model` (the module already imports `MyBackend`, `SmallRng`, `Mark`, `Bot`, `WinProb`, `phi`):

```rust
    fn random_thermo(seed: u64, width: usize) -> WinModel<MyBackend> {
        let device = Default::default();
        <MyBackend as burn::tensor::backend::Backend>::seed(&device, seed);
        WinModelConfig::new()
            .with_width(width)
            .with_thermo(Some(true))
            .init::<MyBackend>(&device)
    }

    /// Real positions: the two boards both ways, and a board with a locked row.
    fn pairs() -> Vec<(State, State)> {
        let (a, b) = boards();
        let mut c = b;
        for n in [2, 3, 4, 5, 6, 12] {
            c.apply_mark(Mark { row: 0, number: n });
        }
        vec![(a, b), (b, a), (c, a), (a, c)]
    }

    fn relabel_t(x: &[f32; WIN_INPUTS_T], code: u8) -> [f32; WIN_INPUTS_T] {
        let mut order = [0usize, 1, 2, 3];
        if code & 1 != 0 {
            order.swap(0, 1);
        }
        if code & 2 != 0 {
            order.swap(2, 3);
        }
        if code & 4 != 0 {
            order = [order[2], order[3], order[0], order[1]];
        }
        let mut y = *x;
        for (slot, &src) in order.iter().enumerate() {
            y[T_TOKEN * slot..T_TOKEN * (slot + 1)].copy_from_slice(&x[T_TOKEN * src..T_TOKEN * (src + 1)]);
        }
        y
    }

    #[test]
    fn thermometer_logit_is_antisymmetric_and_colour_invariant() {
        let model = random_thermo(21, 16);
        let xs: Vec<_> = pairs().iter().map(|(me, opp)| thermo_input(me, opp)).collect();
        let swapped: Vec<_> = xs.iter().map(swap_thermo).collect();
        let base = model_logits_t(&model, &xs);
        for (z, zs) in base.iter().zip(model_logits_t(&model, &swapped)) {
            assert!((z + zs).abs() < 1e-5, "z {z} vs swapped {zs}");
        }
        for code in 0..8u8 {
            let ys: Vec<_> = xs.iter().map(|x| relabel_t(x, code)).collect();
            for (z, zr) in base.iter().zip(model_logits_t(&model, &ys)) {
                assert!((z - zr).abs() < 1e-5, "relabelling {code}: {z} vs {zr}");
            }
        }
    }

    #[test]
    fn manual_net_matches_burn_for_a_thermometer_net() {
        let model = random_thermo(22, 16);
        let net = ManualWinNet::from_model(&model);
        assert!(net.is_thermo() && model.is_thermo());
        let xs: Vec<_> = pairs().iter().map(|(me, opp)| thermo_input(me, opp)).collect();
        for ((me, opp), zb) in pairs().iter().zip(model_logits_t(&model, &xs)) {
            let zm = net.logit_states(me, opp);
            assert!((zb - zm).abs() < 1e-4, "burn {zb} vs manual {zm}");
        }
    }

    #[test]
    #[should_panic(expected = "a thermometer net takes logit_states")]
    fn a_thermometer_net_refuses_plain_inputs() {
        let net = ManualWinNet::from_model(&random_thermo(23, 8));
        let (a, b) = boards();
        net.logit(&win_input(&a, &b));
    }

    #[test]
    fn a_plain_net_gives_the_same_logit_from_states() {
        let net = ManualWinNet::from_model(&random_model(24, 16));
        assert!(!net.is_thermo());
        for (me, opp) in pairs() {
            assert_eq!(net.logit_states(&me, &opp), net.logit(&win_input(&me, &opp)));
        }
    }

    #[test]
    fn widening_keeps_the_function() {
        let device = Default::default();
        let plain = random_model(25, 16);
        let wide = widen_to_thermo(&plain, &device);
        assert!(wide.is_thermo() && !plain.is_thermo());
        let (p, t) = (ManualWinNet::from_model(&plain), ManualWinNet::from_model(&wide));
        for (me, opp) in pairs() {
            let (zp, zt) = (p.logit_states(&me, &opp), t.logit_states(&me, &opp));
            assert!((zp - zt).abs() < 1e-5, "plain {zp} vs widened {zt}");
        }
    }

    #[test]
    fn win_strategy_plays_a_thermometer_net() {
        let bot = WinStrategy::from_model(&random_thermo(26, 16));
        let (a, b) = boards();
        let values = bot.evaluate_batch(&[a, b], &[b]);
        assert_eq!(values[0], probit_of_logit(bot.net.logit_states(&a, &b)));
        assert_eq!(values[1], probit_of_logit(bot.net.logit_states(&b, &b)));
        let p = bot.win_prob_multi(&[(&a, std::slice::from_ref(&a))])[0];
        assert!((p - 0.5).abs() < 1e-6, "p = {p}");
    }

    #[test]
    fn old_model_dirs_load_and_thermometer_dirs_round_trip() {
        let base = std::env::temp_dir().join(format!("qwixxer-win-dirs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (a, b) = boards();
        for thermo in [false, true] {
            let dir = base.join(if thermo { "thermo" } else { "old" });
            std::fs::create_dir_all(&dir).unwrap();
            let dir = dir.to_string_lossy().into_owned();
            let model = if thermo { random_thermo(27, 8) } else { random_model(27, 8) };
            model
                .clone()
                .save_file(format!("{dir}/model"), &CompactRecorder::new())
                .unwrap();
            if thermo {
                WinModelConfig::new()
                    .with_width(8)
                    .with_thermo(Some(true))
                    .save(format!("{dir}/config.json"))
                    .unwrap();
            } else {
                // A config.json written before `thermo` existed.
                std::fs::write(format!("{dir}/config.json"), "{\n  \"width\": 8\n}").unwrap();
                let cfg = WinModelConfig::load(format!("{dir}/config.json")).unwrap();
                assert_eq!((cfg.width, cfg.thermo, cfg.is_thermo()), (8, None, false));
            }
            let loaded = WinStrategy::load(&dir);
            assert_eq!(loaded.net.is_thermo(), thermo);
            // The file is half precision: close, not equal.
            let want = ManualWinNet::from_model(&model).logit_states(&a, &b);
            let got = loaded.net.logit_states(&a, &b);
            assert!((want - got).abs() < 0.05, "thermo {thermo}: {want} vs {got}");
        }
        std::fs::remove_dir_all(&base).unwrap();
    }
```

In `src/dqn/win_pi.rs` `mod tests`, after `validate_refuses_bad_settings_before_any_work`:

```rust
    #[test]
    fn win_pi_refuses_a_thermometer_init() {
        let base = tmp("thermo-init");
        let init = format!("{base}/init");
        let device = Default::default();
        let mcfg = WinModelConfig::new().with_width(4).with_thermo(Some(true));
        save_win(&init, &mcfg, &mcfg.init::<MyBackend>(&device));
        let err = tiny_cfg(format!("{base}/run"), Some(init)).validate().unwrap_err();
        assert!(err.contains("plain nets only"), "{err}");
        std::fs::remove_dir_all(&base).unwrap();
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib --release -- thermometer_logit manual_net_matches_burn_for a_thermometer_net_refuses a_plain_net_gives widening_keeps win_strategy_plays_a_thermometer old_model_dirs win_pi_refuses`
Expected: compile errors — `no method named 'with_thermo'`, `cannot find function 'model_logits_t'`, `no method named 'logit_states'`, `cannot find function 'widen_to_thermo'`.

- [ ] **Step 3: Implement the model changes in `src/dqn/win.rs`**

Imports: change the burn import to

```rust
use burn::{
    module::Param,
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    record::CompactRecorder,
    tensor::TensorData,
};
```

Replace `WinModelConfig` and its `init`:

```rust
#[derive(Config, Debug)]
pub struct WinModelConfig {
    /// Encoder width W (the head is 2W → W).
    #[config(default = 32)]
    pub width: usize,
    /// Thermometer inputs ([`thermo_input`]). `None` in model dirs written
    /// before it existed: an `Option` because burn's config loader rejects
    /// a missing plain field.
    pub thermo: Option<bool>,
}

impl WinModelConfig {
    pub fn is_thermo(&self) -> bool {
        self.thermo == Some(true)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> WinModel<B> {
        let w = self.width;
        let (token, globals) = if self.is_thermo() {
            (T_TOKEN, T_GLOBALS)
        } else {
            (TOKEN, GLOBALS)
        };
        let lin = |d_in: usize, d_out: usize| LinearConfig::new(d_in, d_out).with_bias(true).init(device);
        WinModel {
            row1: lin(token, w),
            row2: lin(w, w),
            pair: lin(w, w),
            head1: lin(w + globals, 2 * w),
            head2: lin(2 * w, w),
            out: lin(w, 1),
            activation: Relu::new(),
        }
    }
}
```

Replace `WinModel::g` (keep `forward` as it is) with:

```rust
    /// Whether the net takes thermometer inputs (its row encoder's width).
    pub fn is_thermo(&self) -> bool {
        self.row1.weight.shape().dims[0] == T_TOKEN
    }

    /// The trunk per input row: every row's encoding e_r `[n, 4, W]` and
    /// the head's last hidden layer h `[n, W]`.
    pub fn trunk(&self, x: Tensor<B, 2>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [n, d] = x.dims();
        let token = self.row1.weight.shape().dims[0];
        let tokens = x.clone().narrow(1, 0, 4 * token).reshape([n, 4, token]);
        let e = self
            .activation
            .forward(self.row2.forward(self.activation.forward(self.row1.forward(tokens))));
        let w = e.dims()[2];
        let asc = e.clone().narrow(1, 0, 2).sum_dim(1);
        let desc = e.clone().narrow(1, 2, 2).sum_dim(1);
        let rows = self.activation.forward(self.pair.forward(asc)) + self.activation.forward(self.pair.forward(desc));
        let h = Tensor::cat(vec![rows.reshape([n, w]), x.narrow(1, 4 * token, d - 4 * token)], 1);
        let h = self.activation.forward(self.head1.forward(h));
        (e, self.activation.forward(self.head2.forward(h)))
    }

    /// g per input row: `[n, inputs]` → `[n, 1]`.
    pub fn g(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.out.forward(self.trunk(x).1)
    }
```

After `model_logits`, add:

```rust
/// Rows of `d` inputs each, flat, as a `[rows, d]` tensor.
pub fn flat_tensor<B: Backend>(flat: &[f32], d: usize, device: &B::Device) -> Tensor<B, 2> {
    Tensor::<B, 1>::from_floats(flat, device).reshape([flat.len() / d, d])
}

/// [`model_logits`] for thermometer inputs.
pub fn model_logits_t(model: &WinModel<MyBackend>, rows: &[[f32; WIN_INPUTS_T]]) -> Vec<f32> {
    let device = Default::default();
    let x: Vec<f32> = rows.iter().flatten().copied().collect();
    let s: Vec<f32> = rows.iter().flat_map(swap_thermo).collect();
    let z = model.forward(
        flat_tensor(&x, WIN_INPUTS_T, &device),
        flat_tensor(&s, WIN_INPUTS_T, &device),
    );
    z.into_data().to_vec::<f32>().unwrap()
}

/// A thermometer net with the same function as the plain `model`: every
/// weight copied, the new input weights (the row bits in the row encoder's
/// first layer, the strike bits in the head's first layer) zero.
pub fn widen_to_thermo<B: Backend>(model: &WinModel<B>, device: &B::Device) -> WinModel<B> {
    assert!(!model.is_thermo(), "already a thermometer net");
    let w = model.row1.weight.shape().dims[1];
    let mut out = model.clone();
    // Weights are input-major: input i's weights are `[i * d_out..][..d_out]`.
    let old = model.row1.weight.val().into_data().to_vec::<f32>().unwrap();
    let mut new = vec![0.0f32; T_TOKEN * w];
    for side in 0..2 {
        for k in 0..4 {
            let (src, dst) = (4 * side + k, T_HALF * side + k);
            new[dst * w..(dst + 1) * w].copy_from_slice(&old[src * w..(src + 1) * w]);
        }
    }
    out.row1.weight = Param::from_tensor(Tensor::from_data(TensorData::new(new, [T_TOKEN, w]), device));
    let d_out = model.head1.weight.shape().dims[1];
    let old = model.head1.weight.val().into_data().to_vec::<f32>().unwrap();
    let mut new = vec![0.0f32; (w + T_GLOBALS) * d_out];
    new[..old.len()].copy_from_slice(&old);
    out.head1.weight = Param::from_tensor(Tensor::from_data(
        TensorData::new(new, [w + T_GLOBALS, d_out]),
        device,
    ));
    out
}
```

In `ManualWinNet`: add a field after `width`:

```rust
    /// Row-token width: [`TOKEN`] (plain) or [`T_TOKEN`] (thermometer).
    token: usize,
```

set it in `from_model` with `token: model.row1.weight.shape().dims[0],`, and replace `g`, `logit` with:

```rust
    fn g(&self, x: &[f32]) -> f32 {
        let (w, t) = (self.width, self.token);
        let (mut h, mut e) = (Vec::with_capacity(2 * w), Vec::with_capacity(w));
        let mut pooled = [vec![0.0f32; w], vec![0.0f32; w]];
        for r in 0..4 {
            self.row1.apply(&x[t * r..t * (r + 1)], &mut h, true);
            self.row2.apply(&h, &mut e, true);
            for (p, v) in pooled[r / 2].iter_mut().zip(&e) {
                *p += v;
            }
        }
        let mut cat = vec![0.0f32; w];
        for p in &pooled {
            self.pair.apply(p, &mut e, true);
            for (c, v) in cat.iter_mut().zip(&e) {
                *c += v;
            }
        }
        cat.extend_from_slice(&x[4 * t..]);
        self.head1.apply(&cat, &mut h, true);
        self.head2.apply(&h, &mut e, true);
        self.out.apply(&e, &mut h, false);
        h[0]
    }

    pub fn is_thermo(&self) -> bool {
        self.token == T_TOKEN
    }

    /// Antisymmetric logit z = g(x) − g(swap x) of a plain input; a
    /// thermometer net takes [`ManualWinNet::logit_states`].
    pub fn logit(&self, x: &[f32; WIN_INPUTS]) -> f32 {
        assert!(!self.is_thermo(), "a thermometer net takes logit_states, not plain inputs");
        self.g(x) - self.g(&swap_input(x))
    }

    /// The logit for `me` against `opp` in the net's own input encoding.
    pub fn logit_states(&self, me: &State, opp: &State) -> f32 {
        if self.is_thermo() {
            let x = thermo_input(me, opp);
            self.g(&x) - self.g(&swap_thermo(&x))
        } else {
            self.logit(&win_input(me, opp))
        }
    }
```

In `WinStrategy`: change the `load` doc to `/// Load \`dir/config.json\` (width and encoding) and \`dir/model.mpk\`.`; in `evaluate_batch_multi` replace `self.net.logit(&win_input(c, &opp))` with `self.net.logit_states(c, &opp)`; in `win_prob_multi` replace `self.net.logit(&win_input(our, &Self::opponent(opps)))` with `self.net.logit_states(our, &Self::opponent(opps))`.

- [ ] **Step 4: Update the other call sites**

`src/dqn/win_train.rs`, in `value_gap` (around line 876): replace

```rust
            sigmoid(win.net.logit(&win_input(&b[0], &b[1])))
```

with

```rust
            sigmoid(win.net.logit_states(&b[0], &b[1]))
```

`src/dqn/win_pi.rs`: change `fn load_win_ad` to `pub(crate) fn load_win_ad`. In `WinPiCfg::validate`, replace the `if let Some(init) = &self.init { … }` block with:

```rust
        if let Some(init) = &self.init {
            for f in ["config.json", "model.mpk"] {
                if !Path::new(&format!("{init}/{f}")).exists() {
                    return Err(format!("--init {init}: {f} not found"));
                }
            }
            match WinModelConfig::load(format!("{init}/config.json")) {
                Ok(c) if c.is_thermo() => {
                    return Err(format!(
                        "--init {init}: win-pi trains plain nets only (this is a thermometer net)"
                    ))
                }
                Ok(_) => {}
                Err(e) => return Err(format!("--init {init}: unreadable config.json ({e:?})")),
            }
        }
```

If the compiler reports `win_input` unused in `win_train.rs`, leave the import: it is still used elsewhere in that file (check with `grep -n "win_input(" src/dqn/win_train.rs`); remove it only if that grep shows no other use.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib --release -- thermometer_logit manual_net_matches_burn_for a_thermometer_net_refuses a_plain_net_gives widening_keeps win_strategy_plays_a_thermometer old_model_dirs win_pi_refuses`
Expected: `test result: ok. 8 passed`.

Run: `cargo check --lib --no-default-features 2>&1 | grep "^warning" | head`
Expected: only the two existing warnings (unused imports in `src/strategy/mod.rs`, unused `gates`) and the summary.

- [ ] **Step 6: Format, full suite, commit**

```bash
rustfmt --edition 2021 src/dqn/win.rs src/dqn/win_train.rs src/dqn/win_pi.rs
cargo test --lib --release 2>&1 | tail -3
```
Expected: `test result: ok. 176 passed` (including `rollout_positions_by_output_is_pinned`, unchanged). Then:

```bash
git add src/dqn/win.rs src/dqn/win_train.rs src/dqn/win_pi.rs
git commit -m "feat(win): thermometer nets: config option, trunk, manual net, widening a plain net

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Labelled samples with auxiliary labels

**Files:**
- Create: `src/dqn/win_data.rs`
- Modify: `src/dqn/mod.rs` (register the module — by hand, never rustfmt this file)
- Modify: `src/dqn/win_distill.rs` (`sample_var` becomes `pub(crate)`; `GroupView` + `diff_stats_views`, with `diff_stats` delegating)

**Interfaces:**
- Consumes (Task 1): `State::to_bytes`. Existing: `CandDecision`, `label_decision`, `rollout_final_states`, `SimGame::outcome`.
- Produces (in `crate::dqn::win_data`):
  - `#[derive(Clone, Copy, Debug)] pub struct Sample { pub me: State, pub opp: State, pub win: f32, pub win_var: f32, pub marks: [f32; 8], pub strikes: [f32; 2], pub diff: f32 }` with exact `PartialEq` (boards byte for byte, every label).
  - `Sample::from_finals(me: State, opp: State, finals: &[Vec<State>]) -> Sample`, `Sample::swapped(&self) -> Sample`, `Sample::floats(&self) -> [f32; 13]`, `Sample::from_floats(me, opp, f: [f32; 13]) -> Sample`.
  - `#[derive(Clone, Debug, PartialEq)] pub struct DecisionData { pub samples: Vec<Sample>, pub pair_var: Vec<f32>, pub swapped: bool }`
  - `pub fn label_decision_full(bot: &impl Bot, d: &CandDecision, k: usize) -> DecisionData`
  - In `win_distill`: `pub(crate) fn sample_var(xs: &[f32]) -> f32`; `pub struct GroupView<'a> { pub labels: Vec<f32>, pub pair_var: &'a [f32], pub swapped: bool }`; `pub fn diff_stats_views(preds: &[Vec<f32>], groups: &[GroupView], k: usize) -> Option<DiffReport>`.

- [ ] **Step 1: Register the module and write the failing tests**

In `src/dqn/mod.rs`, after the `pub mod win_pi;` line:

```rust
#[cfg(feature = "dqn")]
pub mod win_data;
```

Create `src/dqn/win_data.rs` with only the tests for now:

```rust
//! Stored datasets for the win net: turn ends and the candidates of the
//! decisions win-search would search, labelled with full-game rollouts
//! (win and auxiliary labels), split into training and held-out games.
//!
//! Design: docs/superpowers/specs/2026-10-19-win-thermo-aux-design.md

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::pair::PairStrategy;
    use crate::dqn::win::win_input;
    use crate::dqn::win_distill::label_decision;
    use crate::state::Mark;
    use crate::strategy::search::{phase2_entry, rollout_final_states};

    #[test]
    fn labels_of_an_ended_position_are_exact() {
        let mut done = State::default();
        for _ in 0..4 {
            done.apply_strike();
        }
        let mut opp = State::default();
        opp.apply_mark(Mark { row: 0, number: 2 });
        let s = Sample::from_finals(done, opp, &[vec![done, opp]]);
        assert_eq!((s.win, s.win_var), (0.0, 0.0));
        assert_eq!(s.marks, [0.0, 0.0, 0.0, 0.0, 1.0 / 12.0, 0.0, 0.0, 0.0]);
        assert_eq!(s.strikes, [1.0, 0.0]);
        assert_eq!(s.diff, -21.0 / 100.0);
    }

    #[test]
    fn labels_average_the_rollouts() {
        let mut a = State::default();
        a.apply_mark(Mark { row: 0, number: 2 });
        a.apply_mark(Mark { row: 0, number: 3 });
        let mut b = State::default();
        b.apply_strike();
        // a: 3 points, b: −5; once each way round.
        let s = Sample::from_finals(State::default(), State::default(), &[vec![a, b], vec![b, a]]);
        assert_eq!((s.win, s.win_var), (0.5, 0.5));
        assert!((s.marks[0] - 1.0 / 12.0).abs() < 1e-7 && (s.marks[4] - 1.0 / 12.0).abs() < 1e-7);
        assert!((s.strikes[0] - 0.125).abs() < 1e-7 && (s.strikes[1] - 0.125).abs() < 1e-7);
        assert!(s.diff.abs() < 1e-7);
    }

    #[test]
    fn a_swapped_sample_is_the_other_players_view() {
        let mut a = State::default();
        a.apply_mark(Mark { row: 1, number: 4 });
        let b = State::default();
        let s = Sample {
            me: a,
            opp: b,
            win: 0.75,
            win_var: 0.2,
            marks: [1., 2., 3., 4., 5., 6., 7., 8.],
            strikes: [0.25, 0.5],
            diff: 0.1,
        };
        let t = s.swapped();
        assert_eq!((t.me.to_bytes(), t.opp.to_bytes()), (b.to_bytes(), a.to_bytes()));
        assert_eq!((t.win, t.win_var, t.diff), (0.25, 0.2, -0.1));
        assert_eq!(t.marks, [5., 6., 7., 8., 1., 2., 3., 4.]);
        assert_eq!(t.strikes, [0.5, 0.25]);
        assert_eq!(t.swapped(), s);
        assert_eq!(Sample::from_floats(s.me, s.opp, s.floats()), s);
    }

    #[test]
    fn full_labels_match_the_win_only_labels() {
        let bot = PairStrategy::load("pair_model");
        let opp = State::default();
        let mut marked = State::default();
        marked.apply_mark(Mark { row: 0, number: 2 });
        let mut done = State::default();
        for _ in 0..4 {
            done.apply_strike();
        }
        let live = vec![
            phase2_entry(State::default(), &[opp]),
            phase2_entry(done, &[opp]),
            phase2_entry(marked, &[opp]),
        ];
        let me_next = vec![phase2_entry(marked, &[opp]), phase2_entry(State::default(), &[opp])];
        for (entries, first_active) in [(live, 1), (me_next, 0)] {
            let d = CandDecision {
                entries,
                first_active,
                seed: 9,
            };
            let (g, full) = (label_decision(&bot, &d, 8), label_decision_full(&bot, &d, 8));
            assert_eq!(full.swapped, g.swapped);
            assert_eq!(full.pair_var, g.pair_var);
            assert_eq!(full.samples.len(), g.samples.len());
            for (s, (x, y)) in full.samples.iter().zip(&g.samples) {
                assert_eq!(win_input(&s.me, &s.opp), *x);
                assert_eq!(s.win, *y);
            }
            // Every auxiliary label swaps with the players, not only win.
            let finals = rollout_final_states(&bot, &d.entries, 9, 8, first_active, false);
            let e = &d.entries[0].0;
            let plain = Sample::from_finals(e[0], e[1], &finals[0]);
            let want = if first_active == 0 { plain.swapped() } else { plain };
            assert_eq!(full.samples[0], want);
        }
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib --release win_data`
Expected: compile errors — `cannot find type 'Sample'`, `cannot find type 'CandDecision'`, `cannot find function 'label_decision_full'`.

- [ ] **Step 3: Refactor `win_distill.rs` for reuse**

Make `sample_var` `pub(crate) fn sample_var`. Replace `diff_stats` with:

```rust
/// What the difference diagnostic reads from one decision: its labels in
/// the samples' frame, the runner-ups' paired variances, and the frame.
pub struct GroupView<'a> {
    pub labels: Vec<f32>,
    pub pair_var: &'a [f32],
    pub swapped: bool,
}

/// `preds[g][i]` = the champion's P for `groups[g].samples[i]` (the
/// samples' frame). `None` when no decision has a runner-up.
pub fn diff_stats(preds: &[Vec<f32>], groups: &[CandGroup], k: usize) -> Option<DiffReport> {
    let views: Vec<GroupView> = groups
        .iter()
        .map(|g| GroupView {
            labels: g.samples.iter().map(|s| s.1).collect(),
            pair_var: &g.pair_var,
            swapped: g.swapped,
        })
        .collect();
    diff_stats_views(preds, &views, k)
}

/// [`diff_stats`] over [`GroupView`]s.
pub fn diff_stats_views(preds: &[Vec<f32>], groups: &[GroupView], k: usize) -> Option<DiffReport> {
    let (mut sq, mut var, mut pairs, mut flips) = (0.0f64, 0.0f64, 0usize, 0usize);
    for (p, g) in preds.iter().zip(groups) {
        let y = &g.labels;
        for i in 1..y.len() {
            let e = (p[i] - p[0]) - (y[i] - y[0]);
            sq += (e * e) as f64;
            var += g.pair_var[i - 1] as f64;
            pairs += 1;
        }
        // Best label for me; ties keep the first (static) candidate.
        let mine: Vec<f32> = y.iter().map(|&v| if g.swapped { 1.0 - v } else { v }).collect();
        let best = (1..mine.len()).fold(0, |b, i| if mine[i] > mine[b] { i } else { b });
        flips += (best != 0) as usize;
    }
    (pairs > 0).then(|| DiffReport {
        decisions: groups.len(),
        diff_rmse: (sq / pairs as f64).sqrt() as f32,
        diff_noise: (var / pairs as f64 / k as f64).sqrt() as f32,
        flip_share: flips as f32 / groups.len() as f32,
    })
}
```

- [ ] **Step 4: Implement the samples in `win_data.rs`**

Above `#[cfg(test)]`:

```rust
use crate::dqn::win_distill::{sample_var, CandDecision};
use crate::state::State;
use crate::strategy::search::rollout_final_states;
use crate::strategy::sim::SimGame;
use crate::strategy::Bot;

/// One labelled position in the turn-end frame (the other player to move
/// next). Every label is a mean over the position's rollouts.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub me: State,
    pub opp: State,
    /// Mean outcome for me (win 1, tie ½, loss 0).
    pub win: f32,
    /// Sample variance of the outcomes (0 for an ended position).
    pub win_var: f32,
    /// Final row totals / 12: mine red, yellow, green, blue, then the opponent's.
    pub marks: [f32; 8],
    /// Final strikes / 4: mine, the opponent's.
    pub strikes: [f32; 2],
    /// Final (my − opponent's) points / 100.
    pub diff: f32,
}

/// Exact equality: the boards byte for byte (`State`'s own `==` is a
/// dominance test) and every label.
impl PartialEq for Sample {
    fn eq(&self, o: &Self) -> bool {
        self.me.to_bytes() == o.me.to_bytes() && self.opp.to_bytes() == o.opp.to_bytes() && self.floats() == o.floats()
    }
}

impl Sample {
    /// Labels from the rollouts' final boards (`finals`: one (me, opp) per
    /// rollout; a single element for an ended position, its exact result).
    pub fn from_finals(me: State, opp: State, finals: &[Vec<State>]) -> Sample {
        let n = finals.len() as f32;
        let outcomes: Vec<f32> = finals.iter().map(|f| SimGame::outcome(f)).collect();
        let (mut marks, mut strikes, mut diff) = ([0.0f32; 8], [0.0f32; 2], 0.0f32);
        for f in finals {
            for side in 0..2 {
                for (r, &t) in f[side].row_totals().iter().enumerate() {
                    marks[4 * side + r] += t as f32 / 12.0 / n;
                }
                strikes[side] += f[side].strikes as f32 / 4.0 / n;
            }
            diff += (f[0].count_points() - f[1].count_points()) as f32 / 100.0 / n;
        }
        Sample {
            me,
            opp,
            win: outcomes.iter().sum::<f32>() / n,
            win_var: sample_var(&outcomes),
            marks,
            strikes,
            diff,
        }
    }

    /// The same sample from the other player's side.
    pub fn swapped(&self) -> Sample {
        let mut marks = [0.0f32; 8];
        marks[..4].copy_from_slice(&self.marks[4..]);
        marks[4..].copy_from_slice(&self.marks[..4]);
        Sample {
            me: self.opp,
            opp: self.me,
            win: 1.0 - self.win,
            win_var: self.win_var,
            marks,
            strikes: [self.strikes[1], self.strikes[0]],
            diff: -self.diff,
        }
    }

    /// The labels as stored: win, win_var, marks, strikes, diff.
    pub fn floats(&self) -> [f32; 13] {
        let mut f = [0.0f32; 13];
        f[0] = self.win;
        f[1] = self.win_var;
        f[2..10].copy_from_slice(&self.marks);
        f[10..12].copy_from_slice(&self.strikes);
        f[12] = self.diff;
        f
    }

    /// The inverse of [`Sample::floats`].
    pub fn from_floats(me: State, opp: State, f: [f32; 13]) -> Sample {
        let mut marks = [0.0f32; 8];
        marks.copy_from_slice(&f[2..10]);
        Sample {
            me,
            opp,
            win: f[0],
            win_var: f[1],
            marks,
            strikes: [f[10], f[11]],
            diff: f[12],
        }
    }
}

/// One labelled decision: a sample per kept candidate (best static value
/// first) and the CRN spread of each runner-up's difference to the first,
/// as [`crate::dqn::win_distill::CandGroup`] holds them.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionData {
    pub samples: Vec<Sample>,
    /// `pair_var[i − 1]`: sample variance of outcome(i) − outcome(0).
    pub pair_var: Vec<f32>,
    /// The entries had me to move next, so the samples are stored from the
    /// opponent's side ([`Sample::swapped`]).
    pub swapped: bool,
}

/// [`crate::dqn::win_distill::label_decision`] with every label: `k`
/// full-game CRN rollouts by `bot` per kept candidate.
pub fn label_decision_full(bot: &impl Bot, d: &CandDecision, k: usize) -> DecisionData {
    let finals = rollout_final_states(bot, &d.entries, d.seed, k, d.first_active, false);
    let swapped = d.first_active == 0;
    let samples = d
        .entries
        .iter()
        .zip(&finals)
        .map(|((e, _), f)| {
            assert_eq!(e.len(), 2, "the win net is 2-player only");
            let s = Sample::from_finals(e[0], e[1], f);
            if swapped {
                s.swapped()
            } else {
                s
            }
        })
        .collect();
    // An ended entry's exact result stands for every rollout.
    let outcomes: Vec<Vec<f32>> = finals
        .iter()
        .map(|f| {
            let o: Vec<f32> = f.iter().map(|states| SimGame::outcome(states)).collect();
            if o.len() == 1 {
                vec![o[0]; k]
            } else {
                o
            }
        })
        .collect();
    let pair_var = outcomes[1..]
        .iter()
        .map(|o| {
            let diffs: Vec<f32> = o.iter().zip(&outcomes[0]).map(|(a, b)| a - b).collect();
            sample_var(&diffs)
        })
        .collect();
    DecisionData {
        samples,
        pair_var,
        swapped,
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib --release win_data`
Expected: `test result: ok. 4 passed`.

Run: `cargo test --lib --release win_distill`
Expected: all 8 existing `win_distill` tests pass (the refactor changed no result).

- [ ] **Step 6: Format, full suite, commit**

```bash
rustfmt --edition 2021 src/dqn/win_data.rs src/dqn/win_distill.rs
cargo test --lib --release 2>&1 | tail -3
```
Expected: `test result: ok. 180 passed`. Then:

```bash
git add src/dqn/mod.rs src/dqn/win_data.rs src/dqn/win_distill.rs
git commit -m "feat(win): samples with auxiliary labels from the rollouts' final boards

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Dataset generation, split, file format and `win-data`

**Files:**
- Modify: `src/dqn/win_data.rs` (generation, split, I/O, `DataCfg`; tests)
- Modify: `src/dqn/win_train.rs` (`TurnEnds` and its fields become `pub(crate)`)
- Modify: `src/dqn/win_pi.rs` (`splitmix64` becomes `pub`)
- Modify: `src/main.rs` (`win-data` subcommand)

**Interfaces:**
- Consumes (Task 3): `Sample`, `DecisionData`, `label_decision_full`. Existing: `TurnEnds { bot, recorded, capture }`, `phase2_entry`, `rollout_final_states`, `GATE_MARGIN`, `win_pi::splitmix64`.
- Produces (in `crate::dqn::win_data`):
  - `#[derive(Clone, Debug, Default, PartialEq)] pub struct Split { pub turns: Vec<Sample>, pub decisions: Vec<DecisionData> }` with `Split::cand_samples(&self) -> usize`.
  - `#[derive(Config, Debug)] pub struct DataMeta { pub player: String, pub seed: u64, pub games: usize, pub rollouts: usize, pub cand_rollouts: usize, pub holdout: f64, pub train_turns: usize, pub train_decisions: usize, pub heldout_turns: usize, pub heldout_decisions: usize }`
  - `#[derive(Debug)] pub struct Dataset { pub meta: DataMeta, pub train: Split, pub heldout: Split }`
  - `pub struct DataCfg { pub player: String, pub out: String, pub positions: usize, pub rollouts: usize, pub cand_rollouts: usize, pub holdout: f64, pub seed: u64 }` with `validate(&self) -> Result<(), String>`.
  - `pub fn held_out(seed: u64, game: u64, share: f64) -> bool`
  - `pub fn generate<P: Bot + Clone + 'static>(cfg: &DataCfg, player: &(dyn Fn() -> P + Sync)) -> Dataset`
  - `pub fn save(dir: &str, ds: &Dataset) -> std::io::Result<()>`, `pub fn load(dir: &str) -> std::io::Result<Dataset>`

- [ ] **Step 1: Write the failing tests**

Add to `win_data.rs` `mod tests` (extend the `use` lines with `use crate::dqn::win_train::{rollout_round_by, CandCfg};` and `use crate::strategy::search::GATE_MARGIN;`):

```rust
    fn tmp(name: &str) -> String {
        let d = std::env::temp_dir().join(format!("qwixxer-win-data-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d.to_string_lossy().into_owned()
    }

    fn cfg(positions: usize, holdout: f64, seed: u64) -> DataCfg {
        DataCfg {
            player: "pair_model".into(),
            out: String::new(),
            positions,
            rollouts: 2,
            cand_rollouts: 4,
            holdout,
            seed,
        }
    }

    #[test]
    fn turn_ends_and_candidates_match_the_win_pi_round() {
        let pair = || PairStrategy::load("pair_model");
        let ds = generate(&cfg(40, 0.0, 11), &pair);
        let round = rollout_round_by(
            40,
            11,
            &pair,
            "pair_model",
            2,
            1.0,
            Some(CandCfg {
                k: 4,
                margin: GATE_MARGIN,
            }),
        );
        assert!(ds.heldout.turns.is_empty() && ds.heldout.decisions.is_empty());
        assert!(ds.train.turns.len() >= 40);
        for (s, (x, y)) in ds.train.turns.iter().zip(&round.positions) {
            assert_eq!(win_input(&s.me, &s.opp), *x);
            assert_eq!(s.win, *y);
        }
        assert_eq!(ds.train.decisions.len(), round.groups.len());
        for (d, g) in ds.train.decisions.iter().zip(&round.groups) {
            assert_eq!((d.swapped, &d.pair_var), (g.swapped, &g.pair_var));
            for (s, (x, y)) in d.samples.iter().zip(&g.samples) {
                assert_eq!(win_input(&s.me, &s.opp), *x);
                assert_eq!(s.win, *y);
            }
        }
        assert_eq!((ds.meta.train_turns, ds.meta.heldout_turns), (ds.train.turns.len(), 0));
    }

    #[test]
    fn generation_is_deterministic_and_splits_whole_games() {
        let pair = || PairStrategy::load("pair_model");
        let (a, b) = (generate(&cfg(300, 0.5, 3), &pair), generate(&cfg(300, 0.5, 3), &pair));
        assert_eq!((&a.train, &a.heldout), (&b.train, &b.heldout));
        assert!(!a.train.turns.is_empty() && !a.heldout.turns.is_empty());
        assert!(a.train.turns.len() + a.heldout.turns.len() >= 300);
        assert_eq!(a.meta.heldout_decisions, a.heldout.decisions.len());
    }

    #[test]
    fn held_out_share_is_close_and_deterministic() {
        let share = (0..10_000u64).filter(|&g| held_out(7, g, 0.1)).count();
        assert!((900..=1100).contains(&share), "{share}");
        assert!((0..1000u64).all(|g| !held_out(7, g, 0.0)));
        assert_eq!(held_out(7, 42, 0.5), held_out(7, 42, 0.5));
    }

    #[test]
    fn a_dataset_round_trips_through_its_files() {
        let dir = tmp("round-trip");
        let ds = generate(&cfg(40, 0.5, 5), &|| PairStrategy::load("pair_model"));
        save(&dir, &ds).unwrap();
        let back = load(&dir).unwrap();
        assert_eq!((&back.train, &back.heldout), (&ds.train, &ds.heldout));
        assert_eq!(
            (back.meta.games, back.meta.rollouts, back.meta.cand_rollouts, back.meta.player.as_str()),
            (ds.meta.games, 2, 4, "pair_model")
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_damaged_file_is_an_error() {
        let dir = tmp("damaged");
        let ds = generate(&cfg(40, 0.5, 5), &|| PairStrategy::load("pair_model"));
        save(&dir, &ds).unwrap();
        let path = format!("{dir}/data.bin");
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 7]).unwrap();
        assert!(load(&dir).is_err(), "truncated");
        let mut foreign = bytes.clone();
        foreign[0] = b'X';
        std::fs::write(&path, &foreign).unwrap();
        assert!(load(&dir).unwrap_err().to_string().contains("not a win-data file"));
        let mut newer = bytes.clone();
        newer[4] = 2;
        std::fs::write(&path, &newer).unwrap();
        assert!(load(&dir).unwrap_err().to_string().contains("unsupported win-data version 2"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn validate_refuses_bad_settings() {
        let base = tmp("validate");
        let ok = DataCfg {
            out: format!("{base}/out"),
            ..cfg(10, 0.1, 1)
        };
        assert_eq!(ok.validate(), Ok(()));
        let err = |c: DataCfg| c.validate().unwrap_err();
        let with = |f: &dyn Fn(&mut DataCfg)| {
            let mut c = DataCfg {
                out: ok.out.clone(),
                ..cfg(10, 0.1, 1)
            };
            f(&mut c);
            c
        };
        assert!(err(with(&|c| c.positions = 0)).contains("--positions"));
        assert!(err(with(&|c| c.rollouts = 1)).contains("--rollouts"));
        assert!(err(with(&|c| c.cand_rollouts = 1)).contains("--cand-rollouts"));
        assert!(err(with(&|c| c.holdout = 1.0)).contains("--holdout"));
        assert!(err(with(&|c| c.player = format!("{base}/nope"))).contains("--player"));
        std::fs::create_dir_all(&ok.out).unwrap();
        assert!(err(with(&|_| {})).contains("already exists"));
        std::fs::remove_dir_all(&base).unwrap();
    }
```

(`player: "pair_model"` passes `validate` because `pair_model/` has `config.json` and `model.mpk`; `generate` takes the bot from its closure, the name only goes into the metadata.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib --release win_data`
Expected: compile errors — `cannot find struct 'DataCfg'`, `cannot find function 'generate'`, `cannot find function 'held_out'`.

- [ ] **Step 3: Open up the two helpers**

`src/dqn/win_train.rs`: change `struct TurnEnds<B: Bot> {` to `pub(crate) struct TurnEnds<B: Bot> {` and its three fields to `pub(crate) bot: B,`, `pub(crate) recorded: …`, `pub(crate) capture: …`.

`src/dqn/win_pi.rs`: change `fn splitmix64` to `pub fn splitmix64`.

- [ ] **Step 4: Implement generation and I/O in `win_data.rs`**

Extend the imports at the top:

```rust
use crate::dqn::win_distill::{sample_var, CandDecision};
use crate::dqn::win_pi::splitmix64;
use crate::dqn::win_train::TurnEnds;
use crate::game::{Game, Player};
use crate::state::State;
use crate::strategy::search::{phase2_entry, rollout_final_states, GATE_MARGIN};
use crate::strategy::sim::SimGame;
use crate::strategy::Bot;
use burn::prelude::*;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::rc::Rc;
```

Add after `label_decision_full`:

```rust
/// One side of the split: turn ends and labelled decisions, in game order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Split {
    pub turns: Vec<Sample>,
    pub decisions: Vec<DecisionData>,
}

impl Split {
    /// Candidate samples over all decisions.
    pub fn cand_samples(&self) -> usize {
        self.decisions.iter().map(|d| d.samples.len()).sum()
    }
}

/// `meta.json`: how the dataset was made, and its sizes.
#[derive(Config, Debug)]
pub struct DataMeta {
    pub player: String,
    pub seed: u64,
    pub games: usize,
    pub rollouts: usize,
    pub cand_rollouts: usize,
    pub holdout: f64,
    pub train_turns: usize,
    pub train_decisions: usize,
    pub heldout_turns: usize,
    pub heldout_decisions: usize,
}

#[derive(Debug)]
pub struct Dataset {
    pub meta: DataMeta,
    pub train: Split,
    pub heldout: Split,
}

/// `win-data` settings.
pub struct DataCfg {
    /// Win model dir that plays the games and the rollouts (metadata only;
    /// [`generate`] takes the bot itself).
    pub player: String,
    pub out: String,
    /// Turn ends to reach; whole games are kept, so the count is reached or just passed.
    pub positions: usize,
    /// Rollouts per turn end.
    pub rollouts: usize,
    /// Rollouts per candidate.
    pub cand_rollouts: usize,
    /// Share of games held out.
    pub holdout: f64,
    pub seed: u64,
}

impl DataCfg {
    /// Every check that can fail, before any work.
    pub fn validate(&self) -> Result<(), String> {
        if self.positions == 0 {
            return Err("--positions must be at least 1".into());
        }
        if self.rollouts < 2 {
            return Err("--rollouts must be at least 2 (the noise floor needs a variance)".into());
        }
        if self.cand_rollouts < 2 {
            return Err("--cand-rollouts must be at least 2".into());
        }
        if !(0.0..1.0).contains(&self.holdout) {
            return Err("--holdout must be in [0, 1)".into());
        }
        for f in ["config.json", "model.mpk"] {
            if !Path::new(&format!("{}/{f}", self.player)).exists() {
                return Err(format!("--player {}: {f} not found", self.player));
            }
        }
        if Path::new(&self.out).exists() {
            return Err(format!("{} already exists: move it aside or pick another --out", self.out));
        }
        Ok(())
    }
}

/// Whether game `game` of a dataset seeded `seed` is held out; about
/// `share` of the games are.
pub fn held_out(seed: u64, game: u64, share: f64) -> bool {
    let h = splitmix64(seed ^ splitmix64(game ^ 0x4E1D_0001));
    ((h >> 11) as f64 / (1u64 << 53) as f64) < share
}

/// One self-play game's labelled turn ends and candidate decisions. Game
/// `g` is seeded `seed + 1000·g` and its turn ends are rolled out from seed
/// `game seed ^ 0x5011_0075`, as in `rollout_round_by`, so the win labels
/// are the ones `win-pi --labels cand` would give.
fn game_samples<P: Bot + Clone + 'static>(player: &P, cfg: &DataCfg, g: u64) -> (Vec<Sample>, Vec<DecisionData>) {
    let game_seed = cfg.seed.wrapping_add(g.wrapping_mul(1000));
    let bufs: [Rc<RefCell<Vec<(State, State)>>>; 2] = Default::default();
    let caps: [Rc<RefCell<Vec<CandDecision>>>; 2] = Default::default();
    let players: Vec<Player> = (0..2)
        .map(|i| {
            let rec = TurnEnds {
                bot: player.clone(),
                recorded: bufs[i].clone(),
                capture: Some((GATE_MARGIN, caps[i].clone())),
            };
            Player::new(
                Box::new(rec),
                Box::new(SmallRng::seed_from_u64(game_seed.wrapping_add(i as u64))),
            )
        })
        .collect();
    Game::new(players).play();
    // The lock-propagated boards, as the turn-end inputs use them.
    let entries: Vec<(Vec<State>, bool)> = bufs
        .iter()
        .flat_map(|b| b.take())
        .map(|(me, opp)| phase2_entry(me, &[opp]))
        .collect();
    let turns = if entries.is_empty() {
        Vec::new()
    } else {
        rollout_final_states(player, &entries, game_seed ^ 0x5011_0075, cfg.rollouts, 1, false)
            .iter()
            .zip(&entries)
            .map(|(f, (b, _))| Sample::from_finals(b[0], b[1], f))
            .collect()
    };
    let decisions = caps
        .iter()
        .flat_map(|c| c.take())
        .map(|d| label_decision_full(player, &d, cfg.cand_rollouts))
        .collect();
    (turns, decisions)
}

/// The dataset of `player`'s self-play (one bot per worker thread): whole
/// games in index order until `cfg.positions` turn ends, each game's turn
/// ends and decisions on one side of the split.
pub fn generate<P: Bot + Clone + 'static>(cfg: &DataCfg, player: &(dyn Fn() -> P + Sync)) -> Dataset {
    let (mut train, mut heldout) = (Split::default(), Split::default());
    let (mut games, mut kept, mut turns) = (0u64, 0usize, 0usize);
    while turns < cfg.positions {
        let per_game = if kept == 0 { 25.0 } else { turns as f64 / kept as f64 };
        let batch = (((cfg.positions - turns) as f64 / per_game).ceil() as u64).max(64);
        let chunk: Vec<(Vec<Sample>, Vec<DecisionData>)> = (games..games + batch)
            .into_par_iter()
            .map_init(player, |player, g| game_samples(&*player, cfg, g))
            .collect();
        for (g, (t, d)) in (games..).zip(chunk) {
            if turns >= cfg.positions {
                break;
            }
            turns += t.len();
            kept += 1;
            let side = if held_out(cfg.seed, g, cfg.holdout) {
                &mut heldout
            } else {
                &mut train
            };
            side.turns.extend(t);
            side.decisions.extend(d);
        }
        games += batch;
    }
    Dataset {
        meta: DataMeta {
            player: cfg.player.clone(),
            seed: cfg.seed,
            games: kept,
            rollouts: cfg.rollouts,
            cand_rollouts: cfg.cand_rollouts,
            holdout: cfg.holdout,
            train_turns: train.turns.len(),
            train_decisions: train.decisions.len(),
            heldout_turns: heldout.turns.len(),
            heldout_decisions: heldout.decisions.len(),
        },
        train,
        heldout,
    }
}

const MAGIC: &[u8; 4] = b"QWXD";
const VERSION: u32 = 1;

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn put_sample(w: &mut impl Write, s: &Sample) -> io::Result<()> {
    w.write_all(&s.me.to_bytes())?;
    w.write_all(&s.opp.to_bytes())?;
    for v in s.floats() {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn get_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn get_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn get_sample(r: &mut impl Read) -> io::Result<Sample> {
    let (mut me, mut opp) = ([0u8; 13], [0u8; 13]);
    r.read_exact(&mut me)?;
    r.read_exact(&mut opp)?;
    let mut f = [0.0f32; 13];
    for v in f.iter_mut() {
        *v = get_f32(r)?;
    }
    Ok(Sample::from_floats(State::from_bytes(me), State::from_bytes(opp), f))
}

fn put_split(w: &mut impl Write, s: &Split) -> io::Result<()> {
    w.write_all(&(s.turns.len() as u64).to_le_bytes())?;
    for t in &s.turns {
        put_sample(w, t)?;
    }
    w.write_all(&(s.decisions.len() as u64).to_le_bytes())?;
    for d in &s.decisions {
        w.write_all(&[d.swapped as u8, d.samples.len() as u8])?;
        for t in &d.samples {
            put_sample(w, t)?;
        }
        for v in &d.pair_var {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Counts come from the file, so nothing is preallocated from them: a
/// damaged count runs into the end of the file instead.
fn get_split(r: &mut impl Read) -> io::Result<Split> {
    let mut s = Split::default();
    for _ in 0..get_u64(r)? {
        s.turns.push(get_sample(r)?);
    }
    for _ in 0..get_u64(r)? {
        let mut h = [0u8; 2];
        r.read_exact(&mut h)?;
        let samples = (0..h[1]).map(|_| get_sample(r)).collect::<io::Result<Vec<_>>>()?;
        let pair_var = (1..h[1]).map(|_| get_f32(r)).collect::<io::Result<Vec<_>>>()?;
        s.decisions.push(DecisionData {
            samples,
            pair_var,
            swapped: h[0] != 0,
        });
    }
    Ok(s)
}

/// Write `dir/meta.json` and `dir/data.bin` (magic, version, then the
/// training and held-out splits; little-endian).
pub fn save(dir: &str, ds: &Dataset) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    ds.meta.save(format!("{dir}/meta.json"))?;
    let mut w = BufWriter::new(File::create(format!("{dir}/data.bin"))?);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    put_split(&mut w, &ds.train)?;
    put_split(&mut w, &ds.heldout)?;
    w.flush()
}

/// Read a dataset written by [`save`].
pub fn load(dir: &str) -> io::Result<Dataset> {
    let meta =
        DataMeta::load(format!("{dir}/meta.json")).map_err(|e| invalid(format!("{dir}/meta.json: {e:?}")))?;
    let mut r = BufReader::new(File::open(format!("{dir}/data.bin"))?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid(format!("{dir}/data.bin is not a win-data file")));
    }
    let mut v = [0u8; 4];
    r.read_exact(&mut v)?;
    let version = u32::from_le_bytes(v);
    if version != VERSION {
        return Err(invalid(format!("{dir}/data.bin: unsupported win-data version {version}")));
    }
    let train = get_split(&mut r)?;
    let heldout = get_split(&mut r)?;
    Ok(Dataset { meta, train, heldout })
}
```

If `ds.meta.save(...)` does not return `io::Result<()>` in burn 0.20.1, map its error with `.map_err(|e| invalid(format!("{e:?}")))?` and note a ruling.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib --release win_data`
Expected: `test result: ok. 10 passed`. If `generation_is_deterministic_and_splits_whole_games` fails only on `!a.heldout.turns.is_empty()`, the seed put every game on one side: change seed 3 to 4 in both calls, ledger a ruling, rerun.

- [ ] **Step 6: Add the `win-data` subcommand to `src/main.rs`**

In `enum Commands`, after `WinPi { … },`:

```rust
    /// Win-net dataset: a player's turn ends and gated candidates with win and auxiliary rollout labels, held out by game
    #[cfg(feature = "dqn")]
    WinData {
        /// Win model dir that plays the games and the rollouts
        #[arg(long)]
        player: String,
        /// Output dir (must not exist): meta.json, data.bin
        #[arg(long)]
        out: String,
        /// Turn ends to reach (whole games are kept)
        #[arg(long, default_value = "400000")]
        positions: usize,
        /// Rollouts per turn end
        #[arg(long, default_value = "64")]
        rollouts: usize,
        /// Rollouts per candidate
        #[arg(long, default_value = "64")]
        cand_rollouts: usize,
        /// Share of games held out
        #[arg(long, default_value = "0.1")]
        holdout: f64,
        #[arg(long, default_value = "42")]
        seed: u64,
    },
```

In the `match`, after the `WinPi` arm:

```rust
        #[cfg(feature = "dqn")]
        Some(Commands::WinData {
            player,
            out,
            positions,
            rollouts,
            cand_rollouts,
            holdout,
            seed,
        }) => {
            let cfg = dqn::win_data::DataCfg {
                player,
                out,
                positions,
                rollouts,
                cand_rollouts,
                holdout,
                seed,
            };
            if let Err(e) = cfg.validate() {
                eprintln!("win-data: {e}");
                std::process::exit(2);
            }
            let start = std::time::Instant::now();
            let win = dqn::win::WinStrategy::load(&cfg.player);
            let ds = dqn::win_data::generate(&cfg, &|| win.clone());
            dqn::win_data::save(&cfg.out, &ds).expect("Failed to write the dataset");
            let m = &ds.meta;
            println!(
                "win-data: {} games; train {} turn ends + {} decisions ({} samples); held out {} + {} ({} samples); {}s",
                m.games,
                m.train_turns,
                m.train_decisions,
                ds.train.cand_samples(),
                m.heldout_turns,
                m.heldout_decisions,
                ds.heldout.cand_samples(),
                start.elapsed().as_secs()
            );
        }
```

Run: `cargo run --release -- win-data --player pair_model --out /nonexistent/x --rollouts 1`
Expected: `win-data: --rollouts must be at least 2 (the noise floor needs a variance)` and exit code 2 (`echo $?` → `2`).

- [ ] **Step 7: Format, full suite, commit**

```bash
rustfmt --edition 2021 src/dqn/win_data.rs src/dqn/win_train.rs src/dqn/win_pi.rs src/main.rs
cargo test --lib --release 2>&1 | tail -3
```
Expected: `test result: ok. 186 passed`. Then:

```bash
git add src/dqn/win_data.rs src/dqn/win_train.rs src/dqn/win_pi.rs src/main.rs
git commit -m "feat(win): win-data: stored datasets with auxiliary labels, held out by game

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Auxiliary heads, metrics, schedule and weight averaging

**Files:**
- Create: `src/dqn/win_arm.rs`
- Modify: `src/dqn/win.rs` (`AuxHeads`, after `widen_to_thermo`)
- Modify: `src/dqn/mod.rs` (register the module, by hand)

**Interfaces:**
- Consumes (Tasks 1–4): `WinModel::{trunk, is_thermo}`, `flat_tensor`, `thermo_input`, `swap_thermo`, `win_input`, `swap_input`, `WIN_INPUTS`, `WIN_INPUTS_T`, `sigmoid`; `Sample`, `DecisionData`, `Split`; `GroupView`, `diff_stats_views`, `DiffReport`, `sample_var`; existing `bce_logits` (`win_train`).
- Produces:
  - in `crate::dqn::win` (compiled without `dqn` too): `#[derive(Module, Debug)] pub struct AuxHeads<B: Backend> { pub marks: Linear<B>, pub strikes: Linear<B>, pub diff: Linear<B> }`, `AuxHeads::new(width: usize, device: &B::Device) -> Self`
  - in `crate::dqn::win_arm`:
  - `#[derive(Module, Debug)] pub struct ArmModel<B: Backend> { pub net: WinModel<B>, pub aux: Option<AuxHeads<B>> }`
  - `pub struct ArmOut<B: Backend> { pub z: Tensor<B, 2>, pub marks: Option<Tensor<B, 2>>, pub strikes: Option<Tensor<B, 2>>, pub diff: Option<Tensor<B, 2>> }` (`[n,1]`, `[n,8]`, `[n,2]`, `[n,1]`); `ArmModel::forward(&self, x, swapped) -> ArmOut<B>`
  - `pub struct Targets<B: Backend> { pub win, pub marks, pub strikes, pub diff }` (same shapes)
  - `pub fn encode(boards: &[(&State, &State)], thermo: bool) -> (Vec<f32>, Vec<f32>, usize)` (inputs, swapped inputs, width)
  - `pub fn tensors<B: Backend>(samples: &[&Sample], thermo: bool, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>, Targets<B>)`
  - `pub fn arm_loss<B: Backend>(out: ArmOut<B>, t: &Targets<B>, w: &[f32]) -> (Tensor<B, 1>, Tensor<B, 1>)` (total, win BCE)
  - `pub fn entropy(w: f32) -> f64`, `pub fn bce(w: f32, p: f32) -> f64`, `pub fn noise_floor(w: f32, v: f32, k: usize) -> f64`
  - `pub fn cosine_lr(lr: f64, step: usize, total: usize) -> f64`
  - `pub fn ema_update(e: &mut [f32], v: &[f32], decay: f32)`; `pub struct Ema` with `new<B: Backend>(m: &ArmModel<B>, decay: f32)`, `update<B: Backend>(&mut self, m: &ArmModel<B>)`, `apply(&self, m: ArmModel<MyBackend>) -> ArmModel<MyBackend>`
  - `#[derive(Default)] pub struct Best { pub metric: Option<f64> }`, `Best::offer(&mut self, m: f64) -> bool`
  - `#[derive(Default)] pub struct Preds { pub p: Vec<f32>, pub marks: Vec<[f32; 8]>, pub strikes: Vec<[f32; 2]>, pub diff: Vec<f32> }`; `pub fn predict(model: &ArmModel<MyBackend>, samples: &[&Sample]) -> Preds`
  - `#[derive(Clone, Debug)] pub struct EvalReport { pub samples: usize, pub excess: f64, pub floor: f64, pub marks: Option<f64>, pub strikes: Option<f64>, pub diff: Option<f64>, pub diffs: Option<DiffReport> }`; `pub fn evaluate(model: &ArmModel<MyBackend>, split: &Split, k_turn: usize, k_cand: usize) -> EvalReport`
  - `pub fn subset(split: &Split, max: usize) -> Split`

- [ ] **Step 1: Register the module and write the failing tests**

In `src/dqn/mod.rs`, after the `pub mod win_data;` line:

```rust
#[cfg(feature = "dqn")]
pub mod win_arm;
```

Create `src/dqn/win_arm.rs`:

```rust
//! One arm of the thermometer/auxiliary experiment: the auxiliary heads,
//! the held-out metrics, the learning-rate schedule, weight averaging and
//! the fit (`win-arm`).
//!
//! Design: docs/superpowers/specs/2026-10-19-win-thermo-aux-design.md

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::win::WinModelConfig;
    use crate::dqn::win_distill::sample_var;
    use crate::state::Mark;
    use rand::{Rng, SeedableRng};

    static SEEDING: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn random_arm(seed: u64, thermo: bool, aux: bool) -> ArmModel<MyBackend> {
        let _guard = SEEDING.lock().unwrap();
        let device = Default::default();
        <MyBackend as Backend>::seed(&device, seed);
        let net = WinModelConfig::new()
            .with_width(8)
            .with_thermo(thermo.then_some(true))
            .init(&device);
        ArmModel {
            net,
            aux: aux.then(|| AuxHeads::new(8, &device)),
        }
    }

    fn boards() -> (State, State) {
        let mut a = State::default();
        for m in [Mark { row: 0, number: 2 }, Mark { row: 0, number: 5 }, Mark { row: 2, number: 12 }] {
            a.apply_mark(m);
        }
        a.apply_strike();
        let mut b = State::default();
        for m in [Mark { row: 1, number: 4 }, Mark { row: 3, number: 9 }, Mark { row: 3, number: 7 }] {
            b.apply_mark(m);
        }
        (a, b)
    }

    fn values(t: Option<Tensor<MyBackend, 2>>) -> Vec<f32> {
        t.unwrap().into_data().to_vec::<f32>().unwrap()
    }

    /// Permute the 4 row tokens of every flat input row.
    fn relabel(x: &[f32], d: usize, token: usize, order: [usize; 4]) -> Vec<f32> {
        let mut y = x.to_vec();
        for row in 0..x.len() / d {
            for (slot, &src) in order.iter().enumerate() {
                let (to, from) = (row * d + token * slot, row * d + token * src);
                y[to..to + token].copy_from_slice(&x[from..from + token]);
            }
        }
        y
    }

    #[test]
    fn aux_predictions_follow_the_players_and_the_rows() {
        let device = Default::default();
        let (a, b) = boards();
        for thermo in [false, true] {
            let m = random_arm(31, thermo, true);
            let run = |x: &[f32], s: &[f32], d: usize| m.forward(flat_tensor(x, d, &device), flat_tensor(s, d, &device));
            let (x, s, d) = encode(&[(&a, &b)], thermo);
            let (xr, sr, _) = encode(&[(&b, &a)], thermo);
            let (o, r) = (run(&x, &s, d), run(&xr, &sr, d));
            let (zo, zr) = (o.z.into_data().to_vec::<f32>().unwrap(), r.z.into_data().to_vec::<f32>().unwrap());
            assert!((zo[0] + zr[0]).abs() < 1e-5);
            let (mo, mr) = (values(o.marks), values(r.marks));
            for i in 0..4 {
                assert!((mo[i] - mr[4 + i]).abs() < 1e-5 && (mo[4 + i] - mr[i]).abs() < 1e-5, "thermo {thermo} row {i}");
            }
            let (so, sr) = (values(o.strikes), values(r.strikes));
            assert!((so[0] - sr[1]).abs() < 1e-5 && (so[1] - sr[0]).abs() < 1e-5);
            let (dd, dr) = (values(o.diff), values(r.diff));
            assert!((dd[0] + dr[0]).abs() < 1e-5);
            // A colour relabelling permutes the row predictions with the rows.
            let token = if thermo { T_TOKEN } else { TOKEN };
            let order = [1usize, 0, 3, 2];
            let base = values(run(&x, &s, d).marks);
            let perm = values(run(&relabel(&x, d, token, order), &relabel(&s, d, token, order), d).marks);
            for side in 0..2 {
                for (slot, &src) in order.iter().enumerate() {
                    assert!((perm[4 * side + slot] - base[4 * side + src]).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn the_loss_adds_each_head_with_its_weight() {
        let device = Default::default();
        let t2 = |v: Vec<f32>, w: usize| Tensor::<MyBackend, 1>::from_floats(v.as_slice(), &device).reshape([1, w]);
        let out = || ArmOut {
            z: t2(vec![0.0], 1),
            marks: Some(t2(vec![0.0; 8], 8)),
            strikes: Some(t2(vec![0.0; 2], 2)),
            diff: Some(t2(vec![0.0], 1)),
        };
        let t = Targets {
            win: t2(vec![1.0], 1),
            marks: t2(vec![0.5; 8], 8),
            strikes: t2(vec![1.0; 2], 2),
            diff: t2(vec![0.2], 1),
        };
        let scalar = |x: Tensor<MyBackend, 1>| x.into_scalar().elem::<f32>();
        let (total, win) = arm_loss(out(), &t, &[1.0, 0.5, 0.5]);
        let ln2 = std::f32::consts::LN_2;
        assert!((scalar(win) - ln2).abs() < 1e-6);
        assert!((scalar(total) - (ln2 + 0.25 + 0.5 + 0.02)).abs() < 1e-6);
        let plain = ArmOut {
            marks: None,
            strikes: None,
            diff: None,
            ..out()
        };
        assert!((scalar(arm_loss(plain, &t, &[1.0, 0.5, 0.5]).0) - ln2).abs() < 1e-6);
    }

    #[test]
    fn metrics_are_finite_at_certain_labels() {
        assert_eq!((entropy(0.0), entropy(1.0)), (0.0, 0.0));
        assert!((entropy(0.5) - std::f64::consts::LN_2).abs() < 1e-12);
        assert!((bce(0.5, 0.5) - std::f64::consts::LN_2).abs() < 1e-7);
        for (w, p) in [(1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0)] {
            assert!(bce(w, p).is_finite(), "bce({w}, {p})");
        }
        assert_eq!((noise_floor(0.0, 0.0, 64), noise_floor(1.0, 0.0, 64)), (0.0, 0.0));
        assert!((noise_floor(0.5, 0.25, 64) - 1.0 / 128.0).abs() < 1e-12);
    }

    #[test]
    fn the_noise_floor_is_what_a_perfect_predictor_scores() {
        let (p, k, n) = (0.3f32, 64usize, 20_000);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(5);
        let (mut excess, mut floor) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let o: Vec<f32> = (0..k).map(|_| (rng.gen::<f32>() < p) as u8 as f32).collect();
            let w = o.iter().sum::<f32>() / k as f32;
            excess += bce(w, p) - entropy(w);
            floor += noise_floor(w, sample_var(&o), k);
        }
        let ratio = excess / floor;
        assert!((ratio - 1.0).abs() < 0.1, "excess {} vs floor {}", excess / n as f64, floor / n as f64);
    }

    #[test]
    fn cosine_schedule_warms_up_then_decays_to_zero() {
        let lr = 1e-3;
        assert!((cosine_lr(lr, 0, 1000) - lr * 0.1).abs() < 1e-15);
        assert!((cosine_lr(lr, 9, 1000) - lr).abs() < 1e-15);
        assert!((cosine_lr(lr, 10, 1000) - lr).abs() < 1e-15);
        assert!((cosine_lr(lr, 505, 1000) - lr * 0.5).abs() < 1e-12);
        assert!(cosine_lr(lr, 999, 1000) < lr * 1e-4);
        assert!(cosine_lr(lr, 0, 1) > 0.0, "a one-step fit still steps");
    }

    #[test]
    fn averaging_and_best_tracking() {
        let mut e = [0.0f32, 2.0];
        ema_update(&mut e, &[1.0, 2.0], 0.5);
        ema_update(&mut e, &[1.0, 2.0], 0.5);
        assert_eq!(e, [0.75, 2.0]);
        let mut best = Best::default();
        let offered: Vec<bool> = [3.0, 1.0, 2.0, 1.0].iter().map(|&m| best.offer(m)).collect();
        assert_eq!(offered, [true, true, false, false]);
        assert_eq!(best.metric, Some(1.0));
    }

    #[test]
    fn averaged_weights_with_no_decay_are_the_live_weights() {
        let m = random_arm(32, true, true);
        let mut ema = Ema::new(&m, 0.0);
        ema.update(&m);
        let back = ema.apply(m.clone());
        let (a, b) = boards();
        let (x, s, d) = encode(&[(&a, &b)], true);
        let device = Default::default();
        let z = |m: &ArmModel<MyBackend>| {
            m.forward(flat_tensor(&x, d, &device), flat_tensor(&s, d, &device))
                .z
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        };
        assert_eq!(z(&back), z(&m));
    }

    #[test]
    fn labels_equal_to_the_predictions_score_no_error() {
        let m = random_arm(33, false, true);
        let (a, b) = boards();
        let raw: Vec<Sample> = [(a, b), (b, a), (a, a), (b, b)]
            .iter()
            .map(|&(me, opp)| Sample {
                me,
                opp,
                win: 0.0,
                win_var: 0.0,
                marks: [0.0; 8],
                strikes: [0.0; 2],
                diff: 0.0,
            })
            .collect();
        let refs: Vec<&Sample> = raw.iter().collect();
        let pr = predict(&m, &refs);
        let labelled: Vec<Sample> = raw
            .iter()
            .enumerate()
            .map(|(i, s)| Sample {
                win: pr.p[i],
                marks: pr.marks[i],
                strikes: pr.strikes[i],
                diff: pr.diff[i],
                ..*s
            })
            .collect();
        let split = Split {
            turns: labelled[..2].to_vec(),
            decisions: vec![DecisionData {
                samples: labelled[2..].to_vec(),
                pair_var: vec![0.0],
                swapped: false,
            }],
        };
        let r = evaluate(&m, &split, 64, 64);
        assert_eq!(r.samples, 4);
        assert!(r.excess.abs() < 1e-6 && r.floor == 0.0, "{r:?}");
        assert!(r.marks.unwrap() < 1e-12 && r.strikes.unwrap() < 1e-12 && r.diff.unwrap() < 1e-12, "{r:?}");
        assert!(r.diffs.unwrap().diff_rmse < 1e-5);
        let plain = random_arm(34, false, false);
        let r = evaluate(&plain, &split, 64, 64);
        assert!(r.marks.is_none() && r.strikes.is_none() && r.diff.is_none());
        let sub = subset(&split, 1);
        assert_eq!((sub.turns.len(), sub.decisions.len()), (1, 0));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib --release win_arm`
Expected: compile errors — `cannot find type 'ArmModel'`, `cannot find function 'encode'`, `cannot find function 'cosine_lr'`.

- [ ] **Step 3: Implement**

In `src/dqn/win.rs`, after `widen_to_thermo`:

```rust
/// Heads on the trunk that play never uses: each row's final marks (one map
/// shared by the rows), final strikes, and the final score difference.
/// Trained by `win-arm --aux`, saved to `aux.mpk` beside the net.
#[derive(Module, Debug)]
pub struct AuxHeads<B: Backend> {
    /// [e_r | h] (2W) → that row's final total / 12.
    pub marks: Linear<B>,
    /// h (W) → final strikes / 4.
    pub strikes: Linear<B>,
    /// h (W) → a; the difference / 100 is a(x) − a(swap x).
    pub diff: Linear<B>,
}

impl<B: Backend> AuxHeads<B> {
    pub fn new(width: usize, device: &B::Device) -> Self {
        let lin = |d_in: usize| LinearConfig::new(d_in, 1).with_bias(true).init(device);
        AuxHeads {
            marks: lin(2 * width),
            strikes: lin(width),
            diff: lin(width),
        }
    }
}
```

Above `#[cfg(test)]` in `win_arm.rs`:

```rust
use crate::dqn::win::{
    flat_tensor, sigmoid, swap_input, swap_thermo, thermo_input, win_input, AuxHeads, WinModel, T_TOKEN, TOKEN,
    WIN_INPUTS, WIN_INPUTS_T,
};
use crate::dqn::win_data::{DecisionData, Sample, Split};
use crate::dqn::win_distill::{diff_stats_views, DiffReport, GroupView};
use crate::dqn::win_train::bce_logits;
use crate::dqn::MyBackend;
use crate::state::State;
use burn::{
    module::Param,
    nn::Linear,
    prelude::*,
    tensor::{ElementConversion, TensorData},
};

/// The net being fitted, with its auxiliary heads when the arm has them.
/// The net goes to `model.mpk`, the heads to `aux.mpk`.
#[derive(Module, Debug)]
pub struct ArmModel<B: Backend> {
    pub net: WinModel<B>,
    pub aux: Option<AuxHeads<B>>,
}

/// Outputs for n input rows: the antisymmetric logit `[n, 1]`, and with
/// heads the row marks `[n, 8]` (mine red…blue, then the opponent's),
/// strikes `[n, 2]` (mine, the opponent's) and difference `[n, 1]`.
pub struct ArmOut<B: Backend> {
    pub z: Tensor<B, 2>,
    pub marks: Option<Tensor<B, 2>>,
    pub strikes: Option<Tensor<B, 2>>,
    pub diff: Option<Tensor<B, 2>>,
}

/// Targets in [`ArmOut`]'s shapes.
pub struct Targets<B: Backend> {
    pub win: Tensor<B, 2>,
    pub marks: Tensor<B, 2>,
    pub strikes: Tensor<B, 2>,
    pub diff: Tensor<B, 2>,
}

impl<B: Backend> ArmModel<B> {
    /// Both orientations run as one batch, as in [`WinModel::forward`]; the
    /// opponent's predictions are the heads on the swapped half.
    pub fn forward(&self, x: Tensor<B, 2>, swapped: Tensor<B, 2>) -> ArmOut<B> {
        let n = x.dims()[0];
        let (e, h) = self.net.trunk(Tensor::cat(vec![x, swapped], 0));
        let g = self.net.out.forward(h.clone());
        let z = g.clone().narrow(0, 0, n) - g.narrow(0, n, n);
        let Some(aux) = &self.aux else {
            return ArmOut {
                z,
                marks: None,
                strikes: None,
                diff: None,
            };
        };
        let w = h.dims()[1];
        let hr = h.clone().reshape([2 * n, 1, w]).repeat_dim(1, 4);
        let m = aux.marks.forward(Tensor::cat(vec![e, hr], 2)).reshape([2 * n, 4]);
        let marks = Tensor::cat(vec![m.clone().narrow(0, 0, n), m.narrow(0, n, n)], 1);
        let s = aux.strikes.forward(h.clone());
        let strikes = Tensor::cat(vec![s.clone().narrow(0, 0, n), s.narrow(0, n, n)], 1);
        let a = aux.diff.forward(h);
        let diff = a.clone().narrow(0, 0, n) - a.narrow(0, n, n);
        ArmOut {
            z,
            marks: Some(marks),
            strikes: Some(strikes),
            diff: Some(diff),
        }
    }
}

/// Inputs and swapped inputs of (me, opp) boards, flat, and their width.
pub fn encode(boards: &[(&State, &State)], thermo: bool) -> (Vec<f32>, Vec<f32>, usize) {
    let (mut x, mut s) = (Vec::new(), Vec::new());
    for &(me, opp) in boards {
        if thermo {
            let v = thermo_input(me, opp);
            x.extend_from_slice(&v);
            s.extend_from_slice(&swap_thermo(&v));
        } else {
            let v = win_input(me, opp);
            x.extend_from_slice(&v);
            s.extend_from_slice(&swap_input(&v));
        }
    }
    (x, s, if thermo { WIN_INPUTS_T } else { WIN_INPUTS })
}

/// Inputs, swapped inputs and targets of `samples` as tensors.
pub fn tensors<B: Backend>(
    samples: &[&Sample],
    thermo: bool,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>, Targets<B>) {
    let boards: Vec<(&State, &State)> = samples.iter().map(|s| (&s.me, &s.opp)).collect();
    let (x, s, d) = encode(&boards, thermo);
    let n = samples.len();
    let col = |f: &dyn Fn(&Sample) -> Vec<f32>, w: usize| {
        let v: Vec<f32> = samples.iter().flat_map(|s| f(s)).collect();
        Tensor::<B, 1>::from_floats(v.as_slice(), device).reshape([n, w])
    };
    let t = Targets {
        win: col(&|s| vec![s.win], 1),
        marks: col(&|s| s.marks.to_vec(), 8),
        strikes: col(&|s| s.strikes.to_vec(), 2),
        diff: col(&|s| vec![s.diff], 1),
    };
    (flat_tensor(&x, d, device), flat_tensor(&s, d, device), t)
}

/// (total loss, win BCE): BCE(win) + w[0]·MSE(marks) + w[1]·MSE(strikes) +
/// w[2]·MSE(difference), the last three only with heads.
pub fn arm_loss<B: Backend>(out: ArmOut<B>, t: &Targets<B>, w: &[f32]) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let win = bce_logits(out.z, t.win.clone());
    let mse = |p: Tensor<B, 2>, y: Tensor<B, 2>| {
        let d = p - y;
        (d.clone() * d).mean()
    };
    let mut total = win.clone();
    if let (Some(m), Some(s), Some(d)) = (out.marks, out.strikes, out.diff) {
        total = total
            + mse(m, t.marks.clone()).mul_scalar(w[0])
            + mse(s, t.strikes.clone()).mul_scalar(w[1])
            + mse(d, t.diff.clone()).mul_scalar(w[2]);
    }
    (total, win)
}

/// Binary entropy in nats; 0·ln 0 = 0.
pub fn entropy(w: f32) -> f64 {
    let h = |q: f64| if q > 0.0 { -q * q.ln() } else { 0.0 };
    h(w as f64) + h(1.0 - w as f64)
}

/// Binary cross-entropy of prediction `p` against label `w`, in nats
/// (`p` clamped away from 0 and 1).
pub fn bce(w: f32, p: f32) -> f64 {
    let (w, p) = (w as f64, (p as f64).clamp(1e-7, 1.0 - 1e-7));
    -(w * p.ln() + (1.0 - w) * (1.0 - p).ln())
}

/// The win excess (BCE − entropy) a perfect predictor scores on a label
/// that is the mean of `k` outcomes with sample variance `v`:
/// v / (2k·w(1 − w)); 0 for a label of exactly 0 or 1.
pub fn noise_floor(w: f32, v: f32, k: usize) -> f64 {
    let w = w as f64;
    if w <= 0.0 || w >= 1.0 {
        return 0.0;
    }
    v as f64 / (2.0 * k as f64 * w * (1.0 - w))
}

/// Learning rate at `step` of `total`: linear warmup over the first 1% of
/// steps (at least one), then a cosine from `lr` down to 0.
pub fn cosine_lr(lr: f64, step: usize, total: usize) -> f64 {
    let warm = (total / 100).max(1);
    if step < warm {
        return lr * (step + 1) as f64 / warm as f64;
    }
    let t = (step - warm) as f64 / (total - warm).max(1) as f64;
    lr * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
}

/// e ← decay·e + (1 − decay)·v.
pub fn ema_update(e: &mut [f32], v: &[f32], decay: f32) {
    for (a, &b) in e.iter_mut().zip(v) {
        *a = decay * *a + (1.0 - decay) * b;
    }
}

fn linears<B: Backend>(m: &ArmModel<B>) -> Vec<&Linear<B>> {
    let n = &m.net;
    let mut v = vec![&n.row1, &n.row2, &n.pair, &n.head1, &n.head2, &n.out];
    if let Some(a) = &m.aux {
        v.extend([&a.marks, &a.strikes, &a.diff]);
    }
    v
}

fn linears_mut<B: Backend>(m: &mut ArmModel<B>) -> Vec<&mut Linear<B>> {
    let n = &mut m.net;
    let mut v = vec![&mut n.row1, &mut n.row2, &mut n.pair, &mut n.head1, &mut n.head2, &mut n.out];
    if let Some(a) = &mut m.aux {
        v.extend([&mut a.marks, &mut a.strikes, &mut a.diff]);
    }
    v
}

/// Every layer's weights then bias, in [`linears`] order.
fn weights<B: Backend>(m: &ArmModel<B>) -> Vec<Vec<f32>> {
    linears(m)
        .into_iter()
        .flat_map(|l| {
            [
                l.weight.val().into_data().to_vec::<f32>().unwrap(),
                l.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().unwrap(),
            ]
        })
        .collect()
}

fn set_linear(l: &mut Linear<MyBackend>, w: &[f32], b: &[f32], device: &<MyBackend as Backend>::Device) {
    let (d_in, d_out) = (l.weight.shape().dims[0], l.weight.shape().dims[1]);
    l.weight = Param::from_tensor(Tensor::from_data(TensorData::new(w.to_vec(), [d_in, d_out]), device));
    l.bias = Some(Param::from_tensor(Tensor::from_data(
        TensorData::new(b.to_vec(), [d_out]),
        device,
    )));
}

/// An exponential moving average of a model's weights, updated after
/// every step; decay 0 keeps the live weights.
pub struct Ema {
    decay: f32,
    values: Vec<Vec<f32>>,
}

impl Ema {
    pub fn new<B: Backend>(m: &ArmModel<B>, decay: f32) -> Self {
        Ema {
            decay,
            values: weights(m),
        }
    }

    pub fn update<B: Backend>(&mut self, m: &ArmModel<B>) {
        let now = weights(m);
        if self.decay == 0.0 {
            self.values = now;
            return;
        }
        for (e, v) in self.values.iter_mut().zip(&now) {
            ema_update(e, v, self.decay);
        }
    }

    /// `m` with the averaged weights.
    pub fn apply(&self, mut m: ArmModel<MyBackend>) -> ArmModel<MyBackend> {
        let device = Default::default();
        for (i, l) in linears_mut(&mut m).into_iter().enumerate() {
            set_linear(l, &self.values[2 * i], &self.values[2 * i + 1], &device);
        }
        m
    }
}

/// The lowest metric offered so far.
#[derive(Default)]
pub struct Best {
    pub metric: Option<f64>,
}

impl Best {
    /// Whether `m` is a new lowest (the first offer always is).
    pub fn offer(&mut self, m: f64) -> bool {
        if self.metric.is_none_or(|b| m < b) {
            self.metric = Some(m);
            true
        } else {
            false
        }
    }
}

/// Predictions for samples, in order; the aux vectors are empty without heads.
#[derive(Default)]
pub struct Preds {
    pub p: Vec<f32>,
    pub marks: Vec<[f32; 8]>,
    pub strikes: Vec<[f32; 2]>,
    pub diff: Vec<f32>,
}

pub fn predict(model: &ArmModel<MyBackend>, samples: &[&Sample]) -> Preds {
    let device = Default::default();
    let thermo = model.net.is_thermo();
    let mut out = Preds::default();
    let v = |t: Tensor<MyBackend, 2>| t.into_data().to_vec::<f32>().unwrap();
    for chunk in samples.chunks(4096) {
        let boards: Vec<(&State, &State)> = chunk.iter().map(|s| (&s.me, &s.opp)).collect();
        let (x, s, d) = encode(&boards, thermo);
        let o = model.forward(flat_tensor(&x, d, &device), flat_tensor(&s, d, &device));
        out.p.extend(v(o.z).into_iter().map(sigmoid));
        if let (Some(m), Some(st), Some(df)) = (o.marks, o.strikes, o.diff) {
            out.marks.extend(v(m).chunks(8).map(|c| <[f32; 8]>::try_from(c).unwrap()));
            out.strikes.extend(v(st).chunks(2).map(|c| <[f32; 2]>::try_from(c).unwrap()));
            out.diff.extend(v(df));
        }
    }
    out
}

/// One split's measurements: win excess (BCE − label entropy) and its
/// noise floor over every sample, each head's MSE, and the difference
/// diagnostic over the decisions.
#[derive(Clone, Debug)]
pub struct EvalReport {
    pub samples: usize,
    pub excess: f64,
    pub floor: f64,
    pub marks: Option<f64>,
    pub strikes: Option<f64>,
    pub diff: Option<f64>,
    pub diffs: Option<DiffReport>,
}

/// `k_turn` / `k_cand`: rollouts behind a turn-end / candidate label.
pub fn evaluate(model: &ArmModel<MyBackend>, split: &Split, k_turn: usize, k_cand: usize) -> EvalReport {
    let cand: Vec<&Sample> = split.decisions.iter().flat_map(|d| &d.samples).collect();
    let all: Vec<&Sample> = split.turns.iter().chain(cand.iter().copied()).collect();
    let ks: Vec<usize> = split
        .turns
        .iter()
        .map(|_| k_turn)
        .chain(cand.iter().map(|_| k_cand))
        .collect();
    let pr = predict(model, &all);
    let n = all.len().max(1) as f64;
    let mean = |f: &dyn Fn(usize) -> f64| (0..all.len()).map(f).sum::<f64>() / n;
    let sq = |a: f32, b: f32| ((a - b) as f64).powi(2);
    let excess = mean(&|i| bce(all[i].win, pr.p[i]) - entropy(all[i].win));
    let floor = mean(&|i| noise_floor(all[i].win, all[i].win_var, ks[i]));
    let has_aux = !pr.diff.is_empty();
    let marks = has_aux.then(|| mean(&|i| (0..8).map(|j| sq(all[i].marks[j], pr.marks[i][j])).sum::<f64>() / 8.0));
    let strikes =
        has_aux.then(|| mean(&|i| (0..2).map(|j| sq(all[i].strikes[j], pr.strikes[i][j])).sum::<f64>() / 2.0));
    let diff = has_aux.then(|| mean(&|i| sq(all[i].diff, pr.diff[i])));
    // The decisions' predictions follow the turn ends', in order.
    let mut at = split.turns.len();
    let mut preds: Vec<Vec<f32>> = Vec::new();
    let views: Vec<GroupView> = split
        .decisions
        .iter()
        .map(|d| {
            preds.push(pr.p[at..at + d.samples.len()].to_vec());
            at += d.samples.len();
            GroupView {
                labels: d.samples.iter().map(|s| s.win).collect(),
                pair_var: &d.pair_var,
                swapped: d.swapped,
            }
        })
        .collect();
    EvalReport {
        samples: all.len(),
        excess,
        floor,
        marks,
        strikes,
        diff,
        diffs: diff_stats_views(&preds, &views, k_cand),
    }
}

/// The first `max` turn ends, and the first decisions up to `max` samples.
pub fn subset(split: &Split, max: usize) -> Split {
    let mut decisions: Vec<DecisionData> = Vec::new();
    let mut n = 0;
    for d in &split.decisions {
        if n + d.samples.len() > max {
            break;
        }
        n += d.samples.len();
        decisions.push(d.clone());
    }
    Split {
        turns: split.turns.iter().take(max).copied().collect(),
        decisions,
    }
}
```

Notes for the implementer:
- `Option::is_none_or` needs Rust 1.82. If `rustc --version` is older, write `self.metric.map_or(true, |b| m < b)` instead.
- If `repeat_dim` does not exist in burn 0.20.1, use `.expand([2 * n, 4, w])` on the `[2n, 1, w]` tensor; ledger the ruling.
- `TOKEN` and `T_TOKEN` are used by the tests only; if the compiler warns that they are unused outside tests, move them into the test module's `use` line instead.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib --release win_arm`
Expected: `test result: ok. 8 passed`.

- [ ] **Step 5: Format, full suite, commit**

```bash
rustfmt --edition 2021 src/dqn/win_arm.rs src/dqn/win.rs
cargo test --lib --release 2>&1 | tail -3
cargo check --lib --no-default-features 2>&1 | grep "^warning" | head
```
Expected: `test result: ok. 194 passed`; the check shows only the two existing warnings. Then:

```bash
git add src/dqn/mod.rs src/dqn/win_arm.rs src/dqn/win.rs
git commit -m "feat(win): auxiliary heads, held-out metrics, cosine schedule and weight averaging

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The fit and `win-arm`

**Files:**
- Modify: `src/dqn/win_arm.rs` (`ArmCfg`, `fit`, `run`, CSV; tests)
- Modify: `src/dqn/win.rs` (`WinModelConfig::aux`, `save_aux`)
- Modify: `src/main.rs` (`win-arm` subcommand)

**Interfaces:**
- Consumes (Tasks 2–5): `load_win_ad`, `save_win` (`win_pi`); `widen_to_thermo`, `WinModelConfig`; `win_data::load`, `Dataset`; everything in Task 5. Existing: `train_epochs`, `Ad` (`win_train`).
- Produces (in `crate::dqn::win_arm`):
  - `pub struct ArmCfg { pub data: String, pub out: String, pub init: Option<String>, pub thermo: bool, pub aux: bool, pub epochs: usize, pub lr: f64, pub wd: f32, pub ema: f32, pub aux_weights: Vec<f32>, pub width: usize, pub seed: u64, pub legacy: bool }` with `validate(&self) -> Result<(), String>`
  - `pub struct FitSummary { pub best_epoch: usize, pub epochs: usize }`
  - `pub fn fit(cfg: &ArmCfg) -> Result<FitSummary, String>`
  - `pub const FIT_HEADER: &str` (the `fit.csv` header)
  - in `crate::dqn::win`: `WinModelConfig { …, pub aux: Option<bool> }` (`Some(true)` = the dir has `aux.mpk`), `WinModelConfig::has_aux(&self) -> bool`, `pub fn save_aux(dir: &str, heads: &AuxHeads<MyBackend>)`

- [ ] **Step 1: Write the failing tests**

Add to `win_arm.rs` `mod tests` (extend the `use` lines with `use crate::dqn::pair::PairStrategy; use crate::dqn::win::{ManualWinNet, WinStrategy}; use crate::dqn::win_data::{generate, load, save, DataCfg}; use crate::dqn::win_pi::{load_win_ad, save_win}; use crate::dqn::win_train::train_epochs; use burn::module::AutodiffModule;`):

```rust
    fn tmp(name: &str) -> String {
        let d = std::env::temp_dir().join(format!("qwixxer-win-arm-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d.to_string_lossy().into_owned()
    }

    /// A tiny dataset by Master with decisions on both sides, and a W=4
    /// plain net to start from.
    fn tiny(base: &str) -> (String, String) {
        let data = format!("{base}/data");
        let cfg = DataCfg {
            player: "pair_model".into(),
            out: data.clone(),
            positions: 120,
            rollouts: 2,
            cand_rollouts: 2,
            holdout: 0.5,
            seed: 3,
        };
        let ds = generate(&cfg, &|| PairStrategy::load("pair_model"));
        assert!(
            !ds.train.decisions.is_empty() && !ds.heldout.decisions.is_empty(),
            "pick another seed: one side has no decisions"
        );
        save(&data, &ds).unwrap();
        let init = format!("{base}/init");
        let device = Default::default();
        let mcfg = WinModelConfig::new().with_width(4);
        save_win(&init, &mcfg, &mcfg.init::<MyBackend>(&device));
        (data, init)
    }

    fn arm(data: &str, out: String, init: Option<String>) -> ArmCfg {
        ArmCfg {
            data: data.into(),
            out,
            init,
            thermo: false,
            aux: false,
            epochs: 2,
            lr: 3e-4,
            wd: 1e-4,
            ema: 0.999,
            aux_weights: vec![1.0, 0.5, 0.5],
            width: 4,
            seed: 42,
            legacy: false,
        }
    }

    #[test]
    fn the_legacy_fit_is_todays_fit() {
        let base = tmp("legacy");
        let (data, init) = tiny(&base);
        let out = format!("{base}/legacy");
        fit(&ArmCfg {
            legacy: true,
            ..arm(&data, out.clone(), Some(init.clone()))
        })
        .unwrap();
        let ds = load(&data).unwrap();
        let samples: Vec<([f32; WIN_INPUTS], f32)> = ds
            .train
            .turns
            .iter()
            .chain(ds.train.decisions.iter().flat_map(|d| &d.samples))
            .map(|s| (win_input(&s.me, &s.opp), s.win))
            .collect();
        let (mcfg, start) = load_win_ad(&init);
        let (want, _) = train_epochs(start, &samples, 2, 3e-4, 42);
        let want_dir = format!("{base}/want");
        save_win(&want_dir, &mcfg, &want.valid());
        let (got, want) = (WinStrategy::load(&out), WinStrategy::load(&want_dir));
        let xs: Vec<_> = samples.iter().map(|s| s.0).collect();
        assert_eq!(got.net.logits(&xs), want.net.logits(&xs));
        let log = std::fs::read_to_string(format!("{out}/fit.csv")).unwrap();
        let rows: Vec<&str> = log.lines().collect();
        assert_eq!(rows[0], FIT_HEADER);
        assert_eq!(rows.len(), 3);
        // The legacy fit saves its last epoch, and no heads.
        assert_eq!(rows[2].split(',').nth(10), Some("1"));
        assert!(!Path::new(&format!("{out}/aux.mpk")).exists());
        assert!(!WinModelConfig::load(format!("{out}/config.json")).unwrap().has_aux());
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn a_thermometer_arm_with_heads_fits_and_saves_a_playable_net() {
        let base = tmp("smoke");
        let (data, init) = tiny(&base);
        let out = format!("{base}/arm");
        let s = fit(&ArmCfg {
            thermo: true,
            aux: true,
            ..arm(&data, out.clone(), Some(init))
        })
        .unwrap();
        assert!((1..=2).contains(&s.best_epoch) && s.epochs == 2);
        let log = std::fs::read_to_string(format!("{out}/fit.csv")).unwrap();
        let rows: Vec<Vec<&str>> = log.lines().skip(1).map(|l| l.split(',').collect()).collect();
        assert_eq!(rows.len(), 2);
        for r in &rows {
            assert_eq!(r.len(), 12);
            assert!(r[4..10].iter().all(|v| !v.is_empty()), "aux and diff columns filled: {r:?}");
        }
        let bot = WinStrategy::load(&out);
        assert!(bot.net.is_thermo());
        // The heads are saved beside the net.
        assert!(Path::new(&format!("{out}/aux.mpk")).exists());
        assert!(WinModelConfig::load(format!("{out}/config.json")).unwrap().has_aux());
        // The widened bit columns trained: some weight of the first progress
        // bit (input 4 of the row encoder) moved off zero.
        let (_, m) = load_win_ad(&out);
        let w = m.row1.weight.val().into_data().to_vec::<f32>().unwrap();
        let width = m.row1.weight.shape().dims[1];
        assert!(w[4 * width..5 * width].iter().any(|&v| v != 0.0));
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn a_from_scratch_arm_fits() {
        let base = tmp("scratch");
        let (data, _) = tiny(&base);
        let out = format!("{base}/arm");
        fit(&ArmCfg {
            thermo: true,
            aux: true,
            epochs: 1,
            ..arm(&data, out.clone(), None)
        })
        .unwrap();
        assert!(WinStrategy::load(&out).net.is_thermo());
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn held_out_games_without_decisions_stop_on_the_win_excess() {
        let base = tmp("no-decisions");
        let (data, init) = tiny(&base);
        let mut ds = load(&data).unwrap();
        ds.heldout.decisions.clear();
        let bare = format!("{base}/bare");
        save(&bare, &ds).unwrap();
        let out = format!("{base}/arm");
        fit(&arm(&bare, out.clone(), Some(init))).unwrap();
        let log = std::fs::read_to_string(format!("{out}/fit.csv")).unwrap();
        for row in log.lines().skip(1) {
            let r: Vec<&str> = row.split(',').collect();
            assert!(r[7].is_empty() && r[8].is_empty() && r[9].is_empty(), "{row}");
        }
        assert!(Path::new(&format!("{out}/model.mpk")).exists());
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn validate_refuses_bad_settings() {
        let base = tmp("validate");
        let (data, init) = tiny(&base);
        let ok = arm(&data, format!("{base}/out"), Some(init.clone()));
        assert_eq!(ok.validate(), Ok(()));
        let err = |c: ArmCfg| c.validate().unwrap_err();
        let base_cfg = || arm(&data, ok.out.clone(), Some(init.clone()));
        assert!(err(ArmCfg { epochs: 0, ..base_cfg() }).contains("--epochs"));
        assert!(err(ArmCfg { ema: 1.0, ..base_cfg() }).contains("--ema"));
        assert!(err(ArmCfg { lr: 0.0, ..base_cfg() }).contains("--lr"));
        assert!(err(ArmCfg { aux_weights: vec![1.0], ..base_cfg() }).contains("--aux-weights"));
        assert!(err(ArmCfg { legacy: true, aux: true, ..base_cfg() }).contains("--legacy-fit"));
        assert!(err(ArmCfg { data: format!("{base}/nope"), ..base_cfg() }).contains("--data"));
        assert!(err(ArmCfg { init: Some(format!("{base}/nope")), ..base_cfg() }).contains("--init"));
        let thermo_init = format!("{base}/thermo");
        let device = Default::default();
        let mcfg = WinModelConfig::new().with_width(4).with_thermo(Some(true));
        save_win(&thermo_init, &mcfg, &mcfg.init::<MyBackend>(&device));
        assert!(err(ArmCfg { init: Some(thermo_init), ..base_cfg() }).contains("add --thermo"));
        // A dataset without held-out games is refused when the fit loads it.
        let mut ds = load(&data).unwrap();
        ds.heldout = Split::default();
        let none = format!("{base}/none");
        save(&none, &ds).unwrap();
        assert!(fit(&arm(&none, format!("{base}/out2"), Some(init.clone()))).unwrap_err().contains("no held-out"));
        std::fs::create_dir_all(&ok.out).unwrap();
        assert!(err(base_cfg()).contains("already exists"));
        std::fs::remove_dir_all(&base).unwrap();
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib --release win_arm`
Expected: compile errors — `cannot find struct 'ArmCfg'`, `cannot find function 'fit'`, `cannot find value 'FIT_HEADER'`.

- [ ] **Step 3: Implement the fit**

In `src/dqn/win.rs`, add a field to `WinModelConfig` after `thermo`:

```rust
    /// The dir also holds trained [`AuxHeads`] in `aux.mpk`. `None` in
    /// older dirs (an `Option` for the same reason as `thermo`).
    pub aux: Option<bool>,
```

a method beside `is_thermo`:

```rust
    pub fn has_aux(&self) -> bool {
        self.aux == Some(true)
    }
```

and after `AuxHeads`:

```rust
/// Write `dir/aux.mpk`; the dir's `config.json` says `aux: true`.
pub fn save_aux(dir: &str, heads: &AuxHeads<MyBackend>) {
    heads
        .clone()
        .save_file(format!("{dir}/aux"), &CompactRecorder::new())
        .expect("Failed to save the auxiliary heads");
}
```

Extend the `win_arm.rs` imports:

```rust
use crate::dqn::win::{save_aux, widen_to_thermo, WinModelConfig};
use crate::dqn::win_data::{self, Dataset};
use crate::dqn::win_pi::{load_win_ad, save_win};
use crate::dqn::win_train::Ad;
use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, AdamWConfig, GradientsParams, Optimizer},
};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use std::path::Path;
```

(merge them into the existing `use crate::dqn::win::{…}` and `use burn::{…}` blocks).

Add before `#[cfg(test)]`:

```rust
/// Training batch size, as `train_epochs`.
pub const BATCH: usize = 1024;

/// Training turn ends and training samples in the training-excess subset.
const TRAIN_SUBSET: usize = 50_000;

pub const FIT_HEADER: &str = "epoch,train_excess,heldout_excess,heldout_floor,heldout_marks,heldout_strikes,\
heldout_diff,diff_rmse,diff_noise,flip_share,best,elapsed_s";

/// `win-arm` settings.
pub struct ArmCfg {
    /// A `win-data` dir.
    pub data: String,
    /// Output model dir (must not exist).
    pub out: String,
    /// Warm start from this win model dir; `None` = from scratch.
    pub init: Option<String>,
    /// Thermometer inputs (a plain `init` is widened).
    pub thermo: bool,
    /// Auxiliary heads.
    pub aux: bool,
    pub epochs: usize,
    /// Peak learning rate (the constant one with `legacy`).
    pub lr: f64,
    /// AdamW weight decay.
    pub wd: f32,
    /// Weight-average decay per step (0 = the live weights).
    pub ema: f32,
    /// Aux loss weights: marks, strikes, difference.
    pub aux_weights: Vec<f32>,
    /// Width of a from-scratch net.
    pub width: usize,
    pub seed: u64,
    /// Today's fit: `train_epochs`' Adam at a constant lr, no weight decay,
    /// no averaging, the last epoch saved.
    pub legacy: bool,
}

impl ArmCfg {
    /// Every check that can fail before the dataset is read.
    pub fn validate(&self) -> Result<(), String> {
        if self.epochs == 0 {
            return Err("--epochs must be at least 1".into());
        }
        if !(self.lr > 0.0) {
            return Err("--lr must be positive".into());
        }
        if !(0.0..1.0).contains(&self.ema) {
            return Err("--ema must be in [0, 1)".into());
        }
        if !(self.wd >= 0.0) {
            return Err("--wd must be at least 0".into());
        }
        if self.aux_weights.len() != 3 || self.aux_weights.iter().any(|&w| !(w >= 0.0)) {
            return Err("--aux-weights takes 3 non-negative weights: marks,strikes,difference".into());
        }
        if self.legacy && (self.thermo || self.aux) {
            return Err("--legacy-fit refits the plain net without aux heads: drop --thermo and --aux".into());
        }
        for f in ["meta.json", "data.bin"] {
            if !Path::new(&format!("{}/{f}", self.data)).exists() {
                return Err(format!("--data {}: {f} not found", self.data));
            }
        }
        match &self.init {
            Some(init) => {
                for f in ["config.json", "model.mpk"] {
                    if !Path::new(&format!("{init}/{f}")).exists() {
                        return Err(format!("--init {init}: {f} not found"));
                    }
                }
                let c = WinModelConfig::load(format!("{init}/config.json"))
                    .map_err(|e| format!("--init {init}: unreadable config.json ({e:?})"))?;
                if c.is_thermo() && !self.thermo {
                    return Err(format!("--init {init} is a thermometer net: add --thermo"));
                }
            }
            None if self.width == 0 => return Err("--width must be at least 1".into()),
            None => {}
        }
        if Path::new(&self.out).exists() {
            return Err(format!("{} already exists: move it aside or pick another --out", self.out));
        }
        Ok(())
    }
}

pub struct FitSummary {
    /// 1-based epoch whose weights were saved.
    pub best_epoch: usize,
    pub epochs: usize,
}

/// Fit one arm (docs/superpowers/specs/2026-10-19-win-thermo-aux-design.md,
/// "Fit"): writes `out/config.json`, `out/model.mpk` (the best epoch's
/// averaged net, without heads) and `out/fit.csv`.
pub fn fit(cfg: &ArmCfg) -> Result<FitSummary, String> {
    cfg.validate()?;
    let ds = win_data::load(&cfg.data).map_err(|e| format!("--data {}: {e}", cfg.data))?;
    if ds.heldout.turns.is_empty() && ds.heldout.decisions.is_empty() {
        return Err(format!(
            "--data {}: no held-out games (generate it with --holdout above 0)",
            cfg.data
        ));
    }
    if ds.train.turns.is_empty() && ds.train.decisions.is_empty() {
        return Err(format!("--data {}: no training games", cfg.data));
    }
    let device = Default::default();
    <Ad as Backend>::seed(&device, cfg.seed);
    let net: WinModel<Ad> = match &cfg.init {
        Some(dir) => {
            let (_, m) = load_win_ad(dir);
            if cfg.thermo && !m.is_thermo() {
                widen_to_thermo(&m, &device)
            } else {
                m
            }
        }
        None => WinModelConfig::new()
            .with_width(cfg.width)
            .with_thermo(cfg.thermo.then_some(true))
            .init(&device),
    };
    let width = net.row1.weight.shape().dims[1];
    let mcfg = WinModelConfig::new()
        .with_width(width)
        .with_thermo(net.is_thermo().then_some(true))
        .with_aux(cfg.aux.then_some(true));
    let model = ArmModel {
        net,
        aux: cfg.aux.then(|| AuxHeads::new(width, &device)),
    };
    std::fs::create_dir_all(&cfg.out).map_err(|e| format!("{}: {e}", cfg.out))?;
    if cfg.legacy {
        run(cfg, &ds, &mcfg, model, AdamConfig::new().init())
    } else {
        run(cfg, &ds, &mcfg, model, AdamWConfig::new().with_weight_decay(cfg.wd).init())
    }
}

fn run<O: Optimizer<ArmModel<Ad>, Ad>>(
    cfg: &ArmCfg,
    ds: &Dataset,
    mcfg: &WinModelConfig,
    mut model: ArmModel<Ad>,
    mut optim: O,
) -> Result<FitSummary, String> {
    let device = Default::default();
    let start = std::time::Instant::now();
    // Turn ends then candidates, as win-pi orders a round's data.
    let samples: Vec<&Sample> = ds
        .train
        .turns
        .iter()
        .chain(ds.train.decisions.iter().flat_map(|d| &d.samples))
        .collect();
    let thermo = model.net.is_thermo();
    let total = cfg.epochs * samples.len().div_ceil(BATCH);
    let mut ema = Ema::new(&model, if cfg.legacy { 0.0 } else { cfg.ema });
    let train_sub = subset(&ds.train, TRAIN_SUBSET);
    let (kt, kc) = (ds.meta.rollouts, ds.meta.cand_rollouts);
    let log_path = format!("{}/fit.csv", cfg.out);
    std::fs::write(&log_path, format!("{FIT_HEADER}\n")).map_err(|e| format!("{log_path}: {e}"))?;
    let mut order: Vec<usize> = (0..samples.len()).collect();
    let (mut best, mut best_model, mut best_epoch, mut step) = (Best::default(), None, 0, 0);
    for epoch in 0..cfg.epochs {
        order.shuffle(&mut SmallRng::seed_from_u64(cfg.seed.wrapping_add(epoch as u64)));
        for chunk in order.chunks(BATCH) {
            let batch: Vec<&Sample> = chunk.iter().map(|&i| samples[i]).collect();
            let (x, s, t) = tensors::<Ad>(&batch, thermo, &device);
            let (loss, _) = arm_loss(model.forward(x, s), &t, &cfg.aux_weights);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            let lr = if cfg.legacy {
                cfg.lr
            } else {
                cosine_lr(cfg.lr, step, total)
            };
            model = optim.step(lr, model, grads);
            ema.update(&model);
            step += 1;
        }
        let avg = ema.apply(model.valid());
        let held = evaluate(&avg, &ds.heldout, kt, kc);
        let train_excess = evaluate(&avg, &train_sub, kt, kc).excess;
        let metric = held.diffs.as_ref().map_or(held.excess, |d| d.diff_rmse as f64);
        let is_best = if cfg.legacy {
            epoch + 1 == cfg.epochs
        } else {
            best.offer(metric)
        };
        if is_best {
            best_model = Some(avg.clone());
            best_epoch = epoch + 1;
        }
        let opt = |v: Option<f64>| v.map(|v| format!("{v:.6}")).unwrap_or_default();
        let d = held.diffs.as_ref();
        let row = format!(
            "{},{train_excess:.6},{:.6},{:.6},{},{},{},{},{},{},{},{}",
            epoch + 1,
            held.excess,
            held.floor,
            opt(held.marks),
            opt(held.strikes),
            opt(held.diff),
            opt(d.map(|d| d.diff_rmse as f64)),
            opt(d.map(|d| d.diff_noise as f64)),
            opt(d.map(|d| d.flip_share as f64)),
            is_best as u8,
            start.elapsed().as_secs()
        );
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&log_path)
                .map_err(|e| format!("{log_path}: {e}"))?;
            writeln!(f, "{row}").map_err(|e| format!("{log_path}: {e}"))?;
        }
        println!(
            "Epoch {:>2}: train excess {train_excess:.5}, held-out excess {:.5} (floor {:.5}){}{}, {}s",
            epoch + 1,
            held.excess,
            held.floor,
            d.map(|d| format!(", diff RMSE {:.4} (noise {:.4})", d.diff_rmse, d.diff_noise))
                .unwrap_or_default(),
            if is_best { " → best" } else { "" },
            start.elapsed().as_secs()
        );
    }
    let best_model: ArmModel<MyBackend> = best_model.expect("at least one epoch");
    save_win(&cfg.out, mcfg, &best_model.net);
    if let Some(heads) = &best_model.aux {
        save_aux(&cfg.out, heads);
    }
    Ok(FitSummary {
        best_epoch,
        epochs: cfg.epochs,
    })
}
```

`save_win` writes `config.json` and `model.mpk` into `cfg.out`, which already holds `fit.csv`; `create_dir_all` there is a no-op. Every other `WinModelConfig::new()` in the code leaves `aux` at `None`, so nothing else changes.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib --release win_arm`
Expected: `test result: ok. 13 passed`. If `tiny` panics with "pick another seed", change its `seed: 3` to 4, ledger a ruling, rerun. If `the_legacy_fit_is_todays_fit` fails on the logits, the two fits diverged: compare `run`'s legacy path with `train_epochs` line by line (shuffle seed, chunk order, loss, lr) before touching anything else (superpowers:systematic-debugging).

- [ ] **Step 5: Add the `win-arm` subcommand to `src/main.rs`**

In `enum Commands`, after `WinData { … },`:

```rust
    /// Win-net fit of one experiment arm on a win-data dataset, measured on its held-out games every epoch
    #[cfg(feature = "dqn")]
    WinArm {
        /// win-data dir
        #[arg(long)]
        data: String,
        /// Output model dir (must not exist): config.json, model.mpk, fit.csv
        #[arg(long)]
        out: String,
        /// Warm start from this win model dir (default: from scratch)
        #[arg(long)]
        init: Option<String>,
        /// Thermometer inputs (widens a plain --init)
        #[arg(long)]
        thermo: bool,
        /// Auxiliary heads: final row marks, strikes, score difference (saved to aux.mpk, never used in play)
        #[arg(long)]
        aux: bool,
        #[arg(long, default_value = "10")]
        epochs: usize,
        /// Peak learning rate (cosine decay to 0)
        #[arg(long, default_value = "3e-4")]
        lr: f64,
        /// AdamW weight decay
        #[arg(long, default_value = "1e-4")]
        wd: f32,
        /// Weight-average decay per step (0 = the live weights)
        #[arg(long, default_value = "0.999")]
        ema: f32,
        /// Aux loss weights: marks,strikes,difference
        #[arg(long, value_delimiter = ',', default_value = "1.0,0.5,0.5")]
        aux_weights: Vec<f32>,
        /// Width of a from-scratch net
        #[arg(long, default_value = "64")]
        width: usize,
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Today's fit as a control: Adam, constant lr, no decay or averaging, last epoch saved
        #[arg(long)]
        legacy_fit: bool,
    },
```

In the `match`, after the `WinData` arm:

```rust
        #[cfg(feature = "dqn")]
        Some(Commands::WinArm {
            data,
            out,
            init,
            thermo,
            aux,
            epochs,
            lr,
            wd,
            ema,
            aux_weights,
            width,
            seed,
            legacy_fit,
        }) => {
            let cfg = dqn::win_arm::ArmCfg {
                data,
                out,
                init,
                thermo,
                aux,
                epochs,
                lr,
                wd,
                ema,
                aux_weights,
                width,
                seed,
                legacy: legacy_fit,
            };
            match dqn::win_arm::fit(&cfg) {
                Ok(s) => println!(
                    "win-arm: saved epoch {} of {} to {}",
                    s.best_epoch, s.epochs, cfg.out
                ),
                Err(e) => {
                    eprintln!("win-arm: {e}");
                    std::process::exit(2);
                }
            }
        }
```

Run: `cargo run --release -- win-arm --data /nonexistent --out /tmp/x --epochs 0`
Expected: `win-arm: --epochs must be at least 1`, exit code 2.

- [ ] **Step 6: Format, full suite, commit**

```bash
rustfmt --edition 2021 src/dqn/win_arm.rs src/dqn/win.rs src/main.rs
cargo test --lib --release 2>&1 | tail -3
```
Expected: `test result: ok. 199 passed`. Then:

```bash
git add src/dqn/win_arm.rs src/dqn/win.rs src/main.rs
git commit -m "feat(win): win-arm: fit one arm with held-out measurement and early stopping; heads saved to aux.mpk

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Saved heads: loading them and a hand-written forward pass

**Files:**
- Modify: `src/dqn/win.rs` (`AuxPred`, `ManualAux`, `ManualWinNet::{with_aux, trunk, aux_states}`, `WinStrategy::load`)
- Modify: `src/dqn/win_arm.rs` (tests)

**Interfaces:**
- Consumes (Tasks 5–6): `AuxHeads`, `WinModelConfig::has_aux`, `aux.mpk`; `ArmModel::forward`, `fit`.
- Produces (in `crate::dqn::win`, compiled without `dqn` too, for a later web explorer):
  - `#[derive(Clone, Copy, Debug, PartialEq)] pub struct AuxPred { pub marks: [f32; 8], pub strikes: [f32; 2], pub diff: f32 }` — final row totals (mine red…blue, then the opponent's), final strikes (mine, the opponent's), final (my − opponent's) points, all in game units (×12, ×4, ×100 of the heads' outputs).
  - `ManualWinNet::with_aux(self, heads: &AuxHeads<MyBackend>) -> Self`
  - `ManualWinNet::aux_states(&self, me: &State, opp: &State) -> Option<AuxPred>` (`None` without heads)
  - `WinStrategy::load` loads `aux.mpk` when `config.json` has `aux: true`. Play never calls the heads.

- [ ] **Step 1: Write the failing tests**

In `src/dqn/win_arm.rs` `mod tests` (add `AuxPred` to the `crate::dqn::win` test import):

```rust
    #[test]
    fn manual_heads_match_the_burn_heads() {
        let device = Default::default();
        let (a, b) = boards();
        for thermo in [false, true] {
            let m = random_arm(35, thermo, true);
            let net = ManualWinNet::from_model(&m.net).with_aux(m.aux.as_ref().unwrap());
            let (x, s, d) = encode(&[(&a, &b)], thermo);
            let o = m.forward(flat_tensor(&x, d, &device), flat_tensor(&s, d, &device));
            let (marks, strikes, diff) = (values(o.marks), values(o.strikes), values(o.diff));
            let got: AuxPred = net.aux_states(&a, &b).unwrap();
            for i in 0..8 {
                assert!((got.marks[i] - 12.0 * marks[i]).abs() < 1e-3, "thermo {thermo} mark {i}");
            }
            for i in 0..2 {
                assert!((got.strikes[i] - 4.0 * strikes[i]).abs() < 1e-3);
            }
            assert!((got.diff - 100.0 * diff[0]).abs() < 1e-2);
            assert!(ManualWinNet::from_model(&m.net).aux_states(&a, &b).is_none());
        }
    }

    #[test]
    fn a_fitted_arm_with_heads_loads_them() {
        let base = tmp("heads");
        let (data, init) = tiny(&base);
        let (with, without) = (format!("{base}/with"), format!("{base}/without"));
        fit(&ArmCfg { aux: true, epochs: 1, ..arm(&data, with.clone(), Some(init.clone())) }).unwrap();
        fit(&ArmCfg { epochs: 1, ..arm(&data, without.clone(), Some(init)) }).unwrap();
        let (a, b) = boards();
        let p = WinStrategy::load(&with).net.aux_states(&a, &b).unwrap();
        assert!(p.marks.iter().chain(&p.strikes).all(|v| v.is_finite()) && p.diff.is_finite());
        assert!(WinStrategy::load(&without).net.aux_states(&a, &b).is_none());
        std::fs::remove_dir_all(&base).unwrap();
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib --release -- manual_heads_match a_fitted_arm_with_heads`
Expected: compile errors — `no method named 'with_aux'`, `cannot find type 'AuxPred'`.

- [ ] **Step 3: Implement in `src/dqn/win.rs`**

After `AuxHeads` / `save_aux`:

```rust
/// What the auxiliary heads predict for a position, in game units: final
/// row totals (mine red, yellow, green, blue, then the opponent's), final
/// strikes (mine, the opponent's), and the final (my − opponent's) points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuxPred {
    pub marks: [f32; 8],
    pub strikes: [f32; 2],
    pub diff: f32,
}

/// Plain-array copy of [`AuxHeads`].
struct ManualAux {
    marks: Dense,
    strikes: Dense,
    diff: Dense,
}
```

In `ManualWinNet`, add a last field:

```rust
    /// Trained auxiliary heads, when the model dir has them (never used in play).
    aux: Option<ManualAux>,
```

set `aux: None,` in `from_model`, and add to `impl ManualWinNet`:

```rust
    /// The net with its auxiliary heads.
    pub fn with_aux(mut self, heads: &AuxHeads<MyBackend>) -> Self {
        self.aux = Some(ManualAux {
            marks: Dense::from_linear(&heads.marks),
            strikes: Dense::from_linear(&heads.strikes),
            diff: Dense::from_linear(&heads.diff),
        });
        self
    }

    /// Each row's encoding and the head's last hidden layer for one input:
    /// [`ManualWinNet::g`] without its output layer. Kept apart from `g`,
    /// which play calls in its hot loop and reuses its buffers.
    fn trunk(&self, x: &[f32]) -> ([Vec<f32>; 4], Vec<f32>) {
        let (w, t) = (self.width, self.token);
        let mut h = Vec::with_capacity(2 * w);
        let mut rows: [Vec<f32>; 4] = Default::default();
        let mut pooled = [vec![0.0f32; w], vec![0.0f32; w]];
        for (r, e) in rows.iter_mut().enumerate() {
            self.row1.apply(&x[t * r..t * (r + 1)], &mut h, true);
            self.row2.apply(&h, e, true);
            for (p, v) in pooled[r / 2].iter_mut().zip(e.iter()) {
                *p += v;
            }
        }
        let (mut e, mut cat) = (Vec::with_capacity(w), vec![0.0f32; w]);
        for p in &pooled {
            self.pair.apply(p, &mut e, true);
            for (c, v) in cat.iter_mut().zip(&e) {
                *c += v;
            }
        }
        cat.extend_from_slice(&x[4 * t..]);
        self.head1.apply(&cat, &mut h, true);
        self.head2.apply(&h, &mut e, true);
        (rows, e)
    }

    /// The auxiliary heads' predictions for `me` against `opp`; `None`
    /// without heads.
    pub fn aux_states(&self, me: &State, opp: &State) -> Option<AuxPred> {
        let aux = self.aux.as_ref()?;
        let (x, s): (Vec<f32>, Vec<f32>) = if self.is_thermo() {
            let x = thermo_input(me, opp);
            (x.to_vec(), swap_thermo(&x).to_vec())
        } else {
            let x = win_input(me, opp);
            (x.to_vec(), swap_input(&x).to_vec())
        };
        let ((ex, hx), (es, hs)) = (self.trunk(&x), self.trunk(&s));
        let one = |d: &Dense, v: &[f32]| {
            let mut o = Vec::with_capacity(1);
            d.apply(v, &mut o, false);
            o[0]
        };
        let mut marks = [0.0f32; 8];
        for r in 0..4 {
            marks[r] = 12.0 * one(&aux.marks, &[ex[r].as_slice(), hx.as_slice()].concat());
            marks[4 + r] = 12.0 * one(&aux.marks, &[es[r].as_slice(), hs.as_slice()].concat());
        }
        Some(AuxPred {
            marks,
            strikes: [4.0 * one(&aux.strikes, &hx), 4.0 * one(&aux.strikes, &hs)],
            diff: 100.0 * (one(&aux.diff, &hx) - one(&aux.diff, &hs)),
        })
    }
```

Replace `WinStrategy::load` with:

```rust
    /// Load `dir/config.json` (width, encoding, heads), `dir/model.mpk`, and
    /// `dir/aux.mpk` when the config says the dir has heads.
    pub fn load(dir: &str) -> Self {
        let device = Default::default();
        let cfg = WinModelConfig::load(format!("{dir}/config.json")).expect("Failed to load win model config");
        let model = cfg
            .init::<MyBackend>(&device)
            .load_file(format!("{dir}/model"), &CompactRecorder::new(), &device)
            .expect("Failed to load win model");
        let mut net = ManualWinNet::from_model(&model);
        if cfg.has_aux() {
            let heads = AuxHeads::<MyBackend>::new(cfg.width, &device)
                .load_file(format!("{dir}/aux"), &CompactRecorder::new(), &device)
                .expect("Failed to load the auxiliary heads");
            net = net.with_aux(&heads);
        }
        WinStrategy { net: Arc::new(net) }
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib --release -- manual_heads_match a_fitted_arm_with_heads`
Expected: `test result: ok. 2 passed`.

Run: `cargo check --lib --no-default-features 2>&1 | grep "^warning" | head`
Expected: only the two existing warnings.

- [ ] **Step 5: Format, full suite, commit**

```bash
rustfmt --edition 2021 src/dqn/win.rs src/dqn/win_arm.rs
cargo test --lib --release 2>&1 | tail -3
```
Expected: `test result: ok. 201 passed`. Then:

```bash
git add src/dqn/win.rs src/dqn/win_arm.rs
git commit -m "feat(win): load saved auxiliary heads and predict with them by hand

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `win-gate` and the end-to-end smoke

**Files:**
- Modify: `src/dqn/win_pi.rs` (`win_gate`; test)
- Modify: `src/main.rs` (`win-gate` subcommand)

**Interfaces:**
- Consumes: existing `head_to_head`, `gate_decision`, `WinStrategy::load`, `PairStrategy::load`.
- Produces: `pub fn win_gate(a: &str, b: &str, games: usize, seed: u64, pair_dir: &str) -> (usize, f64, f64)` — (games played, a's share, z); `b == "master"` plays Master from `pair_dir`.

- [ ] **Step 1: Write the failing test**

In `src/dqn/win_pi.rs` `mod tests`:

```rust
    #[test]
    fn win_gate_of_a_net_against_itself_is_even() {
        let base = tmp("gate");
        let dir = format!("{base}/net");
        let device = Default::default();
        let mcfg = WinModelConfig::new().with_width(4).with_thermo(Some(true));
        save_win(&dir, &mcfg, &mcfg.init::<MyBackend>(&device));
        assert_eq!(win_gate(&dir, &dir, 8, 7, "pair_model"), (8, 0.5, 0.0));
        let (games, share, _) = win_gate(&dir, "master", 4, 7, "pair_model");
        assert!(games == 4 && (0.0..=1.0).contains(&share));
        std::fs::remove_dir_all(&base).unwrap();
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib --release win_gate_of_a_net`
Expected: compile error — `cannot find function 'win_gate'`.

- [ ] **Step 3: Implement**

In `src/dqn/win_pi.rs`, after `run_gate`:

```rust
/// `win-gate`: `games` paired games (both seats per dice set) of the win
/// net in dir `a` against the win net in dir `b`, or Master from
/// `pair_dir` when `b` is "master". Returns (games played, a's share, z).
pub fn win_gate(a: &str, b: &str, games: usize, seed: u64, pair_dir: &str) -> (usize, f64, f64) {
    let pairs = (games / 2) as u64;
    let a = WinStrategy::load(a);
    let scores = if b == "master" {
        head_to_head(&|| a.clone(), &|| PairStrategy::load(pair_dir), 0..pairs, seed)
    } else {
        let b = WinStrategy::load(b);
        head_to_head(&|| a.clone(), &|| b.clone(), 0..pairs, seed)
    };
    let (_, share, z) = gate_decision(&scores, f64::INFINITY, true);
    (scores.len() * 2, share, z)
}
```

In `src/main.rs` `enum Commands`, after `WinArm { … },`:

```rust
    /// Paired head-to-head between two win model dirs, or a win model dir and Master
    #[cfg(feature = "dqn")]
    WinGate {
        /// Win model dir
        #[arg(long)]
        a: String,
        /// Win model dir, or "master" for the pair net in pair_model/
        #[arg(long)]
        b: String,
        /// Games (even: both seats on every dice set)
        #[arg(long, default_value = "200000")]
        games: usize,
        #[arg(long, default_value = "42")]
        seed: u64,
    },
```

and in the `match`:

```rust
        #[cfg(feature = "dqn")]
        Some(Commands::WinGate { a, b, games, seed }) => {
            if games < 2 || games % 2 != 0 {
                eprintln!("win-gate: --games must be even and at least 2");
                std::process::exit(2);
            }
            for dir in [&a, &b] {
                if dir.as_str() != "master" && !std::path::Path::new(&format!("{dir}/model.mpk")).exists() {
                    eprintln!("win-gate: {dir}/model.mpk not found");
                    std::process::exit(2);
                }
            }
            let start = std::time::Instant::now();
            let (n, share, z) = dqn::win_pi::win_gate(&a, &b, games, seed, "pair_model");
            println!(
                "win-gate: {n} games, {a} vs {b}: {:.2}% for {a}, z {z:.2}, {}s",
                share * 100.0,
                start.elapsed().as_secs()
            );
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test --lib --release win_gate_of_a_net`
Expected: `test result: ok. 1 passed`.

- [ ] **Step 5: Format, full suite, commit**

```bash
rustfmt --edition 2021 src/dqn/win_pi.rs src/main.rs
cargo test --lib --release 2>&1 | tail -3
```
Expected: `test result: ok. 202 passed`. Then:

```bash
git add src/dqn/win_pi.rs src/main.rs
git commit -m "feat(win): win-gate: paired head-to-head between win model dirs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: End-to-end smoke (no commit)**

Uses the local `win_model/` (the user's net; read only) and the scratchpad dir `$S` (the session's scratchpad path).

```bash
cargo build --release
./target/release/qwixxer win-data --player win_model --out $S/smoke-data --positions 300 --rollouts 4 --cand-rollouts 4 --holdout 0.3
./target/release/qwixxer win-arm --data $S/smoke-data --init win_model --thermo --aux --epochs 2 --out $S/smoke-arm
./target/release/qwixxer win-gate --a $S/smoke-arm --b win_model --games 200
QWIXXER_WIN_MODEL=$S/smoke-arm ./target/release/qwixxer bench win pair -n 100
```

Expected:
- `win-data: N games; train … turn ends + … decisions (… samples); held out … ; Ns` with both sides non-zero.
- Two `Epoch` lines, each with a held-out excess, a floor and a diff RMSE, then `win-arm: saved epoch k of 2 to …`.
- `win-gate: 200 games, … z …` (the share is near 50%: two epochs on 300 turn ends barely move the net).
- `bench` plays 100 games and prints its usual summary without panicking.
- `ls $S/smoke-arm` shows `aux.mpk` beside `config.json`, `model.mpk` and `fit.csv`.

Record the four output lines in the ledger. Delete `$S/smoke-*` afterwards.
