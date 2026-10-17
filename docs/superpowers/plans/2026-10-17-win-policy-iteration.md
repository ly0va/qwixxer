# Win Policy Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `win-pi` command that runs gated policy iteration on the win net: the champion's rollouts label its own games' turn ends, a warm-started candidate is fitted to a window of recent rounds, and the candidate replaces the champion only if a sequential paired head-to-head says it is better.

**Architecture:** New module `src/dqn/win_pi.rs` (feature `dqn`) holding pure helpers (round seeds, the data window, the gate decision), the paired head-to-head bench, the sequential gate, `WinPiCfg` with validation, and the loop `policy_iteration`. It reuses `rollout_positions_by`, `train_epochs`, `bench_vs_pair`, `outcome` and `Ad` from `src/dqn/win_train.rs`. `src/main.rs` gets the `win-pi` subcommand.

**Tech Stack:** Rust 2021, burn 0.20.1 (ndarray backend, autodiff), rand 0.8 `SmallRng`, rayon.

**Spec:** `docs/superpowers/specs/2026-10-17-win-policy-iteration-design.md`

## Global Constraints

- 2-player only; same `WinModel` architecture; width from `--init`'s config, else `--width` (default 64).
- Labels: pure rollout P(win) (`blend` = 1.0) from the champion's own games; Master's static term unused.
- Gate: sequential paired head-to-head, blocks of `gate_block` games (default 100 000), cap `gate_max` (default 1 000 000); promote when z > `gate_z` (default 2.0); reject when z < 0, at the cap, or when the per-pair std is 0. z = (share − 0.5)/(std of pair scores / √pairs).
- Defaults: `--rounds 10 --positions 300000 --rollouts 64 --window 2 --epochs 20 --lr 3e-4 --master-games 200000 --seed 42 --dir win_pi`; a from-scratch round 1 (no `--init`) fits at lr 1e-3.
- `training.csv` header exactly: `round,positions,loss,gate_games,share_vs_champion,z,promoted,share_vs_master,elapsed_s`; round-0 row (only with `--init` and `master_games` > 0): `0,,,,,,,<share>,<elapsed>`.
- Refuse to start if `--dir` exists; no resume.
- Not touched: `win-train`, `win-fit`, `pair_model/`, `champion.txt`, `dqn_model/`, the web app. Don't edit `AGENTS.md`; never commit `.superpowers/`, `AGENTS.md`, `web-ui.txt`, `win_training.csv`, `win_model/`.
- Formatting: `rustfmt --edition 2021 src/dqn/win_pi.rs src/main.rs` only — never `cargo fmt`, never rustfmt `pair.rs`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Check the test output before committing; never chain a commit after a test command.

## Review Focus

- Round 1 without `--init` whose fresh fit loses to Master: the run must stop cleanly with a message and a CSV row, not panic, and leave no `champion/`. (Test in Task 3.)
- `gate_max` not a multiple of `gate_block`: the last block is truncated so the gate never plays more than `gate_max` games. (Test in Task 2.)
- Thread count / block split must not change head-to-head results: dice depend only on seed and pair index. (Test in Task 2.)
- `--init` pointing at a dir without `config.json` / `model.mpk`, or a missing `pair_model`: refused before any work with a clear message. (Test in Task 3.)
- A rejected round keeps the champion and still adds its data to the window. (Window test in Task 1; loop code in Task 3.)

---

### Task 1: Module scaffold and pure helpers

**Files:**
- Create: `src/dqn/win_pi.rs`
- Modify: `src/dqn/mod.rs` (after `pub mod win_train;`, line ~25)

**Interfaces:**
- Produces:
  - `pub const POS: u64 = 1; pub const FIT: u64 = 2; pub const GATE: u64 = 3;`
  - `pub fn round_seed(seed: u64, round: usize, purpose: u64) -> u64`
  - `pub fn push_window<T>(window: &mut VecDeque<Vec<T>>, round: Vec<T>, keep: usize)`
  - `#[derive(Debug, Clone, Copy, PartialEq)] pub enum Gate { Promote, Reject, Continue }`
  - `pub fn gate_decision(scores: &[f32], z_min: f64, at_max: bool) -> (Gate, f64, f64)` — returns (decision, share, z)

- [ ] **Step 1: Register the module and write the failing tests**

In `src/dqn/mod.rs`, after the `win_train` lines add:

```rust
#[cfg(feature = "dqn")]
pub mod win_pi;
```

Create `src/dqn/win_pi.rs`:

```rust
//! Gated policy iteration for the win net: the champion's rollouts label its
//! own games, a warm-started candidate is fitted to a window of recent
//! rounds, and a sequential paired head-to-head decides promotion.
//!
//! Design: docs/superpowers/specs/2026-10-17-win-policy-iteration-design.md

use std::collections::VecDeque;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_seeds_are_deterministic_and_distinct() {
        assert_eq!(round_seed(42, 3, POS), round_seed(42, 3, POS));
        let mut seen = std::collections::HashSet::new();
        for round in 0..50 {
            for purpose in [POS, FIT, GATE] {
                assert!(seen.insert(round_seed(42, round, purpose)), "collision at {round}/{purpose}");
            }
        }
        assert_ne!(round_seed(42, 1, POS), round_seed(43, 1, POS));
    }

    #[test]
    fn window_keeps_the_last_rounds() {
        let mut w: VecDeque<Vec<u32>> = VecDeque::new();
        push_window(&mut w, vec![1], 2);
        push_window(&mut w, vec![2, 2], 2);
        assert_eq!(w.iter().flatten().count(), 3);
        // A third round drops the oldest; every round's data enters,
        // whatever its gate decided.
        push_window(&mut w, vec![3], 2);
        assert_eq!(w, VecDeque::from(vec![vec![2, 2], vec![3]]));
    }

    #[test]
    fn gate_promotes_a_clear_winner() {
        let scores: Vec<f32> = (0..200).map(|i| if i % 2 == 0 { 1.0 } else { 0.5 }).collect();
        let (g, share, z) = gate_decision(&scores, 2.0, false);
        assert_eq!(g, Gate::Promote);
        assert!((share - 0.75).abs() < 1e-9);
        assert!(z > 2.0);
    }

    #[test]
    fn gate_rejects_a_net_behind() {
        let scores: Vec<f32> = (0..200).map(|i| if i % 2 == 0 { 0.0 } else { 0.5 }).collect();
        assert_eq!(gate_decision(&scores, 2.0, false).0, Gate::Reject);
    }

    #[test]
    fn gate_continues_when_undecided_and_rejects_at_the_cap() {
        // Share just above ½, z ≈ 0.05: keep playing, or give up at the cap.
        let mut scores: Vec<f32> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        scores.push(0.75);
        let (g, _, z) = gate_decision(&scores, 2.0, false);
        assert!(z > 0.0 && z < 2.0, "z = {z}");
        assert_eq!(g, Gate::Continue);
        assert_eq!(gate_decision(&scores, 2.0, true).0, Gate::Reject);
    }

    #[test]
    fn gate_rejects_zero_variance() {
        assert_eq!(gate_decision(&[0.5; 10], 2.0, false).0, Gate::Reject);
        assert_eq!(gate_decision(&[0.5], 2.0, false).0, Gate::Reject);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib --release win_pi 2>&1 | tail -20`
Expected: compile errors — `round_seed`, `push_window`, `gate_decision`, `Gate`, `POS` not found.

- [ ] **Step 3: Implement the helpers**

Insert above `#[cfg(test)]` in `src/dqn/win_pi.rs`:

```rust
/// Seed purposes for [`round_seed`].
pub const POS: u64 = 1;
pub const FIT: u64 = 2;
pub const GATE: u64 = 3;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The seed for one purpose (positions, fit, gate) of one round, all
/// derived from the run's `--seed`.
pub fn round_seed(seed: u64, round: usize, purpose: u64) -> u64 {
    splitmix64(seed ^ splitmix64(((round as u64) << 8) | purpose))
}

/// Append a round's data and keep only the last `keep` rounds.
pub fn push_window<T>(window: &mut VecDeque<Vec<T>>, round: Vec<T>, keep: usize) {
    window.push_back(round);
    while window.len() > keep {
        window.pop_front();
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gate {
    Promote,
    Reject,
    Continue,
}

/// The sequential gate's decision over the pair scores so far (each in
/// [0, 1], the candidate's mean over a pair's two seats): returns the
/// decision, the share and z = (share − ½) / (pair-score std / √pairs).
/// Promote when z > `z_min`; reject when z < 0, at the cap (`at_max`), or
/// when the pair scores have no spread (z is then reported as 0).
pub fn gate_decision(scores: &[f32], z_min: f64, at_max: bool) -> (Gate, f64, f64) {
    let n = scores.len() as f64;
    let share = scores.iter().map(|&s| s as f64).sum::<f64>() / n.max(1.0);
    if scores.len() < 2 {
        return (Gate::Reject, share, 0.0);
    }
    let var = scores.iter().map(|&s| (s as f64 - share).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (var / n).sqrt();
    if se == 0.0 {
        return (Gate::Reject, share, 0.0);
    }
    let z = (share - 0.5) / se;
    let g = if z > z_min {
        Gate::Promote
    } else if z < 0.0 || at_max {
        Gate::Reject
    } else {
        Gate::Continue
    };
    (g, share, z)
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib --release win_pi 2>&1 | tail -20`
Expected: `test result: ok. 6 passed` (plus no new warnings other than the pre-existing unused `double` in `src/state.rs`).

- [ ] **Step 5: Format and commit**

```bash
rustfmt --edition 2021 src/dqn/win_pi.rs
git add src/dqn/win_pi.rs src/dqn/mod.rs
git commit -m "feat(win): win_pi helpers: round seeds, data window, sequential gate decision

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Paired head-to-head and the sequential gate

**Files:**
- Modify: `src/dqn/win_pi.rs`

**Interfaces:**
- Consumes: `gate_decision`, `Gate` (Task 1); `crate::dqn::win_train::outcome(me: isize, opp: isize) -> f32`.
- Produces:
  - `pub fn head_to_head<A: Bot + Clone + 'static, B: Bot + Clone + 'static>(a: &(dyn Fn() -> A + Sync), b: &(dyn Fn() -> B + Sync), pairs: std::ops::Range<u64>, seed: u64) -> Vec<f32>` — one score per pair: `a`'s mean outcome over its two seats.
  - `pub struct GateResult { pub promoted: bool, pub games: usize, pub share: f64, pub z: f64 }`
  - `pub fn run_gate<A: Bot + Clone + 'static, B: Bot + Clone + 'static>(cand: &(dyn Fn() -> A + Sync), champ: &(dyn Fn() -> B + Sync), seed: u64, z_min: f64, block: usize, max: usize) -> GateResult`

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module of `src/dqn/win_pi.rs`:

```rust
    use crate::dqn::win::{WinModelConfig, WinStrategy};
    use crate::dqn::MyBackend;
    use burn::prelude::*;

    fn tiny_net(seed: u64) -> WinStrategy {
        let device = Default::default();
        <MyBackend as Backend>::seed(&device, seed);
        WinStrategy::from_model(&WinModelConfig::new().with_width(4).init::<MyBackend>(&device))
    }

    #[test]
    fn a_bot_against_itself_scores_exactly_half_per_pair() {
        let a = tiny_net(1);
        let scores = head_to_head(&|| a.clone(), &|| a.clone(), 0..8, 7);
        assert_eq!(scores, vec![0.5; 8]);
    }

    #[test]
    fn head_to_head_is_deterministic_and_split_independent() {
        let (a, b) = (tiny_net(1), tiny_net(2));
        let whole = head_to_head(&|| a.clone(), &|| b.clone(), 0..12, 7);
        assert_eq!(whole, head_to_head(&|| a.clone(), &|| b.clone(), 0..12, 7));
        let mut split = head_to_head(&|| a.clone(), &|| b.clone(), 0..5, 7);
        split.extend(head_to_head(&|| a.clone(), &|| b.clone(), 5..12, 7));
        assert_eq!(whole, split);
        assert!(whole.iter().any(|&s| s != 0.5), "different nets should differ somewhere");
        // Swapping the bots mirrors every pair score.
        let mirrored = head_to_head(&|| b.clone(), &|| a.clone(), 0..12, 7);
        for (x, y) in whole.iter().zip(&mirrored) {
            assert!((x + y - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn gate_on_equal_bots_rejects_after_one_block() {
        let a = tiny_net(1);
        let r = run_gate(&|| a.clone(), &|| a.clone(), 7, 2.0, 10, 40);
        assert!(!r.promoted);
        assert_eq!(r.games, 10);
    }

    #[test]
    fn gate_never_plays_past_its_cap() {
        let (a, b) = (tiny_net(1), tiny_net(2));
        // z_min unreachable and the cap not a multiple of the block: the
        // gate must stop at exactly `max` games unless it rejects earlier.
        let r = run_gate(&|| a.clone(), &|| b.clone(), 7, f64::INFINITY, 8, 14);
        assert!(!r.promoted);
        assert!(r.games <= 14 && r.games % 2 == 0, "games = {}", r.games);
        // Stopping before the cap is only allowed on a reject (z < 0, or
        // z reported as 0 for zero spread).
        assert!(r.games == 14 || r.z <= 0.0, "games = {}, z = {}", r.games, r.z);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib --release win_pi 2>&1 | tail -20`
Expected: compile errors — `head_to_head`, `run_gate` not found.

- [ ] **Step 3: Implement**

Add to the imports at the top of `src/dqn/win_pi.rs`:

```rust
use crate::dqn::win_train::outcome;
use crate::game::{Game, Player};
use crate::strategy::{Bot, Strategy};
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;
```

Add below `gate_decision`:

```rust
/// Paired games between `a` and `b`, one score per pair index: both games
/// of a pair share the dice streams (`seed`, pair index, seat), `a` in seat
/// 0 then seat 1, and the score is `a`'s mean outcome (win 1, tie ½). The
/// dice depend only on `seed` and the pair index, so splitting `pairs`
/// into blocks or changing the thread count never changes a score. Bots
/// are built once per worker thread.
pub fn head_to_head<A: Bot + Clone + 'static, B: Bot + Clone + 'static>(
    a: &(dyn Fn() -> A + Sync),
    b: &(dyn Fn() -> B + Sync),
    pairs: std::ops::Range<u64>,
    seed: u64,
) -> Vec<f32> {
    pairs
        .into_par_iter()
        .map_init(
            || (a(), b()),
            |(a, b), p| {
                let dice = |seat: u64| Box::new(SmallRng::seed_from_u64(seed.wrapping_add(p * 2 + seat)));
                let mut total = 0.0;
                for rot in 0..2 {
                    let x: Box<dyn Strategy> = Box::new(a.clone());
                    let y: Box<dyn Strategy> = Box::new(b.clone());
                    let players = if rot == 0 {
                        vec![Player::new(x, dice(0)), Player::new(y, dice(1))]
                    } else {
                        vec![Player::new(y, dice(0)), Player::new(x, dice(1))]
                    };
                    let mut game = Game::new(players);
                    game.play();
                    let s: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
                    total += outcome(s[rot], s[1 - rot]);
                }
                total / 2.0
            },
        )
        .collect()
}

pub struct GateResult {
    pub promoted: bool,
    pub games: usize,
    pub share: f64,
    pub z: f64,
}

/// The spec's sequential gate: play `block` games at a time (the last
/// block cut so the total never exceeds `max`) and stop as soon as
/// [`gate_decision`] promotes or rejects.
pub fn run_gate<A: Bot + Clone + 'static, B: Bot + Clone + 'static>(
    cand: &(dyn Fn() -> A + Sync),
    champ: &(dyn Fn() -> B + Sync),
    seed: u64,
    z_min: f64,
    block: usize,
    max: usize,
) -> GateResult {
    let (block_pairs, max_pairs) = ((block / 2) as u64, (max / 2) as u64);
    let mut scores: Vec<f32> = Vec::new();
    loop {
        let start = scores.len() as u64;
        let end = (start + block_pairs).min(max_pairs);
        scores.extend(head_to_head(cand, champ, start..end, seed));
        let (g, share, z) = gate_decision(&scores, z_min, end >= max_pairs);
        if g != Gate::Continue {
            return GateResult {
                promoted: g == Gate::Promote,
                games: scores.len() * 2,
                share,
                z,
            };
        }
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib --release win_pi 2>&1 | tail -20`
Expected: `test result: ok. 10 passed`.
If `a_bot_against_itself_scores_exactly_half_per_pair` fails, the win bot is not deterministic for fixed dice — stop and investigate with superpowers:systematic-debugging (the gate's pairing assumes it); do not loosen the test.

- [ ] **Step 5: Format and commit**

```bash
rustfmt --edition 2021 src/dqn/win_pi.rs
git add src/dqn/win_pi.rs
git commit -m "feat(win): paired head-to-head bench and the sequential promotion gate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Config, validation and the policy-iteration loop

**Files:**
- Modify: `src/dqn/win_pi.rs`

**Interfaces:**
- Consumes: Task 1 and 2 items; from `win_train.rs`: `pub type Ad`, `train_epochs(model: WinModel<Ad>, data: &[([f32; WIN_INPUTS], f32)], epochs: usize, lr: f64, seed: u64) -> (WinModel<Ad>, f32)`, `bench_vs_pair(win: &WinStrategy, pair_dir: &str, games: usize) -> f64`, `rollout_positions_by<P: Bot + Clone + 'static>(count: usize, seed: u64, player: &(dyn Fn() -> P + Sync), pair_dir: &str, k: usize, blend: f32) -> Vec<([f32; WIN_INPUTS], f32)>`; `PairStrategy::load(dir: &str) -> PairStrategy` (Clone); `WinStrategy::from_model(&WinModel<MyBackend>)`.
- Produces:
  - `pub struct WinPiCfg { pub dir: String, pub init: Option<String>, pub width: usize, pub rounds: usize, pub positions: usize, pub rollouts: usize, pub window: usize, pub epochs: usize, pub lr: f64, pub gate_z: f64, pub gate_block: usize, pub gate_max: usize, pub master_games: usize, pub seed: u64, pub pair_dir: String }`
  - `impl WinPiCfg { pub fn validate(&self) -> Result<(), String> }`
  - `pub const FRESH_LR: f64 = 1e-3;`
  - `pub fn policy_iteration(cfg: &WinPiCfg) -> Option<WinStrategy>` — the final champion; `None` when round 1 (no `--init`) fails to beat Master.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module:

```rust
    use crate::dqn::win::WinModel;

    fn tmp(name: &str) -> String {
        let d = std::env::temp_dir().join(format!("qwixxer-win-pi-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d.to_string_lossy().into_owned()
    }

    fn tiny_cfg(dir: String, init: Option<String>) -> WinPiCfg {
        WinPiCfg {
            dir,
            init,
            width: 4,
            rounds: 2,
            positions: 30,
            rollouts: 2,
            window: 2,
            epochs: 1,
            lr: 3e-4,
            gate_z: 2.0,
            gate_block: 4,
            gate_max: 8,
            master_games: 2,
            seed: 42,
            pair_dir: "pair_model".into(),
        }
    }

    #[test]
    fn validate_refuses_bad_settings_before_any_work() {
        let base = tmp("validate");
        let ok = tiny_cfg(format!("{base}/run"), None);
        assert_eq!(ok.validate(), Ok(()));
        let err = |c: WinPiCfg| c.validate().unwrap_err();
        assert!(err(WinPiCfg { rollouts: 0, ..tiny_cfg(ok.dir.clone(), None) }).contains("--rollouts"));
        assert!(err(WinPiCfg { window: 0, ..tiny_cfg(ok.dir.clone(), None) }).contains("--window"));
        assert!(err(WinPiCfg { rounds: 0, ..tiny_cfg(ok.dir.clone(), None) }).contains("--rounds"));
        assert!(err(WinPiCfg { gate_block: 3, ..tiny_cfg(ok.dir.clone(), None) }).contains("--gate-block"));
        assert!(err(WinPiCfg { gate_max: 2, ..tiny_cfg(ok.dir.clone(), None) }).contains("--gate-max"));
        assert!(err(tiny_cfg(ok.dir.clone(), Some(format!("{base}/nope")))).contains("--init"));
        assert!(err(WinPiCfg { pair_dir: format!("{base}/nope"), ..tiny_cfg(ok.dir.clone(), None) }).contains("pair model"));
        std::fs::create_dir_all(&ok.dir).unwrap();
        assert!(err(tiny_cfg(ok.dir.clone(), None)).contains("already exists"));
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn two_rounds_from_an_init_net_write_the_run() {
        let base = tmp("smoke");
        let init = format!("{base}/init");
        let device = Default::default();
        <MyBackend as Backend>::seed(&device, 3);
        let mcfg = WinModelConfig::new().with_width(4);
        let model: WinModel<MyBackend> = mcfg.init(&device);
        save_win(&init, &mcfg, &model);
        let cfg = tiny_cfg(format!("{base}/run"), Some(init));
        assert!(policy_iteration(&cfg).is_some());
        for d in ["round-1", "round-2", "champion"] {
            assert!(std::path::Path::new(&format!("{}/{d}/model.mpk", cfg.dir)).exists(), "{d}");
            assert!(std::path::Path::new(&format!("{}/{d}/config.json", cfg.dir)).exists(), "{d}");
        }
        let csv = std::fs::read_to_string(format!("{}/training.csv", cfg.dir)).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "round,positions,loss,gate_games,share_vs_champion,z,promoted,share_vs_master,elapsed_s"
        );
        assert_eq!(lines.len(), 4, "{csv}");
        assert!(lines[1].starts_with("0,,,,,,,"), "{}", lines[1]);
        for (i, row) in lines[2..].iter().enumerate() {
            let f: Vec<&str> = row.split(',').collect();
            assert_eq!(f.len(), 9, "{row}");
            assert_eq!(f[0], (i + 1).to_string());
            assert!(f[6] == "0" || f[6] == "1", "{row}");
            assert!(!f[7].is_empty(), "{row}");
        }
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn a_fresh_start_that_loses_to_master_stops_cleanly() {
        let base = tmp("fresh");
        // A barely trained W=4 net cannot beat Master over 8 games with z > 2.
        let cfg = WinPiCfg { master_games: 0, ..tiny_cfg(format!("{base}/run"), None) };
        assert!(policy_iteration(&cfg).is_none());
        assert!(std::path::Path::new(&format!("{}/round-1/model.mpk", cfg.dir)).exists());
        assert!(!std::path::Path::new(&format!("{}/champion", cfg.dir)).exists());
        let csv = std::fs::read_to_string(format!("{}/training.csv", cfg.dir)).unwrap();
        let rows: Vec<&str> = csv.lines().skip(1).collect();
        assert_eq!(rows.len(), 1, "{csv}");
        assert_eq!(rows[0].split(',').nth(6), Some("0"));
        std::fs::remove_dir_all(&base).unwrap();
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib --release win_pi 2>&1 | tail -20`
Expected: compile errors — `WinPiCfg`, `policy_iteration`, `save_win` not found.

- [ ] **Step 3: Implement**

Add to the imports:

```rust
use crate::dqn::pair::PairStrategy;
use crate::dqn::win::{WinModel, WinModelConfig, WinStrategy};
use crate::dqn::win_train::{bench_vs_pair, rollout_positions_by, train_epochs, Ad};
use crate::dqn::MyBackend;
use burn::{module::AutodiffModule, prelude::*, record::CompactRecorder};
use std::path::Path;
```

(merge with the existing `use crate::dqn::win_train::outcome;` into one `use crate::dqn::win_train::{…}` line.)

Add below `run_gate`:

```rust
/// Learning rate of a from-scratch round 1 (no `--init`), as `win-fit`.
pub const FRESH_LR: f64 = 1e-3;

pub struct WinPiCfg {
    pub dir: String,
    /// Starting champion (a win model dir); `None` = Master plays round 1.
    pub init: Option<String>,
    /// Width of a from-scratch net; ignored with `init`.
    pub width: usize,
    pub rounds: usize,
    /// New turn-end positions per round.
    pub positions: usize,
    /// Rollouts per position.
    pub rollouts: usize,
    /// Rounds of positions the fit trains on.
    pub window: usize,
    pub epochs: usize,
    /// Warm-start learning rate (rounds with a win-net champion).
    pub lr: f64,
    pub gate_z: f64,
    pub gate_block: usize,
    pub gate_max: usize,
    /// Yardstick games against Master per round (0 = none).
    pub master_games: usize,
    pub seed: u64,
    pub pair_dir: String,
}

impl WinPiCfg {
    /// Every check that can fail, before any work.
    pub fn validate(&self) -> Result<(), String> {
        if self.rounds == 0 {
            return Err("--rounds must be at least 1".into());
        }
        if self.rollouts == 0 {
            return Err("--rollouts must be at least 1".into());
        }
        if self.window == 0 {
            return Err("--window must be at least 1".into());
        }
        if self.gate_block < 2 || self.gate_block % 2 != 0 {
            return Err("--gate-block must be even and at least 2".into());
        }
        if self.gate_max < self.gate_block {
            return Err("--gate-max must be at least --gate-block".into());
        }
        if !Path::new(&format!("{}/model.mpk", self.pair_dir)).exists() {
            return Err(format!("pair model not found at {}/model.mpk", self.pair_dir));
        }
        if let Some(init) = &self.init {
            for f in ["config.json", "model.mpk"] {
                if !Path::new(&format!("{init}/{f}")).exists() {
                    return Err(format!("--init {init}: {f} not found"));
                }
            }
        }
        if Path::new(&self.dir).exists() {
            return Err(format!("{} already exists: move it aside or pick another --dir", self.dir));
        }
        Ok(())
    }
}

/// Write a loadable win model dir (`config.json`, `model.mpk`).
pub fn save_win(dir: &str, mcfg: &WinModelConfig, model: &WinModel<MyBackend>) {
    std::fs::create_dir_all(dir).expect("Failed to create the win model dir");
    mcfg.save(format!("{dir}/config.json")).expect("Failed to save win model config");
    model
        .clone()
        .save_file(format!("{dir}/model"), &CompactRecorder::new())
        .expect("Failed to save win model");
}

fn load_win_ad(dir: &str) -> (WinModelConfig, WinModel<Ad>) {
    let device = Default::default();
    let mcfg = WinModelConfig::load(format!("{dir}/config.json")).expect("Failed to load win model config");
    let model = mcfg
        .init::<Ad>(&device)
        .load_file(format!("{dir}/model"), &CompactRecorder::new(), &device)
        .expect("Failed to load win model");
    (mcfg, model)
}

/// The spec's loop. Returns the final champion, or `None` when a
/// from-scratch round 1 fails to beat Master (nothing to continue from).
pub fn policy_iteration(cfg: &WinPiCfg) -> Option<WinStrategy> {
    if let Err(e) = cfg.validate() {
        panic!("{e}");
    }
    let device = Default::default();
    let start = std::time::Instant::now();
    std::fs::create_dir_all(&cfg.dir).expect("Failed to create the run dir");
    let log_path = format!("{}/training.csv", cfg.dir);
    std::fs::write(
        &log_path,
        "round,positions,loss,gate_games,share_vs_champion,z,promoted,share_vs_master,elapsed_s\n",
    )
    .expect("Failed to write log");
    let log = |row: String| {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&log_path)
            .expect("Failed to open log");
        writeln!(f, "{row}").ok();
    };
    let pct = |s: Option<f64>| s.map(|s| format!("{:.2}%", s * 100.0)).unwrap_or_else(|| "-".into());
    let csv = |s: Option<f64>| s.map(|s| format!("{s:.4}")).unwrap_or_default();
    let yardstick = |m: &WinModel<Ad>| {
        (cfg.master_games > 0)
            .then(|| bench_vs_pair(&WinStrategy::from_model(&m.valid()), &cfg.pair_dir, cfg.master_games))
    };
    let champion_dir = format!("{}/champion", cfg.dir);

    // `None` = Master is the champion (round 1 of a from-scratch run).
    let (mcfg, mut champion): (WinModelConfig, Option<WinModel<Ad>>) = match &cfg.init {
        Some(dir) => {
            let (mcfg, model) = load_win_ad(dir);
            (mcfg, Some(model))
        }
        None => (WinModelConfig::new().with_width(cfg.width), None),
    };
    if let Some(m) = &champion {
        save_win(&champion_dir, &mcfg, &m.valid());
        if cfg.master_games > 0 {
            let s = yardstick(m);
            println!("Round  0 (init): vs Master {}", pct(s));
            log(format!("0,,,,,,,{},{}", csv(s), start.elapsed().as_secs()));
        }
    }

    let mut window: VecDeque<Vec<([f32; crate::dqn::win::WIN_INPUTS], f32)>> = VecDeque::new();
    for round in 1..=cfg.rounds {
        // 1. Generate: the champion's own turn ends, labelled by its rollouts.
        let pos_seed = round_seed(cfg.seed, round, POS);
        let data = match &champion {
            None => rollout_positions_by(
                cfg.positions,
                pos_seed,
                &|| PairStrategy::load(&cfg.pair_dir),
                &cfg.pair_dir,
                cfg.rollouts,
                1.0,
            ),
            Some(m) => {
                let win = WinStrategy::from_model(&m.valid());
                rollout_positions_by(cfg.positions, pos_seed, &|| win.clone(), &cfg.pair_dir, cfg.rollouts, 1.0)
            }
        };
        // 2. Window: every round's data enters, whatever its gate decides.
        push_window(&mut window, data, cfg.window);
        let train: Vec<_> = window.iter().flatten().copied().collect();

        // 3. Fit: warm start from the champion, or from scratch against Master.
        let fit_seed = round_seed(cfg.seed, round, FIT);
        let (init_model, lr) = match &champion {
            Some(m) => (m.clone(), cfg.lr),
            None => {
                <Ad as Backend>::seed(&device, fit_seed);
                (mcfg.init::<Ad>(&device), FRESH_LR)
            }
        };
        let (candidate, loss) = train_epochs(init_model, &train, cfg.epochs, lr, fit_seed);
        save_win(&format!("{}/round-{round}", cfg.dir), &mcfg, &candidate.valid());

        // 4. Gate: candidate vs champion, paired, sequential.
        let cand = WinStrategy::from_model(&candidate.valid());
        let gate_seed = round_seed(cfg.seed, round, GATE);
        let g = match &champion {
            None => run_gate(
                &|| cand.clone(),
                &|| PairStrategy::load(&cfg.pair_dir),
                gate_seed,
                cfg.gate_z,
                cfg.gate_block,
                cfg.gate_max,
            ),
            Some(m) => {
                let champ = WinStrategy::from_model(&m.valid());
                run_gate(&|| cand.clone(), &|| champ.clone(), gate_seed, cfg.gate_z, cfg.gate_block, cfg.gate_max)
            }
        };
        let verdict = if g.promoted { "promoted" } else { "kept" };
        let row = |master: Option<f64>| {
            format!(
                "{round},{},{loss:.5},{},{:.4},{:.2},{},{},{}",
                train.len(),
                g.games,
                g.share,
                g.z,
                g.promoted as u8,
                csv(master),
                start.elapsed().as_secs()
            )
        };
        if g.promoted {
            save_win(&champion_dir, &mcfg, &candidate.valid());
            champion = Some(candidate);
        } else if champion.is_none() {
            println!(
                "Round {round:>2}: fresh fit lost the gate to Master ({:.2}%, z {:.2}): no win net to continue from",
                g.share * 100.0,
                g.z
            );
            log(row(None));
            return None;
        }

        // 5. Yardstick (logged only) and 6. log.
        let master = yardstick(champion.as_ref().expect("a win-net champion after round 1"));
        println!(
            "Round {round:>2}: {} positions, loss {loss:.4}, gate {} games {:.2}% z {:.2} → {verdict}, champion vs Master {}, {}s",
            train.len(),
            g.games,
            g.share * 100.0,
            g.z,
            pct(master),
            start.elapsed().as_secs()
        );
        log(row(master));
    }
    champion.map(|m| WinStrategy::from_model(&m.valid()))
}
```

Note: `positions` in the CSV is the number of positions the round's fit trained on (the whole window), the most useful count for reading the loss.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib --release win_pi 2>&1 | tail -20`
Expected: `test result: ok. 13 passed`.
If `a_fresh_start_that_loses_to_master_stops_cleanly` reports a promotion (z > 2 over 4 pairs against Master is not expected from a W=4 net trained one epoch on ~30 positions), report it rather than changing the thresholds.

- [ ] **Step 5: Run the whole library suite**

Run: `cargo test --lib --release 2>&1 | tail -5`
Expected: `test result: ok. 146 passed` (133 before + 13 new). Read the output; any failure goes in the report by name.

- [ ] **Step 6: Format and commit**

```bash
rustfmt --edition 2021 src/dqn/win_pi.rs
git add src/dqn/win_pi.rs
git commit -m "feat(win): policy_iteration: rollout-labelled rounds, windowed warm-start fits, gated promotion

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The `win-pi` CLI subcommand

**Files:**
- Modify: `src/main.rs` — `Commands` enum (after `WinFit { … }`, ~line 403) and the dispatch `match` (after the `Some(Commands::WinFit { … })` arm, ~line 925)

**Interfaces:**
- Consumes: `dqn::win_pi::{WinPiCfg, policy_iteration}` (Task 3).

- [ ] **Step 1: Add the subcommand**

In the `Commands` enum, after the `WinFit { … },` variant:

```rust
    /// Win-net policy iteration: rollout labels from the champion's own games, gated promotion
    #[cfg(feature = "dqn")]
    WinPi {
        /// Output dir (must not exist): round-<i>/, champion/, training.csv
        #[arg(long, default_value = "win_pi")]
        dir: String,
        /// Starting champion (a win model dir, e.g. win_model); without it Master plays round 1
        #[arg(long)]
        init: Option<String>,
        /// Width of a from-scratch net (ignored with --init)
        #[arg(long, default_value = "64")]
        width: usize,
        #[arg(long, default_value = "10")]
        rounds: usize,
        /// New turn-end positions per round
        #[arg(long, default_value = "300000")]
        positions: usize,
        /// Rollouts per position
        #[arg(long, default_value = "64")]
        rollouts: usize,
        /// Rounds of positions each fit trains on
        #[arg(long, default_value = "2")]
        window: usize,
        /// Epochs per fit
        #[arg(long, default_value = "20")]
        epochs: usize,
        /// Warm-start learning rate (a from-scratch round 1 uses 1e-3)
        #[arg(long, default_value = "3e-4")]
        lr: f64,
        /// Promote when z = (share − ½)/se exceeds this
        #[arg(long, default_value = "2.0")]
        gate_z: f64,
        /// Gate games per block (even)
        #[arg(long, default_value = "100000")]
        gate_block: usize,
        /// Gate games at most
        #[arg(long, default_value = "1000000")]
        gate_max: usize,
        /// Champion vs Master games per round, logged only (0 = skip)
        #[arg(long, default_value = "200000")]
        master_games: usize,
        #[arg(long, default_value = "42")]
        seed: u64,
    },
```

In the dispatch `match`, after the `WinFit` arm:

```rust
        #[cfg(feature = "dqn")]
        Some(Commands::WinPi {
            dir,
            init,
            width,
            rounds,
            positions,
            rollouts,
            window,
            epochs,
            lr,
            gate_z,
            gate_block,
            gate_max,
            master_games,
            seed,
        }) => {
            let cfg = dqn::win_pi::WinPiCfg {
                dir,
                init,
                width,
                rounds,
                positions,
                rollouts,
                window,
                epochs,
                lr,
                gate_z,
                gate_block,
                gate_max,
                master_games,
                seed,
                pair_dir: "pair_model".into(),
            };
            if let Err(e) = cfg.validate() {
                eprintln!("win-pi: {e}");
                std::process::exit(2);
            }
            dqn::win_pi::policy_iteration(&cfg);
        }
```

- [ ] **Step 2: Build and check the help text**

Run: `cargo run --release -- win-pi --help 2>&1 | tail -40`
Expected: the flags above with their defaults (`--gate-z`, `--gate-block`, `--gate-max`, `--master-games`, …).

- [ ] **Step 3: Check refusal**

Run: `cargo run --release -- win-pi --init /nonexistent 2>&1 | tail -3; echo "exit $?"`
Expected: `win-pi: --init /nonexistent: config.json not found`. (The exit code printed is the pipe's; the message is what matters.)

- [ ] **Step 4: Smoke run from the user's current best net (read-only)**

Run (SCRATCH = the session scratchpad dir):
```bash
cargo run --release -- win-pi --init win_model --rounds 2 --positions 200 --rollouts 2 --epochs 1 \
  --gate-block 20 --gate-max 40 --master-games 20 --dir $SCRATCH/pi_smoke 2>&1 | tail -5
cat $SCRATCH/pi_smoke/training.csv; ls $SCRATCH/pi_smoke
```
Expected: a `Round  0 (init)` line, two `Round  N:` lines; `training.csv` with header + 3 rows; dirs `champion round-1 round-2`. `win_model/` is only read — `git status` still shows it unchanged and untracked.

- [ ] **Step 5: Run the whole library suite, format, commit**

Run: `cargo test --lib --release 2>&1 | tail -5`
Expected: `test result: ok. 146 passed`. Read it before committing.

```bash
rustfmt --edition 2021 src/main.rs
git add src/main.rs
git commit -m "feat(win): win-pi subcommand for gated policy iteration

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
