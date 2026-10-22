# Qwixxer Architecture

## Project Structure

Qwixxer is a Rust workspace with a native binary/library crate and a WASM web crate.

```
qwixxer/
  Cargo.toml          -- lib + bin crate, feature flags: dqn, parallel
  src/
    lib.rs             -- Public module re-exports
    main.rs            -- CLI entry point (clap subcommands: play, bench, solo, evolve, dqn-train, dqn-selfplay)
    state.rs           -- Game state, rows, marks, moves, scoring, display
    game.rs            -- Game loop, players, DiceSource trait
    strategy/mod.rs    -- Strategy trait + Interactive, Conservative, Opportunist, Random
    bot.rs             -- Genetic algorithm: DNA, genes, Population, evolution
    dqn/
      mod.rs           -- DQN model, feature extraction, DqnStrategy, P(win) ranking helpers
      train.rs         -- TD(lambda) targets, decoupled mu/sigma losses, self-play loop
    mcts.rs            -- Monte Carlo tree search with pluggable rollout policy
  web/
    Cargo.toml         -- WASM crate (qwixxer-web), depends on qwixxer with default-features = false
    build.sh           -- wasm-pack + wasm-opt -Oz into web/pkg/
    src/
      lib.rs           -- wasm-bindgen exports: WasmGame (one game, JSON in/out)
      session.rs       -- Session: one human-vs-bot game, turn flow, snapshots, views
      selection.rs     -- The human's legal selections (passive white sum, active options)
      bots.rs          -- Master+ / Master / Student / Newbie, embedded champion + pair net
      bench.rs         -- bench(bot, games, seed): bot decision latency
      main.ts          -- Mounts App.svelte
      engine/
        protocol.ts    -- TS mirror of the session's JSON types, Engine interface
        client.ts      -- WorkerEngine: promise RPC over postMessage
        engine.worker.ts -- Web Worker entry: wires the wasm pkg into serve.ts
        serve.ts       -- Worker message handler: wasm load (with retry), one WasmGame
      game/
        rules.ts       -- Pure sheet/confirm/log model (lookups over engine data only)
        controller.svelte.ts -- GameController: turn sequencing, pacing, tips, persistence
        keys.ts        -- keyAction: pure page-wide keyboard decisions
        layout.ts      -- phoneLayout / deskScale: pure layout sizing
        *.test.ts      -- Vitest unit tests
      test/client-env.ts -- Vitest environment compiling runes for the client
      ui/              -- Svelte 5 components (App, Desktop, Phone, sheets, dice, log, ...)
      styles/theme.css -- Light/dark tokens, self-hosted Barlow fonts
    scripts/bench.mjs  -- Node runner for the latency bench
    index.html         -- Single page
    vite.config.ts     -- Svelte plugin, ES-module worker, base path /qwixxer/, Vitest config
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `dqn` | Full DQN training (burn with train support, serde, rayon) |
| `parallel` | Rayon parallelism for benchmarks and GA training |

Default features are `dqn` + `parallel` for the native binary. Burn (ndarray backend) is always available for inference. The web crate depends on the root crate with `default-features = false` (no rayon, no training code) to keep the WASM binary small (~2 MB, built for speed with wasm-opt). `mimalloc`, the native binary's global allocator, is a target dependency for non-wasm32 only (its C build fails on wasm32).

---

## Game Engine

### Core Types (`state.rs`)

**`State`** -- A player's game state: 4 rows + a strike count (0-4).

**`Row`** (private) -- Per-row state with three fields:
- `ascending: bool` -- Red/Yellow ascend (2 to 12); Green/Blue descend (12 to 2)
- `total: u8` -- Number of marks in the row (0-12; locking adds a bonus mark)
- `free: Option<u8>` -- Next markable number, or `None` if locked

**`Mark`** -- A specific marking action: `{ row: usize, number: u8 }`.

**`Move`** -- One of three variants:
- `Strike` -- Take a penalty (-5 points)
- `Single(Mark)` -- Mark one number
- `Double(Mark, Mark)` -- Mark two numbers (white sum + color sum, active turn only)

### Move Generation

`State::generate_moves(dice: [u8; 6])` produces all legal active-player moves. The 6 dice are ordered `[W1, W2, R, Y, G, B]`. The method:
1. Computes the white sum (`W1 + W2`)
2. For each row where `can_mark(white_sum)` is true, generates a white-only `Single`
3. For each row `i` and each white die `w`, checks if `w + color[i]` is markable, generating color-only `Single` moves
4. Cross-products white moves with color moves on different rows for `Double` moves, plus same-row doubles where marking white first opens up a color mark
5. Deduplicates via `itertools::unique()`
6. Does NOT include `Strike` -- callers add it themselves

`State::generate_opponent_moves(number: u8)` produces only `Single` marks for the white sum. Passive players cannot use colored dice and can legally skip without taking a strike.

### Row Locking Rules

A row can be locked by marking its terminal number (12 for ascending, 2 for descending), but only if the player already has 5+ marks in that row. Locking adds a bonus mark (`total += 2`: one for the terminal number, one for the lock). A locked row has `free = None` and can no longer be marked by anyone.

### Scoring

Triangular number scoring: `n` marks = `n * (n + 1) / 2` points. Each strike subtracts 5. Maximum per row is 12 marks (including lock bonus) = 78 points.

### State Metrics

- `blanks()` -- Skipped positions across all unlocked rows (marks that were jumped over)
- `probability()` -- Probability that two random dice sum to any free value (for passive turns)
- `probability_n(n)` -- Probability that at least one of `n` opponents rolls a usable sum

### Game Loop (`game.rs`)

**`Player`** wraps a `Strategy`, a `State`, and a `DiceSource`. Each player has their own dice (supports manual input for humans, `SmallRng` for bots).

**`DiceSource`** trait with one method: `fn roll(&mut self) -> [u8; 6]`. Implemented for `SmallRng` (random) and `ManualDice` (stdin).

**`Game::play()`** runs the full game loop:

```
while !game_over():
    active_player rolls dice
    notify active_player of opponent states (observe_opponents)
    active_player picks a move (your_move) -> apply

    for each other player (in seat order):
        notify of opponent states (observe_opponents)
        passive player optionally marks white sum (opponents_move) -> apply
        accumulate locks from passive player's state

    propagate ALL accumulated locks to ALL players
    advance active_player to next seat
```

**Game-over conditions** (checked before each turn):
1. Any player has 4 strikes
2. Any player has 2+ locked rows

The game rotates the active player each turn. `Game::new_from_turn()` allows starting mid-game from a specific seat (used by MCTS rollouts).

---

## Strategy Trait (`strategy/mod.rs`)

```rust
pub trait Strategy: std::fmt::Debug {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move;
    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move>;
    fn is_interactive(&self) -> bool { false }
    fn observe_opponents(&mut self, _our_score: isize, _opponents: &[State]) {}
}
```

- `your_move` -- Called when this player is active. Receives the full 6-die roll. Must return a `Move` (including `Strike`).
- `opponents_move` -- Called when another player is active. Receives just the white sum and the current global lock state. Returns `Some(Move::Single(...))` to mark, or `None` to skip. No penalty for skipping.
- `is_interactive` -- Returns `true` for the human player (suppresses verbose logging).
- `observe_opponents` -- Called before each decision with the player's current score and all opponents' states. Default is a no-op; overridden by DQN to track opponent context.

---

## Strategies Implemented

### Hand-Crafted

**Random** -- Picks a uniformly random legal move. Baseline for benchmarking.

**Conservative** -- Minimizes new blanks. A `Bot` on the shared pipeline (so it weighs whole turns, with the meta-rules and pruning) whose value is fewest blanks, then highest score, with a strike costing slightly more than `max_new_blanks` (default 3) blanks; on the opponent's roll the free skip means it marks only without skipping. The web's Newbie. (Before it joined the pipeline it chose the white-sum mark and the colour mark separately and won ~25% vs GA; now ~34%.)

**Opportunist** -- Probability-maximizing with blank discipline. Always locks when possible. Caps blanks (2 on active, 1 on passive), then picks the move maximizing `State::probability()` (the chance of getting a useful white sum on future opponent turns). The strongest hand-crafted strategy.

**Interactive** -- Terminal-based human player. Shows the board, dice, numbered move list, and reads input. Groups moves into "Single" and "Both" sections for readability.

### Genetic Algorithm Champion (`bot.rs`)

**DNA** -- A weight vector over 4 gene functions, evaluated as a linear combination:

| Gene | Function | Champion Weight |
|------|----------|----------------|
| `weighted_prob` | `sum(P(rolling free) * (total + 1))` per active row | +0.560 |
| `strikes` | Strike count | -0.826 |
| `concentration` | `sum(total^2)` across rows | +0.021 |
| `blanks` | Total skipped positions | -0.063 |

The `instinct(&self, state: &State) -> f64` method computes the weighted sum. Weights are L2-normalized.

**Meta-rules**: Before evaluating moves, `State::apply_meta_rules()` applies smart lock, smart strike, and move pruning (shared with DQN). The GA also tracks `score_gap` via `observe_opponents`.

**Move selection**: For active turns, evaluates `instinct()` on each resulting state (including Strike) and picks the maximum. For passive turns, also compares against the skip state (with locks applied) to decide whether marking is worthwhile.

### DQN Neural Net (`dqn/mod.rs`, `dqn/train.rs`)

**Architecture**: 25-input MLP with ReLU activations, two-head output.

```
Input (25) -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(2)
                                                           |    |
                                                           mu   log sigma^2
```

Total parameters: `25*128 + 128 + 128*64 + 64 + 64*2 + 2 = 11,778`.

The model outputs two scalars per state: `mu(s)` (expected final score, TD value learner) and `log sigma^2(s)` (log-variance of the final-score distribution). At inference, `log sigma^2` is clamped to `[LOG_VAR_MIN, LOG_VAR_MAX] = [-5.0, 10.0]`.

**Move ranking via P(win)**: Instead of picking `argmax mu(s')`, the DQN ranks candidates with `win_rank_score(mu_us, log_var_us, mu_opp, log_var_opp) = (mu_us - mu_opp) / sqrt(sigma^2_us + sigma^2_opp)` — monotonic in the Gaussian-approximation probability `P(score_us > score_opp)`. The leading opponent's `(mu_opp, log_var_opp)` is obtained by a forward pass on the opponent's actual state, batched with all our candidates in one forward call.

**Per-candidate opponent context**: Each candidate post-move state rebuilds *both* its own `OpponentContext` (from the post-move our_score) and the opponent's view (from the opponent's score and our post-move state). This matters because context-derived features (score_gap, max_opp_strikes, max_opp_locks, max_opp_total_progress) change across candidates — different moves leave different gaps. Both our and opp feature vectors are batched together (`2N` rows) and evaluated in a single forward pass per decision.

Uses a HashMap cache keyed by quantized features for repeated states.

**Meta-rules** (`State::apply_meta_rules`): Before the model evaluates moves, shared strategic rules are applied:
- **Smart lock**: Always lock to win the game (2+ locked rows and ahead). Always lock the first row. Never lock into a game-ending loss.
- **Smart strike**: With 3 strikes, strike to end the game if ahead. Never strike into a loss (unless forced).
- **Move pruning**: Remove dominated singles (same row: keep closest to free pointer; cross-row: equal blanks + progress, prefer higher total marks).

These meta-rules are shared by DQN, GA, and training code via `State::apply_meta_rules()` and `State::find_smart_lock()`.

**Opponent awareness**: `observe_opponents()` updates an `OpponentContext` with `num_opponents`, `max_opponent_strikes`, and `score_gap_to_leader`, which feed into the last 3 input features.

### Feature Extraction (25 features)

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0-3 | Row progress (how far the free pointer has advanced) | 0.0-1.0, locked=1.0 |
| 4-7 | Row mark counts | /11 |
| 8-11 | Row locked flags | 0 or 1 |
| 12-15 | Per-row weighted probability: `P(rolling free) * (total+1)` | /(6*11) |
| 16 | Our strikes | /3 (max in-play; 4 ends the game) |
| 17 | Our blanks | /40 |
| 18 | Number of opponents | /4 |
| 19 | Score gap to leader | /100, clamped to [-1, 1] |
| 20 | Max strikes across all players (ours + opponents) | /3 |
| 21 | Max locks across all players | capped at 1 (2 ends the game) |
| 22 | Max total row-progress across all players (game-duration proxy) | 0.0-1.0 |
| 23 | Aggregate probability over usable dice sums | 0.0-1.0 |
| 24 | Total lockable rows across all players | /8 |

Features 20-24 are game-end-proximity signals: they let the network predict how much of the game remains, which is the dominant driver of remaining-score variance.

### Pair Network (`dqn/pair.rs`, `dqn/pair_train.rs`)

Experimental successor to the DQN: a joint two-board MLP (45 -> 128 -> 64 -> 2)
that predicts the distribution of the *future score differential*
`final_diff - current_diff` as `(mu, log sigma^2)`. Inputs are two 20-feature
board blocks (per-row progress/marks/locked/weighted-prob with the lock-rule
fix, strikes, blanks, aggregate wprob, lockable rows) plus 5 pair-level
features (current diff, opponent count, all-opponent summaries). Candidates
are ranked by `(current_diff + mu) / sigma` against the leading opponent.
Training is pure self-play with per-opponent TD(lambda=0.8) chains; every
sample is also emitted board-swapped with negated targets (pairwise
consistency); color permutations are applied in the batcher to both blocks.
Passive skips are recorded (the old recorder dropped them). Model dir:
`pair_model/`. Design doc: `docs/superpowers/specs/2026-06-10-pair-network-design.md`.

**Win net** (`src/dqn/win.rs`, `src/dqn/win_train.rs`; experimental,
2-player only) predicts P(win) directly. Inputs are the pair net's two
board blocks regrouped as 4 row tokens (a row's 4 features for each
player) and 9 globals. A shared row encoder sums within the colour pairs,
a shared pair encoder sums over both pairs, and a head outputs g(me, opp);
the logit is z = g(me, opp) − g(opp, me), so P(a ahead) + P(b ahead) = 1
and the 8 colour relabellings leave z unchanged, both exactly. Trained from
scratch by TD(λ = 0.8) self-play on win/loss (tie ½) with targets rebuilt
by the current net every iteration (`win-train`); `win-fit` checks the
width against today's P(win). `WinStrategy` reports values in probit units
(Φ⁻¹ of its P(win)) so the search's gate and blend apply unchanged. Bench
bots `win` / `win-search`, model dir `win_model/`. Design doc:
`docs/superpowers/specs/2026-10-15-win-net-design.md`.

### Pair-Search (`strategy/search.rs`, `strategy/sim.rs`)

Decision-time search on top of the pair network. At gated active decisions
(top-2 static gap below GATE_MARGIN, or endgame proximity: any locked row, 3
strikes, or a locking candidate), the top-2 candidates are re-ranked by
truncated rollouts: complete the current turn deterministically, simulate one
full round with every player played greedily by the value net
(lockstep-batched through `Bot::evaluate_batch_multi`), then score each
sampled future as the exact outcome if the game ended or the net's win
probability (`WinProb` trait) at the horizon. 64 samples per candidate with
common-random-number dice derived by hashing the decision context, so the bot
is stateless-deterministic and benches are byte-reproducible. Meta-rules stay
single-source: the shared bot_impl pipelines expose pure-logic `*_choices()`
consumed by both the blanket `Bot -> Strategy` impl and the search bot. The
simulator is pinned to `Game::play` by an equivalence test covering both
strike- and lock-terminated games. Calibration diagnostic:
`cargo run --release --example search_calibration -- 300`.

Measured (50k-game paired benches): 59.8% vs GA (static pair bot: 59.2%);
head-to-head vs the static pair bot +1.6-1.7% win margin, replicated on two
seeds. Cost: ~100x the static bot (~27 games/s on 8 cores); optimization
deferred until needed (see EXPERIMENTS.md Phase 13).

**Expert iteration** (`pair-train --search`): training games are generated with
the search bot as every player's policy (the e-coin fires before search, so
exploring moves skip it); recording, TD targets, and losses are unchanged — the
net distills search-improved trajectories. Recommended run (executed on a
dedicated machine): 5k games/iter, 5 epochs, 500k-game per-iteration static
benchmark (the primary metric; paired SE ~0.05%). Design doc:
docs/superpowers/specs/2026-06-11-expert-iteration-design.md.

**SearchConfig** (`strategy/search.rs`): `k_samples` (truncated-search
samples), `search_passive` (also search gated passive decisions via
`passive_entry`), `lock_k` (adjudicate forced safe locks against the best
non-lock alternative and runner-up lock by full-game rollouts), `stratify`
(stratified first-roll white dice; did not help), `blend` (pick by
`(1 − λ)·Φ(static z) + λ·rollout mean` instead of the rollout mean alone;
spec `blend=λ` / `noblend`, 0.7 in production since Phase 25), `seq` /
`stop` (sequential sampling: start at `seq` samples per candidate and double
up to `k_samples` until every blended score difference to the static pick
exceeds `stop` × λ·SE of the CRN-paired mean difference; spec `seq=N` /
`noseq`, `stop=C`; production 32 → 512 with stop 2 since Phase 26; with
more than two candidates it is successive elimination: each round, every
candidate the leader beats by `stop` SEs stops sampling), `cands`
(candidates searched, spec `cands=N`, production 3 since Phase 27, always
printed in the spec; `K_CANDIDATES` = 2 stays for `legacy()` and the
endgame gate's lock check) and `gate` (close-gate margin, spec
`gate=G`, default `GATE_MARGIN` = 0.15; distillation keeps the constant).
Rollout sample `k`'s dice depend only on (decision seed, k, player), so
sequential batches compose into exactly the fixed-k sample set.
`Default` = production
= `k=512,passive,lock=256,blend=0.7,seq=32,cands=3` since Phase 27
(Phase 26 was the same with 2 candidates, printed without `cands`; Phase 21 was
`k=128,passive,lock=256` with fixed k and no blend; a spec that sets `k`
stays sequential unless it adds `noseq`); `SearchConfig::legacy()` =
`k=128,nopassive,nolock,noblend,noseq,cands=2`, the
Phase 13–20 search (active-only, forced locks), still used by the
`pair-train --search` expert and the divergence harness.
Bench bots `pair-search-exp` / `pair-search-exp-b` read a spec such as
`nopassive,nolock` from `QWIXXER_SEARCH` / `QWIXXER_SEARCH_B`. Measured:
production reaches 62.8% vs GA (100k games, seed 42; Phase 26), Phase 21
production 62.4% (legacy 60.7%).
Static pair: 60.1% vs GA on the Phase 22 checkpoint. Checkpoint comparisons
use `pair-alt` / `pair-search-alt`, which load `QWIXXER_ALT_MODEL`.

### Monte Carlo Tree Search (`mcts.rs`)

**`MonteCarlo`** runs `N` rollout simulations per candidate move (default 500). For each move:
1. Apply the move to a copy of the current state
2. Simulate `N` complete games from that point using a rollout policy
3. Average the final scores across simulations

The rollout policy is pluggable via `RolloutFactory = Arc<dyn Fn() -> Box<dyn Strategy> + Send + Sync>`. The standard configuration uses the GA champion as the rollout policy.

**Parallelism**: Rollouts run in parallel via `rayon::par_iter` when the `parallel` feature is enabled.

**Passive turns**: Delegates to the rollout strategy's `opponents_move` for fast evaluation (no MC simulation on passive turns).

**Always-lock rule**: Same as GA and DQN.

---

## Training Pipeline

### Genetic Algorithm Evolution

**Population** -- A vector of `DNA` individuals, each with random initial weights.

**`next_generation()`**:
1. Run 1,000 4-player tournaments per generation (parallelized with rayon)
2. In each tournament, 4 randomly-selected DNA bots play a full game
3. Fitness = normalized score relative to the game's top scorer
4. Accumulate fitness across all tournaments
5. **Elitism**: The #1 individual carries forward unchanged
6. Remaining slots: fitness-proportional selection of two parents, uniform crossover (each weight randomly from either parent), then mutation (20% chance per weight of adding uniform noise in [-0.1, +0.1])
7. L2-normalize the weight vector

**Hyperparameters**: Population 100, 200 generations, 1,000 simulations/generation, mutation rate 0.2, elitism = 1.

### DQN Training

**Phase 1: MC-supervised pretraining** (`generate_training_data` + `train`):
1. Play 1,500 games using the GA champion
2. At each active-turn decision point, evaluate all candidate moves with 500 MC rollouts
3. Record `(features_after_move, mc_average_score)` tuples
4. Train the MLP for 50 epochs with MSE loss, Adam optimizer (lr=1e-3), batch size 1024

**Phase 2: Self-play RL** (`self_play_train`):
1. Can start from pretrained model or from scratch (random weights)
2. For each iteration (default 40, 20,000 games per iteration):
   a. Play games with epsilon-greedy exploration (epsilon decays from 0.2 to 0.07)
   b. 4 training configurations (5,000 games each): 3p vs 2 GA, 4p vs 3 GA, 3p vs GA + self-play copy, 4p vs 2 GA + self-play copy
   c. Record `(state_features, TD_target, final_score)` tuples at each decision point for player 0 (the DQN agent)
   d. Compute TD(lambda=0.8) targets: `G_t = (1-0.8) * V(s_{t+1}) + 0.8 * G_{t+1}`, with `G_{n-1} = final_score`
   e. Add new samples to a replay buffer (last 3 iterations retained, VecDeque)
   f. Retrain for 10 epochs on the full replay buffer (Adam lr=4e-4, batch 1024)
   g. Optionally benchmark against GA each iteration (`--bench N`)
   h. Save per-iteration checkpoint (iter-N.mpk)

**Decoupled loss (mu + sigma)**: The joint loss is `L = L_mu + alpha * L_sigma` with `alpha = 1.0` (see `SIGMA_LOSS_WEIGHT`).
- `L_mu = mean((G_t - mu)^2)` — MSE on the TD(lambda) target.
- `L_sigma = 0.5 * mean(log_var + (final_score - mu.detach())^2 * exp(-log_var))` — Gaussian NLL against the **final-score residual** (not the TD target). `mu.detach()` prevents sigma's gradient from corrupting mu training. Training sigma against TD residuals was tried first and failed (corr = -0.44 vs empirical MC sigma), because G_t has lower variance than the true final-score distribution; training against final-score residuals gives an unbiased estimator (corr = +0.91, see `examples/sigma_validation.rs`).

**Data augmentation**: The batcher randomly swaps symmetric row features (3 independent swaps, 8 permutations): red↔yellow, green↔blue, ascending↔descending pairs.

**Reproducibility**: All RNG is seeded from a global `TRAIN_SEED` constant. Per-game seeds are deterministic (`TRAIN_SEED + iteration * games + game_idx`). Burn's weight initialization is seeded. Training is reproducible across runs (identical training scores, model weights may differ slightly due to floating-point non-determinism in burn's optimizer).

**Training infrastructure**: Self-play games are parallelized with rayon. Models are pre-cloned (Arc-backed NdArray tensors make cloning cheap) instead of saving/loading from disk. LTO and `target-cpu=native` enabled for release builds.

**Key finding**: With P(win) ranking and 25-feature inputs, win rate peaks at **60.3% vs GA** around iterations 4-15 and then plateaus while avg score keeps climbing. The upper ceiling is structural — empirical Monte Carlo rollouts confirm that within-decision sigma variation across candidates is typically only 4-5%, so the P(win) ranker degenerates to argmax-mu at most decisions. Early stopping at the win-rate peak still produces the best checkpoint.

---

## Web UI Architecture

Browser Qwixx against one bot, specified in
`docs/superpowers/specs/2026-10-13-web-turn-flow-design.md` (turn flow, legal
squares, pacing, visual design). The Rust side owns every rule; the Svelte 5 +
TypeScript frontend (Vite) only looks up what the engine reports. A state
explorer (`#/explore`, spec
`docs/superpowers/specs/2026-10-14-state-explorer-design.md`) evaluates any
two-sheet position; see "State explorer" below.

### WASM Crate (`web/src/`)

**`Session`** (`session.rs`) is one human-vs-bot game. The human is player 0,
the bot player 1, and the bot starts every game. It follows the turn order of
`src/game.rs` (both phase-1 marks decided on the pre-roll boards and applied
together, locks propagate, the game may end, then the active player's phase 2).
Phases:

```
bot_to_roll -> passive -> you_to_roll -> active -> bot_to_roll -> ...   (any step may end in over)
```

- `bot_to_roll`: `bot_roll()` rolls and decides the bot's white sum, kept hidden.
- `passive`: `submit_passive(pick | null)` applies your white sum (or pass) and
  the bot's phase 1, then the bot's colour mark or strike. Returns a `RollRecord`.
- `you_to_roll`: `you_roll()` rolls and decides the bot's white sum, which is
  revealed (`view.bot_white`) because your legal set depends on it.
- `active`: `submit_active(picks, strike)` applies one of the engine's legal
  selections (`view.options.selections`) or a strike. Returns a `RollRecord`.
- `over`: with a reason (`locks`, `you_strikes`, `bot_strikes`).

Per-cell display marks (`[[bool; 11]; 4]` per player) are the source of truth;
`State` is derived from them, so marked and skipped cells stay distinguishable.
`selection.rs` computes the human's legal selections (spec "Which squares are
legal"): phase-1 marks judged on the pre-roll boards, phase 2 after both
phase-1 marks and lock propagation, no colour mark or strike once two rows are
locked. The session serialises to a `Snapshot`, so a game survives a reload.

**Bots** (`bots.rs`, engine ids in brackets): **Master+** (`master_plus`,
default) = pair-search over the pair net with `SearchConfig::parse(MASTER_PLUS_SPEC)`,
where `MASTER_PLUS_SPEC = ""` is the native production config (see the latency
table below); **Master** (`master`) = the static pair net; **Student**
(`student`) = the GA champion; **Newbie** (`newbie`) = the hand-written
Conservative. The header's "⋯" menu (`Menu.svelte`, both layouts) holds New
game (two clicks mid-game), Explorer (Back to game in
the explorer) and the dark-mode switch; on desktop, where the Play / Explore
tabs switch views, it drops Explorer and Back to game. `champion.txt` and `pair_model/model.mpk` are
embedded at compile time via `include_bytes!`, so rebuild after changing them.

**`WasmGame`** (`lib.rs`) wraps a `Session` for wasm-bindgen with JSON in/out:
`new(bot, seed)`, `restore(bot, snapshot, seed)`, `view()`, `snapshot()`,
`bot_roll()`, `submit_passive(pick_json)`, `you_roll()`,
`submit_active(picks_json, strike)`. `bench.rs` exports `bench(bot, games, seed)`.

### Web bot latency

Measured with `./web/build.sh && node web/scripts/bench.mjs <bot> 10` (Node
22, single thread, seed 1). The bench (`web/src/bench.rs`, exported as
`bench(bot, games, seed)`) plays games against random legal human choices and
times every call into the bot (all three decision phases, including trivial
ones). Besides the web bots it takes `win` and `win_search`, the win net and
its search, which the UI does not offer yet.

2026-10-22 (win-search; the other rows 2026-10-21), `opt-level = 3`, `-C target-feature=+simd128`, `wasm-opt -O3`
(`.wasm` 2.1 MB):

| Bot | Decisions | mean ms | p50 ms | p95 ms | max ms |
|---|---|---|---|---|---|
| Master+ (pair-search, `MASTER_PLUS_SPEC = ""`) | 291 | 7.11 | 0.07 | 38.48 | 109.76 |
| win-search (`SearchConfig::win()`: gate 0.05, `noendgame`, `gapcands`) | 290 | 12.93 | 0.09 | 82.83 | 345.65 |
| Master (pair net) | 297 | 0.03 | 0.02 | 0.10 | 1.22 |
| win | 294 | 0.15 | 0.09 | 0.42 | 1.76 |
| Student (GA champion) | 297 | 0.01 | 0.00 | 0.03 | 1.02 |

The earlier size-optimised build (`opt-level = 'z'`, `-Oz`, no SIMD, 1.0 MB)
ran Master+ at mean 20.99 / p95 122.90 / max 316.39 ms; win-search, first
measured at `opt-level = 3` without SIMD, at mean 60.50 / p95 286.20 / max
1131.33 ms. The win
net's dense tile is 8 outputs on wasm (two simd128 vectors; 16, two AVX2
vectors, natively): its 4 × 16 accumulators fill all 16 wasm registers and
spill, and the 8-wide tile is 2.4× faster for win-search (of 4/8/16 wide and
2/4/8 inputs per pass, 8 × 4 is fastest). Any tile is bit-identical
(`a_batched_layer_matches_the_plain_one_at_every_tile_width`). Mean ms by
build, Master+ / win-search: `z` + simd 21.5 / 106, `s` + simd 20.3 / 92,
`2` + simd 9.3 / 65, `3` + simd 7.1 / 22. Later exact changes took
win-search to 19.45 / 92.21 / 328.50 (ReLU as a comparison, globals-only
inputs), then the win-search gate of 0.05 (same moves; see EXPERIMENTS,
"Win-search gate") to 15.13 / 91.94 / 328.81, then dropping its endgame
gate and shortlisting by the gate (EXPERIMENTS, "Endgame gate: dropped",
"Shortlist by the gate") to the row above; the slowest decisions are
close ones, which both still search. The bar is p95 ≤ 400 ms and max
≤ 2500 ms.

### Frontend (`web/src/`)

**Worker engine.** The worker (`engine.worker.ts`, logic in `serve.ts`) holds one `WasmGame` (and the explorer's `WasmExplorer`), so bot
decisions (Master+ can take hundreds of ms) never stall animations.
`WorkerEngine` (`client.ts`) implements the `Engine` interface of
`protocol.ts` as promise RPC over `postMessage` (`start`, `restore`,
`snapshot`, `botRoll`, `submitPassive`, `youRoll`, `submitActive`, and the
explorer's `explore`);
`protocol.ts` mirrors the session's serde JSON types. Arguments are deep-copied
to plain JSON data before `postMessage`, since callers pass Svelte `$state`
Proxies, which cannot be structured-cloned. Failures reject with an
`EngineError` whose `kind` is `snapshot` (the worker loaded and `restore`
rejected the snapshot itself; `Session::restore` validates it) or `engine`
(anything else, e.g. the wasm failed to load). A failed wasm load is retried
on the next call; calls to a dead worker reject until `reset()` terminates it
and spawns a new one.

**`GameController`** (`game/controller.svelte.ts`) is the Svelte 5 rune state
for one page: it sequences turns against the `Engine`, enforces the pacing
minimums (`PACE`: tumble 600 ms with faces every 75 ms, bot think 500 ms,
colour-mark stagger 450 ms, hand-over 900 ms; beats overlap the engine call
and stretch if the bot is slower), stages the bot's marks (pencil, then ink),
keeps the selection, tips and log, and surfaces engine errors. After your
roll, `pendingDice` shows the rolled dice during the bot's think, before the
view (with the bot's revealed white sum) lands. The log resets per game.
`game/rules.ts` is the pure sheet / confirm-button / log model; it only looks
up the engine's lists and never re-derives legality.

**Persistence** (`localStorage`): `qwixxer.save.v1` holds the snapshot, log,
roll count, current and next bot; it is written after each resolved step and
restored on load (a game waiting on the bot's roll resumes its turn). Your
roll is saved as soon as the engine has rolled, before the tumble and think
finish, so a reload cannot re-roll. If the engine rejects a saved snapshot
(`kind: 'snapshot'`), the save is discarded and a new game starts; if the
engine fails to load, the controller retries once on a fresh worker, then
keeps the save and shows the error. After any engine failure, New game
recreates the worker. The stored preference for the next bot wins over the
one in the save.
`qwixxer.prefs.v1` holds the theme and the bot for the next game (changing the
bot applies from the next game).

**Layouts.** `App.svelte` picks `Desktop.svelte` (paper sheet, targets
1280×800) or `Phone.svelte` (sheet turned sideways, portrait 390×844) by the
media query `(max-width: 1099px), (orientation: portrait)`. Windows smaller
than 1280×800 (laptops at 1366×768, landscape tablets) get the desktop scaled
down by `deskScale` = min(1, width / 1280, height / 800) with a CSS transform,
inside a box of the scaled size, so the page never scrolls. On the phone the
bot's sheet, the bot picker and the log live in a modal drawer.
`game/layout.ts` (`phoneLayout`) sizes the phone: 48 px squares shrinking to a
28 px floor; if even that does not fit, the page scrolls vertically rather
than clipping. `game/keys.ts` (`keyAction`) decides page-wide keys: Enter
confirms, Space rolls, Escape clears; keyboard-focused controls and form
fields keep their native keys; while the drawer is open Escape closes it and
Enter/Space never act on the game.

**State explorer.** `explorer.rs` (`WasmExplorer` in `lib.rs`,
`explore(position_json)`) takes both sheets, the rows closed by tapping their
lock (rows closed by marking are derived and unioned in), the dice and whose
roll. It returns the pair net's P(you win), margin (score difference + μ) and
σ, the result for a finished game, and for a roll every legal turn of yours
(from `selection.rs`) ranked by the pair net's P(win) after it, and tagged
with the ids of the bots that would take it in your seat. On your roll the
opponent's simultaneous white-sum mark is predicted as Master predicts it
(`predict_opp_white`) and used throughout: legality, the P(win) and every
bot's phase 2; the answer reports it as `opponent_white`. When Master+'s or
Master's move isn't first it carries a note: `rolled_out` (Master+'s search
preferred it) or `rule` (a meta-rule forced it), from
`SearchBot::explain_active_phase1` / `explain_passive_phase1` (which decide
exactly as `Strategy` does and also report whether the move was forced and
the blended score of each candidate searched) and the pipeline's
`Decision::Forced`. The worker creates one `WasmExplorer` on the first
`explore` call and keeps it beside the game's `WasmGame`, so both views share
one worker. `ExplorerController` (`game/explorer.svelte.ts`) holds the
position (`game/position.ts`) and keeps one engine call in flight: edits made
meanwhile coalesce into one call for the latest position, and an outdated
answer never lands. Links are `#/explore?p=<code>`, where `game/link.ts`
packs the position into 117 bits as a version character plus 20 base64url
characters; the app loads a link once, then rewrites the address to
`#/explore`. The position is remembered as a link code in
`qwixxer.explorer.v1`. The explorer's "Explore current game" button (beside Swap
sides) copies the game's current view into it.

**Explorer analysis** (spec
`docs/superpowers/specs/2026-10-15-explorer-analysis-design.md`). The
Position card's win chance and margin are a play-out estimate, not the net's
P(win): `Explorer::analyze(position, start, count)` plays out
`start .. start + count` from the position before the roll (dice ignored),
both seats Master without its passive lookahead through `BatchedRollouts`,
for 4 turns (`PLAY_OUT_TURNS`), then values each by the net: P(win) and the
aux head's predicted final score difference, or the result where the game
ended. Play-out k's dice are seeded from the position and k, so batches
compose and replay identically. It returns the sum and sum of squares of
the win values and the sum of the score differences. `ExplorerController` runs one
`analyze` batch at a time while the explorer is on screen (route and tab
visible), sizing batches toward 100 ms (8 to 512 games, back to 8 on a new
position), back to back until ± < 2 points (plus-four estimate), then resting
as long as each batch took (half a core), adding replies to
totals keyed by the position without its dice (a dice edit keeps them; any
other edit restarts them; a reply for an older position is dropped). The win
chance is the mean win value, with ± = 1.96 standard errors from the
values' spread; it shows one
decimal once ± < 0.5 points. The move list shows the net's P(win) after each
move as a percentage with one decimal ("54.6%"). Throughput in WASM (Node, `node
web/scripts/analyze-bench.mjs`, 2026-10-22): about 1,250 play-outs/s from the
empty position; ± 1 point there takes about 0.4 s (25 s with full play-outs).

**Look.** Light/dark tokens in `styles/theme.css`, self-hosted Barlow and
Barlow Condensed (`@fontsource/*`). Squares, strike boxes and actions are real
`<button>`s with descriptive labels and `aria-pressed` / `aria-disabled`
(never `disabled`, so pressing an inactive control can show a tip).

**Tests.** Rust: `cd web && cargo test --release` (session, selection and
explorer edge cases). TypeScript: `npm test` (Vitest: rules, controller and
explorer controller with a fake engine, position, link codec, route, worker
RPC and handler, keys, layout) and `npm run check` (svelte-check).
Vitest runs in a custom environment (`src/test/client-env.ts`) that compiles
runes for the client, so `$state` values are Proxies as in the browser; the
fake engine `structuredClone`s its arguments like `postMessage`. CI
(`.github/workflows/test.yml`) runs both the Rust and the Vitest suites.

---

## Benchmarking (`main.rs`)

The `bench` subcommand runs N games between 2+ bot types with seat rotation to cancel position bias.

**Seat rotation**: Game `i` rotates seats by `i % num_players`, so each bot plays from every position equally across the benchmark run.

**Paired dice (common random numbers)**: Dice streams are a deterministic function of `(--seed, pair, seat)`, where a "pair" is the block of `num_players` consecutive games covering all rotations. Within a pair, each seat replays the same dice stream, so luck attaches to the seat and cancels across the rotations. Consequences:
- The same `--seed` (default 42) replays the identical game set — reruns are byte-identical for deterministic bots, and two models can be compared on exactly the same games (no dice variance between experiments).
- Different `--seed` values give independent samples (the base seed is hashed with SplitMix64 so stream sets of nearby seeds are disjoint).
- The 99% CI is computed over per-pair means (the pair is the independent sampling unit), which both stays honest under within-pair correlation and captures the pairing gain. Measured: ~1.3x efficiency in 1v1; ~0.8x in 2v2 (with 4 rotations the seat luck already cancels, while pair-shared luck correlates the games — the paired CI accounts for it).
- The per-iteration training benchmark (`benchmark_vs_ga`) uses a fixed base seed, so winrate differences between iterations are purely model-driven.

**Parallelism**: When the `parallel` feature is enabled, games run on a rayon thread pool. Each game constructs its own strategies and RNG (no shared mutable state). Results are collected into a `Vec` and aggregated serially.

**Aggregation**: Tracks wins (outright, no ties), total points, and tie count per bot. Reports win rate percentage and average points. When multiple bots share a strategy, prints aggregate stats where ties between same-strategy bots count as wins for that strategy. For 1v1 and 2-strategy matchups, shows 99% confidence interval (SE = sqrt(p*(1-p)/N), z=2.576).

**Solo mode**: Runs single-player games for each strategy, reporting average/min/max scores.

**Scale**: 500,000 games between DQN and GA complete in ~32 seconds with rayon parallelization and DQN batch+cache optimizations.

---

## PWA Support

The web app is installable as a Progressive Web App with offline caching:
- `web/public/manifest.json` — app name, theme color, icons
- `web/public/sw.js` — service worker with stale-while-revalidate caching
- WASM binary built for speed: `opt-level=3`, `simd128`, `strip`, `panic='abort'`, and `wasm-opt -O3` (~2 MB)
- Build script: `web/build.sh` runs wasm-pack + wasm-opt in one step
