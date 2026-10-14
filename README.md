# qwixxer

A bot framework for the dice game [Qwixx](https://en.wikipedia.org/wiki/Qwixx): hand-crafted strategies, a genetic algorithm, value networks trained by self-play, decision-time search, and a terminal and a web UI.

**[Play in your browser](https://ly0va.github.io/qwixxer/)**: no installation needed. Four opponents, from **Newbie** to **Master+**, on a paper-sheet board that works on desktop and phone (light and dark).

## Bots

| Bot | Web name | Description | vs GA (1v1) |
|---|---|---|---|
| **Pair-Search** | Master+ | Pair net + gated rollout search (see below) | **~62.8%** |
| **Pair** | Master | Joint two-board value net, static play | ~60.1% |
| **DQN** | | TD(λ) single-board value net + Gaussian P(win) ranking | ~60.3% |
| **GA Champion** | Student | Genetically evolved 4-weight heuristic | baseline |
| **MCTS** | | Monte Carlo tree search with GA rollouts | ~GA level |
| **Opportunist** | Newbie* | Max weighted probability, blank caps, always lock | ~30% |
| **Conservative** | | Skip at most 2 numbers per mark | ~25% |
| **Random** | | Uniform random legal move | ~0% |

Win rates are from deterministic paired-dice benches; the full history is in `docs/EXPERIMENTS.md`.

\* The web's Newbie is the Opportunist with the shared pipeline's pruning (it never picks a mark another available mark dominates); the win rate above is the unpruned CLI `opportunist`.

All non-trivial bots share one decision pipeline (`src/strategy/bot_impl.rs`) with **meta-rules**:
- **Smart lock / smart strike**: end the game when it is won, never end it into a loss.
- **Move pruning**: drop moves dominated by another (same row closer to the free pointer, equal progress with a higher total, post-state dominance).

## Pair net

A two-board MLP (45 inputs → 128 → 64 → 2): per-board features for both players plus pair-level features. It predicts the distribution of the *future* score differential, `(μ, log σ²)`, and ranks candidate moves by `(current_diff + μ) / σ`, which is monotone in a Gaussian P(win) against the leading opponent. Trained by self-play (`pair-train`), then improved by expert iteration and distillation from the search bot.

## Pair-Search

Decision-time search on top of the pair net (`src/strategy/search.rs`), the production `pair-search` bot:

- **Gate**: search only close decisions (top-2 static gap < 0.15) or the endgame (a locked row, 3 strikes, or a locking candidate); active and passive (white-sum) decisions both qualify.
- **Candidates**: the top 3 static moves, each with its turn completed deterministically.
- **Rollouts**: one full round per sample with common random numbers across candidates; the leaf is the exact outcome if the game ended, else the net's win probability.
- **Sequential sampling**: start at 32 samples per candidate and double up to 512 until the leader beats each rival by 2 standard errors; decisively worse candidates stop early.
- **Blend**: the pick score is `0.3 · Φ(static z) + 0.7 · rollout mean`.
- **Lock adjudication**: a forced safe lock is checked against the best alternative with 256 full-game rollouts each.

The search config is a string (`k=512,passive,lock=256,blend=0.7,seq=32,cands=3` in production); `pair-search-exp` reads an experimental one from `QWIXXER_SEARCH`.

## DQN

A TD(λ) value network (closer to TD-Gammon than DQN): 25 single-board features → 128 → 64 → `(μ, log σ²)` of the final score, ranking moves by Gaussian P(win) against the leading opponent. Trained by self-play with replay, colour-symmetry augmentation and per-candidate opponent context. Superseded by the pair net, but still a strong baseline.

## Genetic algorithm

Evolves a 4-weight vector over gene functions (`weighted_prob`, `strikes`, `concentration`, `blanks`): 100 individuals, 200 generations, 1,000 simulated 4-player games per generation (rayon-parallel). Fitness is the score relative to the game's top scorer.

## Usage

```bash
# Play in the terminal (human vs bots)
cargo run --release -- play pair-search
cargo run --release -- play ga opportunist          # vs 2 bots

# Benchmark bots against each other (seat rotation, paired dice)
cargo run --release -- bench pair-search ga -n 100000
cargo run --release -- bench pair dqn ga ga -n 100000   # aggregate stats by bot
QWIXXER_SEARCH="k=128,noseq" cargo run --release -- bench pair-search-exp pair-search -n 50000

# Single-player score benchmark across all bots
cargo run --release -- solo -n 10000

# Training
cargo run --release -- evolve                         # GA champion
cargo run --release -- dqn-selfplay -i 40 -b 100000   # DQN self-play
cargo run --release -- pair-train -i 40 -b 100000 -c  # pair net self-play
cargo run --release -- pair-train --search -g 5000 -e 5 -b 500000 -c   # expert iteration
```

`cargo run --release -- <command> --help` lists every option and bot name.

Model artifacts (committed, embedded in the web build):
- `champion.txt`: GA weights
- `dqn_model/model.mpk`: DQN weights
- `pair_model/model.mpk`: pair-net weights

## Web UI

Svelte 5 + TypeScript + Vite. The game rules run in Rust compiled to WASM inside a Web Worker, so bot thinking never blocks the animations. PWA-installable and playable offline. Deployed to GitHub Pages by `.github/workflows/deploy.yml` on every push to `main`. An explorer (`#/explore`) evaluates any two-sheet position and, for a roll, ranks your moves and shows which one each bot would pick; positions can be shared as links. Build and development instructions are in `web/README.md`; the design is in `docs/superpowers/specs/2026-10-13-web-turn-flow-design.md`.

## Project structure

```
src/
  state.rs         Game state, moves, scoring, lock rules, move pruning
  game.rs          Game loop, players, dice sources
  strategy/        Strategy trait, hand-written bots, shared bot pipeline (bot_impl.rs),
                   pair-search (search.rs), batched rollout simulation (sim.rs)
  bot.rs           GA: DNA, genes, population, evolution
  dqn/             DQN (mod.rs, train.rs), pair net (pair.rs, pair_train.rs)
  mcts.rs          Monte Carlo tree search
  main.rs          CLI (clap subcommands)
examples/          Analysis tools: divergence and feature screens, sigma probes,
                   score distributions, search calibration
web/
  src/*.rs         WASM crate: game session, legal selections, bots, explorer, latency bench
  src/engine/      Worker RPC to the WASM engine
  src/game/        Pure UI model, controller, keyboard and layout helpers
  src/ui/          Svelte components (desktop and phone layouts)
docs/
  ARCHITECTURE.md  Technical reference
  EXPERIMENTS.md   Chronological experiment log
```

## Feature flags

| Feature | Purpose |
|---------|---------|
| `dqn` | Training code (burn with train support, serde, rayon); default |
| `parallel` | Rayon parallelism for benchmarks and training; default |

The web crate builds the root crate with `default-features = false` (inference only).

## Rules

Qwixx is a dice game for 2–5 players. Each turn the active player rolls 6 dice (2 white, 4 coloured). Every player may mark the white sum in any row; the active player may also mark one white + colour sum in that colour's row. Numbers are marked left to right (Red and Yellow ascending, Green and Blue descending). Locking a row requires 5 marks before marking its last number. If the active player marks nothing, they take a strike. The game ends when someone has 4 strikes or two rows are locked. A row with n marks scores n(n+1)/2 points; each strike costs 5.

## Disclaimer

An unofficial fan project. Qwixx is a trademark of Nürnberger-Spielkarten-Verlag GmbH, published in North America by Gamewright. This project is not affiliated with or endorsed by either.
