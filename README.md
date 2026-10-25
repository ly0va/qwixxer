# qwixxer

A bot framework for the dice game [Qwixx](https://boardgamegeek.com/boardgame/131260/qwixx): hand-crafted strategies, a genetic algorithm, value networks trained by self-play, decision-time search, and a terminal and a web UI.

**[Play in your browser](https://ly0va.github.io/qwixxer/)**: no installation needed. Four opponents, from **Newbie** to **Master+**, on a paper-sheet board that works on desktop and phone (light and dark). An explorer ranks the moves in any position, and a game review shows whether a game was won or lost by luck or by play.

## Bots

| Bot | Web name | Description | vs GA (1v1) |
|---|---|---|---|
| **Win-Search** | Master+ | Win net + gated rollout search (see below) | **~63.7%** |
| **Win** | Master | Direct P(win) net (2 players), static play with passive lookahead | ~63.3% |
| **Pair-Search** | | Pair net + gated rollout search | ~62.7% |
| **Pair** | | Joint two-board value net, static play | ~60.1% |
| **DQN** | | TD(λ) single-board value net + Gaussian P(win) ranking | ~60.3% |
| **GA Champion** | Student | Genetically evolved 4-weight heuristic | baseline |
| **MCTS** | | Monte Carlo tree search with GA rollouts | ~GA level |
| **Opportunist** | | Max weighted probability, blank caps, always lock | ~30% |
| **Conservative** | Newbie | Fewest skipped numbers over the whole turn (strike rather than skip 4+; none on the opponent's roll) | ~34% |
| **Random** | | Uniform random legal move | ~0% |

Win rates are from deterministic paired-dice benches; the full history is in `docs/EXPERIMENTS.md`.

All non-trivial bots share one decision pipeline (`src/strategy/bot_impl.rs`) with **meta-rules**:
- **Smart lock / smart strike**: end the game when it is won, never end it into a loss. The win net judges safe locks itself instead of always taking them.
- **Move pruning**: drop moves dominated by another (same row closer to the free pointer, equal progress with a higher total, post-state dominance).

## Win net

The production net for two players (`src/dqn/win.rs`): it predicts P(win) directly. Inputs are both boards regrouped as 4 row tokens (one row of both players, with thermometer-coded progress and marks) and 9 globals. A shared row encoder, summed within the colour pairs, feeds a shared pair encoder summed over both pairs, then a head g(me, opp); the logit is g(me, opp) − g(opp, me), so the two players' chances add up to 1 and the colour relabellings leave it unchanged, both exactly. Auxiliary heads predict final row marks, strikes and score difference (used by the explorer, never in play), and a tempo head values moving next.

Training (`win-data`, `win-arm`, `win-gate`): the champion plays itself, every turn end and every decision the search would search is labelled by 64 full-game rollouts, and a net is fitted from scratch on those labels and promoted only if it beats the champion in paired games.

**Passive lookahead**: a white-sum mark on the opponent's roll is valued at the end of the opponent's turn (its predicted white mark, then its colour choice against the boards it sees), so locks and game ends caused by that turn count.

## Search

Decision-time search on top of a static net (`src/strategy/search.rs`). Master+ runs it on the win net (`SearchConfig::win()`):

- **Gate**: search only close decisions, where the top-2 static gap is under 0.05 (probit units); active and passive decisions both qualify.
- **Candidates**: those within the gate of the best (2 to 4), each with its turn completed deterministically.
- **Rollouts**: one full round per sample with common random numbers across candidates; the leaf is the exact outcome if the game ended, else the net's win probability.
- **Sequential sampling**: start at 32 samples per candidate and double up to 512 until the leader beats each rival by 2 standard errors; decisively worse candidates stop early.
- **Blend**: the pick score is `0.3 · Φ(static z) + 0.7 · rollout mean`.

Pair-search (`SearchConfig::default()`) also searches the endgame regardless of the gap, takes the top 3 candidates, uses a 0.15 gate and adjudicates forced safe locks with 256 full-game rollouts each. The search config is a string (e.g. `k=512,passive,lock=256,blend=0.7,seq=32,cands=3`); `pair-search-exp` and `win-search-exp` read an experimental one from `QWIXXER_SEARCH`.

## Pair net

A two-board MLP (45 inputs → 128 → 64 → 2): per-board features for both players plus pair-level features. It predicts the distribution of the *future* score differential, `(μ, log σ²)`, and ranks candidate moves by `(current_diff + μ) / σ`, which is monotone in a Gaussian P(win) against the leading opponent. Trained by self-play (`pair-train`), then improved by expert iteration and distillation from the search bot. Unlike the win net it handles more than two players.

## DQN

A TD(λ) value network (closer to TD-Gammon than DQN): 25 single-board features → 128 → 64 → `(μ, log σ²)` of the final score, ranking moves by Gaussian P(win) against the leading opponent. Trained by self-play with replay, colour-symmetry augmentation and per-candidate opponent context. Superseded by the pair net, but still a strong baseline.

## Genetic algorithm

Evolves a 4-weight vector over gene functions (`weighted_prob`, `strikes`, `concentration`, `blanks`): 100 individuals, 200 generations, 1,000 simulated 4-player games per generation (rayon-parallel). Fitness is the score relative to the game's top scorer.

## Usage

```bash
# Play in the terminal (human vs bots)
cargo run --release -- play win-search
cargo run --release -- play ga opportunist          # vs 2 bots (the win bots are 2-player only)

# Benchmark bots against each other (seat rotation, paired dice)
cargo run --release -- bench win-search ga -n 100000
cargo run --release -- bench pair dqn ga ga -n 100000   # aggregate stats by bot
QWIXXER_SEARCH="k=128,noseq" cargo run --release -- bench win-search-exp win-search -n 50000

# Single-player score benchmark across all bots
cargo run --release -- solo -n 10000

# Training
cargo run --release -- evolve                         # GA champion
cargo run --release -- dqn-selfplay -i 40 -b 100000   # DQN self-play
cargo run --release -- pair-train -i 40 -b 100000 -c  # pair net self-play
cargo run --release -- win-data --help                # win net: rollout-labelled dataset
cargo run --release -- win-arm --help                 # win net: fit from scratch
cargo run --release -- win-gate --help                # win net: paired-game promotion gate
```

`cargo run --release -- <command> --help` lists every option and bot name. The current win-net recipe (exact flags) is in `docs/EXPERIMENTS.md`, Phase 28.

Model artifacts (committed, embedded in the web build):
- `champion.txt`: GA weights
- `dqn_model/model.mpk`: DQN weights
- `pair_model/model.mpk`: pair-net weights
- `win_model/`: win-net config, weights, auxiliary and tempo heads

## Web UI

Svelte 5 + TypeScript + Vite. The game rules and bots run in Rust compiled to WASM inside a Web Worker, so bot thinking never blocks the animations. PWA-installable and playable offline; nothing leaves the browser. Deployed to GitHub Pages by `.github/workflows/deploy.yml` on every push to `main`.

- **Play**: two-player games against Newbie, Student, Master or Master+.
- **Explorer** (`#/explore`): evaluates any two-sheet position; for a roll, ranks your moves by win chance (the top ones refined one roll ahead), shows which move each bot would pick, the opponent's predicted turn and how lucky the roll was. Positions can be shared as links.
- **Game review** (`#/review`): replays a finished game with an eval bar and graph, splits the result into dice luck and play, and for each mistake shows what it cost and the best move. A whole game fits in a short link (`?g=…`).

Build and development instructions are in `web/README.md`; the designs are in `docs/superpowers/specs/` (turn flow, explorer, roll luck, game review).

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

## License

Copyright (C) 2026 Lev Potomkin. Licensed under the [GNU Affero General Public License v3.0](LICENSE) or later. That covers the code and the trained networks; it grants no rights in Qwixx itself.
