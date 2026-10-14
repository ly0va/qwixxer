# Qwixx Web UI

Browser Qwixx against four bots: Master+, Master, Student and Newbie. The game runs
as a Rust session compiled to WASM inside a Web Worker; the UI is Svelte 5 +
TypeScript on Vite. Design and turn flow:
`docs/superpowers/specs/2026-10-13-web-turn-flow-design.md`; architecture:
`docs/ARCHITECTURE.md` ("Web UI Architecture").

## State explorer

`#/explore` (the desktop Play / Explore tabs, or the "⋯" menu's Explorer and,
in a game, "Explore this position") sets up any two-sheet position and asks
the engine about it. Spec: `docs/superpowers/specs/2026-10-14-state-explorer-design.md`.

- `src/explorer.rs` (`WasmExplorer` in `src/lib.rs`): the pair net's P(win)
  and margin for the position, and for a roll every legal move of yours
  ranked by the pair net's P(win) after it, judged as Master judges it
  (with the opponent's predicted white-sum mark on your roll, which every bot
  also sees), tagged with the move each bot would make in your seat; an (i)
  tooltip explains Master+'s or Master's move when it isn't first. Runs as the `explore` command of the same worker as the game.
- `src/game/position.ts`: the editable position (marks, strikes, tap-closed
  rows, dice, whose roll) and its sheet model; `src/game/link.ts`: the
  21-character link code (`#/explore?p=<code>`); `src/game/explorer.svelte.ts`:
  `ExplorerController` (edits, one engine call in flight with edits coalesced,
  the position remembered in `localStorage`, Copy link); `src/game/route.ts`:
  hash routing.
- `src/ui/`: `ExplorerDesktop` / `ExplorerPhone`, with `ExSheet`,
  `PositionCard`, `RollCard`, `MovesList`, `StrikeBoxes` and `Tabs`.

## Prerequisites

- Rust toolchain with `wasm32-unknown-unknown` target: `rustup target add wasm32-unknown-unknown`
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/): `cargo install wasm-pack`
- [binaryen](https://github.com/WebAssembly/binaryen) (for wasm-opt): `apt install binaryen`
- Node.js 22.12+ (Vite 8 needs 20.19+, Vitest 5 needs 22.12+)

## Build WASM

From the repo root, compile the WASM crate into `web/pkg/` and run wasm-opt:

```bash
./web/build.sh
```

This embeds `champion.txt` (Student, GA weights) and `pair_model/model.mpk`
(pair net for Master and Master+) into the WASM binary via `include_bytes!`.
Rebuild after changing either file.

## Frontend

```bash
cd web
npm install
npm run dev      # dev server: http://localhost:5173/qwixxer/
npm test         # Vitest unit tests
npm run check    # svelte-check (types)
npm run build    # production build into web/dist/
```

The Rust session tests: `cd web && cargo test --release`.

## Bot latency

```bash
./web/build.sh && node web/scripts/bench.mjs master_plus 10
```

Plays 10 games against random legal human choices and reports per-decision
latency of the bot (`master_plus`, `master` or `student`). Results are recorded
in `docs/ARCHITECTURE.md` ("Web bot latency").

## Deploy to GitHub Pages

Push to `main` — the GitHub Actions workflow (`.github/workflows/deploy.yml`) builds and deploys automatically.
