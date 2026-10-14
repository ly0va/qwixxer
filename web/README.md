# Qwixx Web UI

Browser Qwixx against four bots: Master+, Master, Student and Newbie. The game runs
as a Rust session compiled to WASM inside a Web Worker; the UI is Svelte 5 +
TypeScript on Vite. Design and turn flow:
`docs/superpowers/specs/2026-10-13-web-turn-flow-design.md`; architecture:
`docs/ARCHITECTURE.md` ("Web UI Architecture").

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
