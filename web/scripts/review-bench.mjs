// Time of the game review's per-roll analysis in WASM, over a whole game.
// Usage (repo root): ./web/build.sh && node web/scripts/review-bench.mjs <game code>
import { readFile } from 'node:fs/promises';
import init, { WasmExplorer } from '../pkg/qwixxer_web.js';

await init({ module_or_path: await readFile(new URL('../pkg/qwixxer_web_bg.wasm', import.meta.url)) });
const code = process.argv[2];
if (!code) throw new Error('usage: node web/scripts/review-bench.mjs <game code>');
const ex = new WasmExplorer();
const game = JSON.parse(ex.review_game(code));
const ms = [];
for (let k = 0; k < game.rolls.length; k++) {
    const t0 = performance.now();
    ex.review_roll(code, k);
    ms.push(performance.now() - t0);
}
const total = ms.reduce((a, b) => a + b, 0);
console.log(`${game.rolls.length} rolls: ${(total / 1000).toFixed(1)} s total, mean ${(total / ms.length).toFixed(0)} ms, slowest ${Math.max(...ms).toFixed(0)} ms per roll`);
