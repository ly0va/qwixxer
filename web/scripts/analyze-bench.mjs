// Play-out throughput of the explorer's analyze call in WASM.
// Usage (repo root): ./web/build.sh && node web/scripts/analyze-bench.mjs
import { readFile } from 'node:fs/promises';
import init, { WasmExplorer } from '../pkg/qwixxer_web.js';

await init({ module_or_path: await readFile(new URL('../pkg/qwixxer_web_bg.wasm', import.meta.url)) });
const empty = { marks: Array.from({ length: 4 }, () => Array(11).fill(false)), strikes: 0 };
const pos = JSON.stringify({ you: empty, opponent: empty, locked: [false, false, false, false], dice: null, roller: 'you' });
const ex = new WasmExplorer();
ex.analyze(pos, 0, 8); // warm up
for (const n of [8, 64, 256]) {
    const t0 = performance.now();
    ex.analyze(pos, 0, n);
    const ms = performance.now() - t0;
    console.log(`${n} play-outs: ${ms.toFixed(0)} ms, ${((n / ms) * 1000).toFixed(0)} play-outs/s`);
}
