// Time of the explorer's luck call in WASM, per position and roller.
// Usage (repo root): ./web/build.sh && node web/scripts/luck-bench.mjs
import { readFile } from 'node:fs/promises';
import init, { WasmExplorer } from '../pkg/qwixxer_web.js';

await init({ module_or_path: await readFile(new URL('../pkg/qwixxer_web_bg.wasm', import.meta.url)) });
const sheet = (rows, strikes) => ({ marks: rows.map((ix) => Array.from({ length: 11 }, (_, i) => ix.includes(i))), strikes });
const positions = {
    empty: [sheet([[], [], [], []], 0), sheet([[], [], [], []], 0), [false, false, false, false]],
    midgame: [sheet([[0, 2, 3], [1, 4], [0, 1], [2]], 1), sheet([[1, 2], [0, 3, 5], [2], [0, 1]], 0), [false, false, false, false]],
};
const ex = new WasmExplorer();
for (const [name, [you, opponent, locked]] of Object.entries(positions)) {
    for (const roller of ['you', 'opponent']) {
        const pos = JSON.stringify({ you, opponent, locked, dice: [3, 4, 2, 5, 6, 1], roller });
        // As the explorer asks: 7 chunks of 3 white pairs.
        let total = 0, slowest = 0, average = 0, l;
        for (let first = 0; first < 21; first += 3) {
            const t0 = performance.now();
            l = JSON.parse(ex.luck(pos, first, 3));
            const ms = performance.now() - t0;
            total += ms; slowest = Math.max(slowest, ms); average += l.average;
        }
        console.log(`${name}, ${roller} roll: ${total.toFixed(0)} ms, slowest chunk ${slowest.toFixed(0)} ms (this ${l.this_roll.toFixed(3)}, average ${average.toFixed(3)})`);
    }
}
