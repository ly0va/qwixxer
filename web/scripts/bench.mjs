// Usage (repo root): ./web/build.sh && node web/scripts/bench.mjs [bot] [games]
import { readFile } from 'node:fs/promises';
import init, { bench } from '../pkg/qwixxer_web.js';

const wasm = await readFile(new URL('../pkg/qwixxer_web_bg.wasm', import.meta.url));
await init({ module_or_path: wasm });
const bot = process.argv[2] ?? 'master_plus';
const games = Number(process.argv[3] ?? 10);
console.log(bot, games, 'games:', bench(bot, games, 1));
