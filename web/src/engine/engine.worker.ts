import init, { WasmGame } from '../../pkg/qwixxer_web.js';
import wasmUrl from '../../pkg/qwixxer_web_bg.wasm?url';

const ready = init({ module_or_path: wasmUrl });
let game: WasmGame | null = null;

function need(): WasmGame {
    if (!game) throw new Error('no game');
    return game;
}
const view = () => JSON.parse(need().view());

function run(cmd: string, args: any[]): unknown {
    switch (cmd) {
        case 'start':
            game?.free();
            game = new WasmGame(args[0], args[1]);
            return view();
        case 'restore':
            game?.free();
            game = WasmGame.restore(args[0], args[1], args[2]);
            return view();
        case 'snapshot':
            return need().snapshot();
        case 'botRoll':
            need().bot_roll();
            return view();
        case 'submitPassive': {
            const record = JSON.parse(need().submit_passive(JSON.stringify(args[0])));
            return { view: view(), record };
        }
        case 'youRoll':
            need().you_roll();
            return view();
        case 'submitActive': {
            const record = JSON.parse(need().submit_active(JSON.stringify(args[0]), args[1]));
            return { view: view(), record };
        }
        default:
            throw new Error(`unknown command ${cmd}`);
    }
}

self.onmessage = async (e: MessageEvent<{ id: number; cmd: string; args: any[] }>) => {
    const { id, cmd, args } = e.data;
    try {
        await ready;
        self.postMessage({ id, ok: true, result: run(cmd, args) });
    } catch (err) {
        self.postMessage({ id, ok: false, error: err instanceof Error ? err.message : String(err) });
    }
};
