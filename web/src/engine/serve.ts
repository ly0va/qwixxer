import type { WasmExplorer, WasmGame } from '../../pkg/qwixxer_web.js';

/** The wasm-bindgen pieces the worker needs (injected so tests don't need the built pkg). */
export interface Wasm {
    init: () => Promise<unknown>;
    WasmGame: {
        new (bot: string, seed: number, botFirst: boolean): WasmGame;
        restore(bot: string, snapshot: string, seed: number): WasmGame;
    };
    WasmExplorer: { new (): WasmExplorer };
}
export type Request = { id: number; cmd: string; args: any[] };
export type Reply = { id: number; ok: boolean; result?: unknown; error?: string; kind?: 'snapshot' | 'engine' };

/**
 * The worker's message handler: loads the wasm (retrying a failed fetch or
 * instantiate on the next call instead of rejecting forever) and runs one
 * game and the explorer. Starts loading at once.
 */
export function serve(wasm: Wasm, post: (r: Reply) => void): (req: Request) => Promise<void> {
    let ready: Promise<unknown> | null = null;
    const load = () =>
        (ready ??= wasm.init().catch((err) => {
            ready = null;
            throw err;
        }));
    load().catch(() => {}); // a failure surfaces on the first call
    let game: WasmGame | null = null;
    let explorer: WasmExplorer | null = null;

    const need = (): WasmGame => {
        if (!game) throw new Error('no game');
        return game;
    };
    const view = () => JSON.parse(need().view());
    const replace = (g: WasmGame) => {
        game?.free();
        game = g;
        return view();
    };

    function run(cmd: string, args: any[]): unknown {
        switch (cmd) {
            case 'start':
                return replace(new wasm.WasmGame(args[0], args[1], args[2]));
            case 'restore':
                return replace(wasm.WasmGame.restore(args[0], args[1], args[2]));
            case 'snapshot':
                return need().snapshot();
            case 'gameCode':
                return need().code();
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
            case 'explore':
                // Created on first use and kept: the game and the explorer share the worker.
                explorer ??= new wasm.WasmExplorer();
                return JSON.parse(explorer.explore(JSON.stringify(args[0])));
            case 'analyze':
                explorer ??= new wasm.WasmExplorer();
                return JSON.parse(explorer.analyze(JSON.stringify(args[0]), args[1], args[2]));
            case 'luck':
                explorer ??= new wasm.WasmExplorer();
                return JSON.parse(explorer.luck(JSON.stringify(args[0]), args[1], args[2]));
            case 'onePly':
                explorer ??= new wasm.WasmExplorer();
                return explorer.move_one_ply(JSON.stringify(args[0]), JSON.stringify(args[1]), args[2]);
            case 'play':
                explorer ??= new wasm.WasmExplorer();
                return JSON.parse(explorer.play(JSON.stringify(args[0]), JSON.stringify(args[1]), args[2]));
            case 'reviewGame':
                explorer ??= new wasm.WasmExplorer();
                return JSON.parse(explorer.review_game(args[0]));
            case 'reviewRoll':
                explorer ??= new wasm.WasmExplorer();
                return JSON.parse(explorer.review_roll(args[0], args[1]));
            default:
                throw new Error(`unknown command ${cmd}`);
        }
    }

    const message = (err: unknown) => (err instanceof Error ? err.message : String(err));

    return async ({ id, cmd, args }) => {
        try {
            await load();
        } catch (err) {
            post({ id, ok: false, error: `engine failed to load: ${message(err)}`, kind: 'engine' });
            return;
        }
        try {
            post({ id, ok: true, result: run(cmd, args) });
        } catch (err) {
            // With the engine loaded, a failed restore means the snapshot itself is bad.
            post({ id, ok: false, error: message(err), kind: cmd === 'restore' ? 'snapshot' : 'engine' });
        }
    };
}
