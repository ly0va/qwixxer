import { EngineError, type Analysis, type Answer, type BotKind, type Engine, type EngineErrorKind, type Luck, type Played, type Position, type ReviewGame, type RollAnalysis, type Sq, type Step, type View } from './protocol';

type Pending = { resolve: (v: any) => void; reject: (e: Error) => void };
type Reply = { id: number; ok: boolean; result?: any; error?: string; kind?: EngineErrorKind };

/** Promise RPC to engine.worker.ts. One worker owns the whole game. */
export class WorkerEngine implements Engine {
    private next = 1;
    private pending = new Map<number, Pending>();
    private dead: Error | null = null;
    private worker!: Worker;
    private spawn: () => Worker;

    constructor(spawn: () => Worker) {
        this.spawn = spawn;
        this.attach();
    }

    private attach() {
        const w = (this.worker = this.spawn());
        this.dead = null;
        w.onmessage = (e: MessageEvent<Reply>) => {
            if (w !== this.worker) return;
            const p = this.pending.get(e.data.id);
            if (!p) return;
            this.pending.delete(e.data.id);
            if (e.data.ok) p.resolve(e.data.result);
            else p.reject(new EngineError(e.data.error ?? 'engine error', e.data.kind ?? 'engine'));
        };
        w.onerror = (e) => w === this.worker && this.fail(new EngineError(e.message || 'engine failed to load', 'engine'));
        w.onmessageerror = () => w === this.worker && this.fail(new EngineError('engine message could not be decoded', 'engine'));
    }

    /** The worker is unusable: reject everything pending and every later call until `reset`. */
    private fail(err: Error) {
        this.dead = err;
        for (const p of this.pending.values()) p.reject(err);
        this.pending.clear();
    }

    reset() {
        this.worker.terminate();
        this.fail(new EngineError('engine was reset', 'engine'));
        this.attach();
    }

    private call<T>(cmd: string, ...args: unknown[]): Promise<T> {
        if (this.dead) return Promise.reject(this.dead);
        // Plain data only: callers may pass Svelte state Proxies, which
        // postMessage can't clone (DataCloneError).
        const plain = JSON.parse(JSON.stringify(args)) as unknown[];
        const id = this.next++;
        return new Promise<T>((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            this.worker.postMessage({ id, cmd, args: plain });
        });
    }

    start(bot: BotKind, seed: number) { return this.call<View>('start', bot, seed); }
    restore(bot: BotKind, snapshot: string, seed: number) { return this.call<View>('restore', bot, snapshot, seed); }
    snapshot() { return this.call<string>('snapshot'); }
    botRoll() { return this.call<View>('botRoll'); }
    submitPassive(pick: Sq | null) { return this.call<Step>('submitPassive', pick); }
    youRoll() { return this.call<View>('youRoll'); }
    submitActive(picks: Sq[], strike: boolean) { return this.call<Step>('submitActive', picks, strike); }
    explore(position: Position) { return this.call<Answer>('explore', position); }
    analyze(position: Position, start: number, count: number) { return this.call<Analysis>('analyze', position, start, count); }
    luck(position: Position, first: number, count: number) { return this.call<Luck>('luck', position, first, count); }
    onePly(position: Position, marks: Sq[], strike: boolean) { return this.call<number>('onePly', position, marks, strike); }
    play(position: Position, marks: Sq[], strike: boolean) { return this.call<Played>('play', position, marks, strike); }
    gameCode() { return this.call<string>('gameCode'); }
    reviewGame(code: string) { return this.call<ReviewGame>('reviewGame', code); }
    reviewRoll(code: string, k: number) { return this.call<RollAnalysis>('reviewRoll', code, k); }
}

export function createEngine(): Engine {
    return new WorkerEngine(() => new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' }));
}
