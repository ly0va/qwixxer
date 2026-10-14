import type { BotKind, Engine, Sq, Step, View } from './protocol';

type Pending = { resolve: (v: any) => void; reject: (e: Error) => void };

/** Promise RPC to engine.worker.ts. One worker owns the whole game. */
export class WorkerEngine implements Engine {
    private next = 1;
    private pending = new Map<number, Pending>();
    private dead: Error | null = null;

    constructor(private worker: Worker) {
        worker.onmessage = (e: MessageEvent<{ id: number; ok: boolean; result?: any; error?: string }>) => {
            const p = this.pending.get(e.data.id);
            if (!p) return;
            this.pending.delete(e.data.id);
            if (e.data.ok) p.resolve(e.data.result);
            else p.reject(new Error(e.data.error));
        };
        worker.onerror = (e) => this.fail(new Error(e.message || 'engine failed to load'));
        worker.onmessageerror = () => this.fail(new Error('engine message could not be decoded'));
    }

    /** The worker is unusable: reject everything pending and every later call. */
    private fail(err: Error) {
        this.dead = err;
        for (const p of this.pending.values()) p.reject(err);
        this.pending.clear();
    }

    private call<T>(cmd: string, ...args: unknown[]): Promise<T> {
        if (this.dead) return Promise.reject(this.dead);
        const id = this.next++;
        return new Promise<T>((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            this.worker.postMessage({ id, cmd, args });
        });
    }

    start(bot: BotKind, seed: number) { return this.call<View>('start', bot, seed); }
    restore(bot: BotKind, snapshot: string, seed: number) { return this.call<View>('restore', bot, snapshot, seed); }
    snapshot() { return this.call<string>('snapshot'); }
    botRoll() { return this.call<View>('botRoll'); }
    submitPassive(pick: Sq | null) { return this.call<Step>('submitPassive', pick); }
    youRoll() { return this.call<View>('youRoll'); }
    submitActive(picks: Sq[], strike: boolean) { return this.call<Step>('submitActive', picks, strike); }
}

export function createEngine(): Engine {
    const worker = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' });
    return new WorkerEngine(worker);
}
