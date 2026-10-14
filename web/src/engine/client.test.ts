import { describe, expect, it } from 'vitest';
import { WorkerEngine } from './client';
import { EngineError } from './protocol';

type Msg = { id: number; cmd: string; args: unknown[] };

/** A Worker stand-in: postMessage structuredClones like the real one, replies via `answer`. */
class FakeWorker {
    onmessage: ((e: { data: unknown }) => void) | null = null;
    onerror: ((e: { message: string }) => void) | null = null;
    onmessageerror: (() => void) | null = null;
    terminated = false;
    seen: Msg[] = [];
    constructor(private answer: (m: Msg) => unknown) {}
    postMessage(data: unknown) {
        const m = structuredClone(data) as Msg;
        this.seen.push(m);
        queueMicrotask(() => this.onmessage?.({ data: { id: m.id, ...(this.answer(m) as object) } }));
    }
    terminate() { this.terminated = true; }
}

const asWorker = (w: FakeWorker) => w as unknown as Worker;

describe('WorkerEngine', () => {
    it('posts plain copies of Proxy arguments (Svelte state)', async () => {
        const w = new FakeWorker(() => ({ ok: true, result: 'done' }));
        const e = new WorkerEngine(() => asWorker(w));
        const picks = new Proxy([new Proxy({ row: 0, n: 3 }, {})], {});
        expect(() => structuredClone(picks)).toThrow();
        await expect(e.submitActive(picks, false)).resolves.toBe('done');
        await expect(e.submitPassive(new Proxy({ row: 1, n: 7 }, {}))).resolves.toBe('done');
        expect(w.seen.map((m) => m.args)).toEqual([[[{ row: 0, n: 3 }], false], [{ row: 1, n: 7 }]]);
    });

    it('reports the error kind from the worker', async () => {
        const w = new FakeWorker((m) => ({ ok: false, error: `bad ${m.cmd}`, kind: m.cmd === 'restore' ? 'snapshot' : 'engine' }));
        const e = new WorkerEngine(() => asWorker(w));
        await expect(e.restore('master', '{}', 1)).rejects.toMatchObject({ kind: 'snapshot', message: 'bad restore' });
        await expect(e.start('master', 1)).rejects.toMatchObject({ kind: 'engine' });
    });

    it('rejects every call after the worker dies, until reset spawns a new one', async () => {
        const workers: FakeWorker[] = [];
        const e = new WorkerEngine(() => {
            const w = new FakeWorker(() => ({ ok: true, result: workers.length }));
            workers.push(w);
            return asWorker(w);
        });
        workers[0].onerror?.({ message: 'module failed to load' });
        const err = await e.start('master', 1).catch((x) => x);
        expect(err).toBeInstanceOf(EngineError);
        expect(err).toMatchObject({ kind: 'engine', message: 'module failed to load' });
        e.reset();
        expect(workers[0].terminated).toBe(true);
        await expect(e.start('master', 1)).resolves.toBe(2);
        expect(workers[1].seen).toHaveLength(1);
    });
});
