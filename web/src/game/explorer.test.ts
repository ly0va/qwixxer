import { describe, expect, it, vi } from 'vitest';
import type { Answer, Engine, Position } from '../engine/protocol';
import type { Store } from './controller.svelte';
import { BAD_LINK, ExplorerController } from './explorer.svelte';
import { encode } from './link';
import { emptyPosition, toggleSquare } from './position';

const answer = (p_win: number): Answer => ({ over: null, eval: { p_win, margin: 0, sigma: 10 }, moves: null, opponent_white: null });
function memStore(init: Record<string, string> = {}): Store & { data: Record<string, string> } {
    const data = { ...init };
    return { data, get: (k) => data[k] ?? null, set: (k, v) => { data[k] = v; } };
}
/** An engine whose explore calls resolve only when the test says so. */
function manualEngine() {
    const calls: { pos: Position; resolve: (a: Answer) => void }[] = [];
    const engine = { explore: vi.fn((pos: Position) => new Promise<Answer>((resolve) => calls.push({ pos: structuredClone(pos), resolve }))) } as unknown as Engine;
    return { engine, calls };
}
const flush = () => new Promise((r) => setTimeout(r, 0));

describe('ExplorerController', () => {
    it('starts empty when storage is empty or garbage, and evaluates on refresh', async () => {
        for (const store of [memStore(), memStore({ 'qwixxer.explorer.v1': 'garbage' })]) {
            const { engine, calls } = manualEngine();
            const x = new ExplorerController(engine, store);
            x.init();
            expect(x.pos).toEqual(emptyPosition());
            expect(calls).toHaveLength(0); // init only restores; the explorer evaluates when entered
            x.refresh();
            expect(calls).toHaveLength(1);
            calls[0].resolve(answer(0.5));
            await flush();
            expect(x.answer?.eval.p_win).toBe(0.5);
            expect(x.pending).toBe(false);
        }
    });

    it('refresh evaluates once, and does nothing while a call is in flight or with a good answer', async () => {
        const { engine, calls } = manualEngine();
        const x = new ExplorerController(engine, memStore());
        x.init();
        x.refresh();
        x.refresh();
        expect(calls).toHaveLength(1);
        calls[0].resolve(answer(0.3));
        await flush();
        expect(x.answer?.eval.p_win).toBe(0.3);
        x.refresh();
        expect(calls).toHaveLength(1);
    });

    it('refresh after an engine failure evaluates again', async () => {
        const engine = { explore: vi.fn().mockRejectedValueOnce(new Error('engine was reset')).mockResolvedValue(answer(0.8)) } as unknown as Engine;
        const x = new ExplorerController(engine, memStore());
        x.init();
        x.refresh();
        await flush();
        expect(x.error).toBe('engine was reset');
        x.refresh();
        await flush();
        expect(engine.explore).toHaveBeenCalledTimes(2);
        expect(x.error).toBe('');
        expect(x.answer?.eval.p_win).toBe(0.8);
    });

    it('evaluates only the latest of rapid edits, and never lands an older answer', async () => {
        const { engine, calls } = manualEngine();
        const x = new ExplorerController(engine, memStore());
        x.init();
        x.refresh();
        x.tapSquare('you', 0, 0);
        x.tapSquare('you', 0, 1);
        x.tapSquare('you', 0, 2);
        expect(calls).toHaveLength(1); // the first call is still in flight
        calls[0].resolve(answer(0.1)); // answer for the empty position: stale
        await flush();
        expect(x.answer).toBeNull();
        expect(x.pending).toBe(true);
        expect(calls).toHaveLength(2);
        expect(calls[1].pos.you.marks[0].slice(0, 3)).toEqual([true, true, true]);
        calls[1].resolve(answer(0.6));
        await flush();
        expect(x.answer?.eval.p_win).toBe(0.6);
        expect(x.pending).toBe(false);
        expect(calls).toHaveLength(2);
    });

    it('remembers the position and restores it', () => {
        const store = memStore();
        const { engine } = manualEngine();
        const x = new ExplorerController(engine, store);
        x.init();
        x.tapStrike('opponent', 1);
        const y = new ExplorerController(manualEngine().engine, store);
        y.init();
        expect(y.pos.opponent.strikes).toBe(2);
    });

    it('loads a link over the remembered position, and keeps it on a bad link', () => {
        const store = memStore();
        const x = new ExplorerController(manualEngine().engine, store, undefined, () => {});
        x.init();
        x.tapSquare('you', 3, 3);
        const linked = toggleSquare(emptyPosition(), 'opponent', 1, 1);
        expect(x.loadCode(encode(linked))).toBe(true);
        expect(x.pos).toEqual(linked);
        expect(x.loadCode('1nonsense')).toBe(false);
        expect(x.pos).toEqual(linked);
        expect(x.notice).toBe(BAD_LINK);
    });

    it('moves the die picker to the next unset die and closes it when all are set', () => {
        const x = new ExplorerController(manualEngine().engine, memStore());
        x.init();
        x.openPicker(4);
        x.pickFace(3);
        expect(x.pos.dice[4]).toBe(3);
        expect(x.picker).toBe(5);
        for (const v of [1, 2, 3, 4, 5]) x.pickFace(v);
        expect(x.picker).toBeNull();
        expect(x.pos.dice.every((d) => d !== null)).toBe(true);
    });

    it('copies a link and says so', async () => {
        const copied: string[] = [];
        const x = new ExplorerController(manualEngine().engine, memStore(), async (t) => { copied.push(t); }, () => {});
        x.init();
        await x.copyLink('https://e.x/qwixxer/');
        expect(copied[0]).toBe(`https://e.x/qwixxer/#/explore?p=${encode(emptyPosition())}`);
        expect(x.notice).toBe('Link copied');
    });

    it('evaluates an edit made during a failing call', async () => {
        const calls: Position[] = [];
        let reject!: (e: Error) => void;
        const engine = {
            explore: vi.fn((pos: Position) => {
                calls.push(structuredClone(pos));
                return calls.length === 1 ? new Promise<Answer>((_, rej) => { reject = rej; }) : Promise.resolve(answer(0.7));
            }),
        } as unknown as Engine;
        const x = new ExplorerController(engine, memStore());
        x.init();
        x.refresh();
        x.tapSquare('you', 2, 0);
        reject(new Error('boom'));
        await flush();
        expect(calls).toHaveLength(2);
        expect(calls[1].you.marks[2][0]).toBe(true);
        expect(x.answer?.eval.p_win).toBe(0.7);
        expect(x.error).toBe('');
        expect(x.pending).toBe(false);
    });

    it('shows an engine failure and recovers on the next edit', async () => {
        const engine = { explore: vi.fn().mockRejectedValueOnce(new Error('boom')).mockResolvedValue(answer(0.4)) } as unknown as Engine;
        const x = new ExplorerController(engine, memStore());
        x.init();
        x.refresh();
        await flush();
        expect(x.error).toBe('boom');
        expect(x.pending).toBe(false);
        x.swap();
        await flush();
        expect(x.error).toBe('');
        expect(x.answer?.eval.p_win).toBe(0.4);
    });
});
