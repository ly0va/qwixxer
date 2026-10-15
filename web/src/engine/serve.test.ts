import { describe, expect, it } from 'vitest';
import { serve, type Reply, type Wasm } from './serve';

class FakeGame {
    view() { return '{"phase":"you_to_roll"}'; }
    free() {}
    static restore(_bot: string, snap: string) {
        if (snap === 'bad') throw new Error('bad snapshot');
        return new FakeGame();
    }
}

class FakeExplorer {
    static made = 0;
    constructor() { FakeExplorer.made++; }
    explore(json: string) {
        const p = JSON.parse(json);
        if (p.bad) throw new Error('bad position');
        return JSON.stringify({ over: null, eval: { p_win: 0.5, margin: 0, sigma: 10 }, moves: null, echo: p.roller });
    }
    analyze(json: string, start: number, count: number) {
        const p = JSON.parse(json);
        if (p.over) throw new Error('the game is over');
        return JSON.stringify({ games: count, wins: start, ties: 0, diff_sum: 0, diff_sq_sum: 0 });
    }
    free() {}
}

function setup(failures: number) {
    const s = { inits: 0, fail: failures };
    const posted: Reply[] = [];
    const wasm = {
        init: async () => {
            s.inits++;
            if (s.fail-- > 0) throw new Error('fetch failed');
        },
        WasmGame: FakeGame,
        WasmExplorer: FakeExplorer,
    } as unknown as Wasm;
    const handle = serve(wasm, (r) => posted.push(r));
    const send = async (id: number, cmd: string, args: unknown[]) => {
        await handle({ id, cmd, args });
        return posted.find((m) => m.id === id);
    };
    return { s, send };
}

describe('engine worker handler', () => {
    it('reports a failed wasm load as an engine error and retries it on the next call', async () => {
        const { s, send } = setup(1);
        // The eager load fails and the first call (which waited on it) reports
        // that; the next call loads afresh.
        expect(await send(1, 'restore', ['master', '{}', 1])).toMatchObject({ ok: false, kind: 'engine', error: 'engine failed to load: fetch failed' });
        expect(await send(2, 'restore', ['master', '{}', 1])).toMatchObject({ ok: true, result: { phase: 'you_to_roll' } });
        expect(s.inits).toBe(2);
        await send(3, 'snapshot', []);
        expect(s.inits).toBe(2); // loaded once, for good
    });

    it('tells a bad snapshot apart from a load failure', async () => {
        const { send } = setup(0);
        expect(await send(1, 'restore', ['master', 'bad', 1])).toMatchObject({ ok: false, kind: 'snapshot', error: 'bad snapshot' });
        expect(await send(2, 'nope', [])).toMatchObject({ ok: false, kind: 'engine' });
    });

    it('answers explore with one lazily created explorer, and reports bad positions as engine errors', async () => {
        const { send } = setup(0);
        FakeExplorer.made = 0;
        expect(await send(1, 'explore', [{ roller: 'you' }])).toMatchObject({ ok: true, result: { echo: 'you', eval: { p_win: 0.5 } } });
        expect(await send(2, 'explore', [{ roller: 'opponent' }])).toMatchObject({ ok: true, result: { echo: 'opponent' } });
        expect(FakeExplorer.made).toBe(1);
        expect(await send(3, 'explore', [{ bad: true }])).toMatchObject({ ok: false, kind: 'engine', error: 'bad position' });
    });
    it('analyze shares the explorer and passes the batch range', async () => {
        const { send } = setup(0);
        const made = FakeExplorer.made;
        const r = await send(1, 'analyze', [{ roller: 'you' }, 40, 25]);
        expect(r).toEqual({ id: 1, ok: true, result: { games: 25, wins: 40, ties: 0, diff_sum: 0, diff_sq_sum: 0 } });
        await send(2, 'explore', [{ roller: 'you' }]);
        expect(FakeExplorer.made).toBe(made + 1);
        const bad = await send(3, 'analyze', [{ over: true }, 0, 10]);
        expect(bad).toMatchObject({ id: 3, ok: false, error: 'the game is over', kind: 'engine' });
    });
});
