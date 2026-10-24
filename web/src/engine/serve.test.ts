import { describe, expect, it } from 'vitest';
import { serve, type Reply, type Wasm } from './serve';

class FakeGame {
    view() { return '{"phase":"you_to_roll"}'; }
    code() { return '1BAAAA'; }
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
        return JSON.stringify({ over: null, eval: { p_win: 0.5, margin: 0 }, moves: null, echo: p.roller });
    }
    analyze(json: string, start: number, count: number) {
        const p = JSON.parse(json);
        if (p.over) throw new Error('the game is over');
        return JSON.stringify({ games: count, win_sum: start, win_sq_sum: 0, diff_sum: 0 });
    }
    luck(json: string, first: number, count: number) {
        const p = JSON.parse(json);
        if (!p.dice) throw new Error('no dice');
        return JSON.stringify({ this_roll: 0.55, average: first + count / 10, echo: p.roller });
    }
    move_one_ply(json: string, marks: string, strike: boolean) {
        if (JSON.parse(json).roller !== 'you') throw new Error('1-ply values are for your roll');
        return JSON.parse(marks).length / 10 + (strike ? 0.5 : 0);
    }
    review_game(code: string) {
        if (code === 'bad') throw new Error('bad code');
        return JSON.stringify({ bot: 'master', bot_first: true, rolls: [], over: null, echo: code });
    }
    review_roll(code: string, k: number) {
        return JSON.stringify({ start: null, this_roll: k / 10, after_move: 0.5, you_cost: 0, bot_cost: 0, echo: code });
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
    it('luck shares the explorer and passes the position', async () => {
        const { send } = setup(0);
        await send(1, 'explore', [{ roller: 'you' }]);
        const made = FakeExplorer.made;
        expect(await send(2, 'luck', [{ roller: 'opponent', dice: [1, 2, 3, 4, 5, 6] }, 6, 3]))
            .toEqual({ id: 2, ok: true, result: { this_roll: 0.55, average: 6.3, echo: 'opponent' } });
        expect(FakeExplorer.made).toBe(made);
        expect(await send(3, 'luck', [{ roller: 'you' }, 0, 3])).toMatchObject({ id: 3, ok: false, error: 'no dice', kind: 'engine' });
    });
    it('review calls share the explorer and pass the code', async () => {
        const { send } = setup(0);
        await send(1, 'explore', [{ roller: 'you' }]);
        const made = FakeExplorer.made;
        expect(await send(2, 'reviewGame', ['1BAAAA'])).toMatchObject({ ok: true, result: { bot: 'master', echo: '1BAAAA' } });
        expect(await send(3, 'reviewRoll', ['1BAAAA', 3])).toMatchObject({ ok: true, result: { this_roll: 0.3, echo: '1BAAAA' } });
        expect(FakeExplorer.made).toBe(made);
        expect(await send(4, 'reviewGame', ['bad'])).toMatchObject({ ok: false, kind: 'engine', error: 'bad code' });
    });
    it('onePly passes the position, the marks and the strike to the explorer', async () => {
        const { send } = setup(0);
        expect(await send(1, 'onePly', [{ roller: 'you' }, [{ row: 0, n: 2 }, { row: 1, n: 3 }], true])).toEqual({ id: 1, ok: true, result: 0.7 });
        expect(await send(2, 'onePly', [{ roller: 'opponent' }, [], false])).toMatchObject({ ok: false, error: '1-ply values are for your roll' });
    });
    it('gameCode asks the game', async () => {
        const { send } = setup(0);
        expect(await send(1, 'gameCode', [])).toMatchObject({ ok: false, error: 'no game' });
        await send(2, 'start', ['master', 1]);
        expect(await send(3, 'gameCode', [])).toEqual({ id: 3, ok: true, result: '1BAAAA' });
    });
    it('analyze shares the explorer and passes the batch range', async () => {
        const { send } = setup(0);
        const made = FakeExplorer.made;
        const r = await send(1, 'analyze', [{ roller: 'you' }, 40, 25]);
        expect(r).toEqual({ id: 1, ok: true, result: { games: 25, win_sum: 40, win_sq_sum: 0, diff_sum: 0 } });
        await send(2, 'explore', [{ roller: 'you' }]);
        expect(FakeExplorer.made).toBe(made + 1);
        const bad = await send(3, 'analyze', [{ over: true }, 0, 10]);
        expect(bad).toMatchObject({ id: 3, ok: false, error: 'the game is over', kind: 'engine' });
    });
});
