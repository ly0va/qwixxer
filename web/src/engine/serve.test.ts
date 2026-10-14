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

function setup(failures: number) {
    const s = { inits: 0, fail: failures };
    const posted: Reply[] = [];
    const wasm = {
        init: async () => {
            s.inits++;
            if (s.fail-- > 0) throw new Error('fetch failed');
        },
        WasmGame: FakeGame,
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
});
