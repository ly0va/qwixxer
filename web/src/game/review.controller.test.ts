import { describe, expect, it, vi } from 'vitest';
import type { Engine, ReviewGame, RollAnalysis } from '../engine/protocol';
import type { Store } from './controller.svelte';
import { LAST_GAME, ReviewController } from './review.svelte';

function memStore(init: Record<string, string> = {}): Store {
    const data = { ...init };
    return { get: (k) => data[k] ?? null, set: (k, v) => { data[k] = v; } };
}
const empty = { marks: [0, 1, 2, 3].map(() => Array(11).fill(false)), strikes: 0 };
const pos = { you: empty, opponent: empty, locked: [false, false, false, false], dice: [1, 2, 3, 4, 5, 6], roller: 'opponent' as const };
const rec = { roller: 'bot' as const, you: [], you_strike: false, bot_white: null, bot_colour: null, bot_strike: true, locked: [] };
const gameOf = (n: number): ReviewGame => ({ bot: 'master', bot_first: true, over: null, rolls: Array.from({ length: n }, () => ({ before: pos, record: rec, after: pos })) });
const analysis = (k: number): RollAnalysis => ({ start: k === 0 ? 0.5 : null, start_var: k === 0 ? 0.002 : null, this_roll: 0.5, after_move: 0.5, after_var: 0.002, you_cost: 0, you_best: { marks: [], strike: false, after: pos }, bot_cost: 0 });

/** reviewRoll resolves (or rejects) only when the test says so. */
function engine(games: Record<string, number>) {
    const rolls: { code: string; k: number; resolve: (a: RollAnalysis) => void; reject: (e: Error) => void }[] = [];
    const e = {
        reviewGame: vi.fn(async (code: string) => {
            if (!(code in games)) throw new Error('bad code');
            return gameOf(games[code]);
        }),
        reviewRoll: vi.fn((code: string, k: number) => new Promise<RollAnalysis>((resolve, reject) => rolls.push({ code, k, resolve, reject }))),
    } as unknown as Engine;
    return { e, rolls };
}
const flush = () => new Promise((r) => setTimeout(r, 0));

describe('ReviewController', () => {
    it('replays the game, then analyses roll by roll, one at a time', async () => {
        const { e, rolls } = engine({ A: 3 });
        const r = new ReviewController(e, memStore());
        r.setActive(true);
        await r.load('A', false);
        expect(r.game?.rolls).toHaveLength(3);
        expect(r.sel).toBe(0);
        await flush();
        expect(rolls.map((x) => x.k)).toEqual([0]);
        rolls[0].resolve(analysis(0));
        await flush();
        expect(rolls.map((x) => x.k)).toEqual([0, 1]);
        expect(r.an[0]?.start).toBe(0.5);
    });

    it('waits while inactive and resumes at the first missing roll', async () => {
        const { e, rolls } = engine({ A: 3 });
        const r = new ReviewController(e, memStore());
        await r.load('A', false);
        await flush();
        expect(rolls).toHaveLength(0);
        r.setActive(true);
        await flush();
        rolls[0].resolve(analysis(0));
        await flush();
        r.setActive(false);
        rolls[1].resolve(analysis(1));
        await flush();
        expect(rolls).toHaveLength(2);
        r.setActive(true);
        await flush();
        expect(rolls.map((x) => x.k)).toEqual([0, 1, 2]);
    });

    it('drops a reply for a game that is no longer shown', async () => {
        const { e, rolls } = engine({ A: 3, B: 2 });
        const r = new ReviewController(e, memStore());
        r.setActive(true);
        await r.load('A', false);
        await flush();
        await r.load('B', true);
        rolls[0].resolve(analysis(0)); // A's roll 0
        await flush();
        expect(r.code).toBe('B');
        expect(r.shared).toBe(true);
        expect(r.an[0]).toBeUndefined();
        const b = rolls.filter((x) => x.code === 'B');
        expect(b.map((x) => x.k)).toEqual([0]);
    });

    it('stops on an engine error and retries', async () => {
        const { e, rolls } = engine({ A: 2 });
        const r = new ReviewController(e, memStore());
        r.setActive(true);
        await r.load('A', false);
        await flush();
        rolls[0].reject(new Error('engine was reset'));
        await flush();
        expect(r.error).toBe('engine was reset');
        expect(rolls).toHaveLength(1);
        r.retry();
        await flush();
        expect(r.error).toBeNull();
        expect(rolls.map((x) => x.k)).toEqual([0, 0]);
    });

    it('a bad code is an error with no game', async () => {
        const { e } = engine({});
        const r = new ReviewController(e, memStore());
        await r.load('nope', true);
        expect(r.game).toBeNull();
        expect(r.error).toBe('bad code');
    });

    it('with no code shows the last finished game, or nothing', async () => {
        const { e } = engine({ L: 2 });
        const r = new ReviewController(e, memStore({ [LAST_GAME]: 'L' }));
        await r.load(null, false);
        expect(r.code).toBe('L');
        const none = engine({});
        const r2 = new ReviewController(none.e, memStore());
        await r2.load(null, false);
        expect(r2.game).toBeNull();
        expect(none.e.reviewGame).not.toHaveBeenCalled();
    });

    it('steps within the game and resets the side on a new selection', async () => {
        const { e } = engine({ A: 3 });
        const r = new ReviewController(e, memStore());
        await r.load('A', false);
        r.step(-1);
        expect(r.sel).toBe(0);
        r.last();
        expect(r.sel).toBe(6);
        r.step(1);
        expect(r.sel).toBe(6);
        r.setSide('best');
        r.select(4);
        expect(r.side).toBe('played');
        r.first();
        expect(r.sel).toBe(0);
        await r.load('A', false); // same code: nothing changes
        expect(r.sel).toBe(0);
    });

    it('show picks the side after a roll\'s move, moving there from its dice point; at the start it does nothing', async () => {
        const { e } = engine({ A: 3 });
        const r = new ReviewController(e, memStore());
        await r.load('A', false);
        r.show('best');
        expect([r.sel, r.side]).toEqual([0, 'played']);
        r.select(3); // roll 2's dice
        r.show('best');
        expect([r.sel, r.side]).toEqual([4, 'best']);
        r.show('played');
        expect([r.sel, r.side]).toEqual([4, 'played']);
    });

    it('clears an old error and the shared flag when there is nothing to show', async () => {
        const { e } = engine({});
        const r = new ReviewController(e, memStore());
        await r.load('nope', true);
        await r.load(null, false);
        expect(r.error).toBeNull();
        expect(r.shared).toBe(false);
        expect(r.code).toBeNull();
    });

    it('retry reopens a game whose replay failed (the engine may have failed to load)', async () => {
        let fail = true;
        const e = {
            reviewGame: vi.fn(async () => { if (fail) throw new Error('engine failed to load'); return gameOf(1); }),
            reviewRoll: vi.fn(() => new Promise(() => {})),
        } as unknown as Engine;
        const r = new ReviewController(e, memStore());
        await r.load('A', false);
        expect(r.error).toBe('engine failed to load');
        fail = false;
        r.retry();
        await flush();
        expect(r.error).toBeNull();
        expect(r.game?.rolls).toHaveLength(1);
    });
});
