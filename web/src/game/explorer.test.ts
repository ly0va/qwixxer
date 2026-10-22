import { describe, expect, it, vi } from 'vitest';
import type { Answer, Engine, Position } from '../engine/protocol';
import type { Store } from './controller.svelte';
import { BAD_LINK, ExplorerController, luckKey, MAX_BATCH, MAX_GAMES, MIN_BATCH } from './explorer.svelte';
import { encode } from './link';
import { analysisKey, emptyPosition, toggleSquare } from './position';

const answer = (p_win: number): Answer => ({ over: null, eval: { p_win, margin: 0 }, moves: null, opponent_white: null, opponent_turn: null });
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
describe('ExplorerController analysis', () => {
    type Call = { kind: 'explore' | 'analyze'; pos: Position; start?: number; count?: number; resolve: (v: any) => void; reject: (e: Error) => void };
    function loopEngine() {
        const calls: Call[] = [];
        const push = (kind: Call['kind'], pos: Position, start?: number, count?: number) =>
            new Promise((resolve, reject) => calls.push({ kind, pos: structuredClone(pos), start, count, resolve, reject }));
        const engine = {
            explore: vi.fn((pos: Position) => push('explore', pos)),
            analyze: vi.fn((pos: Position, start: number, count: number) => push('analyze', pos, start, count)),
        } as unknown as Engine;
        const reply = (games: number, wins: number) => {
            const c = calls.filter((c) => c.kind === 'analyze').at(-1)!;
            c.resolve({ games, win_sum: wins, win_sq_sum: wins, diff_sum: games });
        };
        return { engine, calls, reply, analyses: () => calls.filter((c) => c.kind === 'analyze') };
    }
    const setup = (t = { now: 0 }) => {
        const e = loopEngine();
        const x = new ExplorerController(e.engine, memStore(), undefined, undefined, () => t.now);
        x.init();
        return { ...e, x, t };
    };

    it('does nothing until active, then accumulates batches from the running game count', async () => {
        const { x, analyses, reply } = setup();
        x.tapSquare('you', 0, 0);
        expect(analyses()).toHaveLength(0);
        x.setActive(true);
        expect(analyses()).toHaveLength(1);
        expect(analyses()[0]).toMatchObject({ start: 0, count: MIN_BATCH });
        expect(analyses()[0].pos.dice).toBeNull();
        reply(MIN_BATCH, 5);
        await flush();
        expect(x.analysis?.totals.games).toBe(MIN_BATCH);
        expect(x.analysing).toBe(true);
        expect(analyses()[1].start).toBe(MIN_BATCH);
        reply(analyses()[1].count!, 1);
        await flush();
        expect(x.analysis?.totals.win_sum).toBe(6);
    });

    it('a mark edit restarts at zero and discards the stale reply; a dice edit does not restart', async () => {
        const { x, analyses, reply } = setup();
        x.setActive(true);
        reply(MIN_BATCH, 3);
        await flush();
        x.openPicker(0);
        x.pickFace(4);
        const c1 = analyses()[1].count!;
        reply(c1, c1 / 2); // about even, so not yet precise enough to rest
        await flush();
        expect(x.analysis?.totals.win_sum).toBe(3 + c1 / 2); // dice-only: kept accumulating
        x.tapSquare('you', 0, 0);
        reply(analyses()[2].count!, 7); // started before the edit: stale
        await flush();
        expect(x.analysis?.totals.win_sum).toBe(3 + c1 / 2);
        expect(x.analysis?.key).not.toBe(analysisKey(x.pos)); // old numbers stay (faded) until fresh ones arrive
        expect(analyses()[3].start).toBe(0);
        reply(analyses()[3].count!, 1);
        await flush();
        expect(x.analysis).toMatchObject({ key: analysisKey(x.pos), totals: { win_sum: 1 } });
    });

    it('never has two batches in flight, and an explore call goes before the next batch', async () => {
        const { x, calls, analyses, reply } = setup();
        x.setActive(true);
        x.tapSquare('you', 0, 0);
        x.tapSquare('you', 0, 1);
        x.setActive(true);
        expect(analyses()).toHaveLength(1);
        reply(MIN_BATCH, 0);
        await flush();
        const kinds = calls.map((c) => c.kind);
        expect(kinds.indexOf('explore')).toBeLessThan(kinds.lastIndexOf('analyze'));
        expect(analyses()).toHaveLength(2);
    });

    it('pauses when inactive and resumes with its totals', async () => {
        const { x, analyses, reply } = setup();
        x.setActive(true);
        x.setActive(false);
        reply(MIN_BATCH, 2);
        await flush();
        expect(x.analysis?.totals.games).toBe(MIN_BATCH); // the in-flight batch still counts
        expect(analyses()).toHaveLength(1);
        expect(x.analysing).toBe(false);
        x.setActive(true);
        expect(analyses()).toHaveLength(2);
        expect(analyses()[1].start).toBe(MIN_BATCH);
    });

    it('sends nothing for a finished game, and restarts when edited back to a live one', async () => {
        const { x, analyses, reply } = setup();
        x.tapStrike('opponent', 3); // 4 strikes
        x.setActive(true);
        expect(analyses()).toHaveLength(0);
        x.tapStrike('opponent', 3); // back to 3
        expect(analyses()).toHaveLength(1);
        reply(MIN_BATCH, 1);
        await flush();
        expect(x.analysis?.totals.games).toBe(MIN_BATCH);
    });

    it('stops quietly on an engine error and starts again on the next edit', async () => {
        const { x, analyses } = setup();
        x.setActive(true);
        analyses()[0].reject(new Error('engine was reset'));
        await flush();
        expect(x.analysing).toBe(false);
        expect(x.error).toBe('');
        expect(analyses()).toHaveLength(1);
        x.tapSquare('you', 0, 0);
        expect(analyses()).toHaveLength(2);
    });

    it('adapts the batch size toward 100 ms within bounds', async () => {
        const t = { now: 0 };
        const { x, analyses, reply } = setup(t);
        x.setActive(true);
        t.now += 10; // 8 games in 10 ms → 80 next
        reply(MIN_BATCH, 4);
        await flush();
        expect(analyses()[1].count).toBe(80);
        t.now += 1000; // 80 games in 1 s → 8 next
        reply(80, 40);
        await flush();
        expect(analyses()[2].count).toBe(MIN_BATCH);
        t.now += 1; // 8 games in 1 ms → 800
        reply(MIN_BATCH, 4);
        await flush();
        expect(analyses()[3].count).toBe(Math.min(800, MAX_BATCH));
        t.now += 1; // that many games in 1 ms → far over the cap
        reply(analyses()[3].count!, analyses()[3].count! / 2);
        await flush();
        expect(analyses()[4].count).toBe(MAX_BATCH);
    });
});

describe('ExplorerController analysis fixes', () => {
    function eng() {
        const an: { count: number; resolve: (v: any) => void }[] = [];
        const engine = {
            explore: vi.fn(() => new Promise(() => {})),
            analyze: vi.fn((_p: Position, _s: number, count: number) => new Promise((resolve) => an.push({ count, resolve }))),
        } as unknown as Engine;
        return { engine, an };
    }

    it('a finished game counts as settled, so its card is not faded', () => {
        const { engine } = eng();
        const x = new ExplorerController(engine, memStore());
        x.init();
        expect(x.settled).toBe(false); // nothing analysed yet
        x.tapStrike('opponent', 3); // 4 strikes: game over
        expect(x.settled).toBe(true);
    });

    it('a position edit restarts the batch size at the minimum', async () => {
        const t = { now: 0 };
        const { engine, an } = eng();
        const x = new ExplorerController(engine, memStore(), undefined, undefined, () => t.now);
        x.init();
        x.setActive(true);
        t.now += 1; // 8 games in 1 ms: the next batch grows
        an[0].resolve({ games: MIN_BATCH, win_sum: 0, win_sq_sum: 0, diff_sum: 0 });
        await flush();
        expect(an[1].count).toBeGreaterThan(MIN_BATCH);
        expect(x.settled).toBe(true);
        x.tapSquare('you', 0, 0);
        expect(x.settled).toBe(false);
        an[1].resolve({ games: an[1].count, win_sum: 0, win_sq_sum: 0, diff_sum: 0 }); // stale
        await flush();
        expect(an[2].count).toBe(MIN_BATCH);
    });
});

describe('ExplorerController luck', () => {
    type Call = { kind: string; pos: Position; resolve: (v: any) => void };
    function setupLuck() {
        const calls: Call[] = [];
        const push = (kind: string) => (pos: Position) => new Promise((resolve) => calls.push({ kind, pos: structuredClone(pos), resolve }));
        const engine = { explore: vi.fn(push('explore')), analyze: vi.fn(push('analyze')), luck: vi.fn(push('luck')) } as unknown as Engine;
        const x = new ExplorerController(engine, memStore());
        x.init();
        const kinds = () => calls.map((c) => c.kind);
        const luck = () => calls.filter((c) => c.kind === 'luck');
        return { x, calls, kinds, luck };
    }

    it('asks right after explore when all dice are set, before the next analysis batch', async () => {
        const { x, kinds, luck } = setupLuck();
        x.setActive(true);
        x.random();
        expect(kinds().slice(-2)).toEqual(['explore', 'luck']);
        expect(luck()[0].pos.dice).toHaveLength(6);
        luck()[0].resolve({ this_roll: 0.55, average: 0.5 });
        await flush();
        expect(x.luck).toEqual({ key: luckKey(x.pos), value: { this_roll: 0.55, average: 0.5 } });
    });

    it('a dice edit asks again and never lands an older reply', async () => {
        const { x, luck } = setupLuck();
        x.random();
        x.openPicker(0);
        x.pickFace(x.pos.dice[0] === 6 ? 1 : 6);
        expect(luck()).toHaveLength(1); // one in flight: the edit waits for it
        luck()[0].resolve({ this_roll: 0.1, average: 0.5 });
        await flush();
        expect(x.luck).toBeNull(); // for the old dice: dropped
        expect(luck()).toHaveLength(2);
        luck()[1].resolve({ this_roll: 0.6, average: 0.5 });
        await flush();
        expect(x.luck).toEqual({ key: luckKey(x.pos), value: { this_roll: 0.6, average: 0.5 } });
    });

    it('clearing the dice clears luck and asks nothing', async () => {
        const { x, luck } = setupLuck();
        x.random();
        luck()[0].resolve({ this_roll: 0.55, average: 0.5 });
        await flush();
        x.clearDice();
        await flush();
        expect(x.luck).toBeNull();
        expect(luck()).toHaveLength(1);
    });

    it('a finished game asks nothing', () => {
        const { x, luck } = setupLuck();
        x.tapStrike('opponent', 0); x.tapStrike('opponent', 1); x.tapStrike('opponent', 2); x.tapStrike('opponent', 3);
        x.random();
        expect(luck()).toHaveLength(0);
    });
});

describe('ExplorerController analysis throttle', () => {
    function setupRest() {
        const t = { now: 0 };
        const an: { start: number; count: number; resolve: (v: any) => void }[] = [];
        const timers: { f: () => void; ms: number }[] = [];
        const engine = {
            explore: vi.fn(() => new Promise(() => {})),
            analyze: vi.fn((_p: Position, start: number, count: number) => new Promise((resolve) => an.push({ start, count, resolve }))),
        } as unknown as Engine;
        const x = new ExplorerController(engine, memStore(), undefined, (f, ms) => void timers.push({ f, ms }), () => t.now);
        x.init();
        // 5,000 games at 50%: ± ≈ 1.4 points, below the 2-point threshold.
        const precise = (games = 5000) => ({ games, win_sum: games / 2, win_sq_sum: games / 2, diff_sum: 0 });
        return { x, an, timers, t, precise };
    }

    it('runs back to back while ± ≥ 2 points', async () => {
        const { x, an, timers } = setupRest();
        x.setActive(true);
        an[0].resolve({ games: 100, win_sum: 50, win_sq_sum: 50, diff_sum: 0 }); // ± ≈ 9.8
        await flush();
        expect(an).toHaveLength(2);
        expect(timers).toHaveLength(0);
    });

    it('below ± 2 points rests as long as the batch took (half a core)', async () => {
        const { x, an, timers, t, precise } = setupRest();
        x.setActive(true);
        t.now += 90;
        an[0].resolve(precise());
        await flush();
        expect(an).toHaveLength(1);
        expect(timers.map((r) => r.ms)).toEqual([90]);
        expect(x.analysing).toBe(true);
        timers[0].f();
        await flush();
        expect(an).toHaveLength(2);
        expect(an[1].start).toBe(5000);
    });

    it('an edit during the rest restarts at once, from zero', async () => {
        const { x, an, precise } = setupRest();
        x.setActive(true);
        an[0].resolve(precise());
        await flush();
        x.tapSquare('you', 0, 0);
        await flush();
        expect(an).toHaveLength(2);
        expect(an[1].start).toBe(0);
    });

    it('a small one-sided sample is not precise enough to rest', async () => {
        const { x, an, timers } = setupRest();
        x.setActive(true);
        an[0].resolve({ games: 8, win_sum: 0, win_sq_sum: 0, diff_sum: -80 }); // ± 0 by the plain formula
        await flush();
        expect(timers).toHaveLength(0);
        expect(an).toHaveLength(2);
    });

    it('stops for good at MAX_GAMES, asking only for the games left', async () => {
        const { x, an, timers, precise } = setupRest();
        x.setActive(true);
        an[0].resolve(precise(MAX_GAMES - 100));
        await flush();
        timers[0].f();
        await flush();
        expect(an[1]).toMatchObject({ start: MAX_GAMES - 100, count: 100 });
        an[1].resolve(precise(100));
        await flush();
        expect(x.analysis?.totals.games).toBe(MAX_GAMES);
        expect(an).toHaveLength(2);
        expect(timers).toHaveLength(1);
        expect(x.analysing).toBe(false);
        x.setActive(false);
        x.setActive(true);
        await flush();
        expect(an).toHaveLength(2); // still done on return
        x.tapSquare('you', 0, 0);
        await flush();
        expect(an).toHaveLength(3); // a new position starts again
        expect(an[2].start).toBe(0);
    });

    it('leaving during the rest sends nothing more', async () => {
        const { x, an, timers, precise } = setupRest();
        x.setActive(true);
        an[0].resolve(precise());
        await flush();
        x.setActive(false);
        await flush();
        expect(x.analysing).toBe(false);
        timers[0].f();
        await flush();
        expect(an).toHaveLength(1);
    });
});
