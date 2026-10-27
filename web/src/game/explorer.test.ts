import { describe, expect, it, vi } from 'vitest';
import type { Answer, Engine, Luck, MoveOut, Position } from '../engine/protocol';
import type { Store } from './controller.svelte';
import { BAD_LINK, ExplorerController, LUCK_CHUNK, luckKey, MAX_BATCH, MAX_GAMES, MIN_BATCH } from './explorer.svelte';
import { encode } from './link';
import { analysisKey, emptyPosition, toggleSquare } from './position';

const ends = { marks: [0, 0, 0, 0], strikes: 0 };
const answer = (p_win: number): Answer => ({ over: null, eval: { p_win, margin: 0, expected: { you: ends, opponent: ends } }, moves: null, opponent_white: null, opponent_turn: null });
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
    /** A luck chunk (no roll values unless `values` gives them). */
    const chunk = (this_roll: number, average: number, values: Luck['values'] = []): Luck => ({ this_roll, average, values });
    type Call = { kind: string; pos: Position; first?: number; count?: number; resolve: (v: any) => void };
    function setupLuck(store = memStore()) {
        const calls: Call[] = [];
        const push = (kind: string) => (pos: Position, first?: number, count?: number) =>
            new Promise((resolve) => calls.push({ kind, pos: JSON.parse(JSON.stringify(pos)), first, count, resolve }));
        const engine = { explore: vi.fn(push('explore')), analyze: vi.fn(push('analyze')), luck: vi.fn(push('luck')) } as unknown as Engine;
        const x = new ExplorerController(engine, store);
        x.init();
        const kinds = () => calls.map((c) => c.kind);
        const luck = () => calls.filter((c) => c.kind === 'luck');
        // Answers every chunk asked for until none is pending: each adds 0.05 to the average.
        const finish = async (this_roll: number) => {
            for (let c = luck().find((c) => !(c as any).done); c; c = luck().find((c) => !(c as any).done)) {
                (c as any).done = true;
                c.resolve(chunk(this_roll, 0.05 * (c.count ?? 0) / 3, [[(c.first ?? 0) / 100, (c.count ?? 0) / 21]]));
                await flush();
            }
        };
        return { x, calls, kinds, luck, finish };
    }

    it('asks right after explore, in chunks of white pairs, and adds them up', async () => {
        const { x, kinds, luck, finish } = setupLuck();
        x.setActive(true);
        x.random();
        expect(kinds().slice(-2)).toEqual(['explore', 'luck']);
        expect(luck()[0]).toMatchObject({ first: 0, count: LUCK_CHUNK });
        expect(luck()[0].pos.dice).toHaveLength(6);
        await finish(0.55);
        expect(luck().map((c) => c.first)).toEqual([0, 3, 6, 9, 12, 15, 18]);
        expect(x.luck?.key).toBe(luckKey(x.pos));
        expect(x.luck?.value.this_roll).toBe(0.55);
        expect(x.luck?.value.average).toBeCloseTo(0.35);
        expect(x.luck?.value.values).toHaveLength(7);
        expect(x.luck?.value.values[6]).toEqual([0.18, 3 / 21]);
        expect(x.luck?.value.values.reduce((a, v) => a + v[1], 0)).toBeCloseTo(1);
    });

    it('nothing shows until the last chunk is in', async () => {
        const { x, luck } = setupLuck();
        x.random();
        luck()[0].resolve(chunk(0.55, 0.05));
        await flush();
        expect(x.luck).toBeNull();
        expect(luck()).toHaveLength(2);
    });

    it('an edit between chunks gets its explore in before the next chunk, which starts over', async () => {
        const { x, calls, kinds, luck, finish } = setupLuck();
        x.random();
        x.openPicker(0);
        x.pickFace(x.pos.dice[0] === 6 ? 1 : 6);
        calls.find((c) => c.kind === 'explore')!.resolve(answer(0.5)); // its coalesced follow-up goes out
        await flush();
        expect(kinds().filter((k) => k === 'explore')).toHaveLength(2);
        expect(kinds().at(-1)).toBe('explore'); // the luck chunk in flight doesn't hold it back
        luck()[0].resolve(chunk(0.1, 0.05));
        await flush();
        expect(luck()[1]).toMatchObject({ first: 0 }); // the new dice, from the start
        (luck()[0] as any).done = true;
        await finish(0.6);
        expect(x.luck?.key).toBe(luckKey(x.pos));
        expect(x.luck?.value.this_roll).toBe(0.6);
    });

    it('entering the explorer asks for the move list first', () => {
        const store = memStore();
        const first = setupLuck(store);
        first.x.random(); // saved with dice
        const { x, kinds } = setupLuck(store);
        x.refresh();
        expect(kinds()).toEqual(['explore', 'luck']);
    });

    it('clearing the dice clears luck and asks nothing', async () => {
        const { x, luck, finish } = setupLuck();
        x.random();
        await finish(0.55);
        const n = luck().length;
        x.clearDice();
        await flush();
        expect(x.luck).toBeNull();
        expect(luck()).toHaveLength(n);
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

describe('ExplorerController 1-ply', () => {
    type Call = { kind: string; args: any[]; resolve: (v: any) => void; reject: (e: Error) => void };
    const mv = (n: number, p_win: number): MoveOut => ({ marks: [{ row: 0, n }], strike: false, pass: false, p_win, bots: [], note: null });
    const withMoves = (moves: MoveOut[]): Answer => ({ ...answer(0.5), moves });
    function setup() {
        const calls: Call[] = [];
        const push = (kind: string) => (...args: any[]) =>
            new Promise((resolve, reject) => calls.push({ kind, args: JSON.parse(JSON.stringify(args)), resolve, reject }));
        const engine = { explore: vi.fn(push('explore')), luck: vi.fn(push('luck')), onePly: vi.fn(push('onePly')) } as unknown as Engine;
        const x = new ExplorerController(engine, memStore());
        x.init();
        const of = (kind: string) => calls.filter((c) => c.kind === kind);
        return { x, of };
    }
    const list = [mv(2, 0.452), mv(3, 0.441), mv(4, 0.43), mv(5, 0.42)];

    it('values your top 3 one at a time, then re-ranks them once', async () => {
        const { x, of } = setup();
        x.random();
        of('explore')[0].resolve(withMoves(list));
        await flush();
        expect(x.refining).toBe(true);
        expect(x.ranked.map((m) => m.marks[0].n)).toEqual([2, 3, 4, 5]);
        expect(of('onePly')).toHaveLength(1);
        expect(of('onePly')[0].args[0].roller).toBe('you');
        expect(of('onePly')[0].args.slice(1)).toEqual([[{ row: 0, n: 2 }], false]);
        of('onePly')[0].resolve(0.4396);
        await flush();
        of('onePly')[1].resolve(0.4398);
        await flush();
        expect(x.ranked.every((m) => !m.refined)).toBe(true); // nothing moves until all three are in
        expect(of('onePly')[2].args[1]).toEqual([{ row: 0, n: 4 }]);
        of('onePly')[2].resolve(0.40);
        await flush();
        expect(of('onePly')).toHaveLength(3);
        expect(x.refining).toBe(false);
        expect(x.ranked.map((m) => [m.marks[0].n, m.p_win, m.refined])).toEqual(
            [[3, 0.4398, true], [2, 0.4396, true], [4, 0.40, true], [5, 0.42, false]]);
    });

    it("the opponent's roll stays static and asks nothing", async () => {
        const { x, of } = setup();
        x.setRoller('opponent');
        x.random();
        of('explore')[0].resolve(answer(0.5)); // outdated by the dice
        await flush();
        of('explore')[1].resolve(withMoves(list));
        await flush();
        expect(of('onePly')).toHaveLength(0);
        expect(x.refining).toBe(false);
        expect(x.ranked.map((m) => m.p_win)).toEqual([0.452, 0.441, 0.43, 0.42]);
    });

    it('an edit drops the old values and refines the new list', async () => {
        const { x, of } = setup();
        x.random();
        of('explore')[0].resolve(withMoves(list));
        await flush();
        x.tapSquare('you', 0, 0);
        of('onePly')[0].resolve(0.1);
        await flush();
        expect(of('onePly')).toHaveLength(1); // the old list's next move isn't asked
        of('explore')[1].resolve(withMoves([mv(9, 0.6)]));
        await flush();
        expect(of('onePly')).toHaveLength(2);
        expect(of('onePly')[1].args[1]).toEqual([{ row: 0, n: 9 }]);
        of('onePly')[1].resolve(0.61);
        await flush();
        expect(x.ranked.map((m) => [m.marks[0].n, m.p_win])).toEqual([[9, 0.61]]);
    });

    it('an engine error leaves the static list', async () => {
        const { x, of } = setup();
        x.random();
        of('explore')[0].resolve(withMoves(list));
        await flush();
        of('onePly')[0].reject(new Error('engine was reset'));
        await flush();
        expect(x.refining).toBe(false);
        expect(x.ranked.map((m) => m.refined)).toEqual([false, false, false, false]);
    });
});

describe('ExplorerController play', () => {
    type Call = { kind: string; args: any[]; resolve: (v: any) => void; reject: (e: Error) => void };
    const mv = (n: number): MoveOut => ({ marks: [{ row: 0, n }], strike: false, pass: false, p_win: 0.5, bots: [], note: null });
    function setup() {
        const calls: Call[] = [];
        const push = (kind: string) => (...args: any[]) =>
            new Promise((resolve, reject) => calls.push({ kind, args: JSON.parse(JSON.stringify(args)), resolve, reject }));
        const engine = { explore: vi.fn(push('explore')), luck: vi.fn(push('luck')), onePly: vi.fn(push('onePly')), play: vi.fn(push('play')) } as unknown as Engine;
        const x = new ExplorerController(engine, memStore());
        x.init();
        const of = (kind: string) => calls.filter((c) => c.kind === kind);
        return { x, of };
    }
    /** The engine's next position: your red n marked, the opponent's yellow 7, their roll, no dice. */
    const next = (from: Position, n: number): Position => {
        const p: Position = JSON.parse(JSON.stringify(from));
        p.you.marks[0][n - 2] = true;
        p.opponent.marks[1][5] = true;
        return { ...p, dice: null, roller: from.roller === 'you' ? 'opponent' : 'you' };
    };
    /** The answer for the position as it stands, so its moves can be played. */
    async function answered(x: ExplorerController, of: (k: string) => Call[]) {
        // An answer for an outdated position is dropped and asked again: answer until one lands.
        for (let i = 0; i < 5 && !x.canPlay; i++) {
            of('explore').at(-1)!.resolve(answer(0.5));
            await flush();
        }
        expect(x.canPlay).toBe(true);
    }
    async function playOne(x: ExplorerController, of: (k: string) => Call[], n: number) {
        await answered(x, of);
        const before = JSON.parse(JSON.stringify(x.pos));
        const done = x.play(mv(n));
        const call = of('play').at(-1)!;
        expect(call.args.slice(1)).toEqual([[{ row: 0, n }], false]);
        call.resolve({ position: next(call.args[0], n), opponent: [{ row: 1, n: 7 }], opponent_strike: false });
        await done;
        return { before, sent: call.args[0] as Position };
    }

    it('plays a move: the next position, what was played, and undo brings the position back', async () => {
        const { x, of } = setup();
        x.random();
        const { before, sent } = await playOne(x, of, 4);
        expect(sent.dice).toEqual(before.dice);
        expect(x.pos.you.marks[0][2]).toBe(true);
        expect(x.pos.opponent.marks[1][5]).toBe(true);
        expect(x.pos.roller).toBe('opponent');
        expect(x.pos.dice).toEqual(Array(6).fill(null));
        expect(x.lastPlay).toMatchObject({ marks: [{ row: 0, n: 4 }], strike: false, opponent: [{ row: 1, n: 7 }], opponentStrike: false });
        x.undo();
        expect(x.pos).toEqual(before);
        expect(x.lastPlay).toBeNull();
    });

    it('undo steps back through every play, showing the one before', async () => {
        const { x, of } = setup();
        x.random();
        const first = await playOne(x, of, 4);
        x.random();
        const second = await playOne(x, of, 5);
        expect(x.lastPlay?.marks).toEqual([{ row: 0, n: 5 }]);
        x.undo();
        expect(x.pos).toEqual(second.before);
        expect(x.lastPlay?.marks).toEqual([{ row: 0, n: 4 }]);
        x.undo();
        expect(x.pos).toEqual(first.before);
        expect(x.lastPlay).toBeNull();
    });

    it('editing the sheets clears the plays; the dice and whose roll keep them', async () => {
        const { x, of } = setup();
        x.random();
        await playOne(x, of, 4);
        x.random();
        x.setRoller('you');
        x.clearDice();
        expect(x.lastPlay?.marks).toEqual([{ row: 0, n: 4 }]);
        for (const edit of [() => x.tapSquare('you', 3, 0), () => x.tapStrike('you', 0), () => x.tapLock(2), () => x.clear('you'), () => x.clearAll(), () => x.swap()]) {
            x.random();
            await playOne(x, of, 6);
            edit();
            expect(x.lastPlay).toBeNull();
        }
    });

    it('a failed play leaves the position and says so', async () => {
        const { x, of } = setup();
        x.random();
        await answered(x, of);
        const before = JSON.parse(JSON.stringify(x.pos));
        const done = x.play(mv(4));
        of('play')[0].reject(new Error('not a legal move'));
        await done;
        expect(x.pos).toEqual(before);
        expect(x.lastPlay).toBeNull();
        expect(x.notice).toBe('Couldn’t play that move');
    });

    it('a play answered after an edit is dropped', async () => {
        const { x, of } = setup();
        x.random();
        await answered(x, of);
        const done = x.play(mv(4));
        x.tapSquare('you', 3, 0);
        const edited = JSON.parse(JSON.stringify(x.pos));
        of('play')[0].resolve({ position: next(of('play')[0].args[0], 4), opponent: [], opponent_strike: false });
        await done;
        expect(x.pos).toEqual(edited);
        expect(x.lastPlay).toBeNull();
    });
});

describe('ExplorerController play guards and badges', () => {
    type Call = { kind: string; args: any[]; resolve: (v: any) => void; reject: (e: Error) => void };
    const mv: MoveOut = { marks: [{ row: 0, n: 4 }], strike: false, pass: false, p_win: 0.5, bots: [], note: null };
    const red = (n: number) => ({ marks: [n, 0, 0, 0], strikes: 0 });
    const withEnds = (n: number): Answer => ({ ...answer(0.5), eval: { p_win: 0.5, margin: 0, expected: { you: red(n), opponent: red(0) } }, moves: [mv] });
    function setup() {
        const calls: Call[] = [];
        const push = (kind: string) => (...args: any[]) =>
            new Promise((resolve, reject) => calls.push({ kind, args: JSON.parse(JSON.stringify(args)), resolve, reject }));
        const engine = { explore: vi.fn(push('explore')), luck: vi.fn(push('luck')), onePly: vi.fn(push('onePly')), play: vi.fn(push('play')) } as unknown as Engine;
        const x = new ExplorerController(engine, memStore());
        x.init();
        const of = (kind: string) => calls.filter((c) => c.kind === kind);
        return { x, of };
    }

    it('plays only from an answer for this position', async () => {
        const { x, of } = setup();
        x.random();
        expect(x.canPlay).toBe(false); // no answer yet
        void x.play(mv);
        expect(of('play')).toHaveLength(0);
        of('explore')[0].resolve(withEnds(3));
        await flush();
        expect(x.canPlay).toBe(true);
        x.tapSquare('you', 3, 0); // the list now belongs to the old position…
        of('explore')[1].reject(new Error('engine was reset')); // …and stays after a failed explore
        await flush();
        expect(x.answer?.moves).toHaveLength(1);
        expect(x.canPlay).toBe(false);
        void x.play(mv);
        expect(of('play')).toHaveLength(0);
    });

    it('plays one move at a time', async () => {
        const { x, of } = setup();
        x.random();
        of('explore')[0].resolve(withEnds(3));
        await flush();
        void x.play(mv);
        expect(x.canPlay).toBe(false);
        void x.play(mv);
        expect(of('play')).toHaveLength(1);
        of('play')[0].reject(new Error('not a legal move'));
        await flush();
        expect(x.canPlay).toBe(true);
    });

    it('shows the badges of the position the answer is for, dimmed until the next one, and none once the game is over', async () => {
        const { x, of } = setup();
        x.refresh();
        of('explore')[0].resolve(withEnds(3));
        await flush();
        expect(x.more('you')).toEqual({ rows: [expect.objectContaining({ text: '+3' }), null, null, null], strikes: null, stale: false });
        x.tapSquare('you', 0, 0); // red 2: the old answer still says three more from an empty row
        expect(x.more('you')).toMatchObject({ rows: [{ text: '+3' }, null, null, null], stale: true });
        of('explore')[1].resolve(withEnds(3));
        await flush();
        expect(x.more('you')).toMatchObject({ rows: [{ text: '+2' }, null, null, null], stale: false });
        for (const k of [0, 1, 2, 3]) x.tapStrike('you', k);
        expect(x.pos.you.strikes).toBe(4);
        expect(x.more('you')).toBeNull();
    });
});
