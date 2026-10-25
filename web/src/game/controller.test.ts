import { describe, expect, it, vi } from 'vitest';
import { EngineError, type Engine, type Step, type View } from '../engine/protocol';
import { GameController, type Store } from './controller.svelte';

const empty = () => ({ marks: [0, 1, 2, 3].map(() => Array(11).fill(false)), strikes: 0, score: 0, row_points: [0, 0, 0, 0] });
const v = (p: Partial<View>): View => ({
    phase: 'bot_to_roll', dice: null, you: empty(), bot: empty(), locked: [false, false, false, false],
    bot_white: null, options: null, passive: [], over: null, ...p,
});
const rec = (p: Partial<Step['record']> = {}): Step['record'] => ({ roller: 'you', you: [], you_strike: false, bot_white: null, bot_colour: null, bot_strike: false, locked: [], ...p });

/**
 * Every call's arguments go through structuredClone, as postMessage does in
 * WorkerEngine: Svelte state Proxies would throw a DataCloneError here.
 */
function fakeEngine(over: Partial<Record<keyof Engine, any>> = {}): Engine {
    const impl: Record<string, any> = {
        start: vi.fn(async () => v({})),
        restore: vi.fn(async () => v({ phase: 'you_to_roll' })),
        snapshot: vi.fn(async () => '{"snap":1}'),
        botRoll: vi.fn(async () => v({ phase: 'passive', dice: [1, 2, 3, 4, 5, 6], passive: [{ row: 0, n: 3 }] })),
        submitPassive: vi.fn(async () => ({ view: v({ phase: 'you_to_roll' }), record: rec({ roller: 'bot' }) })),
        youRoll: vi.fn(async () => v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: { selections: [[{ row: 0, n: 3 }]], strike: true, pass: false } })),
        submitActive: vi.fn(async () => ({ view: v({ phase: 'bot_to_roll' }), record: rec() })),
        ...over,
    };
    const e: Record<string, any> = { reset: vi.fn() };
    for (const [k, f] of Object.entries(impl)) {
        e[k] = vi.fn((...args: unknown[]) => f(...structuredClone(args)));
        // Keep the test's own mock reachable for implementation-specific checks.
        (e[k] as any).impl = f;
    }
    return e as Engine;
}
function memStore(init: Record<string, string> = {}): Store & { data: Record<string, string> } {
    const data = { ...init };
    return { data, get: (k) => data[k] ?? null, set: (k, val) => { data[k] = val; } };
}
const instant = async () => {};

describe('GameController', () => {
    it('runs with client-compiled runes ($state arrays are Proxies)', async () => {
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: { selections: [[{ row: 0, n: 3 }]], strike: true, pass: false } })) });
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 3, bot: 'master', nextBot: 'master' }) }), instant, () => 1);
        await c.init();
        c.tapSquare({ row: 0, n: 3 });
        // If this stops throwing, the test environment no longer matches the browser.
        expect(() => structuredClone(c.sel)).toThrow();
    });

    it('plays a whole roll of each kind through a postMessage-like engine', async () => {
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'you_to_roll' })) });
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 2, bot: 'master', nextBot: 'master' }) }), instant, () => 1);
        await c.init();
        await c.roll();
        expect(c.view?.phase).toBe('active');
        c.tapSquare({ row: 0, n: 3 });
        await c.confirm(); // submitActive, then the bot's roll lands in passive
        expect(c.error).toBe('');
        expect(e.submitActive).toHaveBeenCalledWith([{ row: 0, n: 3 }], false);
        expect(c.view?.phase).toBe('passive');
        c.tapSquare({ row: 0, n: 3 });
        await c.confirm(); // submitPassive with your white-sum square
        expect(c.error).toBe('');
        expect(e.submitPassive).toHaveBeenCalledWith({ row: 0, n: 3 });
        expect(c.view?.phase).toBe('you_to_roll');
    });

    it('starts a new game with the bot rolling first', async () => {
        const e = fakeEngine();
        const c = new GameController(e, memStore(), instant, () => 1);
        await c.init();
        expect(e.start).toHaveBeenCalledWith('master_plus', 1);
        expect(e.botRoll).toHaveBeenCalledOnce();
        expect(c.view?.phase).toBe('passive');
        expect(c.rollNo).toBe(1);
    });

    it('submits once when confirm is pressed twice while busy', async () => {
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: { selections: [[{ row: 0, n: 3 }]], strike: true, pass: false } })) });
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 3, bot: 'master', nextBot: 'master' }) }), instant, () => 1);
        await c.init();
        c.tapSquare({ row: 0, n: 3 });
        await Promise.all([c.confirm(), c.confirm()]);
        expect(e.submitActive).toHaveBeenCalledOnce();
        expect(e.submitActive).toHaveBeenCalledWith([{ row: 0, n: 3 }], false);
    });

    it('shows a tip instead of submitting an empty selection', async () => {
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: { selections: [[{ row: 0, n: 3 }]], strike: true, pass: false } })) });
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 3, bot: 'master', nextBot: 'master' }) }), instant, () => 1);
        await c.init();
        await c.confirm();
        expect(e.submitActive).not.toHaveBeenCalled();
        expect(c.tip).toBe('Select a square to mark, or the strike box to take a strike.');
    });

    it('applies a bot change only to the next game, and persists it', async () => {
        const store = memStore();
        const e = fakeEngine();
        const c = new GameController(e, store, instant, () => 1);
        await c.init();
        c.setBot('student');
        expect(c.bot).toBe('master_plus');
        expect(JSON.parse(store.data['qwixxer.prefs.v1']).nextBot).toBe('student');
        await c.newGame();
        expect(e.start).toHaveBeenLastCalledWith('student', 1);
        expect(c.bot).toBe('student');
    });

    it('restores a saved game and resumes a pending bot roll', async () => {
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'bot_to_roll' })) });
        const saved = { snapshot: '{"s":1}', log: [{ n: 4, lines: [] }], rollNo: 4, bot: 'master', nextBot: 'master' };
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify(saved) }), instant, () => 9);
        await c.init();
        expect(e.restore).toHaveBeenCalledWith('master', '{"s":1}', 9);
        expect(e.start).not.toHaveBeenCalled();
        expect(e.botRoll).toHaveBeenCalledOnce();
        expect(c.log[c.log.length - 1].n).toBe(4);
        expect(c.bot).toBe('master');
    });

    it('surfaces engine failures and stays usable', async () => {
        const e = fakeEngine({ start: vi.fn(async () => { throw new Error('wasm failed'); }) });
        const c = new GameController(e, memStore(), instant, () => 1);
        await c.init();
        expect(c.error).toContain('wasm failed');
        expect(c.busy).toBe(false);
    });

    it('clears the selection on Escape', async () => {
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: { selections: [[{ row: 0, n: 3 }]], strike: true, pass: false } })) });
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 1, bot: 'master', nextBot: 'master' }) }), instant, () => 1);
        await c.init();
        c.tapSquare({ row: 0, n: 3 });
        c.clearSelection();
        expect(c.sel).toEqual([]);
        expect(c.strikeSel).toBe(false);
    });

    it('explains a blocked square in a locking row', async () => {
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'active', dice: [6, 6, 2, 1, 1, 1], bot_white: { row: 0, n: 12 }, options: { selections: [[{ row: 0, n: 12 }]], strike: true, pass: false } })) });
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 1, bot: 'master', nextBot: 'master' }) }), instant, () => 1);
        await c.init();
        c.tapSquare({ row: 0, n: 8 });
        expect(c.tip).toContain('Red is locking');
    });

    it('discards a corrupt saved game and starts a new one', async () => {
        const store = memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: 'old', log: [], rollNo: 2, bot: 'master', nextBot: 'master' }) });
        const e = fakeEngine({ restore: vi.fn(async () => { throw new EngineError('bad snapshot', 'snapshot'); }) });
        const c = new GameController(e, store, instant, () => 1);
        await c.init();
        expect(e.restore).toHaveBeenCalledOnce();
        expect(e.start).toHaveBeenCalledOnce();
        expect(c.error).toBe('');
        expect(c.view?.phase).toBe('passive');
        expect(c.rollNo).toBe(1);
    });

    it('settles the roll animation before surfacing a failed roll', async () => {
        const e = fakeEngine({ youRoll: vi.fn(async () => { throw new Error('roll failed'); }) });
        const later = () => new Promise<void>((r) => setTimeout(r, 0));
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 2, bot: 'master', nextBot: 'master' }) }), later, () => 1);
        await c.init();
        expect(c.view?.phase).toBe('you_to_roll');
        await c.roll();
        await new Promise((r) => setTimeout(r, 20));
        expect(c.error).toContain('roll failed');
        expect(c.thinking).toBe(false);
        expect(c.rolling).toBe(false);
        expect(c.shownDice).toBe(null);
        expect(c.busy).toBe(false);
    });

    it('persists a bot change made mid-roll only to prefs', async () => {
        let finish: (v: View) => void = () => {};
        const e = fakeEngine({ youRoll: vi.fn(() => new Promise<View>((r) => { finish = r; })) });
        const store = memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 2, bot: 'master', nextBot: 'master' }) });
        const c = new GameController(e, store, instant, () => 1);
        await c.init();
        const before = store.data['qwixxer.save.v1'];
        const snaps = (e.snapshot as any).mock.calls.length;
        const pending = c.roll();
        expect(c.busy).toBe(true);
        c.setBot('student');
        await Promise.resolve();
        expect((e.snapshot as any).mock.calls.length).toBe(snaps);
        expect(store.data['qwixxer.save.v1']).toBe(before);
        expect(JSON.parse(store.data['qwixxer.prefs.v1']).nextBot).toBe('student');
        finish(v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: { selections: [[{ row: 0, n: 3 }]], strike: true, pass: false } }));
        await pending;
    });
    it('shows the rolled dice while the bot thinks, before revealing the view', async () => {
        const waits: { ms: number; done: () => void }[] = [];
        const sleep = (ms: number) => new Promise<void>((r) => waits.push({ ms, done: r }));
        const flush = () => new Promise((r) => setTimeout(r, 0));
        const e = fakeEngine();
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 2, bot: 'master', nextBot: 'master' }) }), sleep, () => 1);
        await c.init();
        let finished = false;
        const pending = c.roll().then(() => (finished = true));
        await flush();
        expect(waits.map((w) => w.ms)).toEqual([600]);
        expect(c.pendingDice).toBe(null);
        waits[0].done();
        await flush();
        expect(waits.map((w) => w.ms)).toEqual([600, 500]);
        expect(finished).toBe(false); // the think is a minimum even with a fast engine
        expect(c.rolling).toBe(false);
        expect(c.thinking).toBe(true);
        expect(c.pendingDice).toEqual([1, 2, 3, 4, 5, 6]);
        expect(c.view?.phase).toBe('you_to_roll');
        waits[1].done();
        await pending;
        expect(c.pendingDice).toBe(null);
        expect(c.view?.phase).toBe('active');
    });
    it('keeps thinking with tumble faces up while a slow engine computes', async () => {
        const waits: { ms: number; done: () => void }[] = [];
        const sleep = (ms: number) => new Promise<void>((r) => waits.push({ ms, done: r }));
        const flush = () => new Promise((r) => setTimeout(r, 0));
        let finish: (v: View) => void = () => {};
        const e = fakeEngine({ youRoll: vi.fn(() => new Promise<View>((r) => { finish = r; })) });
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 2, bot: 'master', nextBot: 'master' }) }), sleep, () => 1);
        await c.init();
        let finished = false;
        const pending = c.roll().then(() => (finished = true));
        await flush();
        waits[0].done(); // tumble over, engine still computing
        await flush();
        expect(waits.map((w) => w.ms)).toEqual([600, 500]); // think overlaps the engine call
        expect(c.rolling).toBe(false);
        expect(c.thinking).toBe(true);
        expect(c.shownDice).not.toBe(null); // tray keeps the last tumble faces, never blank
        expect(c.pendingDice).toBe(null);
        waits[1].done(); // minimum think over, engine still busy
        await flush();
        expect(finished).toBe(false);
        expect(c.thinking).toBe(true);
        expect(c.view?.phase).toBe('you_to_roll');
        finish(v({ phase: 'active', dice: [6, 5, 4, 3, 2, 1], options: { selections: [[{ row: 0, n: 11 }]], strike: true, pass: false } }));
        await pending;
        expect(c.thinking).toBe(false);
        expect(c.shownDice).toBe(null);
        expect(c.pendingDice).toBe(null);
        expect(c.view?.phase).toBe('active');
        expect(waits.length).toBe(2);
    });

    it('keeps the save when the engine fails to load, after one retry on a fresh worker', async () => {
        const saved = JSON.stringify({ snapshot: '{"s":1}', log: [], rollNo: 5, bot: 'master', nextBot: 'master' });
        const store = memStore({ 'qwixxer.save.v1': saved });
        const e = fakeEngine({ restore: vi.fn(async () => { throw new EngineError('engine failed to load: fetch', 'engine'); }) });
        const c = new GameController(e, store, instant, () => 1);
        await c.init();
        expect(e.restore).toHaveBeenCalledTimes(2);
        expect(e.reset).toHaveBeenCalledOnce();
        expect(e.start).not.toHaveBeenCalled();
        expect(store.data['qwixxer.save.v1']).toBe(saved);
        expect(c.error).toContain('engine failed to load');
        expect(c.busy).toBe(false);
        // New game (the error button) recreates the worker before starting.
        await c.newGame();
        expect(e.reset).toHaveBeenCalledTimes(2);
        expect(e.start).toHaveBeenCalledOnce();
        expect(c.error).toBe('');
        expect(c.view?.phase).toBe('passive');
    });

    it('resumes the save when the retry on a fresh worker loads', async () => {
        let calls = 0;
        const restore = vi.fn(async () => {
            if (calls++ === 0) throw new EngineError('engine failed to load: fetch', 'engine');
            return v({ phase: 'you_to_roll' });
        });
        const e = fakeEngine({ restore });
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 5, bot: 'master', nextBot: 'master' }) }), instant, () => 1);
        await c.init();
        expect(e.reset).toHaveBeenCalledOnce();
        expect(e.start).not.toHaveBeenCalled();
        expect(c.error).toBe('');
        expect(c.rollNo).toBe(5);
        expect(c.view?.phase).toBe('you_to_roll');
    });

    it('recreates the engine on New game after a failed call', async () => {
        let calls = 0;
        const e = fakeEngine({ start: vi.fn(async () => {
            if (calls++ === 0) throw new EngineError('engine failed to load', 'engine');
            return v({});
        }) });
        const c = new GameController(e, memStore(), instant, () => 1);
        await c.init();
        expect(c.error).toContain('engine failed to load');
        expect(e.reset).not.toHaveBeenCalled();
        await c.newGame();
        expect(e.reset).toHaveBeenCalledOnce();
        expect(c.error).toBe('');
        expect(c.view?.phase).toBe('passive');
        await c.newGame(); // healthy again: no more resets
        expect(e.reset).toHaveBeenCalledOnce();
    });

    it('lets the stored preference win over the saved game for the next bot', async () => {
        const e = fakeEngine();
        const c = new GameController(e, memStore({
            'qwixxer.prefs.v1': JSON.stringify({ theme: 'dark', nextBot: 'student' }),
            'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 2, bot: 'master', nextBot: 'master' }),
        }), instant, () => 1);
        await c.init();
        expect(c.bot).toBe('master');
        expect(c.nextBot).toBe('student');
    });

    it('saves the rolled game as soon as the engine has rolled, before the think ends', async () => {
        const waits: { ms: number; done: () => void }[] = [];
        const sleep = (ms: number) => new Promise<void>((r) => waits.push({ ms, done: r }));
        const flush = () => new Promise((r) => setTimeout(r, 0));
        let rolled = false;
        const e = fakeEngine({
            youRoll: vi.fn(async () => { rolled = true; return v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: { selections: [[{ row: 0, n: 3 }]], strike: true, pass: false } }); }),
            snapshot: vi.fn(async () => (rolled ? '{"after":1}' : '{"before":1}')),
        });
        const store = memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 2, bot: 'master', nextBot: 'master' }) });
        const c = new GameController(e, store, sleep, () => 1);
        await c.init();
        const pending = c.roll();
        await flush();
        expect(waits.map((w) => w.ms)).toEqual([600]); // still tumbling
        const early = JSON.parse(store.data['qwixxer.save.v1']);
        expect(early).toMatchObject({ snapshot: '{"after":1}', rollNo: 3 });
        waits[0].done();
        await flush();
        expect(c.thinking).toBe(true);
        waits[1].done();
        await pending;
        expect(JSON.parse(store.data['qwixxer.save.v1'])).toMatchObject({ snapshot: '{"after":1}', rollNo: 3 });
        expect(c.rollNo).toBe(3);
    });

    it('says to unselect first when another legal square conflicts with the selection', async () => {
        const opts = { selections: [[{ row: 0, n: 3 }], [{ row: 1, n: 9 }]], strike: true, pass: false };
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: opts })) });
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 1, bot: 'master', nextBot: 'master' }) }), instant, () => 1);
        await c.init();
        c.tapSquare({ row: 0, n: 3 });
        c.tapSquare({ row: 1, n: 9 });
        expect(c.tip).toBe('Tap your selected square to unselect it first.');
        c.tapSquare({ row: 2, n: 5 }); // not legal at all
        expect(c.tip).toBe('That square isn’t available with this roll.');
    });

    it('says to unselect first when switching white-sum squares on the bot’s turn', async () => {
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'passive', dice: [1, 2, 3, 4, 5, 6], passive: [{ row: 0, n: 3 }, { row: 3, n: 3 }] })) });
        const c = new GameController(e, memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 1, bot: 'master', nextBot: 'master' }) }), instant, () => 1);
        await c.init();
        c.tapSquare({ row: 0, n: 3 });
        c.tapSquare({ row: 3, n: 3 });
        expect(c.tip).toBe('Tap your selected square to unselect it first.');
        c.tapSquare({ row: 2, n: 8 });
        expect(c.tip).toContain('you may only mark the white sum (3)');
    });

    it('stores the finished game’s code for review, once', async () => {
        const over = v({ phase: 'over', over: 'locks' });
        const e = fakeEngine({
            restore: vi.fn(async () => v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: { selections: [[{ row: 0, n: 3 }]], strike: true, pass: false } })),
            submitActive: vi.fn(async () => ({ view: over, record: rec() })),
            gameCode: vi.fn(async () => '1BCODE'),
        });
        const store = memStore({ 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 3, bot: 'master', nextBot: 'master' }) });
        const c = new GameController(e, store, instant, () => 1);
        await c.init();
        c.tapSquare({ row: 0, n: 3 });
        await c.confirm();
        await new Promise((r) => setTimeout(r, 0));
        expect(e.gameCode).toHaveBeenCalledTimes(1);
        expect(c.lastCode).toBe('1BCODE');
        expect(store.data['qwixxer.lastGame']).toBe('1BCODE');
    });

    it('a game without a code (saved before rolls were recorded) offers no review', async () => {
        const e = fakeEngine({
            restore: vi.fn(async () => v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: { selections: [[{ row: 0, n: 3 }]], strike: true, pass: false } })),
            submitActive: vi.fn(async () => ({ view: v({ phase: 'over', over: 'locks' }), record: rec() })),
            gameCode: vi.fn(async () => { throw new EngineError('this game was saved before rolls were recorded', 'engine'); }),
        });
        const store = memStore({ 'qwixxer.lastGame': '1OLD', 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 3, bot: 'master', nextBot: 'master' }) });
        const c = new GameController(e, store, instant, () => 1);
        await c.init();
        expect(c.lastCode).toBe('1OLD');
        c.tapSquare({ row: 0, n: 3 });
        await c.confirm();
        await new Promise((r) => setTimeout(r, 0));
        expect(c.lastCode).toBeNull();
        expect(c.error).toBe('');
        // A review of "the last game" must not open the one before it.
        expect(store.data['qwixxer.lastGame'] || null).toBeNull();
    });

    it('a finished game restored on reload keeps its own code, not the previous game’s', async () => {
        const e = fakeEngine({ restore: vi.fn(async () => v({ phase: 'over', over: 'locks' })), gameCode: vi.fn(async () => '1NEW') });
        const store = memStore({ 'qwixxer.lastGame': '1OLD', 'qwixxer.save.v1': JSON.stringify({ snapshot: '{}', log: [], rollNo: 9, bot: 'master', nextBot: 'master' }) });
        const c = new GameController(e, store, instant, () => 1);
        await c.init();
        await new Promise((r) => setTimeout(r, 0));
        expect(c.lastCode).toBe('1NEW');
        expect(store.data['qwixxer.lastGame']).toBe('1NEW');
    });
});
