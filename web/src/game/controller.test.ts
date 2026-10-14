import { describe, expect, it, vi } from 'vitest';
import type { Engine, Step, View } from '../engine/protocol';
import { GameController, type Store } from './controller.svelte';

const empty = () => ({ marks: [0, 1, 2, 3].map(() => Array(11).fill(false)), strikes: 0, score: 0, row_points: [0, 0, 0, 0] });
const v = (p: Partial<View>): View => ({
    phase: 'bot_to_roll', dice: null, you: empty(), bot: empty(), locked: [false, false, false, false],
    bot_white: null, options: null, passive: [], over: null, ...p,
});
const rec = (p: Partial<Step['record']> = {}): Step['record'] => ({ roller: 'you', you: [], you_strike: false, bot_white: null, bot_colour: null, bot_strike: false, locked: [], ...p });

function fakeEngine(over: Partial<Record<keyof Engine, any>> = {}): Engine {
    return {
        start: vi.fn(async () => v({})),
        restore: vi.fn(async () => v({ phase: 'you_to_roll' })),
        snapshot: vi.fn(async () => '{"snap":1}'),
        botRoll: vi.fn(async () => v({ phase: 'passive', dice: [1, 2, 3, 4, 5, 6], passive: [{ row: 0, n: 3 }] })),
        submitPassive: vi.fn(async () => ({ view: v({ phase: 'you_to_roll' }), record: rec({ roller: 'bot' }) })),
        youRoll: vi.fn(async () => v({ phase: 'active', dice: [1, 2, 3, 4, 5, 6], options: { selections: [[{ row: 0, n: 3 }]], strike: true, pass: false } })),
        submitActive: vi.fn(async () => ({ view: v({ phase: 'bot_to_roll' }), record: rec() })),
        ...over,
    } as Engine;
}
function memStore(init: Record<string, string> = {}): Store & { data: Record<string, string> } {
    const data = { ...init };
    return { data, get: (k) => data[k] ?? null, set: (k, val) => { data[k] = val; } };
}
const instant = async () => {};

describe('GameController', () => {
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
        const e = fakeEngine({ restore: vi.fn(async () => { throw new Error('bad snapshot'); }) });
        const c = new GameController(e, store, instant, () => 1);
        await c.init();
        expect(e.restore).toHaveBeenCalledOnce();
        expect(e.start).toHaveBeenCalledOnce();
        expect(c.error).toBe('');
        expect(c.view?.phase).toBe('passive');
        expect(c.rollNo).toBe(1);
    });
});
