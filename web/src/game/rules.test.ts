import { describe, expect, it } from 'vitest';
import type { View } from '../engine/protocol';
import { clickable, confirmState, logEntry, predictedTurn, sheetModel, strikeOpen, type SheetUi } from './rules';

const empty = () => ({ marks: [0, 1, 2, 3].map(() => Array(11).fill(false)), strikes: 0, score: 0, row_points: [0, 0, 0, 0] });
function view(p: Partial<View>): View {
    return {
        phase: 'active', dice: [3, 4, 1, 1, 1, 1], you: empty(), bot: empty(), locked: [false, false, false, false],
        bot_white: null, options: { selections: [], strike: true, pass: false }, passive: [], over: null, ...p,
    };
}
const ui = (p: Partial<SheetUi> = {}): SheetUi => ({ sel: [], strikeSel: false, committing: false, fresh: {}, hidden: [], ...p });
const mark = (v: View, who: 'you' | 'bot', row: number, idxs: number[]) => idxs.forEach((i) => (v[who].marks[row][i] = true));

describe('clickable / confirm', () => {
    const v = view({ options: { selections: [[{ row: 0, n: 7 }], [{ row: 0, n: 7 }, { row: 1, n: 5 }], [{ row: 2, n: 5 }]], strike: true, pass: false } });
    it('narrows to partners after the first pick', () => {
        expect(clickable(v, [], false, { row: 2, n: 5 })).toBe(true);
        expect(clickable(v, [{ row: 0, n: 7 }], false, { row: 1, n: 5 })).toBe(true);
        expect(clickable(v, [{ row: 0, n: 7 }], false, { row: 2, n: 5 })).toBe(false);
        expect(clickable(v, [], true, { row: 2, n: 5 })).toBe(false);
    });
    it('labels and validates the confirm button', () => {
        expect(confirmState(v, [], false)).toEqual({ label: 'Mark', ok: false });
        expect(confirmState(v, [{ row: 0, n: 7 }], false)).toEqual({ label: 'Mark 1', ok: true });
        expect(confirmState(v, [{ row: 1, n: 5 }, { row: 0, n: 7 }], false)).toEqual({ label: 'Mark 2', ok: true });
        expect(confirmState(v, [], true)).toEqual({ label: 'Strike', ok: true });
        expect(confirmState(view({ options: { selections: [], strike: false, pass: true } }), [], false)).toEqual({ label: 'Pass', ok: true });
        expect(confirmState(view({ phase: 'passive' }), [], false)).toEqual({ label: 'Pass', ok: true });
        expect(confirmState(view({ phase: 'over' }), [], false)).toEqual({ label: 'New game', ok: true });
        expect(strikeOpen(v, [])).toBe(true);
        expect(strikeOpen(v, [{ row: 0, n: 7 }])).toBe(false);
    });
});

describe('sheetModel', () => {
    it('shades the lock in pencil while its row-ending number is pencilled, ink once committed', () => {
        const v = view({ options: { selections: [[{ row: 0, n: 12 }]], strike: true, pass: false } });
        mark(v, 'you', 0, [0, 1, 2, 3, 4]); // red 2..6: the 12 is lockable
        expect(sheetModel(v, 'you', ui())[0].lock).toBe('open');
        expect(sheetModel(v, 'you', ui({ sel: [{ row: 0, n: 12 }] }))[0].lock).toBe('pencil');
        expect(sheetModel(v, 'you', ui({ sel: [{ row: 0, n: 12 }], committing: true }))[0].lock).toBe('locked_by');
        // The bot's revealed row-ending mark shades its own lock; your sheet shows the row locking.
        const b = view({ bot_white: { row: 2, n: 2 } });
        mark(b, 'bot', 2, [0, 1, 2, 3, 4]);
        expect(sheetModel(b, 'bot', ui())[2].lock).toBe('pencil');
        expect(sheetModel(b, 'you', ui())[2].lock).toBe('locking');
    });
    it('greys skipped squares and the terminal number until five marks, pencil included', () => {
        const v = view({ options: { selections: [[{ row: 0, n: 8 }]], strike: true, pass: false } });
        mark(v, 'you', 0, [0, 1, 2, 4]); // red 2,3,4,6 → 5 is skipped
        let rows = sheetModel(v, 'you', ui());
        expect(rows[0].cells[3].look).toBe('skipped');
        expect(rows[0].cells[10].look).toBe('skipped');
        expect(rows[0].lock).toBe('grey');
        rows = sheetModel(v, 'you', ui({ sel: [{ row: 0, n: 8 }] }));
        expect(rows[0].cells[6].mark).toBe('pencil');
        expect(rows[0].lock).toBe('open');
        expect(rows[0].cells[10].look).not.toBe('skipped');
    });
    it('marks legal squares heavy and other open ones soft while choosing', () => {
        const v = view({ options: { selections: [[{ row: 1, n: 7 }]], strike: true, pass: false } });
        const rows = sheetModel(v, 'you', ui());
        expect(rows[1].cells[5].look).toBe('legal');
        expect(rows[1].cells[5].clickable).toBe(true);
        expect(rows[1].cells[6].look).toBe('soft');
        expect(sheetModel(view({ phase: 'you_to_roll', options: null }), 'you', ui())[1].cells[6].look).toBe('normal');
    });
    it('shows a bot lock as pending on both sheets and greys locked rows', () => {
        const v = view({ bot_white: { row: 0, n: 12 } });
        mark(v, 'bot', 0, [0, 1, 2, 3, 4]);
        expect(sheetModel(v, 'you', ui())[0].band).toBe('locking');
        const bot = sheetModel(v, 'bot', ui());
        expect(bot[0].band).toBe('locking');
        expect(bot[0].cells[10].mark).toBe('pencil');
        const done = view({ phase: 'bot_to_roll', options: null, locked: [true, false, false, false] });
        mark(done, 'bot', 0, [0, 1, 2, 3, 4, 10]);
        expect(sheetModel(done, 'you', ui())[0].band).toBe('locked');
        expect(sheetModel(done, 'you', ui())[0].cells[5].look).toBe('out');
        expect(sheetModel(done, 'you', ui())[0].lock).toBe('locked_other');
        expect(sheetModel(done, 'bot', ui())[0].lock).toBe('locked_by');
    });
    it('hides staged bot marks and inks committing selections', () => {
        const v = view({ phase: 'passive', options: null, passive: [{ row: 0, n: 7 }] });
        mark(v, 'bot', 3, [0]);
        expect(sheetModel(v, 'bot', ui({ hidden: ['bot:3:12'] }))[3].cells[0].mark).toBe('none');
        const r = sheetModel(v, 'you', ui({ sel: [{ row: 0, n: 7 }], committing: true }));
        expect(r[0].cells[5]).toMatchObject({ mark: 'ink', anim: 'ink' });
    });
});

describe('fix round 1', () => {
    it('keeps a row open while the mark that locks it is hidden', () => {
        const v = view({ phase: 'bot_to_roll', options: null, locked: [true, false, false, false] });
        mark(v, 'bot', 0, [0, 1, 2, 3, 4, 10]);
        const h = ui({ hidden: ['bot:0:12'] });
        for (const who of ['you', 'bot'] as const) {
            const r = sheetModel(v, who, h)[0];
            expect(r.band).toBe('normal');
            expect(r.cells[5].look).not.toBe('out');
            expect(r.lock).not.toBe('locked_other');
        }
        expect(sheetModel(v, 'bot', h)[0].cells[10].mark).toBe('none');
        expect(sheetModel(v, 'you', ui())[0].band).toBe('locked');
    });
    it('lets the legal cue win over terminal greying', () => {
        const v = view({ options: { selections: [[{ row: 0, n: 7 }, { row: 0, n: 12 }]], strike: true, pass: false } });
        mark(v, 'you', 0, [0, 1, 2, 3]);
        const c = sheetModel(v, 'you', ui())[0].cells[10];
        expect(c.clickable).toBe(true);
        expect(c.look).toBe('legal');
    });
});

describe('log', () => {
    it('puts the roller first and adds lock and strike chips', () => {
        const e = logEntry(7, { roller: 'bot', you: [], you_strike: false, bot_white: { row: 0, n: 12 }, bot_colour: null, bot_strike: false, locked: [0] });
        expect(e.n).toBe(7);
        expect(e.lines[0].who).toBe('Bot');
        expect(e.lines[0].chips).toEqual([{ kind: 'num', row: 0, n: 12 }, { kind: 'lock' }]);
        expect(e.lines[1]).toMatchObject({ who: 'You', chips: [], empty: '–' });
        const s = logEntry(8, { roller: 'you', you: [], you_strike: true, bot_white: null, bot_colour: null, bot_strike: false, locked: [] });
        expect(s.lines[0].chips).toEqual([{ kind: 'strike' }]);
    });
});

describe('predictedTurn', () => {
    it('words the opponent\'s predicted turn on their roll', () => {
        const w = { row: 0, n: 7 }, c = { row: 2, n: 9 };
        expect(predictedTurn({ white: w, colour: c, strike: false })).toEqual({
            verb: 'marks',
            chips: [{ kind: 'num', row: 0, n: 7 }, { kind: 'num', row: 2, n: 9 }],
        });
        expect(predictedTurn({ white: null, colour: c, strike: false }).chips).toHaveLength(1);
        expect(predictedTurn({ white: null, colour: null, strike: true })).toEqual({ verb: 'takes', chips: [{ kind: 'strike' }] });
        expect(predictedTurn({ white: null, colour: null, strike: false })).toEqual({ verb: 'passes', chips: [] });
    });
});
