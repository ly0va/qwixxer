import type { OpponentTurn, RollRecord, Sq, View } from '../engine/protocol';

export const ROWS = ['red', 'yellow', 'green', 'blue'] as const;
export const NAME = ['Red', 'Yellow', 'Green', 'Blue'];

export const numbers = (row: number): number[] => {
    const a = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    return row < 2 ? a : a.slice().reverse();
};
export const cellIndex = (row: number, n: number) => (row < 2 ? n - 2 : 12 - n);
export const isTerminal = (s: Sq) => cellIndex(s.row, s.n) === 10;
export const sqKey = (s: Sq) => `${s.row}:${s.n}`;
export const selKey = (sel: Sq[]) => sel.map(sqKey).sort().join('|');
export const has = (list: Sq[], s: Sq) => list.some((x) => x.row === s.row && x.n === s.n);

/** Can `s` still complete a legal selection, given the current one? (Lookup only.) */
export function clickable(view: View, sel: Sq[], strikeSel: boolean, s: Sq): boolean {
    if (view.phase === 'active' && view.options) {
        if (strikeSel || has(sel, s)) return false;
        return view.options.selections.some((x) => has(x, s) && sel.every((p) => has(x, p)));
    }
    if (view.phase === 'passive') return sel.length === 0 && has(view.passive, s);
    return false;
}

export function confirmState(view: View, sel: Sq[], strikeSel: boolean): { label: string; ok: boolean } {
    if (view.phase === 'over') return { label: 'New game', ok: true };
    if (view.phase === 'passive') return { label: sel.length ? 'Mark' : 'Pass', ok: true };
    if (view.phase === 'active' && view.options) {
        if (strikeSel) return { label: 'Strike', ok: view.options.strike };
        const k = selKey(sel);
        const ok = sel.length > 0 ? view.options.selections.some((x) => selKey(x) === k) : view.options.pass;
        const label = sel.length === 2 ? 'Mark 2' : sel.length === 1 ? 'Mark 1' : view.options.pass ? 'Pass' : 'Mark';
        return { label, ok };
    }
    return { label: 'Mark', ok: false };
}

export const strikeOpen = (view: View, sel: Sq[]) => view.phase === 'active' && !!view.options?.strike && sel.length === 0;

export type CellLook = 'normal' | 'legal' | 'soft' | 'skipped' | 'out';
export interface CellModel { n: number; sq: Sq; mark: 'none' | 'ink' | 'pencil'; anim: '' | 'draw' | 'ink'; look: CellLook; clickable: boolean; label: string }
export type LockLook = 'grey' | 'open' | 'locking' | 'pencil' | 'locked_by' | 'locked_other';
export interface RowModel { row: number; color: (typeof ROWS)[number]; band: 'normal' | 'locking' | 'locked'; cells: CellModel[]; lock: LockLook; label: string }
export interface SheetUi { sel: Sq[]; strikeSel: boolean; committing: boolean; fresh: Record<string, 'draw' | 'ink'>; hidden: string[] }

export function sheetModel(view: View, who: 'you' | 'bot', ui: SheetUi): RowModel[] {
    const sheet = view[who];
    const isYou = who === 'you';
    const choosing = isYou && (view.phase === 'active' || view.phase === 'passive');
    const botWhite = view.phase === 'active' ? view.bot_white : null;
    const lockingRow = botWhite && isTerminal(botWhite) ? botWhite.row : -1;
    const pencils: Sq[] = isYou ? ui.sel : botWhite ? [botWhite] : [];
    return ROWS.map((color, r) => {
        const ink = sheet.marks[r].map((m, i) => m && !ui.hidden.includes(`${who}:${sqKey({ row: r, n: numbers(r)[i] })}`));
        const last = ink.lastIndexOf(true);
        const pencilHere = pencils.filter((p) => p.row === r).length;
        const lockable = ink.filter(Boolean).length + pencilHere >= 5;
        const term = numbers(r)[10];
        const locked = view.locked[r] && !ui.hidden.includes(`you:${r}:${term}`) && !ui.hidden.includes(`bot:${r}:${term}`);
        const cells: CellModel[] = numbers(r).map((n, i) => {
            const sq = { row: r, n };
            const isPencil = has(pencils, sq);
            const committed = isYou && ui.committing && isPencil;
            const mark: CellModel['mark'] = ink[i] || committed ? 'ink' : isPencil ? 'pencil' : 'none';
            const canClick = isYou && !ink[i] && clickable(view, ui.sel, ui.strikeSel, sq);
            let look: CellLook = 'normal';
            if (locked && !ink[i]) look = 'out';
            else if (canClick) look = 'legal';
            else if (i < last && !ink[i]) look = 'skipped';
            else if (i === 10 && mark === 'none' && !lockable) look = 'skipped';
            else if (choosing && mark === 'none') look = canClick ? 'legal' : 'soft';
            const anim = committed ? 'ink' : mark === 'pencil' ? 'draw' : mark === 'ink' ? (ui.fresh[`${who}:${sqKey(sq)}`] ?? '') : '';
            const state = mark === 'ink' ? ', marked' : mark === 'pencil' ? (isYou ? ', selected' : ', marked this roll') : canClick ? ', available' : '';
            return { n, sq, mark, anim, look, clickable: canClick, label: `${NAME[r]} ${n}${state}` };
        });
        // The lock counts as a mark: pencil-shaded while its row-ending number
        // is pencilled on this sheet, solid ink once that mark is inked.
        const end = cells[10].mark;
        const lock: LockLook = end === 'ink' ? 'locked_by' : locked ? 'locked_other' : end === 'pencil' ? 'pencil'
            : r === lockingRow ? 'locking' : lockable ? 'open' : 'grey';
        const band = locked ? 'locked' : r === lockingRow ? 'locking' : 'normal';
        return { row: r, color, band, cells, lock, label: `${NAME[r]} row${locked ? ', locked' : r === lockingRow ? ', locking this roll' : ''}` };
    });
}

export type Chip = { kind: 'num'; row: number; n: number } | { kind: 'strike' } | { kind: 'lock' };
export interface LogLine { who: 'You' | 'Bot'; chips: Chip[]; empty: string }
export interface LogEntry { n: number; lines: LogLine[] }

export function chipsFor(marks: Sq[], strike: boolean): Chip[] {
    if (strike) return [{ kind: 'strike' }];
    return marks.flatMap((m): Chip[] => (isTerminal(m) ? [{ kind: 'num', row: m.row, n: m.n }, { kind: 'lock' }] : [{ kind: 'num', row: m.row, n: m.n }]));
}

/** The rolling opponent's predicted turn (explorer, their roll) as a verb and its chips. */
export function predictedTurn(t: OpponentTurn): { verb: 'marks' | 'takes' | 'passes'; chips: Chip[] } {
    const marks = [t.white, t.colour].filter((x): x is Sq => !!x);
    const chips = chipsFor(marks, t.strike);
    return { verb: t.strike ? 'takes' : chips.length ? 'marks' : 'passes', chips };
}

export function logEntry(n: number, rec: RollRecord): LogEntry {
    const botMarks = [rec.bot_white, rec.bot_colour].filter((x): x is Sq => !!x);
    const you: LogLine = { who: 'You', chips: chipsFor(rec.you, rec.you_strike), empty: '–' };
    const bot: LogLine = { who: 'Bot', chips: chipsFor(botMarks, rec.bot_strike), empty: '–' };
    return { n, lines: rec.roller === 'you' ? [you, bot] : [bot, you] };
}
