import type { Analysis, Position, Roller, SheetSnap, View } from '../engine/protocol';
import { NAME, numbers, ROWS, type CellLook, type CellModel, type LockLook, type RowModel } from './rules';

/** The explorer's two sides (spec "The position"). */
export type Who = 'you' | 'opponent';

/**
 * The explorer's position. `dice` per die (null = unset); `locked` = rows
 * closed by tapping their lock (rows closed by marking are derived).
 */
export interface ExPosition { you: SheetSnap; opponent: SheetSnap; locked: boolean[]; dice: (number | null)[]; roller: Roller }

const emptySheet = (): SheetSnap => ({ marks: [0, 1, 2, 3].map(() => Array(11).fill(false)), strikes: 0 });
export const emptyPosition = (): ExPosition => ({ you: emptySheet(), opponent: emptySheet(), locked: [false, false, false, false], dice: Array(6).fill(null), roller: 'you' });

// Positions may be Svelte state Proxies: copy through JSON, not structuredClone.
const copy = (p: ExPosition): ExPosition => JSON.parse(JSON.stringify(p));

/** The engine's lock rule: the row's end number marked, with five or more marks. */
export const closesRow = (s: SheetSnap, r: number) => s.marks[r][10] && s.marks[r].filter(Boolean).length >= 5;
export const closedRows = (p: ExPosition) => [0, 1, 2, 3].map((r) => p.locked[r] || closesRow(p.you, r) || closesRow(p.opponent, r));

export function toggleSquare(p: ExPosition, who: Who, row: number, i: number): ExPosition {
    const q = copy(p);
    q[who].marks[row][i] = !q[who].marks[row][i];
    return q;
}

/** Tap box k: strikes become k + 1, or k if box k was the last one taken. */
export function tapStrike(p: ExPosition, who: Who, k: number): ExPosition {
    const q = copy(p);
    q[who].strikes = q[who].strikes === k + 1 ? k : k + 1;
    return q;
}

/** A row closed by marking reopens only by unmarking its end number. */
export function toggleLock(p: ExPosition, row: number): ExPosition {
    if (closesRow(p.you, row) || closesRow(p.opponent, row)) return p;
    const q = copy(p);
    q.locked[row] = !q.locked[row];
    return q;
}

export function clearSheet(p: ExPosition, who: Who): ExPosition {
    const q = copy(p);
    q[who] = emptySheet();
    q.locked = [false, false, false, false];
    return q;
}
export const clearSheets = (p: ExPosition): ExPosition => clearSheet(clearSheet(p, 'you'), 'opponent');

export function swapSides(p: ExPosition): ExPosition {
    const q = copy(p);
    [q.you, q.opponent] = [q.opponent, q.you];
    q.roller = q.roller === 'you' ? 'opponent' : 'you';
    return q;
}

export function setRoller(p: ExPosition, r: Roller): ExPosition {
    const q = copy(p);
    q.roller = r;
    return q;
}

export function setDie(p: ExPosition, i: number, v: number | null): ExPosition {
    const q = copy(p);
    q.dice[i] = v;
    return q;
}
export function randomDice(p: ExPosition, rand: () => number = Math.random): ExPosition {
    const q = copy(p);
    q.dice = q.dice.map(() => 1 + Math.floor(rand() * 6));
    return q;
}
export function clearDice(p: ExPosition): ExPosition {
    const q = copy(p);
    q.dice = Array(6).fill(null);
    return q;
}
/** The next unset die after `after` (wrapping), or null when all are set. */
export function nextUnsetDie(p: ExPosition, after: number): number | null {
    for (let k = 1; k <= 6; k++) {
        const i = (after + k) % 6;
        if (p.dice[i] === null) return i;
    }
    return null;
}

export function toEngine(p: ExPosition): Position {
    const q = copy(p);
    const dice = q.dice.every((d) => d !== null) ? (q.dice as number[]) : null;
    return { you: q.you, opponent: q.opponent, locked: q.locked, dice, roller: q.roller };
}

/**
 * Spec "Explore this position": the game's boards (You → You, Bot →
 * Opponent) before the current roll's marks; while a roll is being
 * resolved, its dice and whose roll it is.
 */
export function fromView(v: View): ExPosition {
    const sheet = (s: View['you']): SheetSnap => ({ marks: s.marks.map((r) => [...r]), strikes: s.strikes });
    const you = sheet(v.you);
    const opponent = sheet(v.bot);
    const rolling = (v.phase === 'active' || v.phase === 'passive') && !!v.dice;
    return {
        you, opponent,
        locked: v.locked.map((l, r) => l && !closesRow(you, r) && !closesRow(opponent, r)),
        dice: rolling ? [...v.dice!] : Array(6).fill(null),
        roller: v.phase === 'passive' || v.phase === 'bot_to_roll' ? 'opponent' : 'you',
    };
}

/** Points per row as the engine counts them: the lock adds a mark. */
export function rowPoints(s: SheetSnap): number[] {
    return [0, 1, 2, 3].map((r) => {
        const n = s.marks[r].filter(Boolean).length + (closesRow(s, r) ? 1 : 0);
        return (n * (n + 1)) / 2;
    });
}
export const sheetTotal = (s: SheetSnap) => rowPoints(s).reduce((a, b) => a + b, 0) - 5 * s.strikes;

/** Rows for SheetRows / SheetColumns: every square tappable, all marks ink. */
export function exploreSheet(p: ExPosition, who: Who): RowModel[] {
    const s = p[who];
    const closed = closedRows(p);
    return ROWS.map((color, r) => {
        const marks = s.marks[r];
        const last = marks.lastIndexOf(true);
        const count = marks.filter(Boolean).length;
        const cells: CellModel[] = numbers(r).map((n, i) => {
            let look: CellLook = 'normal';
            if (closed[r] && !marks[i]) look = 'out';
            else if (i < last && !marks[i]) look = 'skipped';
            else if (i === 10 && !marks[i] && count < 5) look = 'skipped';
            return { n, sq: { row: r, n }, mark: marks[i] ? 'ink' : 'none', anim: '', look, clickable: true, label: `${NAME[r]} ${n}${marks[i] ? ', marked' : ''}` };
        });
        const lock: LockLook = closesRow(s, r) ? 'locked_by' : closed[r] ? 'locked_other' : count >= 5 ? 'open' : 'grey';
        return { row: r, color, band: closed[r] ? 'locked' : 'normal', cells, lock, label: `${NAME[r]} row${closed[r] ? ', locked' : ''}` };
    });
}

export const percent = (p: number, digits: 0 | 1) => `${(p * 100).toFixed(digits)}%`;

export function marginText(margin: number): string {
    const m = Math.round(margin);
    if (m > 0) return `You ahead by about ${m}`;
    if (m < 0) return `Opponent ahead by about ${-m}`;
    return 'Level';
}

export function overText(o: { you: number; opponent: number }): string {
    if (o.you > o.opponent) return `You win ${o.you} – ${o.opponent}`;
    if (o.you < o.opponent) return `Opponent wins ${o.opponent} – ${o.you}`;
    return `Draw ${o.you} – ${o.opponent}`;
}

/** Spec "Controller": analysis restarts when this changes (dice don't count). */
export const analysisKey = (p: ExPosition) => JSON.stringify([p.you, p.opponent, p.locked, p.roller]);

/** Four strikes on a sheet or two closed rows: nothing to play out. */
export const isOver = (p: ExPosition) =>
    p.you.strikes >= 4 || p.opponent.strikes >= 4 || closedRows(p).filter(Boolean).length >= 2;

export const ZERO_ANALYSIS: Analysis = { games: 0, win_sum: 0, win_sq_sum: 0, diff_sum: 0 };

export const addAnalysis = (a: Analysis, b: Analysis): Analysis => ({
    games: a.games + b.games,
    win_sum: a.win_sum + b.win_sum,
    win_sq_sum: a.win_sq_sum + b.win_sq_sum,
    diff_sum: a.diff_sum + b.diff_sum,
});

/** Mean value and ± (1.96 standard errors), the spread taken from the values. */
export function winChance(t: Analysis): { p: number; pm: number } {
    const p = t.win_sum / t.games;
    const v = Math.max(0, t.win_sq_sum / t.games - p * p);
    return { p, pm: 1.96 * Math.sqrt(v / t.games) };
}

/** Mean final score difference, for `marginText`. */
export const analysisMargin = (t: Analysis) => t.diff_sum / t.games;

/** Whole percent until ± < 0.5 points, then one decimal. */
export const chanceText = (w: { p: number; pm: number }) => percent(w.p, w.pm < 0.005 ? 1 : 0);

export const gamesText = (w: { p: number; pm: number }, games: number) =>
    `± ${(w.pm * 100).toFixed(1)} · ${games.toLocaleString('en-US')} ${games === 1 ? 'game' : 'games'}`;

/** The move list's number: the net's P(win) as a percentage, one decimal. */
export const moveChance = (p: number) => percent(p, 1);
