import type { Position, ReviewGame, RollAnalysis, Sq } from '../engine/protocol';
import { encode } from './link';
import { closesRow, exploreSheet, type ExPosition, type Who } from './position';
import { chipsFor, type Chip, type RowModel } from './rules';

/**
 * Spec `2026-10-24-game-review-design.md`, "The page": the review page's
 * numbers and pictures from a replayed game and its per-roll analyses.
 * Points: j = 0 is the start, 2k+1 after roll k's dice, 2k+2 after its move
 * (k from 0); a game of n rolls has 2n+1 points.
 */
export type Side = 'played' | 'best';
export type Analyses = (RollAnalysis | undefined)[];

/** What the human is called: "You" in your own game, "Player" in a shared one. */
export interface Names { you: string; You: string; your: string; Your: string }
export const names = (shared: boolean): Names =>
    shared ? { you: 'Player', You: 'Player', your: 'Player’s', Your: 'Player’s' } : { you: 'you', You: 'You', your: 'your', Your: 'Your' };

/** Signed points of win chance with one decimal: "+3.2", "−4.6" (U+2212). */
export const signed = (v: number) => {
    const r = Math.round(v * 1000) / 10; // the sign of what is shown: −0.04 points is "0.0"
    return `${r > 0 ? '+' : r < 0 ? '−' : ''}${Math.abs(r).toFixed(1)}`;
};

export function pointValue(an: Analyses, j: number): number | null {
    if (j === 0) return an[0]?.start ?? null;
    const a = an[Math.floor((j - 1) / 2)];
    if (!a) return null;
    return j % 2 === 1 ? a.this_roll : a.after_move;
}

/** f(this roll) − E[f]; E[f] is the previous roll's after-move value (roll 1: the start). */
export function rollLuck(an: Analyses, k: number): number | null {
    const a = an[k];
    const prev = k === 0 ? an[0]?.start : an[k - 1]?.after_move;
    return a && prev != null ? a.this_roll - prev : null;
}

/** The variance of roll k's luck: from the start (roll 1) or the previous roll's after-move. */
export function rollVar(an: Analyses, k: number): number | null {
    return (k === 0 ? an[0]?.start_var : an[k - 1]?.after_var) ?? null;
}

export function totals(an: Analyses, n: number): { luck: number; var: number; you: number; bot: number; done: number } {
    const t = { luck: 0, var: 0, you: 0, bot: 0, done: 0 };
    for (let k = 0; k < n; k++) {
        const a = an[k];
        if (!a) continue;
        t.done++;
        t.luck += rollLuck(an, k) ?? 0;
        t.var += rollVar(an, k) ?? 0;
        t.you += a.you_cost;
        t.bot += a.bot_cost;
    }
    return t;
}

/**
 * Spec "Luck score": the game's luck over its standard deviation (z), so
 * games compare whatever the dice could swing; the word is z's band
 * (rounded as shown). No percentile: its spread depends on how well you play.
 */
export function luckScore(luck: number, variance: number): { z: number; word: string; short: string; text: string } {
    const z = variance > 0 ? luck / Math.sqrt(variance) : 0;
    const r = Math.round(z * 10) / 10;
    const word = r <= -1.5 ? 'Very unlucky' : r <= -0.5 ? 'Unlucky' : r < 0.5 ? 'Average' : r < 1.5 ? 'Lucky' : 'Very lucky';
    const sign = r > 0 ? '+' : r < 0 ? '−' : '';
    const short = `z ${sign}${Math.abs(r).toFixed(1)}`;
    return { z, word, short, text: `${word} · ${short}` };
}

/**
 * The bot's tile: its play, signed for you. Its mistakes count for you; on
 * your roll its white-sum mark can also beat the prediction luck used, which
 * counts against you. The tone and note follow the sign shown.
 */
export function botTile(bot: number, name: string, nm: Names) {
    const value = signed(bot);
    const against = value.startsWith('−');
    return {
        label: `${name}’s play`, short: 'Bot’s play', value,
        tone: against ? 'unlucky' : 'lucky', note: against ? `against ${nm.you}` : `in ${nm.your} favour`,
    };
}

/** Spec "Verdict": luck against the net of both players' mistakes. */
export function verdict(t: { luck: number; you: number; bot: number; done: number }, n: number, youWon: boolean | null, nm: Names): string {
    if (t.done < n) return `Analysing… ${t.done} of ${n}`;
    const skill = t.bot - t.you;
    if (Math.abs(t.luck) > Math.abs(skill)) return 'Mostly the dice';
    return (youWon ?? skill > 0) ? `Mostly ${nm.your} play` : `Mostly ${nm.your} mistakes`;
}

export const fromEngine = (p: Position): ExPosition => ({
    you: p.you, opponent: p.opponent, locked: [...p.locked],
    dice: p.dice ? [...p.dice] : Array(6).fill(null), roller: p.roller,
});

/** Below this a cost rounds to 0.0 points. */
export const TIE = 0.0005;

/**
 * How your choice on a roll compares with the best: your own move ("best"),
 * a different move within TIE ("excellent"), or one that cost you more.
 */
export type Grade = 'best' | 'excellent' | 'cost';

const sameMove = (a: Sq[], b: Sq[]) =>
    a.length === b.length && a.every((q) => b.some((r) => r.row === q.row && r.n === q.n));

/**
 * The best move on roll k as the page shows it, with your choice's grade:
 * "Best move" only ever sits next to your own move. null: not analysed yet.
 */
export function bestShown(game: ReviewGame, an: Analyses, k: number): { marks: Sq[]; strike: boolean; after: Position; grade: Grade } | null {
    const a = an[k];
    if (!a) return null;
    const r = game.rolls[k];
    const strike = r.record.roller === 'you' && r.record.you_strike;
    if (sameMove(a.you_best.marks, r.record.you) && a.you_best.strike === strike) {
        return { marks: r.record.you, strike, after: r.after, grade: 'best' };
    }
    return { ...a.you_best, grade: a.you_cost < TIE ? 'excellent' : 'cost' };
}

/**
 * The sheets at point j: after a move, the played or the best outcome; `prev`
 * is the position before the roll's marks, so the difference is this roll's
 * marks (drawn in pencil).
 */
export function positionAt(game: ReviewGame, an: Analyses, j: number, side: Side): { pos: ExPosition; prev: ExPosition } {
    if (j === 0 || game.rolls.length === 0) {
        const empty = () => ({ marks: [0, 1, 2, 3].map(() => Array(11).fill(false)), strikes: 0 });
        const p: Position = { you: empty(), opponent: empty(), locked: [false, false, false, false], dice: null, roller: game.bot_first ? 'opponent' : 'you' };
        return { pos: fromEngine(p), prev: fromEngine(p) };
    }
    const k = Math.floor((j - 1) / 2);
    const roll = game.rolls[k];
    if (j % 2 === 1) return { pos: fromEngine(roll.before), prev: fromEngine(roll.before) };
    const best = side === 'best' ? bestShown(game, an, k)?.after : undefined;
    return { pos: fromEngine(best ?? roll.after), prev: fromEngine({ ...roll.before, dice: null }) };
}

/** A read-only sheet; squares marked since `prev` are pencil marks. */
export function reviewSheet(prev: ExPosition, pos: ExPosition, who: Who): RowModel[] {
    return exploreSheet(pos, who).map((row) => ({
        ...row,
        cells: row.cells.map((c, i) => ({
            ...c,
            clickable: false,
            mark: c.mark === 'ink' && !prev[who].marks[row.row][i] ? 'pencil' : c.mark,
        })),
        lock: closesRow(pos[who], row.row) && !closesRow(prev[who], row.row) ? 'pencil' : row.lock,
    }));
}

export interface ReviewLine { who: string; you: boolean; chips: Chip[]; empty: string; cost: string | null }
export interface ReviewEntry { k: number; n: number; lines: ReviewLine[]; luck: string }

const cost = (c: number | undefined) => (c !== undefined && c >= TIE ? signed(-c) : null);

/** The game log, roller first, with each roll's luck and each choice's cost. */
export function reviewLog(game: ReviewGame, an: Analyses, nm: Names): ReviewEntry[] {
    return game.rolls.map((roll, k) => {
        const r = roll.record;
        const a = an[k];
        const you: ReviewLine = { who: nm.You, you: true, chips: chipsFor(r.you, r.you_strike), empty: '–', cost: cost(a?.you_cost) };
        const botMarks = [r.bot_white, r.bot_colour].filter((x) => x !== null);
        const bot: ReviewLine = { who: 'Bot', you: false, chips: chipsFor(botMarks, r.bot_strike), empty: '–', cost: cost(a?.bot_cost) };
        const l = rollLuck(an, k);
        return { k, n: k + 1, lines: r.roller === 'you' ? [you, bot] : [bot, you], luck: l === null ? 'luck …' : `luck ${signed(l)}` };
    });
}

/** Your mistakes from this many points of win chance get a dot on the graph. */
export const DOT_COST = 0.02;

/** The graph's data: win chance per point, running luck per point (null: not yet), and points with a dot. */
export function graphSeries(game: ReviewGame, an: Analyses): { values: (number | null)[]; luck: (number | null)[]; dots: number[] } {
    const n = game.rolls.length;
    const values = Array.from({ length: 2 * n + 1 }, (_, j) => pointValue(an, j));
    const luck: (number | null)[] = [an[0] ? 0 : null];
    let sum: number | null = 0;
    for (let k = 0; k < n; k++) {
        const l = rollLuck(an, k);
        sum = sum !== null && l !== null ? sum + l : null;
        luck.push(sum, sum);
    }
    const dots = game.rolls.flatMap((_, k) => ((an[k]?.you_cost ?? 0) >= DOT_COST ? [2 * k + 2] : []));
    return { values, luck, dots };
}

export function stepLabel(j: number, n: number): { title: string; sub: string } {
    if (j === 0) return { title: 'Start', sub: 'before the first roll' };
    return { title: `Roll ${Math.floor((j - 1) / 2) + 1} of ${n}`, sub: j % 2 === 1 ? 'after the roll' : 'after the move' };
}

/** Roll k in the explorer: the position before its marks, with the dice and the roller. */
export const explorerHref = (game: ReviewGame, k: number) => '#/explore?p=' + encode(fromEngine(game.rolls[k].before));
export const gameHref = (code: string) => '#/review?g=' + code;
