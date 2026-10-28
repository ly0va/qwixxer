import type { Position, ReviewGame, RollAnalysis, Sq } from '../engine/protocol';
import { encode } from './link';
import { closesRow, exploreSheet, fromEngine, type ExPosition, type Who } from './position';
import { chipsFor, type Chip, type RowModel } from './rules';

/**
 * Spec `2026-10-24-game-review-design.md`, "The page": the review page's
 * numbers and pictures from a replayed game and its per-roll analyses.
 * Points: j = 0 is the start, 2k+1 after roll k's dice, 2k+2 after its move
 * (k from 0); a game of n rolls has 2n+1 points.
 */
export type Side = 'played' | 'best';
export type Analyses = (RollAnalysis | undefined)[];

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
/** A z as shown, one decimal: "z +1.4", "z −0.9". */
const zText = (z: number) => {
    const r = Math.round(z * 10) / 10;
    return `z ${r > 0 ? '+' : r < 0 ? '−' : ''}${Math.abs(r).toFixed(1)}`;
};

export function luckScore(luck: number, variance: number): { z: number; word: string; short: string; text: string } {
    const z = variance > 0 ? luck / Math.sqrt(variance) : 0;
    const r = Math.round(z * 10) / 10;
    const word = r <= -1.5 ? 'Very unlucky' : r <= -0.5 ? 'Unlucky' : r < 0.5 ? 'Average' : r < 1.5 ? 'Lucky' : 'Very lucky';
    const short = zText(z);
    return { z, word, short, text: `${word} · ${short}` };
}

/**
 * The bot's tile: its play, signed for you. Its mistakes count for you; on
 * your roll its white-sum mark can also beat the prediction luck used, which
 * counts against you. The tone and note follow the sign shown.
 */
export function botTile(bot: number, name: string) {
    const value = signed(bot);
    const against = value.startsWith('−');
    return {
        label: `${name}’s play`, short: 'Bot’s play', value,
        tone: against ? 'unlucky' : 'lucky', note: against ? 'against you' : 'in your favour',
    };
}

/** The summary's progress line while the analysis runs; nothing once it's done. */
export const progress = (t: { done: number }, n: number) => (t.done < n ? `Analysing… ${t.done} of ${n}` : '');

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

/** A choice's cost as a gain for its chooser; the bot's can be a gain (its white-sum mark beat the prediction). */
const cost = (c: number | undefined) => (c !== undefined && Math.abs(c) >= TIE ? signed(-c) : null);

/** The game log, roller first, with each roll's luck and each choice's cost. */
export function reviewLog(game: ReviewGame, an: Analyses): ReviewEntry[] {
    return game.rolls.map((roll, k) => {
        const r = roll.record;
        const a = an[k];
        const you: ReviewLine = { who: 'You', you: true, chips: chipsFor(r.you, r.you_strike), empty: '–', cost: cost(a?.you_cost) };
        const botMarks = [r.bot_white, r.bot_colour].filter((x) => x !== null);
        const bot: ReviewLine = { who: 'Bot', you: false, chips: chipsFor(botMarks, r.bot_strike), empty: '–', cost: cost(a?.bot_cost) };
        const l = rollLuck(an, k);
        return { k, n: k + 1, lines: r.roller === 'you' ? [you, bot] : [bot, you], luck: l === null ? 'luck …' : `luck ${signed(l)}` };
    });
}

/** Your mistakes from this many points of win chance get a dot on the graph. */
export const DOT_COST = 0.02;

/**
 * The graph's data per point (null: not yet): win chance; running luck and
 * its running z (luck so far over the root of its variance so far); skill
 * (the bot's play less your mistakes so far, changing at each move); and
 * the points with a dot on the win chance graph.
 */
export function graphSeries(game: ReviewGame, an: Analyses) {
    const n = game.rolls.length;
    const values = Array.from({ length: 2 * n + 1 }, (_, j) => pointValue(an, j));
    const luck: (number | null)[] = [an[0] ? 0 : null];
    const luckZ: (number | null)[] = [an[0] ? 0 : null];
    const skill: (number | null)[] = [an[0] ? 0 : null];
    let sum: number | null = 0;
    let vsum = 0;
    let sk: number | null = 0;
    for (let k = 0; k < n; k++) {
        const a = an[k];
        const l = rollLuck(an, k);
        sum = sum !== null && l !== null ? sum + l : null;
        vsum += rollVar(an, k) ?? 0;
        const z = sum === null ? null : luckScore(sum, vsum).z;
        luck.push(sum, sum);
        luckZ.push(z, z);
        skill.push(a ? sk : null);
        sk = sk !== null && a ? sk + a.bot_cost - a.you_cost : null;
        skill.push(sk);
    }
    const dots = game.rolls.flatMap((_, k) => ((an[k]?.you_cost ?? 0) >= DOT_COST ? [2 * k + 2] : []));
    return { values, luck, luckZ, skill, dots };
}

/** Spec "The graph": one card, four views. */
export type GraphKind = 'win' | 'skill' | 'luck' | 'z';
export const GRAPHS: { kind: GraphKind; tab: string }[] = [
    { kind: 'win', tab: 'Win chance' }, { kind: 'skill', tab: 'Skill' }, { kind: 'luck', tab: 'Dice luck' }, { kind: 'z', tab: 'Luck · z' },
];

/** A mistake from this many points of win chance gets a dot on the skill graph. */
const SKILL_DOT = 0.005;

export type Tone = 'you' | 'bot';
export interface GraphModel {
    title: string;
    /** The title on a phone. */
    short: string;
    values: (number | null)[];
    /** The points the curve passes through: where the value changes, up to the last analysed point. */
    knots: number[];
    lo: number;
    hi: number;
    /** Filled from the bottom (win chance), else above and below 0 in the luck colours. */
    area: boolean;
    guides: { v: number; dash: boolean }[];
    bands: { from: number; to: number }[];
    labels: { v: number; text: string; right: boolean }[];
    dots: { j: number; tone: Tone }[];
    legend: { text: string; tone: Tone }[];
    /** The selected point's bubble. */
    tag: (v: number) => string;
}

export function graphModel(game: ReviewGame, an: Analyses, kind: GraphKind, botName: string): GraphModel {
    const g = graphSeries(game, an);
    const values = kind === 'win' ? g.values : kind === 'skill' ? g.skill : kind === 'luck' ? g.luck : g.luckZ;
    const missing = values.findIndex((v) => v === null);
    const done = missing < 0 ? values.length - 1 : missing - 1;
    const knots = values.flatMap((_, j) => (j > done ? [] : j === 0 || j === done || kind === 'win' || (kind === 'skill' ? j % 2 === 0 : j % 2 === 1) ? [j] : []));
    const base = { values, knots, bands: [], labels: [], dots: [], legend: [] };
    if (kind === 'win') {
        return {
            ...base, title: 'Your win chance', short: 'Your win chance', lo: 0, hi: 1, area: true, guides: [{ v: 0.5, dash: true }],
            dots: g.dots.filter((j) => j <= done).map((j) => ({ j, tone: 'you' })), legend: [{ text: 'your mistakes', tone: 'you' }],
            tag: (v) => `${Math.round(v * 100)}%`,
        };
    }
    if (kind === 'z') {
        return {
            ...base, title: 'Dice luck compared with a typical game', short: 'Luck vs a typical game', lo: -3, hi: 3, area: false,
            guides: [1.5, 0.5, -0.5, -1.5].map((v) => ({ v, dash: true })), bands: [{ from: -0.5, to: 0.5 }],
            labels: ([['Very lucky', 2.25], ['Lucky', 1], ['Average', 0], ['Unlucky', -1], ['Very unlucky', -2.25]] as const).map(([text, v]) => ({ v, text, right: true })),
            tag: zText,
        };
    }
    // Points of win chance: a scale in whole tens of points, at least ±10.
    const most = Math.max(0, ...values.slice(0, done + 1).map((v) => Math.abs(v ?? 0)));
    const top = Math.max(10, Math.ceil(Math.round(most * 1000) / 10 / 10) * 10);
    const scale = { lo: -top / 100, hi: top / 100, area: false, guides: [{ v: 0, dash: false }], labels: [{ v: top / 100, text: `+${top}`, right: false }, { v: -top / 100, text: `−${top}`, right: false }], tag: signed };
    if (kind === 'luck') return { ...base, ...scale, title: 'Dice luck so far', short: 'Dice luck so far' };
    return {
        ...base, ...scale, title: 'Win chance won or lost by play', short: 'Won or lost by play',
        dots: game.rolls.flatMap((_, k): { j: number; tone: Tone }[] => {
            const a = an[k];
            if (!a || 2 * k + 2 > done) return [];
            return [...(Math.abs(a.bot_cost) >= SKILL_DOT ? [{ j: 2 * k + 2, tone: 'bot' as const }] : []), ...(a.you_cost >= SKILL_DOT ? [{ j: 2 * k + 2, tone: 'you' as const }] : [])];
        }),
        legend: [{ text: 'You', tone: 'you' }, { text: botName, tone: 'bot' }],
    };
}

export function stepLabel(j: number, n: number): { title: string; sub: string } {
    if (j === 0) return { title: 'Start', sub: 'before the first roll' };
    return { title: `Roll ${Math.floor((j - 1) / 2) + 1} of ${n}`, sub: j % 2 === 1 ? 'after the roll' : 'after the move' };
}

/** Roll k in the explorer: the position before its marks, with the dice and the roller. */
export const explorerHref = (game: ReviewGame, k: number) => '#/explore?p=' + encode(fromEngine(game.rolls[k].before));
export const gameHref = (code: string) => '#/review?g=' + code;

/** Copies the game `code`'s review link (`base`: the app's address) and says how it went, for a notice. */
export async function copyGameLink(code: string, base: string, copy: (t: string) => Promise<void>): Promise<string> {
    try {
        await copy(base + gameHref(code));
        return 'Link copied';
    } catch {
        return 'Couldn’t copy the link';
    }
}
