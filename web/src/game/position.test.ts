import { describe, expect, it } from 'vitest';
import type { MoveOut, View } from '../engine/protocol';
import {
    clearSheet, closedRows, emptyPosition, exploreSheet, fromView, marginText, nextUnsetDie, overText, percent, randomDice,
    setDie, sheetTotal, swapSides, tapStrike, toEngine, toggleLock, toggleSquare,
    addAnalysis, analysisKey, analysisMargin, chanceText, gamesText, isOver, luckHistogram, luckText, moreBadges, moveChance, rankMoves, refineCount, winChance, ZERO_ANALYSIS,
} from './position';

const markRed = (idx: number[]) => idx.reduce((p, i) => toggleSquare(p, 'you', 0, i), emptyPosition());

describe('position edits', () => {
    it('toggles squares without touching the original', () => {
        const p0 = emptyPosition();
        const p1 = toggleSquare(p0, 'you', 2, 3);
        expect(p1.you.marks[2][3]).toBe(true);
        expect(p0.you.marks[2][3]).toBe(false);
        expect(toggleSquare(p1, 'you', 2, 3).you.marks[2][3]).toBe(false);
    });
    it('sets strikes from the tapped box, and the last taken box removes itself', () => {
        let p = tapStrike(emptyPosition(), 'opponent', 2);
        expect(p.opponent.strikes).toBe(3);
        p = tapStrike(p, 'opponent', 2);
        expect(p.opponent.strikes).toBe(2);
        expect(tapStrike(p, 'opponent', 0).opponent.strikes).toBe(1);
    });
    it('closes a row by marking its end number with five marks, or by tapping its lock', () => {
        expect(closedRows(markRed([0, 1, 2, 10]))[0]).toBe(false); // four marks: not closed
        const byMark = markRed([0, 1, 2, 3, 10]);
        expect(closedRows(byMark)[0]).toBe(true);
        expect(toggleLock(byMark, 0)).toEqual(byMark); // closed by marking: the lock tap does nothing
        const byTap = toggleLock(emptyPosition(), 1);
        expect(closedRows(byTap)).toEqual([false, true, false, false]);
        expect(closedRows(toggleLock(byTap, 1))[1]).toBe(false);
    });
    it('clears a sheet and reopens rows closed by tapping', () => {
        const p = clearSheet(toggleLock(tapStrike(markRed([0, 1]), 'you', 0), 3), 'you');
        expect(p.you.marks.flat().some(Boolean)).toBe(false);
        expect(p.you.strikes).toBe(0);
        expect(p.locked).toEqual([false, false, false, false]);
    });
    it('swaps sides and whose roll; twice is the identity', () => {
        const p = tapStrike(markRed([4]), 'you', 1);
        const s = swapSides(p);
        expect(s.opponent.marks[0][4]).toBe(true);
        expect(s.opponent.strikes).toBe(2);
        expect(s.roller).toBe('opponent');
        expect(swapSides(s)).toEqual(p);
    });
    it('sends dice to the engine only when all six are set', () => {
        let p = setDie(emptyPosition(), 0, 3);
        expect(toEngine(p).dice).toBeNull();
        expect(nextUnsetDie(p, 0)).toBe(1);
        p = randomDice(p, () => 0.999);
        expect(toEngine(p).dice).toEqual([6, 6, 6, 6, 6, 6]);
        expect(nextUnsetDie(p, 0)).toBeNull();
    });
});

describe('from the game', () => {
    const empty = () => ({ marks: [0, 1, 2, 3].map(() => Array(11).fill(false)), strikes: 0, score: 0, row_points: [0, 0, 0, 0] });
    const view = (p: Partial<View>): View => ({ phase: 'bot_to_roll', dice: null, you: empty(), bot: empty(), locked: [false, false, false, false], bot_white: null, options: null, passive: [], over: null, ...p });
    it('copies the boards, and the dice and roller while a roll is resolved', () => {
        const v = view({ phase: 'passive', dice: [1, 2, 3, 4, 5, 6] });
        v.bot.marks[1][4] = true;
        v.you.strikes = 2;
        const p = fromView(v);
        expect(p.opponent.marks[1][4]).toBe(true);
        expect(p.you.strikes).toBe(2);
        expect(p.dice).toEqual([1, 2, 3, 4, 5, 6]);
        expect(p.roller).toBe('opponent');
        expect(fromView(view({ phase: 'active', dice: [1, 1, 1, 1, 1, 1] })).roller).toBe('you');
    });
    it('copies no dice between rolls', () => {
        const p = fromView(view({ phase: 'you_to_roll', dice: [1, 2, 3, 4, 5, 6] }));
        expect(p.dice).toEqual([null, null, null, null, null, null]);
        expect(p.roller).toBe('you');
    });
});

describe('sheet model and text', () => {
    it('inks marks, greys squares left of the last mark and outs closed rows', () => {
        const p = toggleLock(markRed([0, 3]), 2);
        const rows = exploreSheet(p, 'you');
        expect(rows[0].cells[0].mark).toBe('ink');
        expect(rows[0].cells[1].look).toBe('skipped');
        expect(rows[0].cells.every((c) => c.clickable)).toBe(true);
        expect(rows[0].lock).toBe('grey');
        expect(rows[2].band).toBe('locked');
        expect(rows[2].cells[0].look).toBe('out');
        expect(rows[2].lock).toBe('locked_other');
    });
    it('shows the lock filled on the sheet that closed the row by marking', () => {
        const p = markRed([0, 1, 2, 3, 10]);
        expect(exploreSheet(p, 'you')[0].lock).toBe('locked_by');
        expect(exploreSheet(p, 'opponent')[0].lock).toBe('locked_other');
    });
    it('scores like the engine: the lock counts as a mark, strikes cost 5', () => {
        const p = tapStrike(markRed([0, 1, 2, 3, 10]), 'you', 0);
        expect(sheetTotal(p.you)).toBe(21 - 5); // 5 marks + lock = 6 → 21
    });
    it('formats chances, margins and results', () => {
        expect(percent(0.3786, 0)).toBe('38%');
        expect(percent(0.41249, 1)).toBe('41.2%');
        expect(marginText(-8.6)).toBe('Opponent ahead by ~9 points');
        expect(marginText(3.2)).toBe('You ahead by ~3 points');
        expect(marginText(1.4)).toBe('You ahead by ~1 point');
        expect(marginText(-0.6)).toBe('Opponent ahead by ~1 point');
        expect(marginText(0.3)).toBe('Level');
        expect(overText({ you: 41, opponent: 33 })).toBe('You win 41 – 33');
        expect(overText({ you: 30, opponent: 33 })).toBe('Opponent wins 33 – 30');
        expect(overText({ you: 20, opponent: 20 })).toBe('Draw 20 – 20');
    });
});

describe('analysis numbers', () => {
    const t = (games: number, win_sum: number, win_sq_sum: number, diff_sum = 0) => ({ games, win_sum, win_sq_sum, diff_sum });

    it('analysisKey ignores dice only', () => {
        const p = emptyPosition();
        const withDice = { ...p, dice: [1, 2, 3, 4, 5, 6] };
        expect(analysisKey(withDice)).toBe(analysisKey(p));
        expect(analysisKey(toggleSquare(p, 'you', 0, 0))).not.toBe(analysisKey(p));
        expect(analysisKey({ ...p, roller: 'opponent' })).not.toBe(analysisKey(p));
        expect(analysisKey({ ...p, locked: [true, false, false, false] })).not.toBe(analysisKey(p));
        expect(analysisKey(swapSides(toggleSquare(p, 'you', 0, 0)))).not.toBe(analysisKey(toggleSquare(p, 'you', 0, 0)));
    });

    it('isOver: four strikes or two closed rows', () => {
        const p = emptyPosition();
        expect(isOver(p)).toBe(false);
        expect(isOver({ ...p, opponent: { ...p.opponent, strikes: 4 } })).toBe(true);
        expect(isOver({ ...p, locked: [true, false, false, false] })).toBe(false);
        expect(isOver({ ...p, locked: [true, false, true, false] })).toBe(true);
    });

    it('adds batches and derives the win chance, ± and margin', () => {
        const s = addAnalysis(addAnalysis(ZERO_ANALYSIS, t(100, 50, 30, 200)), t(300, 170, 90, 600));
        expect(s).toEqual(t(400, 220, 120, 800));
        // Values between 0 and 1: ± comes from their spread, here
        // 120/400 − 0.55² = 0.0025, not 0.55 · 0.45.
        const w = winChance(s);
        expect(w.p).toBeCloseTo(0.55);
        expect(w.pm).toBeCloseTo(1.96 * Math.sqrt(0.0025 / 400));
        expect(analysisMargin(s)).toBe(2);
    });

    it('formats: whole percent until ± < 0.5 points, then one decimal', () => {
        expect(chanceText({ p: 0.5432, pm: 0.012 })).toBe('54%');
        expect(chanceText({ p: 0.5432, pm: 0.0049 })).toBe('54.3%');
        expect(gamesText({ p: 0.54, pm: 0.012 }, 3200)).toBe('± 1.2 · 3,200 games');
        expect(gamesText({ p: 0.54, pm: 0.3 }, 1)).toBe('± 30.0 · 1 game');
    });

    it('luck chip text: a word, the signed points with one decimal, "Average" near zero', () => {
        expect(luckText({ this_roll: 0.553, average: 0.521 })).toEqual({ word: 'Lucky', value: '+3.2', tone: 'lucky', detail: 'This roll 55.3% · average 52.1%' });
        expect(luckText({ this_roll: 0.494, average: 0.521 })).toMatchObject({ word: 'Unlucky', value: '−2.7', tone: 'unlucky' });
        expect(luckText({ this_roll: 0.5212, average: 0.521 })).toMatchObject({ word: 'Average', value: '', tone: 'average' });
    });

    it('the luck histogram: bars on the luck axis, trimmed to the rolls there are', () => {
        const bins = Array(100).fill(0);
        bins[40] = 0.25; // 40–41%
        bins[42] = 0.5;
        bins[47] = 0.25;
        const h = luckHistogram({ this_roll: 0.425, average: 0.42, bins, below: 0.25, above: 0.25 });
        expect(h.bars.map((b) => [b.lo, b.share, b.current])).toEqual([[-2, 0.25, false], [-1, 0, false], [0, 0.5, true], [1, 0, false], [2, 0, false], [3, 0, false], [4, 0, false], [5, 0.25, false]].map(([lo, s, c]) => [expect.closeTo(lo as number), s, c]));
        expect(h.bars.map((b) => b.tone)).toEqual(['unlucky', 'unlucky', 'lucky', 'lucky', 'lucky', 'lucky', 'lucky', 'lucky']);
        expect(h.min).toBeCloseTo(-2);
        expect(h.max).toBeCloseTo(6);
        expect(h.ticks).toEqual([0, 5]);
        expect(h.tallest).toBe(0.5);
        expect(h.rank).toBe('Better than 25% of rolls');
    });

    it('the rank says "worse than" when more rolls beat this one', () => {
        const h = luckHistogram({ this_roll: 0.3, average: 0.42, bins: Array(100).fill(0.01), below: 0.03, above: 0.96 });
        expect(h.rank).toBe('Worse than 96% of rolls');
        expect(h.ticks).toEqual([-40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]);
    });

    it('a move shows P(win) as a percentage with one decimal', () => {
        expect(moveChance(0.5464)).toBe('54.6%');
        expect(moveChance(0)).toBe('0.0%');
        expect(moveChance(1)).toBe('100.0%');
    });
});

describe('rankMoves', () => {
    const mv = (n: number, p_win: number, extra: Partial<MoveOut> = {}): MoveOut =>
        ({ marks: [{ row: 0, n }], strike: false, pass: false, p_win, bots: [], note: null, ...extra });
    const list = [mv(2, 0.452, { bots: ['master_plus', 'master'] }), mv(3, 0.441, { note: 'rule', bots: ['student'] }), mv(4, 0.43), mv(5, 0.42), mv(6, 0.41)];

    it('without 1-ply values is the static list', () => {
        expect(rankMoves(list, null).map((m) => [m.marks[0].n, m.p_win, m.refined])).toEqual(
            [[2, 0.452, false], [3, 0.441, false], [4, 0.43, false], [5, 0.42, false], [6, 0.41, false]]);
    });

    it('re-sorts the top 3 by their 1-ply values and keeps the rest below, static', () => {
        const r = rankMoves(list, [0.4396, 0.4398, 0.40]);
        expect(r.map((m) => [m.marks[0].n, m.p_win, m.refined])).toEqual(
            [[3, 0.4398, true], [2, 0.4396, true], [4, 0.40, true], [5, 0.42, false], [6, 0.41, false]]);
    });

    it('keeps the bot tags, drops the note on the new first row, and breaks ties by the static order', () => {
        const r = rankMoves(list, [0.43, 0.43, 0.43]);
        expect(r.map((m) => m.marks[0].n)).toEqual([2, 3, 4, 5, 6]);
        const s = rankMoves(list, [0.40, 0.45, 0.41]);
        expect(s[0].marks[0].n).toBe(3);
        expect(s[0].note).toBeNull();
        expect(s[0].bots).toEqual(['student']);
        expect(s.map((m) => m.marks[0].n)).toEqual([3, 4, 2, 5, 6]);
        expect(s[2].bots).toEqual(['master_plus', 'master']);
    });

    it('takes the moves tied with the third too, as the review does', () => {
        const tied = [mv(2, 0.5), mv(3, 0.45), mv(4, 0.4), mv(5, 0.4), mv(6, 0.39)];
        expect(refineCount(tied)).toBe(4);
        expect(refineCount(list)).toBe(3);
        expect(refineCount([mv(2, 0.5)])).toBe(1);
        const r = rankMoves(tied, [0.3, 0.3, 0.3, 0.6]);
        expect(r.map((m) => [m.marks[0].n, m.refined])).toEqual([[5, true], [2, true], [3, true], [4, true], [6, false]]);
    });

    it('handles fewer than 3 moves', () => {
        expect(rankMoves([mv(2, 0.5)], [0.6]).map((m) => [m.p_win, m.refined])).toEqual([[0.6, true]]);
    });
});

describe('moreBadges', () => {
    const tip = (what: string) => `About ${what} by the end of the game (an estimate)`;
    it('rounds the marks each row still gets, and shows none for a closed row or nothing more', () => {
        let p = markRed([0, 1, 2]); // red: 3 marks
        p = toggleSquare(p, 'you', 1, 0); // yellow: 1 mark
        p = toggleLock(p, 3); // blue closed
        const b = moreBadges(p, 'you', { marks: [5.4, 1.3, 2.6, 4], strikes: 0 });
        expect(b.rows).toEqual([
            { text: '+2', tip: tip('2 more red marks') },
            null,
            { text: '+3', tip: tip('3 more green marks') },
            null,
        ]);
        expect(moreBadges(p, 'you', { marks: [4.4, 1, 0, 0], strikes: 0 }).rows[0]).toEqual({ text: '+1', tip: tip('1 more red mark') });
    });
    it('gives strikes one decimal, from 0.1 up, and none at four strikes', () => {
        let p = tapStrike(emptyPosition(), 'you', 0); // 1 strike
        const at = (strikes: number, q = p) => moreBadges(q, 'you', { marks: [0, 0, 0, 0], strikes }).strikes;
        expect(at(1.46)).toEqual({ text: '+0.5', tip: tip('0.5 more strikes') });
        expect(at(2.04)).toEqual({ text: '+1.0', tip: tip('1.0 more strike') });
        expect(at(1.04)).toBeNull();
        p = tapStrike(p, 'you', 3);
        expect(at(4.2, p)).toBeNull();
    });
    it('never expects more than twelve marks in a row', () => {
        const ten = markRed([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]); // red 2–11
        expect(moreBadges(ten, 'you', { marks: [12.7, 0, 0, 0], strikes: 0 }).rows[0]?.text).toBe('+2');
    });
    it('never expects more than the row can still take', () => {
        const at = (idx: number[], n: number) => moreBadges(markRed(idx), 'you', { marks: [n, 0, 0, 0], strikes: 0 }).rows[0]?.text ?? null;
        expect(at([0, 9], 4.6)).toBeNull(); // red 2 and 11: only 12 is left, and it needs five marks
        expect(at([0, 1, 2, 7], 9)).toBe('+4'); // 2 3 4 9: then 10, 11 (six marks), 12 and its lock: 8 at most
        expect(at([0, 1, 2, 7], 7.4)).toBe('+3');
        expect(at([5], 12)).toBe('+6'); // 7: 8 9 10 11 (five marks), then 12 and its lock
    });
    it('never expects more than four strikes in all', () => {
        const three = tapStrike(emptyPosition(), 'you', 2);
        expect(moreBadges(three, 'you', { marks: [0, 0, 0, 0], strikes: 4.4 }).strikes?.text).toBe('+1.0');
    });
});
