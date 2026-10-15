import { describe, expect, it } from 'vitest';
import type { View } from '../engine/protocol';
import {
    clearSheet, closedRows, emptyPosition, exploreSheet, fromView, marginText, nextUnsetDie, overText, percent, randomDice,
    setDie, sheetTotal, swapSides, tapStrike, toEngine, toggleLock, toggleSquare,
    addAnalysis, analysisKey, analysisMargin, chanceText, gamesText, isOver, score, winChance, ZERO_ANALYSIS,
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
        expect(marginText({ margin: -8.6, sigma: 14.9 })).toBe('Opponent ahead by about 9 (± 15)');
        expect(marginText({ margin: 3.2, sigma: 10.4 })).toBe('You ahead by about 3 (± 10)');
        expect(marginText({ margin: 0.3, sigma: 12 })).toBe('Level (± 12)');
        expect(overText({ you: 41, opponent: 33 })).toBe('You win 41 – 33');
        expect(overText({ you: 30, opponent: 33 })).toBe('Opponent wins 33 – 30');
        expect(overText({ you: 20, opponent: 20 })).toBe('Draw 20 – 20');
    });
});

describe('analysis numbers', () => {
    const t = (games: number, wins: number, ties: number, diff_sum = 0, diff_sq_sum = 0) => ({ games, wins, ties, diff_sum, diff_sq_sum });

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
        const s = addAnalysis(addAnalysis(ZERO_ANALYSIS, t(100, 50, 10, 200, 10000)), t(300, 170, 20, 600, 30000));
        expect(s).toEqual(t(400, 220, 30, 800, 40000));
        const w = winChance(s);
        expect(w.p).toBeCloseTo((220 + 15) / 400);
        expect(w.pm).toBeCloseTo(1.96 * Math.sqrt(w.p * (1 - w.p) / 400));
        const m = analysisMargin(s);
        expect(m.margin).toBe(2);
        expect(m.sigma).toBeCloseTo(Math.sqrt(100 - 4));
    });

    it('formats: whole percent until ± < 0.5 points, then one decimal', () => {
        expect(chanceText({ p: 0.5432, pm: 0.012 })).toBe('54%');
        expect(chanceText({ p: 0.5432, pm: 0.0049 })).toBe('54.3%');
        expect(gamesText({ p: 0.54, pm: 0.012 }, 3200)).toBe('± 1.2 · 3,200 games');
        expect(gamesText({ p: 0.54, pm: 0.3 }, 1)).toBe('± 30.0 · 1 game');
    });

    it('score is P(win) × 1000, rounded', () => {
        expect(score(0.5464)).toBe(546);
        expect(score(0)).toBe(0);
        expect(score(1)).toBe(1000);
    });
});
