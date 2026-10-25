import { describe, expect, it } from 'vitest';
import type { Position, ReviewGame, RollAnalysis, RollRecord, SheetSnap } from '../engine/protocol';
import { decode } from './link';
import {
    bestShown, botTile, graphModel, GRAPHS, explorerHref, gameHref, graphSeries, pointValue, positionAt, reviewLog, reviewSheet, luckScore, rollLuck, signed, stepLabel, totals, progress,
} from './review';

const sheet = (marked: [number, number][] = [], strikes = 0): SheetSnap => {
    const marks = [0, 1, 2, 3].map(() => Array(11).fill(false));
    for (const [r, i] of marked) marks[r][i] = true;
    return { marks, strikes };
};
const pos = (you: SheetSnap, opponent: SheetSnap, roller: Position['roller'], dice: number[] | null = null): Position =>
    ({ you, opponent, locked: [false, false, false, false], dice, roller });
const rec = (r: Partial<RollRecord>): RollRecord =>
    ({ roller: 'bot', you: [], you_strike: false, bot_white: null, bot_colour: null, bot_strike: false, locked: [], ...r });

// Roll 1 (bot): bot marks red 7 (index 5), you pass. Roll 2 (you): you mark
// yellow 9 (index 7), bot passes. Roll 3 (bot): bot strikes, you pass.
const s0 = pos(sheet(), sheet(), 'opponent');
const s1 = pos(sheet(), sheet([[0, 5]]), 'you');
const s2 = pos(sheet([[1, 7]]), sheet([[0, 5]]), 'opponent');
const s3 = pos(sheet([[1, 7]]), sheet([[0, 5]], 1), 'you');
const game: ReviewGame = {
    bot: 'master', bot_first: true, over: null,
    rolls: [
        { before: { ...s0, dice: [3, 4, 1, 1, 1, 1] }, record: rec({ bot_white: { row: 0, n: 7 } }), after: s1 },
        { before: { ...s1, dice: [4, 5, 1, 1, 1, 1] }, record: rec({ roller: 'you', you: [{ row: 1, n: 9 }] }), after: s2 },
        { before: { ...s2, dice: [1, 1, 1, 1, 1, 1] }, record: rec({ bot_strike: true }), after: s3 },
    ],
};
const best = (after: Position) => ({ marks: [{ row: 0, n: 9 }], strike: false, after });
const an: RollAnalysis[] = [
    { start: 0.48, start_var: 0.002, this_roll: 0.45, after_move: 0.44, after_var: 0.003, you_cost: 0, you_best: best(s1), bot_cost: 0.01 },
    { start: null, start_var: null, this_roll: 0.5, after_move: 0.46, after_var: 0.004, you_cost: 0.046, you_best: best(pos(sheet([[0, 7]]), sheet([[0, 5]]), 'opponent')), bot_cost: 0 },
];

describe('review points', () => {
    it('maps points to start, after the roll, after the move', () => {
        expect([0, 1, 2, 3, 4, 5, 6].map((j) => pointValue(an, j))).toEqual([0.48, 0.45, 0.44, 0.5, 0.46, null, null]);
    });
    it('luck is this roll minus the previous after-move (or the start)', () => {
        expect(rollLuck(an, 0)).toBeCloseTo(0.45 - 0.48);
        expect(rollLuck(an, 1)).toBeCloseTo(0.5 - 0.44);
        expect(rollLuck(an, 2)).toBeNull();
    });
    it('totals count only analysed rolls', () => {
        const t = totals(an, 3);
        expect(t.done).toBe(2);
        expect(t.luck).toBeCloseTo(-0.03 + 0.06);
        expect(t.you).toBeCloseTo(0.046);
        expect(t.bot).toBeCloseTo(0.01);
        expect(t.var).toBeCloseTo(0.002 + 0.003); // roll 1: the start's; roll 2: roll 1's after-move
    });
    it('luck score: total luck over its standard deviation, with a word', () => {
        expect(luckScore(0.03, 0.005)).toEqual({ z: 0.03 / Math.sqrt(0.005), word: 'Average', short: 'z +0.4', text: 'Average · z +0.4' });
        expect(luckScore(-0.06, 0.004).word).toBe('Unlucky'); // z −0.95
        expect(luckScore(0.04, 0.0016).word).toBe('Lucky'); // z +1.0
        expect(luckScore(0.2, 0.01)).toMatchObject({ word: 'Very lucky', text: 'Very lucky · z +2.0' });
        expect(luckScore(-0.2, 0.01)).toMatchObject({ word: 'Very unlucky', short: 'z −2.0', text: 'Very unlucky · z −2.0' });
        expect(luckScore(0.05, 0.01).word).toBe('Lucky'); // z +0.5: the band starts there
        expect(luckScore(-0.15, 0.01).word).toBe('Very unlucky'); // z −1.5
        expect(luckScore(0.1, 0)).toMatchObject({ z: 0, word: 'Average' }); // the dice could change nothing
    });
    it('progress shows while analysing, nothing once done', () => {
        expect(progress({ done: 1 }, 3)).toBe('Analysing… 1 of 3');
        expect(progress({ done: 3 }, 3)).toBe('');
    });
    it('the bot tile is its play, signed for you: mistakes in your favour, better-than-predicted marks against you', () => {
        expect(botTile(0.05, 'Master+')).toEqual({ label: 'Master+’s play', short: 'Bot’s play', value: '+5.0', tone: 'lucky', note: 'in your favour' });
        expect(botTile(-0.008, 'Master+')).toMatchObject({ value: '−0.8', tone: 'unlucky', note: 'against you' });
        expect(botTile(-0.0004, 'Master+')).toMatchObject({ value: '0.0', tone: 'lucky', note: 'in your favour' });
    });
    it('a value that rounds to 0.0 has no sign', () => {
        expect(signed(-0.0004)).toBe('0.0');
        expect(signed(0.0004)).toBe('0.0');
        expect(signed(-0.0006)).toBe('−0.1');
        const tiny = [{ ...an[0], this_roll: an[0].start! - 0.0003 }];
        expect(reviewLog(game, tiny)[0].luck).toBe('luck 0.0');
    });
    it('steps are labelled by roll and phase', () => {
        expect(stepLabel(46, 36)).toEqual({ title: 'Roll 23 of 36', sub: 'after the move' });
        expect(stepLabel(45, 36)).toEqual({ title: 'Roll 23 of 36', sub: 'after the roll' });
        expect(stepLabel(0, 36)).toEqual({ title: 'Start', sub: 'before the first roll' });
    });
});

describe('review sheets', () => {
    it('after the move shows the played or the best outcome, compared with the position before', () => {
        const played = positionAt(game, an, 4, 'played');
        expect(played.pos.you.marks[1][7]).toBe(true);
        expect(played.prev.you.marks[1][7]).toBe(false);
        const b = positionAt(game, an, 4, 'best');
        expect(b.pos.you.marks[0][7]).toBe(true); // red 9
        expect(b.pos.you.marks[1][7]).toBe(false);
        expect(b.prev).toEqual(played.prev);
    });
    it('after the roll shows the position before the marks, with the dice', () => {
        const p = positionAt(game, an, 3, 'played');
        expect(p.pos.dice).toEqual([4, 5, 1, 1, 1, 1]);
        expect(p.pos.you.marks[1][7]).toBe(false);
        expect(positionAt(game, an, 0, 'played').pos.roller).toBe('opponent');
    });
    it('marks this roll in pencil, read-only', () => {
        const { pos: p, prev } = positionAt(game, an, 4, 'played');
        const rows = reviewSheet(prev, p, 'you');
        const pencil = rows.flatMap((r) => r.cells.filter((c) => c.mark === 'pencil').map((c) => `${r.row}:${c.n}`));
        expect(pencil).toEqual(['1:9']);
        expect(rows.every((r) => r.cells.every((c) => !c.clickable))).toBe(true);
    });
});

describe('review log', () => {
    it('puts costs on the chooser’s line and luck on each roll', () => {
        const log = reviewLog(game, an);
        expect(log.map((e) => e.n)).toEqual([1, 2, 3]);
        expect(log[1].lines[0]).toMatchObject({ who: 'You', you: true, cost: '−4.6' });
        expect(log[1].lines[1]).toMatchObject({ who: 'Bot', cost: null });
        expect(log[0].lines[0]).toMatchObject({ who: 'Bot', cost: '−1.0' });
        expect(log[0].luck).toBe('luck −3.0');
        expect(log[1].luck).toBe('luck +6.0');
        expect(log[2].luck).toBe('luck …');
        expect(log[1].lines[0].who).toBe('You');
    });
});

describe('review links', () => {
    it('opens a roll in the explorer before its marks, with dice and roller', () => {
        const href = explorerHref(game, 1);
        expect(href.startsWith('#/explore?p=')).toBe(true);
        const p = decode(href.slice('#/explore?p='.length))!;
        expect(p.dice).toEqual([4, 5, 1, 1, 1, 1]);
        expect(p.roller).toBe('you');
        expect(p.opponent.marks[0][5]).toBe(true);
        expect(gameHref('1BAAAA')).toBe('#/review?g=1BAAAA');
    });
});

describe('graph series', () => {
    it('values per point, running luck per point, and dots on your costly moves', () => {
        const g = graphSeries(game, an);
        expect(g.values).toEqual([0.48, 0.45, 0.44, 0.5, 0.46, null, null]);
        expect(g.luck[0]).toBe(0);
        expect(g.luck[1]).toBeCloseTo(-0.03);
        expect(g.luck[2]).toBeCloseTo(-0.03);
        expect(g.luck[3]).toBeCloseTo(0.03);
        expect(g.luck[5]).toBeNull();
        expect(g.dots).toEqual([4]); // roll 2 cost you 4.6 points (≥ 2)
        // Running z: luck so far over the root of its variance so far, on both of a roll's points.
        expect(g.luckZ[0]).toBe(0);
        expect(g.luckZ[1]).toBeCloseTo(-0.03 / Math.sqrt(0.002));
        expect(g.luckZ[2]).toBeCloseTo(-0.03 / Math.sqrt(0.002));
        expect(g.luckZ[3]).toBeCloseTo(0.03 / Math.sqrt(0.005));
        expect(g.luckZ[5]).toBeNull();
    });
});

describe('the best move shown', () => {
    it('your own move as the best is "best", shown as played', () => {
        const same = [an[0], { ...an[1], you_cost: 0, you_best: { marks: [{ row: 1, n: 9 }], strike: false, after: game.rolls[1].after } }];
        const b = bestShown(game, same, 1)!;
        expect(b.grade).toBe('best');
        expect(b.marks).toEqual([{ row: 1, n: 9 }]);
        expect(b.after).toEqual(game.rolls[1].after);
    });
    it('a different move within 0.05 points is "excellent", and the better move is shown', () => {
        const close = [an[0], { ...an[1], you_cost: 0.0001 }];
        const b = bestShown(game, close, 1)!;
        expect(b.grade).toBe('excellent');
        expect(b.marks).toEqual([{ row: 0, n: 9 }]);
        expect(b.after).toEqual(an[1].you_best.after);
        expect(positionAt(game, close, 4, 'best').pos).toEqual(positionAt(game, an, 4, 'best').pos);
    });
    it('any larger cost is a cost, with the better move shown', () => {
        const b = bestShown(game, an, 1)!;
        expect(b.grade).toBe('cost');
        expect(b.marks).toEqual([{ row: 0, n: 9 }]);
        expect(bestShown(game, an, 2)).toBeNull(); // not analysed yet
    });
});

describe('graph views', () => {
    const model = (k: Parameters<typeof graphModel>[2]) => graphModel(game, an, k, 'Master');
    it('the tabs, win chance first and skill second', () => {
        expect(GRAPHS.map((g) => g.tab)).toEqual(['Win chance', 'Skill', 'Dice luck', 'Luck · z']);
    });
    it('skill: the bot\'s play less your mistakes so far, changing at each move', () => {
        expect(graphSeries(game, an).skill).toEqual([0, 0, 0.01, 0.01, expect.closeTo(-0.036), null, null]);
    });
    it('win chance: every analysed point is a knot, 0–1, a dashed 50% line, dots on costly moves', () => {
        const m = model('win');
        expect(m).toMatchObject({ title: 'Your win chance', knots: [0, 1, 2, 3, 4], lo: 0, hi: 1, area: true, guides: [{ v: 0.5, dash: true }], dots: [{ j: 4, tone: 'you' }] });
        expect(m.tag(0.456)).toBe('46%');
    });
    it('dice luck: knots after each roll\'s dice (and the last analysed point), a scale in tens of points', () => {
        const m = model('luck');
        expect(m).toMatchObject({ title: 'Dice luck so far', knots: [0, 1, 3, 4], lo: -0.1, hi: 0.1, area: false, guides: [{ v: 0, dash: false }] });
        expect(m.labels.map((l) => l.text)).toEqual(['+10', '−10']);
        expect(m.tag(-0.03)).toBe('−3.0');
    });
    it('skill: knots after each move, dots for both players\' mistakes', () => {
        const m = model('skill');
        expect(m).toMatchObject({ title: 'Win chance won or lost by play', short: 'Won or lost by play', knots: [0, 2, 4] });
        expect(m.dots).toEqual([{ j: 2, tone: 'bot' }, { j: 4, tone: 'you' }]);
        expect(m.legend).toEqual([{ text: 'You', tone: 'you' }, { text: 'Master', tone: 'bot' }]);
        expect(m.tag(-0.036)).toBe('−3.6');
    });
    it('luck · z: ±3 with the luck score\'s bands', () => {
        const m = model('z');
        expect(m).toMatchObject({ title: 'Dice luck compared with a typical game', short: 'Luck vs a typical game', lo: -3, hi: 3, bands: [{ from: -0.5, to: 0.5 }] });
        expect(m.guides.map((g) => g.v)).toEqual([1.5, 0.5, -0.5, -1.5]);
        expect(m.labels.map((l) => l.text)).toEqual(['Very lucky', 'Lucky', 'Average', 'Unlucky', 'Very unlucky']);
        expect(m.tag(1.37)).toBe('z +1.4');
    });
    it('nothing analysed: no knots', () => {
        expect(graphModel(game, [], 'luck', 'Master').knots).toEqual([]);
    });
});
