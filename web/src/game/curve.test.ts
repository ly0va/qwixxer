import { describe, expect, it } from 'vitest';
import { monotone } from './curve';

describe('monotone curve', () => {
    it('passes through every knot', () => {
        const c = monotone([0, 10, 20, 30], [5, 8, 2, 2]);
        expect(c.d.startsWith('M0.0,5.0')).toBe(true);
        expect(c.d.endsWith('30.0,2.0')).toBe(true);
        [[0, 5], [10, 8], [20, 2], [30, 2]].forEach(([x, y]) => expect(c.at(x)).toBeCloseTo(y));
    });
    it('stays flat between equal knots and never overshoots', () => {
        const c = monotone([0, 10, 20, 30, 40], [0, 0, 10, 10, 3]);
        for (let x = 0; x <= 10; x += 0.5) expect(c.at(x)).toBeCloseTo(0);
        for (let x = 10; x <= 20; x += 0.5) {
            expect(c.at(x)).toBeGreaterThanOrEqual(-1e-9);
            expect(c.at(x)).toBeLessThanOrEqual(10 + 1e-9);
        }
        for (let x = 20; x <= 30; x += 0.5) expect(c.at(x)).toBeCloseTo(10);
    });
    it('is a curve, not straight lines', () => {
        expect(monotone([0, 10, 20], [0, 2, 10]).d).toContain('C');
    });
    it('one knot is a point; none is nothing', () => {
        const c = monotone([4], [7]);
        expect(c.d).toBe('M4.0,7.0');
        expect(c.at(4)).toBe(7);
        expect(monotone([], []).d).toBe('');
    });
});
