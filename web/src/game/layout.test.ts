import { describe, expect, it } from 'vitest';
import { deskScale, PHONE_FIXED, PHONE_FIXED_SHORT, phoneLayout } from './layout';

// Height the phone page needs for a given layout (see layout.ts).
const needed = (l: ReturnType<typeof phoneLayout>) => (l.short ? PHONE_FIXED_SHORT : PHONE_FIXED) + 12 * l.size;

describe('phoneLayout', () => {
    it('uses 48 px squares and the design spacing on a 390×844 phone', () => {
        const l = phoneLayout(390, 844);
        expect(l).toMatchObject({ size: 48, short: false, scroll: false });
        expect(needed(l)).toBeLessThanOrEqual(844);
        expect(l.die).toBe(36);
        expect(l.botSize).toBe(24);
    });
    it('fits 375×667 at 36 px with tightened spacing, no scrolling', () => {
        const l = phoneLayout(375, 667);
        expect(l).toMatchObject({ size: 36, short: true, scroll: false });
        expect(needed(l)).toBeLessThanOrEqual(667);
    });
    it('keeps every square reachable on a 375×553 phone (Safari iPhone SE)', () => {
        const l = phoneLayout(375, 553);
        expect(l.size).toBeGreaterThanOrEqual(28);
        expect(l.scroll || needed(l) <= 553).toBe(true);
    });
    it('scrolls at the 28 px floor on 320×480', () => {
        const l = phoneLayout(320, 480);
        expect(l).toMatchObject({ size: 28, scroll: true });
    });
    it('keeps the dice row and the side column within the width', () => {
        for (const w of [320, 360, 375, 390, 430, 800]) {
            const l = phoneLayout(w, 844);
            const content = Math.min(w, 430) - 24;
            expect(110 + 6 * l.die + 36).toBeLessThanOrEqual(content);
            expect(content - (4 * l.size + 62) - 10).toBeGreaterThanOrEqual(98);
            expect(12 * l.botSize + 96).toBeLessThanOrEqual(Math.min(w, 430));
        }
    });
});

describe('deskScale', () => {
    it('keeps the design size when it fits', () => {
        expect(deskScale(1280, 800)).toBe(1);
        expect(deskScale(1920, 1080)).toBe(1);
    });
    it('scales down to the tighter of width and height', () => {
        expect(deskScale(1366, 650)).toBe(0.8125); // 1366×768 laptop, browser chrome included
        expect(deskScale(1100, 700)).toBeCloseTo(0.859375, 6);
        expect(deskScale(1440, 760)).toBe(0.95);
    });
});
