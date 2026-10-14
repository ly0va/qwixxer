import { describe, expect, it } from 'vitest';
import { decode, encode } from './link';
import { emptyPosition, type ExPosition } from './position';

function randomPosition(rand: () => number): ExPosition {
    const p = emptyPosition();
    for (const s of [p.you, p.opponent]) {
        s.marks = s.marks.map((r) => r.map(() => rand() < 0.4));
        s.strikes = Math.floor(rand() * 5);
    }
    p.locked = p.locked.map(() => rand() < 0.2);
    p.dice = p.dice.map(() => (rand() < 0.2 ? null : 1 + Math.floor(rand() * 6)));
    p.roller = rand() < 0.5 ? 'you' : 'opponent';
    return p;
}

describe('link codec', () => {
    it('round-trips random positions in 21 URL-safe characters', () => {
        let seed = 7;
        const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
        for (let k = 0; k < 300; k++) {
            const p = randomPosition(rand);
            const code = encode(p);
            expect(code).toMatch(/^1[A-Za-z0-9_-]{20}$/);
            expect(decode(code)).toEqual(p);
        }
    });
    it('rejects malformed codes', () => {
        const good = encode(emptyPosition());
        for (const bad of ['', '1', good.slice(0, -1), good + 'A', '2' + good.slice(1), good.slice(0, -1) + '*', '1' + '/'.repeat(20)]) {
            expect(decode(bad)).toBeNull();
        }
    });
    it('rejects out-of-range strikes and dice', () => {
        // encode writes any 3-bit value; decode must reject strikes > 4 and face 7.
        const p = emptyPosition();
        p.opponent.strikes = 7;
        expect(decode(encode(p))).toBeNull();
        const q = emptyPosition();
        q.dice[2] = 7;
        expect(decode(encode(q))).toBeNull();
    });
});
