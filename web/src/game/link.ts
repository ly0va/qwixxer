import { emptyPosition, type ExPosition } from './position';

/**
 * Spec "Links and memory": a position as a short URL-safe code, a version
 * character then base64url of 117 bits (see the plan's layout): both
 * sheets' marks and strikes, tap-closed rows, dice (0 = unset), roller.
 */
const VERSION = '1';
const BYTES = 15;

class Bits {
    bytes = new Uint8Array(BYTES);
    pos = 0;
    put(v: number, n: number) {
        for (let k = n - 1; k >= 0; k--, this.pos++) if ((v >> k) & 1) this.bytes[this.pos >> 3] |= 0x80 >> (this.pos & 7);
    }
    get(n: number): number {
        let v = 0;
        for (let k = 0; k < n; k++, this.pos++) v = (v << 1) | ((this.bytes[this.pos >> 3] >> (7 - (this.pos & 7))) & 1);
        return v;
    }
}

export function encode(p: ExPosition): string {
    const b = new Bits();
    for (const s of [p.you, p.opponent]) {
        for (const row of s.marks) for (const m of row) b.put(m ? 1 : 0, 1);
        b.put(s.strikes, 3);
    }
    for (const l of p.locked) b.put(l ? 1 : 0, 1);
    for (const d of p.dice) b.put(d ?? 0, 3);
    b.put(p.roller === 'opponent' ? 1 : 0, 1);
    const b64 = btoa(String.fromCharCode(...b.bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    return VERSION + b64;
}

export function decode(code: string): ExPosition | null {
    if (!/^1[A-Za-z0-9_-]{20}$/.test(code)) return null;
    let raw: string;
    try {
        raw = atob(code.slice(1).replace(/-/g, '+').replace(/_/g, '/'));
    } catch {
        return null;
    }
    const b = new Bits();
    b.bytes = Uint8Array.from(raw, (c) => c.charCodeAt(0));
    const p = emptyPosition();
    for (const s of [p.you, p.opponent]) {
        s.marks = s.marks.map((row) => row.map(() => b.get(1) === 1));
        s.strikes = b.get(3);
        if (s.strikes > 4) return null;
    }
    p.locked = p.locked.map(() => b.get(1) === 1);
    for (let i = 0; i < 6; i++) {
        const d = b.get(3);
        if (d > 6) return null;
        p.dice[i] = d === 0 ? null : d;
    }
    p.roller = b.get(1) === 1 ? 'opponent' : 'you';
    return p;
}
