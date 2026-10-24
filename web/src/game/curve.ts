/**
 * A smooth curve through knots (xs increasing): a monotone cubic
 * (Fritsch–Carlson), so between two knots it never leaves their range and a
 * run of equal knots stays flat. `d` is an SVG path; `at(x)` its height at x
 * (clamped to the knots' span).
 */
export function monotone(xs: number[], ys: number[]): { d: string; at: (x: number) => number } {
    const n = xs.length;
    const f = (v: number) => v.toFixed(1);
    if (n === 0) return { d: '', at: () => NaN };
    if (n === 1) return { d: `M${f(xs[0])},${f(ys[0])}`, at: () => ys[0] };
    const s = xs.slice(1).map((x, i) => (ys[i + 1] - ys[i]) / (x - xs[i]));
    const m = xs.map((_, i) => (i === 0 ? s[0] : i === n - 1 ? s[n - 2] : s[i - 1] * s[i] <= 0 ? 0 : (s[i - 1] + s[i]) / 2));
    for (let i = 0; i < n - 1; i++) {
        if (s[i] === 0) {
            m[i] = m[i + 1] = 0;
            continue;
        }
        const a = m[i] / s[i], b = m[i + 1] / s[i], h = a * a + b * b;
        if (h > 9) {
            const t = 3 / Math.sqrt(h);
            m[i] = t * a * s[i];
            m[i + 1] = t * b * s[i];
        }
    }
    let d = `M${f(xs[0])},${f(ys[0])}`;
    for (let i = 0; i < n - 1; i++) {
        const dx = (xs[i + 1] - xs[i]) / 3;
        d += ` C${f(xs[i] + dx)},${f(ys[i] + m[i] * dx)} ${f(xs[i + 1] - dx)},${f(ys[i + 1] - m[i + 1] * dx)} ${f(xs[i + 1])},${f(ys[i + 1])}`;
    }
    const at = (x: number) => {
        const c = Math.max(xs[0], Math.min(xs[n - 1], x));
        let i = 0;
        while (i < n - 2 && c > xs[i + 1]) i++;
        const h = xs[i + 1] - xs[i], t = (c - xs[i]) / h, t2 = t * t, t3 = t2 * t;
        return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * m[i + 1];
    };
    return { d, at };
}
