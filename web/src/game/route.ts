/**
 * Spec "Navigation": `#/explore` (optionally `?p=<code>`) is the explorer,
 * `#/review` (optionally `?g=<game code>`) the game review; anything else the game.
 * (The about page is a separate static page, `about/`.)
 */
export type Route = { view: 'play' } | { view: 'explore'; code: string | null } | { view: 'review'; code: string | null };

export function parseRoute(hash: string): Route {
    const r = /^#\/review(?:\?(.*))?$/.exec(hash);
    if (r) return { view: 'review', code: new URLSearchParams(r[1] ?? '').get('g') };
    const m = /^#\/explore(?:\?(.*))?$/.exec(hash);
    if (!m) return { view: 'play' };
    return { view: 'explore', code: new URLSearchParams(m[1] ?? '').get('p') };
}
