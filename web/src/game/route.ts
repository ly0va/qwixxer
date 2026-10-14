/** Spec "Navigation": `#/explore` (optionally `?p=<code>`) is the explorer; anything else the game. */
export type Route = { view: 'play' } | { view: 'explore'; code: string | null };

export function parseRoute(hash: string): Route {
    const m = /^#\/explore(?:\?(.*))?$/.exec(hash);
    if (!m) return { view: 'play' };
    return { view: 'explore', code: new URLSearchParams(m[1] ?? '').get('p') };
}
