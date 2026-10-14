import type { Environment } from 'vitest/runtime';

/**
 * Node globals, but modules go through Vite's client environment, so Svelte
 * compiles `.svelte.ts` runes for the browser: `$state` arrays and objects are
 * Proxies, as in the real app (they can't be posted to a Worker as-is).
 */
export default <Environment>{
    name: 'client-node',
    viteEnvironment: 'client',
    setup: () => ({ teardown() {} }),
};
