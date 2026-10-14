import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
    base: '/qwixxer/',
    plugins: [svelte()],
    worker: { format: 'es' },
    build: { outDir: 'dist', assetsInlineLimit: 0 },
    server: { fs: { allow: ['.'] } },
    test: {
        // Client-compiled runes: `$state` values are Proxies, as in the browser,
        // so a test engine that structuredClones its arguments (like
        // postMessage) catches state leaking into the worker RPC.
        environment: './src/test/client-env.ts',
        include: ['src/**/*.test.ts'],
    },
});
