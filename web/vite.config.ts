import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
    base: '/qwixxer/',
    plugins: [svelte()],
    worker: { format: 'es' },
    build: { outDir: 'dist', assetsInlineLimit: 0 },
    server: { fs: { allow: ['.'] } },
    test: { environment: 'node', include: ['src/**/*.test.ts'] },
});
