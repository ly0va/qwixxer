import { defineConfig } from 'vitest/config';
import { createServer, type Plugin, type ViteDevServer } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

const base = '/qwixxer/';

/**
 * The about page is a real path (about/, not #/about) so search engines index
 * it, and it is rendered to HTML here so its text is in the page itself. The
 * dev server renders it on each request; a build starts a throwaway SSR server
 * for the one render.
 */
function aboutPage(): Plugin {
    let dev: ViteDevServer | undefined;
    return {
        name: 'qwixxer-about-page',
        configureServer(server) {
            dev = server;
        },
        transformIndexHtml: {
            order: 'pre',
            async handler(html) {
                if (!html.includes('<!--about-->')) return html;
                const server = dev ?? (await createServer({ configFile: false, base, plugins: [svelte()], appType: 'custom', server: { middlewareMode: true } }));
                try {
                    const { renderAbout } = await server.ssrLoadModule('/src/about.render.ts');
                    return html.replace('<!--about-->', renderAbout());
                } finally {
                    if (!dev) await server.close();
                }
            },
        },
    };
}

export default defineConfig({
    base,
    plugins: [svelte(), aboutPage()],
    worker: { format: 'es' },
    build: {
        outDir: 'dist',
        assetsInlineLimit: 0,
        rollupOptions: { input: { main: 'index.html', about: 'about/index.html' } },
    },
    server: { fs: { allow: ['.'] } },
    test: {
        // Client-compiled runes: `$state` values are Proxies, as in the browser,
        // so a test engine that structuredClones its arguments (like
        // postMessage) catches state leaking into the worker RPC.
        environment: './src/test/client-env.ts',
        include: ['src/**/*.test.ts'],
    },
});
