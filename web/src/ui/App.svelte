<script lang="ts">
    import { onMount } from 'svelte';
    import { createEngine } from '../engine/client';
    import { GameController } from '../game/controller.svelte';
    import { ExplorerController } from '../game/explorer.svelte';
    import { LAST_GAME, ReviewController } from '../game/review.svelte';
    import { keyAction } from '../game/keys';
    import { parseRoute, type Route } from '../game/route';
    import Desktop from './Desktop.svelte';
    import ExplorerDesktop from './ExplorerDesktop.svelte';
    import ExplorerPhone from './ExplorerPhone.svelte';
    import type { Nav } from './nav';
    import Phone from './Phone.svelte';

    const store = { get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
        set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } } };
    // One worker for both: the game and the explorer share the engine.
    const engine = createEngine();
    const c = new GameController(engine, store);
    const x = new ExplorerController(engine, store);
    const rv = new ReviewController(engine, store);
    // Spec "Implementation notes · Window sizes": below ~1100 px wide or in portrait, the phone layout.
    const mq = window.matchMedia('(max-width: 1099px), (orientation: portrait)');
    let phone = $state(mq.matches);
    // The phone's bot/log drawer (bound from Phone): page-wide keys must not reach the game under it.
    let drawer = $state(false);
    // The initial view comes from the address, so a shared link never flashes
    // the game first; a link's code itself loads in sync() on mount.
    const first = parseRoute(location.hash);
    let route = $state<Route>(first.view === 'explore' ? { view: 'explore', code: null } : first);

    // A link (`?p=`) loads once, then the address becomes plain #/explore so
    // a reload keeps your edits instead of reloading the link.
    function sync() {
        const r = parseRoute(location.hash);
        if (r.view === 'explore' && r.code !== null) {
            x.loadCode(r.code);
            history.replaceState(null, '', '#/explore');
            route = { view: 'explore', code: null };
        } else route = r;
        // A review keeps its ?g= code in the address: the code never changes, so a reload shows the same game.
        // It is shared unless it is your own last game.
        if (route.view === 'review') void rv.load(route.code, route.code !== null && route.code !== store.get(LAST_GAME));
        // The explorer evaluates when entered (loadCode already has).
        if (route.view === 'explore') x.refresh();
        c.menuOpen = false;
        x.picker = null;
        activity();
    }
    // Spec "Controller · Pause": analysis runs only while the explorer is on screen.
    const activity = () => {
        const visible = document.visibilityState === 'visible';
        x.setActive(route.view === 'explore' && visible);
        rv.setActive(route.view === 'review' && visible);
    };
    const nav: Nav = {
        get current() { return route.view; },
        play: () => { if (route.view === 'play') return; history.pushState(null, '', location.pathname + location.search); sync(); },
        explorer: () => { location.hash = '#/explore'; },
        review: () => { location.hash = '#/review'; },
    };

    onMount(() => {
        c.init();
        x.init();
        sync();
        const f = () => { phone = mq.matches; if (!phone) drawer = false; };
        mq.addEventListener('change', f);
        window.addEventListener('hashchange', sync);
        window.addEventListener('popstate', sync);
        document.addEventListener('visibilitychange', activity);
        return () => { mq.removeEventListener('change', f); window.removeEventListener('hashchange', sync); window.removeEventListener('popstate', sync); document.removeEventListener('visibilitychange', activity); };
    });
    $effect(() => { document.documentElement.dataset.theme = c.theme; });

    // Spec "Implementation notes · Keyboard": Enter confirms, Space rolls,
    // Escape clears. Only keyboard-focused controls and form fields keep their
    // native Enter/Space (see keys.ts).
    function focusVisible(el: Element): boolean {
        try { return el.matches(':focus-visible'); } catch { return false; }
    }
    function onKey(e: KeyboardEvent) {
        const t = e.target instanceof Element ? e.target : null;
        const field = !!t?.closest('select, input, textarea');
        const keyboardFocus = !!t && t !== document.body && focusVisible(t);
        const action = keyAction(e.key, { field, keyboardFocus, modal: (phone && drawer) || c.menuOpen, explore: route.view === 'explore', review: route.view === 'review' }, c.view?.phase);
        if (action && action !== 'clear' && action !== 'close') e.preventDefault();
        if (action === 'confirm') c.confirm();
        else if (action === 'roll') c.roll();
        else if (action === 'clear') c.clearSelection();
        else if (action === 'prev') rv.step(-1);
        else if (action === 'next') rv.step(1);
        else if (action === 'first') rv.first();
        else if (action === 'last') rv.last();
        else if (action === 'close') { if (c.menuOpen) c.menuOpen = false; else if (x.picker !== null) x.picker = null; else drawer = false; }
    }
</script>

<svelte:window onkeydown={onKey} />
{#if route.view === 'review'}
    <p>Review</p>
{:else if route.view === 'explore'}
    {#if phone}<ExplorerPhone {c} {x} {nav} />{:else}<ExplorerDesktop {c} {x} {nav} />{/if}
{:else if phone}<Phone {c} {nav} bind:drawer />{:else}<Desktop {c} {nav} />{/if}
