<script lang="ts">
    import { onMount } from 'svelte';
    import { createEngine } from '../engine/client';
    import { GameController } from '../game/controller.svelte';
    import { keyAction } from '../game/keys';
    import Desktop from './Desktop.svelte';
    import Phone from './Phone.svelte';

    const store = { get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
        set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } } };
    const c = new GameController(createEngine(), store);
    // Spec "Implementation notes · Window sizes": below ~1100 px wide or in portrait, the phone layout.
    const mq = window.matchMedia('(max-width: 1099px), (orientation: portrait)');
    let phone = $state(mq.matches);
    // The phone's bot/log drawer (bound from Phone): page-wide keys must not reach the game under it.
    let drawer = $state(false);
    onMount(() => {
        c.init();
        const f = () => { phone = mq.matches; if (!phone) drawer = false; };
        mq.addEventListener('change', f);
        return () => mq.removeEventListener('change', f);
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
        const action = keyAction(e.key, { field, keyboardFocus, modal: phone && drawer }, c.view?.phase);
        if (action && action !== 'clear' && action !== 'close') e.preventDefault();
        if (action === 'confirm') c.confirm();
        else if (action === 'roll') c.roll();
        else if (action === 'clear') c.clearSelection();
        else if (action === 'close') drawer = false;
    }
</script>

<svelte:window onkeydown={onKey} />
{#if phone}<Phone {c} bind:drawer />{:else}<Desktop {c} />{/if}
