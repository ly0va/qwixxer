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
    onMount(() => {
        c.init();
        const f = () => (phone = mq.matches);
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
        const action = keyAction(e.key, { field, keyboardFocus }, c.view?.phase);
        if (action && action !== 'clear') e.preventDefault();
        if (action === 'confirm') c.confirm();
        else if (action === 'roll') c.roll();
        else if (action === 'clear') c.clearSelection();
    }
</script>

<svelte:window onkeydown={onKey} />
{#if phone}<Phone {c} />{:else}<Desktop {c} />{/if}
