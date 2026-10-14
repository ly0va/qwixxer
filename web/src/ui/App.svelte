<script lang="ts">
    import { onMount } from 'svelte';
    import { createEngine } from '../engine/client';
    import { GameController } from '../game/controller.svelte';
    import Desktop from './Desktop.svelte';

    const store = { get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
        set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } } };
    const c = new GameController(createEngine(), store);
    onMount(() => { c.init(); });
    $effect(() => { document.documentElement.dataset.theme = c.theme; });

    // Spec "Implementation notes · Keyboard": Enter confirms, Space rolls,
    // Escape clears. A focused button keeps its native Enter/Space click.
    function onKey(e: KeyboardEvent) {
        const t = e.target as HTMLElement | null;
        if (t && (t.closest('button') || t.closest('select') || t.closest('input'))) return;
        if (e.key === 'Enter') { e.preventDefault(); c.confirm(); }
        else if (e.key === ' ' && c.view?.phase === 'you_to_roll') { e.preventDefault(); c.roll(); }
        else if (e.key === 'Escape') c.clearSelection();
    }
</script>

<svelte:window onkeydown={onKey} />
<Desktop {c} />
