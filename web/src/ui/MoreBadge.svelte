<script lang="ts" module>
    let ids = 0;
</script>

<script lang="ts">
    import type { Badge } from '../game/position';
    import MarkX from './MarkX.svelte';
    // What the net expects a row (or the strikes) to add by the end of the
    // game: "+2" and a pencil cross, explained by a tooltip on hover, focus
    // or tap. `wide`: the tooltip spans the nearest positioned ancestor
    // (the phone's narrow columns) instead of hanging from the badge.
    let { badge, wide = false, stale = false }: { badge: Badge; wide?: boolean; stale?: boolean } = $props();
    const id = `more-${++ids}`;
    let open = $state(false);
    function outside(e: MouseEvent) {
        if (open && !(e.target instanceof Element && e.target.closest(`#${id}`))) open = false;
    }
</script>

<svelte:document onclick={outside} />

<span class="more" class:wide class:open class:stale {id}>
    <button type="button" aria-label={badge.tip} aria-expanded={open} onclick={() => (open = !open)}>
        <span>{badge.text}</span><span class="cross"><MarkX pencil /></span>
    </button>
    <span class="tip" role="tooltip">{badge.tip}</span>
</span>

<style>
    .more { position: relative; display: inline-flex; transition: opacity 0.15s; }
    .more.wide { position: static; }
    .stale { opacity: 0.5; }
    button { height: 24px; padding: 0 2px; border: none; border-radius: 6px; background: none; display: flex; align-items: center; gap: 1px; font-family: var(--cond); font-weight: 700; font-size: 16px; color: var(--muted); cursor: pointer; white-space: nowrap; }
    .open button { background: var(--log-bg); box-shadow: inset 0 0 0 1.5px var(--faint); }
    .cross { position: relative; width: 16px; height: 16px; }
    .cross :global(svg) { inset: 0; width: 100%; height: 100%; }
    .tip { display: none; position: absolute; z-index: 4; top: calc(100% + 4px); right: 0; width: max-content; max-width: 240px; box-sizing: border-box; padding: 8px 10px; border-radius: 9px; background: var(--ink); color: var(--ground); font-size: 13px; font-weight: 500; line-height: 1.35; box-shadow: 0 6px 20px var(--shadow); }
    .wide .tip { left: 6px; right: 6px; width: auto; max-width: none; top: calc(100% + 2px); }
    /* Tap toggles it; hover only where there is a real pointer (a tap leaves
       :hover stuck on touch screens), and focus only from the keyboard. */
    .open .tip, .more:has(:focus-visible) .tip { display: block; }
    @media (hover: hover) { .more:hover .tip { display: block; } }
</style>
