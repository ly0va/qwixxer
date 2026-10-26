<script lang="ts" module>
    let ids = 0;
</script>

<script lang="ts">
    import type { luckText } from '../game/position';
    // A roll's luck: "Lucky +1.2", toned, and the percentages behind it in a
    // tooltip on hover, focus or tap (the same bubble as MoreBadge's). null:
    // not analysed yet. `size`: the review strip (sm), the explorer roll card
    // on a phone (md) or on desktop (lg). `start`: the tooltip hangs from the
    // pill's left edge (a pill near the card's left) instead of its right.
    let { luck, stale = false, size = 'lg', live = false, start = false }: {
        luck: ReturnType<typeof luckText> | null; stale?: boolean; size?: 'sm' | 'md' | 'lg'; live?: boolean; start?: boolean;
    } = $props();
    const id = `luck-${++ids}`;
    let open = $state(false);
    function outside(e: MouseEvent) {
        if (open && !(e.target instanceof Element && e.target.closest(`#${id}`))) open = false;
    }
</script>

<svelte:document onclick={outside} />

<span class="wrap {size}" class:start class:open class:stale {id} aria-live={live ? 'polite' : undefined}>
    {#if luck}
        <button type="button" class="luck {luck.tone}" aria-expanded={open} aria-describedby="{id}-tip" onclick={() => (open = !open)}>
            {luck.word}{#if luck.value}<b aria-label="{luck.value} points">{luck.value}</b>{/if}
        </button>
        <span class="tip" role="tooltip" id="{id}-tip">{luck.detail}</span>
    {:else}
        <span class="luck pending">Luck …</span>
    {/if}
</span>

<style>
    .wrap { position: relative; flex: none; display: inline-flex; transition: opacity 0.15s; }
    .stale { opacity: 0.6; }
    .luck { height: 28px; padding: 0 11px; border: none; border-radius: 14px; display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; font: inherit; font-weight: 700; font-size: 14px; cursor: pointer; }
    .luck b { font-family: var(--cond); font-weight: 800; font-size: 18px; }
    .md .luck { height: 26px; padding: 0 9px; font-size: 13px; gap: 5px; }
    .md .luck b { font-size: 17px; }
    .sm .luck { height: 24px; padding: 0 9px; font-size: 13px; gap: 5px; }
    .sm .luck b { font-size: 16px; }
    .luck.lucky { color: var(--lucky-ink); background: color-mix(in srgb, var(--lucky-ink) 14%, var(--paper)); }
    .luck.unlucky { color: var(--unlucky-ink); background: color-mix(in srgb, var(--unlucky-ink) 14%, var(--paper)); }
    .luck.average { color: var(--ink); background: var(--btn-muted-bg); }
    .luck.pending { color: var(--muted); font-weight: 600; background: var(--btn-muted-bg); cursor: default; }
    .open .luck { box-shadow: inset 0 0 0 1.5px currentColor; }
    .tip { display: none; position: absolute; z-index: 4; top: calc(100% + 4px); right: 0; width: max-content; max-width: 240px; box-sizing: border-box; padding: 8px 10px; border-radius: 9px; background: var(--ink); color: var(--ground); font-size: 13px; font-weight: 500; line-height: 1.35; white-space: normal; box-shadow: 0 6px 20px var(--shadow); }
    .start .tip { right: auto; left: 0; }
    /* Tap toggles it; hover only where there is a real pointer (a tap leaves
       :hover stuck on touch screens), and focus only from the keyboard. */
    .open .tip, .wrap:has(:focus-visible) .tip { display: block; }
    @media (hover: hover) { .wrap:hover .tip { display: block; } }
</style>
