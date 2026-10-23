<script lang="ts">
    import { rowPoints, sheetTotal, type ExPosition, type Who } from '../game/position';
    import { reviewSheet } from '../game/review';
    import SheetColumns from './SheetColumns.svelte';
    import SheetRows from './SheetRows.svelte';
    import StrikeBoxes from './StrikeBoxes.svelte';
    // A read-only sheet at the selected point; this roll's marks in pencil.
    let { name, prev, pos, who, compact = false, size = 40, strikeSize = 30 }: { name: string; prev: ExPosition; pos: ExPosition; who: Who; compact?: boolean; size?: number; strikeSize?: number } = $props();
    const s = $derived(pos[who]);
    const rows = $derived(reviewSheet(prev, pos, who));
    const breakdown = $derived(rowPoints(s).join(' + ') + (s.strikes ? ` − ${5 * s.strikes}` : ''));
    const pencil = $derived(s.strikes > prev[who].strikes ? s.strikes - 1 : -1);
</script>

<section class="card" class:compact aria-label="{name} sheet">
    <div class="top">
        <h2>{name}</h2>
        <b class="total">{sheetTotal(s)}</b>
        {#if !compact}
            <span class="muted">{breakdown}</span>
            <div class="right"><span class="label">Strikes</span><StrikeBoxes count={s.strikes} size={strikeSize} label={name} {pencil} /></div>
        {/if}
    </div>
    {#if compact}
        <SheetColumns {rows} {size} compact />
        <div class="strikes"><StrikeBoxes count={s.strikes} size={strikeSize} label={name} {pencil} /></div>
    {:else}
        <SheetRows {rows} {size} />
    {/if}
</section>

<style>
    .card { display: flex; flex-direction: column; gap: 8px; background: var(--paper); border-radius: 14px; padding: 10px 14px 12px; box-shadow: 0 1px 0 var(--edge), 0 8px 24px var(--shadow); }
    .card.compact { flex: 1; min-width: 0; gap: 6px; padding: 8px 4px; align-items: center; }
    .top { display: flex; align-items: center; gap: 12px; align-self: stretch; }
    .compact .top { padding: 0 4px; }
    h2 { margin: 0; font-family: var(--cond); font-size: 22px; font-weight: 700; }
    .compact h2 { font-size: 19px; }
    .total { font-family: var(--cond); font-weight: 800; font-size: 22px; }
    .compact .total { margin-left: auto; font-size: 20px; }
    .muted { font-size: 13px; color: var(--muted); }
    .right { margin-left: auto; display: flex; align-items: center; gap: 6px; }
    .label { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-right: 2px; }
    .strikes { padding-top: 2px; }
</style>
