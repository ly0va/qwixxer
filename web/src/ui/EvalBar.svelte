<script lang="ts">
    // Spec "The page": your win chance at the selected point. Desktop: a
    // column beside the sheets, filled from the top like your sheet above
    // the bot's, with each number at its sheet's end; phone: a thin bar
    // edge to edge. null: not analysed yet.
    let { p, vertical }: { p: number | null; vertical: boolean } = $props();
    const pct = $derived(p === null ? null : Math.round(p * 100));
    // A number sits on the fill only when the fill reaches past it (about 8% of the column).
    const youOnFill = $derived(pct !== null && pct >= 8);
    const themOnFill = $derived(pct !== null && pct > 92);
</script>

<div class="bar" class:vertical role="img" aria-label={pct === null ? 'Win chance not analysed yet' : `Win chance ${pct}%`}>
    <span class="fill" style="{vertical ? 'height' : 'width'}: {pct ?? 0}%"></span>
    {#if vertical}
        <span class="you" class:on={youOnFill}>{pct ?? '…'}</span>
        <span class="them" class:on={themOnFill}>{pct === null ? '' : 100 - pct}</span>
    {/if}
</div>

<style>
    .bar { position: relative; display: flex; background: var(--btn-muted-bg); overflow: hidden; height: 6px; }
    .fill { background: var(--ink); transition: width 0.2s, height 0.2s; }
    .vertical { width: 30px; height: auto; flex: none; border-radius: 8px; flex-direction: column; justify-content: flex-start; box-shadow: 0 1px 0 var(--edge); }
    .them, .you { position: absolute; left: 0; right: 0; text-align: center; font-family: var(--cond); font-size: 15px; }
    .them { bottom: 6px; font-weight: 700; color: var(--muted); }
    .you { top: 6px; font-weight: 800; color: var(--ink); }
    .you.on, .them.on { color: var(--ground); }
</style>
