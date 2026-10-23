<script lang="ts">
    // Spec "The page": your win chance at the selected point. Desktop: a
    // column beside the sheets with the number at its foot; phone: a thin
    // bar edge to edge. null: not analysed yet.
    let { p, vertical }: { p: number | null; vertical: boolean } = $props();
    const pct = $derived(p === null ? null : Math.round(p * 100));
</script>

<div class="bar" class:vertical role="img" aria-label={pct === null ? 'Win chance not analysed yet' : `Win chance ${pct}%`}>
    <span class="fill" style="{vertical ? 'height' : 'width'}: {pct ?? 0}%"></span>
    {#if vertical}
        <span class="them">{pct === null ? '' : 100 - pct}</span>
        <span class="you">{pct ?? '…'}</span>
    {/if}
</div>

<style>
    .bar { position: relative; display: flex; background: var(--btn-muted-bg); overflow: hidden; height: 6px; }
    .fill { background: var(--ink); transition: width 0.2s, height 0.2s; }
    .vertical { width: 30px; height: auto; flex: none; border-radius: 8px; flex-direction: column; justify-content: flex-end; box-shadow: 0 1px 0 var(--edge); }
    .them, .you { position: absolute; left: 0; right: 0; text-align: center; font-family: var(--cond); font-size: 15px; }
    .them { top: 6px; font-weight: 700; color: var(--muted); }
    .you { bottom: 6px; font-weight: 800; color: var(--ground); }
</style>
