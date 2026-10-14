<script lang="ts">
    import type { ExplorerController } from '../game/explorer.svelte';
    import { marginText, overText, percent } from '../game/position';
    let { x, compact = false }: { x: ExplorerController; compact?: boolean } = $props();
    const a = $derived(x.answer);
</script>

<section class="card" class:compact class:stale={x.pending} aria-label="Position" aria-busy={x.pending}>
    {#if a?.over}
        <h2>Game over</h2>
        <b class="big">{overText(a.over)}</b>
    {:else if a}
        {#if !compact}<div class="top"><h2>Position</h2><span class="muted">before the roll</span></div>{/if}
        <div class="line"><b class="big">{percent(a.eval.p_win, 0)}</b><span class="what">chance you win</span>
            {#if !compact}<span class="margin">{marginText(a.eval)}</span>{/if}</div>
        <div class="bar" role="img" aria-label="You {percent(a.eval.p_win, 0)}"><span style="width: {a.eval.p_win * 100}%"></span></div>
        {#if compact}<span class="margin">{marginText(a.eval)} · before the roll</span>{/if}
    {:else}
        <span class="muted">{x.error ? `Something went wrong: ${x.error}` : 'Thinking…'}</span>
    {/if}
    {#if a && x.error}<span class="err">Something went wrong: {x.error}</span>{/if}
</section>

<style>
    .card { display: flex; flex-direction: column; gap: 8px; background: var(--paper); border-radius: 14px; padding: 12px 16px; box-shadow: 0 1px 0 var(--edge); transition: opacity 0.15s; }
    .card.compact { gap: 7px; padding: 10px 14px; }
    .stale { opacity: 0.6; }
    .top { display: flex; align-items: baseline; justify-content: space-between; }
    h2 { margin: 0; font-family: var(--cond); font-size: 20px; font-weight: 700; }
    .line { display: flex; align-items: baseline; gap: 10px; }
    .big { font-family: var(--cond); font-weight: 800; font-size: 40px; line-height: 1; }
    .compact .big { font-size: 36px; }
    .what { font-weight: 600; font-size: 16px; }
    .margin { margin-left: auto; font-size: 14px; color: var(--soft); }
    .compact .margin { margin-left: 0; font-size: 13px; }
    .muted { font-size: 13px; color: var(--muted); }
    .err { font-size: 13px; color: var(--strike-red); }
    .bar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--btn-muted-bg); }
    .compact .bar { height: 8px; }
    .bar span { background: var(--ink); }
</style>
