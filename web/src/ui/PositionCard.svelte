<script lang="ts">
    import type { ExplorerController } from '../game/explorer.svelte';
    import { analysisMargin, chanceText, gamesText, marginText, overText, winChance } from '../game/position';
    let { x, compact = false }: { x: ExplorerController; compact?: boolean } = $props();
    const a = $derived(x.answer);
    const t = $derived(x.analysis?.totals ?? null);
    // Numbers for an older position stay, faded, until the first batch for this one.
    const fresh = $derived(x.settled);
    const w = $derived(t ? winChance(t) : null);
    const m = $derived(t ? marginText(analysisMargin(t)) : '');
</script>

<section class="card" class:compact class:stale={!fresh} aria-label="Position" aria-busy={!fresh}>
    {#if a?.over}
        <h2>Game over</h2>
        <b class="big">{overText(a.over)}</b>
    {:else if t && w}
        {#if !compact}<div class="top"><h2>Position</h2><span class="muted">before the roll</span></div>{/if}
        <div class="line"><b class="big">{chanceText(w)}</b><span class="what">chance you win</span>
            {#if !compact}<span class="margin">{m}</span>{/if}</div>
        <div class="bar" role="img" aria-label="You {chanceText(w)}"><span style="width: {w.p * 100}%"></span></div>
        <span class="games">{#if x.analysing && fresh}<span class="dot" aria-hidden="true"></span>{/if}{gamesText(w, t.games)}</span>
        {#if compact}<span class="margin">{m} · before the roll</span>{/if}
    {:else}
        <span class="muted">{x.error ? `Something went wrong: ${x.error}` : '…'}</span>
    {/if}
    {#if (t || a) && x.error}<span class="err">Something went wrong: {x.error}</span>{/if}
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
    .games { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ink); animation: pulse 1.2s ease-in-out infinite; }
    @keyframes pulse { 50% { opacity: 0.25; } }
    @media (prefers-reduced-motion: reduce) { .dot { animation: none; } }
</style>
