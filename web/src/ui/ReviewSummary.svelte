<script lang="ts">
    import { BOT_NAMES } from '../engine/protocol';
    import { DOT_COST, luckScore, names, signed, totals, verdict } from '../game/review';
    import type { ReviewController } from '../game/review.svelte';
    // Spec "The page": the result, the verdict, and the three totals.
    let { r, compact = false }: { r: ReviewController; compact?: boolean } = $props();
    const nm = $derived(names(r.shared));
    const g = $derived(r.game);
    const n = $derived(g?.rolls.length ?? 0);
    const t = $derived(totals(r.an, n));
    const won = $derived(g?.over ? (g.over.you === g.over.opponent ? null : g.over.you > g.over.opponent) : null);
    const result = $derived(!g ? '' : !g.over ? 'Game not finished' : won === null ? 'Draw' : won ? `${nm.You} won` : `${BOT_NAMES[g.bot]} won`);
    const costly = $derived(r.an.filter((a) => a && a.you_cost >= DOT_COST).length);
    const soFar = $derived(t.done < n);
    const tiles = $derived([
        { label: 'Dice luck', short: 'Dice luck', value: signed(t.luck), tone: t.luck >= 0 ? 'lucky' : 'unlucky', note: soFar ? 'so far' : luckScore(t.luck, t.var).text },
        { label: `${nm.Your} mistakes`, short: `${nm.Your} errors`, value: signed(-t.you), tone: 'unlucky', note: soFar ? 'so far' : `${costly} costing 2+ pts` },
        { label: `${g ? BOT_NAMES[g.bot] : 'Bot'}’s mistakes`, short: 'Bot errors', value: signed(t.bot), tone: 'lucky', note: soFar ? 'so far' : `in ${nm.your} favour` },
    ]);
</script>

<section class="card" class:compact aria-label="Summary">
    {#if g}
        <div class="head">
            <h2>{result}</h2>
            {#if g.over}<span class="score">{g.over.you} – {g.over.opponent}</span>{/if}
            <span class="verdict">{r.error ? '' : verdict(t, n, won, nm)}</span>
        </div>
        <div class="tiles">
            {#each tiles as tile}
                <div class="tile">
                    <span class="lab">{compact ? tile.short : tile.label}</span>
                    <b class={tile.tone}>{tile.value}</b>
                    {#if !compact}<span class="note">{tile.note}</span>{/if}
                </div>
            {/each}
        </div>
    {/if}
    {#if r.error}
        <div class="error" role="alert">
            <span>{g ? `Analysis stopped: ${r.error}` : `Couldn’t open this game: ${r.error}`}</span>
            <button type="button" onclick={() => r.retry()}>Retry</button>
        </div>
    {:else if !g && !r.code}
        <p class="empty">Finish a game or open a shared link to review it.</p>
    {/if}
</section>

<style>
    .card { display: flex; flex-direction: column; gap: 10px; background: var(--paper); border-radius: 14px; padding: 12px 16px; box-shadow: 0 1px 0 var(--edge); }
    .card.compact { gap: 8px; padding: 10px 12px; }
    .head { display: flex; align-items: baseline; gap: 10px; }
    h2 { margin: 0; font-family: var(--cond); font-size: 22px; font-weight: 700; }
    .compact h2 { font-size: 20px; }
    .score { font-family: var(--cond); font-weight: 700; font-size: 20px; color: var(--soft); }
    .verdict { margin-left: auto; font-size: 14px; font-weight: 600; }
    .tiles { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
    .compact .tiles { gap: 6px; }
    .tile { display: flex; flex-direction: column; gap: 2px; padding: 8px 10px; border-radius: 10px; background: var(--log-bg); }
    .compact .tile { padding: 6px 8px; gap: 1px; }
    .lab { font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 1px; white-space: nowrap; }
    .compact .lab { font-size: 11px; letter-spacing: 0.6px; }
    .tile b { font-family: var(--cond); font-weight: 800; font-size: 26px; line-height: 1.05; }
    .compact .tile b { font-size: 22px; }
    .lucky { color: var(--lucky-ink); }
    .unlucky { color: var(--unlucky-ink); }
    .note { font-size: 12px; color: var(--muted); }
    .error { display: flex; align-items: center; gap: 10px; font-size: 14px; color: var(--unlucky-ink); }
    .error button { margin-left: auto; height: 32px; padding: 0 12px; border-radius: 8px; border: none; background: var(--ground); color: var(--ink); font-weight: 600; cursor: pointer; }
    .empty { margin: 0; color: var(--muted); font-size: 15px; }
</style>
