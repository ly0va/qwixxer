<script lang="ts" module>
    let ids = 0;
</script>

<script lang="ts">
    import { BOT_NAMES, type ReviewGame } from '../engine/protocol';
    import { GRAPHS, graphModel, names } from '../game/review';
    import type { ReviewController } from '../game/review.svelte';
    import EvalGraph from './EvalGraph.svelte';
    // Spec "The graph": one card for the game's graphs, a switch on top
    // (win chance, skill, dice luck, luck · z), the view's title and key,
    // then the graph. The selected point is shared by every view.
    let { r, game, width, height, compact = false }: { r: ReviewController; game: ReviewGame; width: number; height: number; compact?: boolean } = $props();
    const id = `graph${++ids}`;
    const nm = $derived(names(r.shared));
    const m = $derived(graphModel(game, r.an, r.graph, nm, BOT_NAMES[game.bot]));
    const pending = $derived(r.an.filter(Boolean).length < game.rolls.length ? `Analysing… ${r.an.filter(Boolean).length} of ${game.rolls.length} rolls` : null);
</script>

<section class="card" class:compact aria-label="Graphs of the game">
    <div class="tabs" role="tablist" aria-label="Graph">
        {#each GRAPHS as g}
            <button type="button" role="tab" aria-selected={r.graph === g.kind} aria-controls={id} class:on={r.graph === g.kind} onclick={() => (r.graph = g.kind)}>{g.tab}</button>
        {/each}
    </div>
    <div class="ghead">
        <h2>{compact ? m.short : m.title}</h2>
        {#if m.legend.length}
            <span class="keys">{#each m.legend as k}<span class="key"><span class="dot {k.tone}"></span>{k.text}</span>{/each}</span>
        {/if}
    </div>
    <div id={id} role="tabpanel">
        <EvalGraph {m} selected={r.sel} {pending} {width} {height} onSelect={(j) => r.select(j)} />
    </div>
</section>

<style>
    .card { display: flex; flex-direction: column; gap: 8px; background: var(--paper); border-radius: 14px; padding: 12px 16px; box-shadow: 0 1px 0 var(--edge); }
    .card.compact { padding: 10px 12px; }
    .tabs { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px; padding: 3px; border-radius: 11px; background: var(--btn-muted-bg); }
    .tabs button { min-height: 32px; padding: 0 4px; border: none; border-radius: 8px; background: transparent; color: var(--muted); font-size: 14px; font-weight: 600; white-space: nowrap; cursor: pointer; }
    .compact .tabs button { min-height: 34px; font-size: 13px; }
    .tabs button.on { background: var(--paper); color: var(--ink); font-weight: 700; box-shadow: 0 1px 0 var(--edge); }
    .ghead { display: flex; align-items: baseline; gap: 10px; min-height: 22px; }
    h2 { margin: 0; font-family: var(--cond); font-size: 18px; font-weight: 700; }
    .compact h2 { font-size: 17px; }
    .keys { margin-left: auto; display: flex; gap: 12px; }
    .key { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); white-space: nowrap; }
    .dot { width: 8px; height: 8px; border-radius: 50%; }
    .dot.you { background: var(--unlucky-ink); }
    .dot.bot { background: var(--soft); }
</style>
