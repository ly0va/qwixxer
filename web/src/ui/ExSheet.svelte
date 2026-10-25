<script lang="ts">
    import type { ExplorerController } from '../game/explorer.svelte';
    import { exploreSheet, moreBadges, rowPoints, sheetTotal, type Who } from '../game/position';
    import { cellIndex } from '../game/rules';
    import MoreBadge from './MoreBadge.svelte';
    import SheetRows from './SheetRows.svelte';
    import StrikeBoxes from './StrikeBoxes.svelte';
    let { x, who }: { x: ExplorerController; who: Who } = $props();
    const name = $derived(who === 'you' ? 'You' : 'Opponent');
    const s = $derived(x.pos[who]);
    const rows = $derived(exploreSheet(x.pos, who));
    const breakdown = $derived(rowPoints(s).join(' + ') + (s.strikes ? ` − ${5 * s.strikes}` : ''));
    // What the net expects this side to add by the end (none once the game is over).
    const more = $derived(x.answer && !x.answer.over ? moreBadges(x.pos, who, x.answer.eval.expected[who]) : null);
</script>

<section class="card" aria-label="{name} sheet">
    <div class="top">
        <h2>{name}</h2>
        <b class="total">{sheetTotal(s)}</b>
        <span class="muted">{breakdown}</span>
        <div class="right">
            <span class="label">Strikes</span>
            <StrikeBoxes count={s.strikes} label={name} onTap={(k) => x.tapStrike(who, k)} />
            {#if more?.strikes}<MoreBadge badge={more.strikes} stale={x.pending} />{/if}
            <button type="button" class="clear" onclick={() => x.clear(who)}>Clear</button>
        </div>
    </div>
    <SheetRows {rows} size={42} onTap={(cell) => x.tapSquare(who, cell.sq.row, cellIndex(cell.sq.row, cell.sq.n))} onLock={(r) => x.tapLock(r)}>
        {#snippet after(r)}{#if more?.rows[r]}<MoreBadge badge={more.rows[r]} stale={x.pending} />{/if}{/snippet}
    </SheetRows>
</section>

<style>
    .card { display: flex; flex-direction: column; gap: 8px; background: var(--paper); border-radius: 14px; padding: 12px 14px; box-shadow: 0 1px 0 var(--edge), 0 8px 24px var(--shadow); }
    .top { display: flex; align-items: center; gap: 12px; }
    h2 { margin: 0; font-family: var(--cond); font-size: 22px; font-weight: 700; }
    .total { font-family: var(--cond); font-weight: 800; font-size: 22px; }
    .muted { font-size: 13px; color: var(--muted); }
    .right { margin-left: auto; display: flex; align-items: center; gap: 6px; }
    .label { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-right: 2px; }
    .clear { height: 34px; margin-left: 8px; padding: 0 12px; border-radius: 9px; border: none; background: var(--ground); color: var(--muted); font-weight: 600; font-size: 14px; cursor: pointer; }
</style>
