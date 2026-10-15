<script lang="ts">
    import type { ExplorerController } from '../game/explorer.svelte';
    import type { GameController } from '../game/controller.svelte';
    import { explorerPhoneSize } from '../game/layout';
    import { exploreSheet, sheetTotal, type Who } from '../game/position';
    import { cellIndex } from '../game/rules';
    import Menu from './Menu.svelte';
    import MovesList from './MovesList.svelte';
    import type { Nav } from './nav';
    import PositionCard from './PositionCard.svelte';
    import RollCard from './RollCard.svelte';
    import SheetColumns from './SheetColumns.svelte';
    import StrikeBoxes from './StrikeBoxes.svelte';
    let { c, x, nav }: { c: GameController; x: ExplorerController; nav: Nav } = $props();
    let w = $state(window.innerWidth);
    const size = $derived(explorerPhoneSize(w));
    // Four strike boxes and their three 5 px gaps fit in one compact sheet column (4·size + 41 px).
    const strikeSize = $derived(Math.min(34, Math.floor((4 * size + 41 - 15) / 4)));
    const base = () => location.origin + location.pathname;
    const sides: { who: Who; name: string }[] = [{ who: 'you', name: 'You' }, { who: 'opponent', name: 'Opponent' }];
</script>

<svelte:window bind:innerWidth={w} />
<div class="xphone">
    <header class="head">
        <div class="word">Explorer</div>
        <div class="right">
            <button type="button" class="icon" aria-label="Copy link to this position" onclick={() => x.copyLink(base())}>
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"></path><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"></path></svg>
            </button>
            <Menu {c} {nav} />
        </div>
    </header>
    {#if x.notice}<div class="notice" role="status">{x.notice}</div>{/if}
    <PositionCard {x} compact />
    <div class="sheets">
        {#each sides as sd (sd.who)}
            {@const rows = exploreSheet(x.pos, sd.who)}
            <section class="side" aria-label="{sd.name} sheet">
                <div class="name"><h2>{sd.name}</h2><b>{sheetTotal(x.pos[sd.who])}</b></div>
                <SheetColumns {rows} {size} compact onTap={(cell) => x.tapSquare(sd.who, cell.sq.row, cellIndex(cell.sq.row, cell.sq.n))} onLock={(r) => x.tapLock(r)} />
                <StrikeBoxes count={x.pos[sd.who].strikes} size={strikeSize} label={sd.name} onTap={(k) => x.tapStrike(sd.who, k)} />
            </section>
        {/each}
    </div>
    <div class="row">
        <button type="button" class="btn" onclick={() => x.swap()}>Swap sides</button>
        <button type="button" class="btn muted" onclick={() => x.clearAll()}>Clear sheets</button>
        <button type="button" class="btn wide" disabled={!c.view} onclick={() => c.view && x.loadView(c.view)}>Explore current game</button>
    </div>
    <RollCard {x} compact />
    <MovesList {x} compact />
</div>

<style>
    .xphone { max-width: 430px; margin: 0 auto; box-sizing: border-box; padding: 12px 12px 20px; display: flex; flex-direction: column; gap: 10px; }
    .head { display: flex; align-items: center; justify-content: space-between; height: 44px; }
    .word { font-family: var(--cond); font-weight: 800; font-size: 26px; letter-spacing: 0.5px; }
    .right { display: flex; align-items: center; gap: 8px; }
    .icon { width: 44px; height: 44px; border-radius: 12px; border: none; background: var(--paper); box-shadow: 0 1px 0 var(--edge); color: var(--ink); display: flex; align-items: center; justify-content: center; padding: 0; cursor: pointer; }
    .notice { font-size: 13px; color: var(--soft); }
    .sheets { display: flex; gap: 8px; justify-content: center; }
    .side { display: flex; flex-direction: column; align-items: center; gap: 6px; }
    .name { align-self: stretch; display: flex; align-items: baseline; justify-content: space-between; padding: 0 4px; }
    .name h2 { margin: 0; font-family: var(--cond); font-size: 19px; font-weight: 700; }
    .name b { font-family: var(--cond); font-weight: 800; font-size: 20px; }
    .row { display: flex; flex-wrap: wrap; gap: 8px; }
    .btn { flex: 1; height: 40px; border-radius: 10px; border: none; background: var(--paper); box-shadow: 0 1px 0 var(--edge); color: var(--ink); font-weight: 600; font-size: 14px; cursor: pointer; }
    .btn.muted { color: var(--muted); }
    .btn.wide { flex-basis: 100%; }
    .btn:disabled { opacity: 0.5; cursor: default; }
</style>
