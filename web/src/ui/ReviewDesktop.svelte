<script lang="ts">
    import { BOT_NAMES } from '../engine/protocol';
    import type { GameController } from '../game/controller.svelte';
    import { DESK_MIN_H, DESK_W, deskFit } from '../game/layout';
    import { pointValue, positionAt } from '../game/review';
    import type { ReviewController } from '../game/review.svelte';
    import EvalBar from './EvalBar.svelte';
    import GraphCard from './GraphCard.svelte';
    import Icon from './Icon.svelte';
    import Menu from './Menu.svelte';
    import type { Nav } from './nav';
    import ReviewLog from './ReviewLog.svelte';
    import ReviewSheet from './ReviewSheet.svelte';
    import ReviewSummary from './ReviewSummary.svelte';
    import SelectedRoll from './SelectedRoll.svelte';
    import Tabs from './Tabs.svelte';
    // Spec `2026-10-24-game-review-design.md`, "The page · Desktop" (board R1).
    let { c, r, nav }: { c: GameController; r: ReviewController; nav: Nav } = $props();
    let w = $state(window.innerWidth);
    let h = $state(window.innerHeight);
    const fit = $derived(deskFit(w, h, DESK_MIN_H.review));
    const at = $derived(r.game ? positionAt(r.game, r.an, r.sel, r.side) : null);
</script>

<svelte:window bind:innerWidth={w} bind:innerHeight={h} />
<div class="fit" style="width: {Math.floor(DESK_W * fit.scale)}px; height: {Math.floor(fit.height * fit.scale)}px">
<div class="desk" style="height: {fit.height}px; transform: scale({fit.scale})">
    <header class="head">
        <div class="left">
            <div class="word">Qwixxer</div>
            <Tabs current="review" onPlay={nav.play} onExplore={nav.explorer} onReview={nav.review} />
        </div>
        <div class="right">
            {#if r.notice}<span class="notice" role="status">{r.notice}</span>{/if}
            {#if r.game}<button type="button" class="btn" onclick={() => r.share()}><Icon name="link" />Share game</button>{/if}
            <Menu {c} {nav} tabs />
        </div>
    </header>
    <div class="body">
        <main>
            {#if r.game && at}
                <div class="board">
                    <EvalBar p={pointValue(r.an, r.sel)} vertical />
                    <div class="sheets">
                        <ReviewSheet name="You" prev={at.prev} pos={at.pos} who="you" />
                        <ReviewSheet name={BOT_NAMES[r.game.bot]} prev={at.prev} pos={at.pos} who="opponent" />
                    </div>
                </div>
                <!-- The roll card takes the column's last pixels, so it ends level with the move list. -->
                <div class="fill"><SelectedRoll {r} /></div>
            {/if}
        </main>
        <aside>
            <ReviewSummary {r} />
            {#if r.game}
                <GraphCard {r} game={r.game} width={476} height={150} />
                <ReviewLog {r} />
            {/if}
        </aside>
    </div>
</div>
</div>

<style>
    .fit { margin: 0 auto; overflow: hidden; }
    .desk { width: 1280px; box-sizing: border-box; padding: 22px 32px; display: flex; flex-direction: column; gap: 14px; overflow: hidden; transform-origin: top left; }
    .head { display: flex; align-items: center; justify-content: space-between; height: 44px; flex: none; }
    .left, .right { display: flex; align-items: center; gap: 20px; }
    .right { gap: 10px; }
    .word { font-family: var(--cond); font-weight: 800; font-size: 30px; letter-spacing: 0.5px; }
    .notice { font-size: 14px; color: var(--soft); }
    .btn { height: 44px; padding: 0 14px; border-radius: 12px; border: none; background: var(--paper); box-shadow: 0 1px 0 var(--edge); color: var(--ink); display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 15px; cursor: pointer; }
    .body { display: flex; gap: 24px; flex: 1; min-height: 0; }
    main { width: 684px; flex: none; display: flex; flex-direction: column; gap: 10px; }
    .board { display: flex; gap: 12px; }
    .sheets { flex: 1; display: flex; flex-direction: column; gap: 10px; }
    .fill { flex: 1; display: flex; flex-direction: column; }
    .fill > :global(section) { flex: 1; }
    aside { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 12px; min-height: 0; }
</style>
