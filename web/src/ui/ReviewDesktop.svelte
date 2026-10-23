<script lang="ts">
    import { BOT_NAMES } from '../engine/protocol';
    import type { GameController } from '../game/controller.svelte';
    import { DESK_H, DESK_W, deskScale } from '../game/layout';
    import { gameHref, graphSeries, names, pointValue, positionAt } from '../game/review';
    import type { ReviewController } from '../game/review.svelte';
    import EvalBar from './EvalBar.svelte';
    import EvalGraph from './EvalGraph.svelte';
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
    const scale = $derived(deskScale(w, h));
    const nm = $derived(names(r.shared));
    const at = $derived(r.game ? positionAt(r.game, r.an, r.sel, r.side) : null);
    const series = $derived(r.game ? graphSeries(r.game, r.an) : null);
    const pending = $derived(r.game && series && series.values.includes(null) ? `Analysing… ${r.an.filter(Boolean).length} of ${r.game.rolls.length} rolls` : null);
    let notice = $state('');
    async function share() {
        if (!r.code) return;
        try {
            await navigator.clipboard.writeText(location.origin + location.pathname + gameHref(r.code));
            notice = 'Link copied';
        } catch {
            notice = 'Couldn’t copy the link';
        }
        setTimeout(() => (notice = ''), 2000);
    }
</script>

<svelte:window bind:innerWidth={w} bind:innerHeight={h} />
<div class="fit" style="width: {Math.floor(DESK_W * scale)}px; height: {Math.floor(DESK_H * scale)}px">
<div class="desk" style="transform: scale({scale})">
    <header class="head">
        <div class="left">
            <div class="word">Qwixxer</div>
            <Tabs current="review" onPlay={nav.play} onExplore={nav.explorer} onReview={nav.review} />
            {#if r.shared && r.game}<span class="pill">Shared game · {nm.You} vs {BOT_NAMES[r.game.bot]}</span>{/if}
        </div>
        <div class="right">
            {#if notice}<span class="notice" role="status">{notice}</span>{/if}
            {#if r.game}<button type="button" class="btn" onclick={share}><Icon name="link" />Share game</button>{/if}
            <Menu {c} {nav} tabs />
        </div>
    </header>
    <div class="body">
        <main>
            {#if r.game && at}
                <div class="board">
                    <EvalBar p={pointValue(r.an, r.sel)} vertical />
                    <div class="sheets">
                        <ReviewSheet name={nm.You} prev={at.prev} pos={at.pos} who="you" />
                        <ReviewSheet name={BOT_NAMES[r.game.bot]} prev={at.prev} pos={at.pos} who="opponent" />
                    </div>
                </div>
                <SelectedRoll {r} />
            {/if}
        </main>
        <aside>
            <ReviewSummary {r} />
            {#if r.game && series}
                <section class="card" aria-label="Win chance over the game">
                    <div class="ghead"><h2>{nm.Your} win chance</h2><span class="key"><span class="dot"></span>{nm.your} mistakes</span></div>
                    <EvalGraph values={series.values} luck={series.luck} dots={series.dots} selected={r.sel} {pending}
                        width={476} height={118} luckHeight={46} onSelect={(j) => r.select(j)} />
                </section>
                <ReviewLog {r} />
            {/if}
        </aside>
    </div>
</div>
</div>

<style>
    .fit { margin: 0 auto; overflow: hidden; }
    .desk { width: 1280px; height: 800px; box-sizing: border-box; padding: 18px 32px 22px; display: flex; flex-direction: column; gap: 14px; overflow: hidden; transform-origin: top left; }
    .head { display: flex; align-items: center; justify-content: space-between; height: 44px; flex: none; }
    .left, .right { display: flex; align-items: center; gap: 20px; }
    .right { gap: 10px; }
    .word { font-family: var(--cond); font-weight: 800; font-size: 30px; letter-spacing: 0.5px; }
    .pill { height: 30px; padding: 0 12px; border-radius: 15px; display: inline-flex; align-items: center; font-size: 14px; font-weight: 600; background: var(--btn-muted-bg); color: var(--soft); }
    .notice { font-size: 14px; color: var(--soft); }
    .btn { height: 44px; padding: 0 14px; border-radius: 12px; border: none; background: var(--paper); box-shadow: 0 1px 0 var(--edge); color: var(--ink); display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 15px; cursor: pointer; }
    .body { display: flex; gap: 24px; flex: 1; min-height: 0; }
    main { width: 684px; flex: none; display: flex; flex-direction: column; gap: 10px; }
    .board { display: flex; gap: 12px; }
    .sheets { flex: 1; display: flex; flex-direction: column; gap: 10px; }
    aside { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 12px; min-height: 0; }
    .card { display: flex; flex-direction: column; gap: 6px; background: var(--paper); border-radius: 14px; padding: 12px 16px; box-shadow: 0 1px 0 var(--edge); }
    .ghead { display: flex; align-items: baseline; gap: 10px; }
    h2 { margin: 0; font-family: var(--cond); font-size: 18px; font-weight: 700; }
    .key { margin-left: auto; display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--unlucky-ink); }
</style>
