<script lang="ts">
    import { BOT_NAMES } from '../engine/protocol';
    import type { GameController } from '../game/controller.svelte';
    import { explorerPhoneSize } from '../game/layout';
    import { gameHref, graphSeries, names, pointValue, positionAt, stepLabel } from '../game/review';
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
    // Spec `2026-10-24-game-review-design.md`, "The page · Phone" (boards R4–R6):
    // one scrolling page; the eval bar and the step bar stay pinned to the bottom.
    let { c, r, nav }: { c: GameController; r: ReviewController; nav: Nav } = $props();
    let w = $state(window.innerWidth);
    const size = $derived(explorerPhoneSize(w));
    const strikeSize = $derived(Math.min(34, Math.floor((4 * size + 41 - 15) / 4)));
    // The graph card's inner width: the page column (≤ 430) less page and card padding.
    const graphW = $derived(Math.min(w, 430) - 48);
    const nm = $derived(names(r.shared));
    const at = $derived(r.game ? positionAt(r.game, r.an, r.sel, r.side) : null);
    const series = $derived(r.game ? graphSeries(r.game, r.an) : null);
    const pending = $derived(r.game && series && series.values.includes(null) ? `Analysing… ${r.an.filter(Boolean).length} of ${r.game.rolls.length} rolls` : null);
    const p = $derived(pointValue(r.an, r.sel));
    const step = $derived(stepLabel(r.sel, r.game?.rolls.length ?? 0));
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
    const LEFT = [{ label: 'First', d: 'M6 5v14M18 6l-7 6 7 6', go: () => r.first() }, { label: 'Previous', d: 'M15 6l-6 6 6 6', go: () => r.step(-1) }];
    const RIGHT = [{ label: 'Next', d: 'M9 6l6 6-6 6', go: () => r.step(1) }, { label: 'Last', d: 'M18 5v14M6 6l7 6-7 6', go: () => r.last() }];
</script>

<svelte:window bind:innerWidth={w} />
<div class="rphone">
    <header class="head">
        <div class="word">Qwixxer</div>
        <div class="right">
            {#if r.game}<button type="button" class="icon" aria-label="Share game" onclick={share}><Icon name="link" size={20} /></button>{/if}
            <Menu {c} {nav} />
        </div>
    </header>
    {#if notice}<div class="notice" role="status">{notice}</div>{/if}
    {#if r.shared && r.game}<div class="pill">Shared game · {nm.You} vs {BOT_NAMES[r.game.bot]}</div>{/if}
    <ReviewSummary {r} compact />
    {#if r.game && series && at}
        <section class="card" aria-label="Win chance over the game">
            <div class="ghead"><h2>{nm.Your} win chance</h2><span class="key"><span class="dot"></span>{nm.your} mistakes</span></div>
            <EvalGraph values={series.values} luck={series.luckZ} luckMax={3} luckLabel="Dice luck · z" dots={series.dots} selected={r.sel} {pending}
                width={graphW} height={92} luckHeight={34} onSelect={(j) => r.select(j)} />
        </section>
        <SelectedRoll {r} compact />
        <div class="sheets">
            <ReviewSheet name={nm.You} prev={at.prev} pos={at.pos} who="you" compact {size} {strikeSize} />
            <ReviewSheet name={BOT_NAMES[r.game.bot]} prev={at.prev} pos={at.pos} who="opponent" compact {size} {strikeSize} />
        </div>
        <div class="log"><ReviewLog {r} /></div>
    {/if}
</div>
{#if r.game}
    <div class="pinned">
        <EvalBar {p} vertical={false} />
        <nav class="steps" aria-label="Step through the game">
            {#each LEFT as s}<button type="button" aria-label={s.label} onclick={s.go}><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d={s.d}></path></svg></button>{/each}
            <div class="where">
                <b>{step.title}</b>
                <!-- Below 380 px the long form doesn't fit between the buttons. -->
                {#if w < 380}<span>{step.sub.replace('the ', '')}{#if p !== null}{' · '}<b class="pct">{Math.round(p * 100)}%</b>{/if}</span>
                {:else}<span>{step.sub}{#if p !== null}{' · '}<b class="pct">{Math.round(p * 100)}%</b> {nm.you} win{/if}</span>{/if}
            </div>
            {#each RIGHT as s}<button type="button" aria-label={s.label} onclick={s.go}><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d={s.d}></path></svg></button>{/each}
        </nav>
    </div>
{/if}

<style>
    .rphone { max-width: 430px; margin: 0 auto; box-sizing: border-box; padding: 12px 12px 96px; display: flex; flex-direction: column; gap: 10px; }
    .head { display: flex; align-items: center; justify-content: space-between; height: 44px; }
    .word { font-family: var(--cond); font-weight: 800; font-size: 26px; letter-spacing: 0.5px; }
    .right { display: flex; align-items: center; gap: 8px; }
    .icon { width: 44px; height: 44px; border-radius: 12px; border: none; background: var(--paper); box-shadow: 0 1px 0 var(--edge); color: var(--ink); display: flex; align-items: center; justify-content: center; padding: 0; cursor: pointer; }
    .notice { font-size: 13px; color: var(--soft); }
    .pill { align-self: flex-start; height: 28px; padding: 0 12px; border-radius: 14px; display: inline-flex; align-items: center; font-size: 13px; font-weight: 600; background: var(--btn-muted-bg); color: var(--soft); }
    .card { display: flex; flex-direction: column; gap: 6px; background: var(--paper); border-radius: 14px; padding: 10px 12px; box-shadow: 0 1px 0 var(--edge); }
    .ghead { display: flex; align-items: baseline; gap: 8px; }
    h2 { margin: 0; font-family: var(--cond); font-size: 17px; font-weight: 700; }
    .key { margin-left: auto; display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--unlucky-ink); }
    .sheets { display: flex; gap: 8px; justify-content: center; }
    .log { display: flex; }
    .pinned { position: fixed; left: 0; right: 0; bottom: 0; z-index: 5; background: var(--paper); box-shadow: 0 -1px 0 var(--edge), 0 -8px 24px var(--shadow); }
    .steps { max-width: 430px; margin: 0 auto; box-sizing: border-box; height: 72px; padding: 10px 12px 14px; display: flex; align-items: center; gap: 6px; }
    .steps button { width: 48px; height: 48px; flex: none; border-radius: 12px; border: none; background: var(--ground); color: var(--ink); display: flex; align-items: center; justify-content: center; padding: 0; cursor: pointer; }
    .where { flex: 1; min-width: 0; display: flex; flex-direction: column; align-items: center; gap: 1px; text-align: center; }
    .where > b { font-family: var(--cond); font-weight: 700; font-size: 17px; }
    .where span { font-size: 12px; color: var(--muted); white-space: nowrap; }
    .pct { color: var(--ink); }
</style>
