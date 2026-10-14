<script lang="ts">
    import { BOT_NAMES, type MoveOut } from '../engine/protocol';
    import type { ExplorerController } from '../game/explorer.svelte';
    import { percent } from '../game/position';
    import { chipsFor } from '../game/rules';
    import ChipView from './Chip.svelte';
    let { x, compact = false }: { x: ExplorerController; compact?: boolean } = $props();
    const moves = $derived(x.answer?.moves ?? null);
    const shown = $derived(moves ? (x.showAll ? moves : moves.slice(0, 6)) : []);
    const noDice = $derived(x.pos.dice.some((d) => d === null));
    // Why a bot plays a move ranked below the top (spec "Moves"), behind an (i).
    const NOTE = {
        rolled_out: 'Master+ played this roll out many times and prefers this move.',
        rule: 'A fixed rule makes the bot play this move: a safe lock or a game-winning end.',
    };
    // The open tooltip (tap on phones; hover and focus show it too).
    let open = $state<number | null>(null);
    function outside(e: MouseEvent) {
        if (open !== null && !(e.target instanceof Element && e.target.closest('.info'))) open = null;
    }
</script>

<svelte:document onclick={outside} />

{#snippet tags(m: MoveOut, i: number)}
    {#each m.bots as b}<span class="tag" class:plus={b === 'master_plus'}>{BOT_NAMES[b]}</span>{/each}
    {#if m.note}
        <span class="info" class:open={open === i}>
            <button type="button" aria-label="Why this move: {NOTE[m.note]}" aria-expanded={open === i} onclick={() => (open = open === i ? null : i)}>
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"></circle><path d="M12 11v6M12 7.5v.5"></path></svg>
            </button>
            <span class="tip" role="tooltip">{NOTE[m.note]}</span>
        </span>
    {/if}
{/snippet}

<!-- A finished game shows its result (PositionCard) and no move list. -->
{#if !x.answer?.over}
<section class="card" class:compact class:stale={x.pending} aria-label="Moves">
    <div class="top">
        <h2>Best moves</h2>
        {#if moves}<span class="muted">{moves.length} legal · your chance to win after each</span>{/if}
    </div>
    {#if noDice}
        <div class="empty">
            <b>Set the dice to rank the moves</b>
            <span>Tap a die to choose its face, or press Random.</span>
        </div>
    {:else if moves}
        <ol>
            {#each shown as m, i}
                <li>
                    <span class="rank">{i + 1}</span>
                    <div class="what">
                        <div class="chips">
                            {#each chipsFor(m.marks, m.strike) as ch}<ChipView chip={ch} />{/each}
                            {#if m.strike}<span class="txt">Strike</span>{:else if m.pass}<span class="txt">Pass</span>{/if}
                            {#if !compact}{@render tags(m, i)}{/if}
                        </div>
                        {#if compact && m.bots.length}<div class="tags">{@render tags(m, i)}</div>{/if}
                    </div>
                    <div class="pct">
                        <b>{percent(m.p_win, 1)}</b>
                        <span class="bar"><span style="width: {m.p_win * 100}%"></span></span>
                    </div>
                </li>
            {/each}
        </ol>
        {#if moves.length > 6 && !x.showAll}
            <button type="button" class="all" onclick={() => (x.showAll = true)}>Show all {moves.length}</button>
        {/if}
    {/if}
</section>
{/if}

<style>
    .card { display: flex; flex-direction: column; gap: 8px; background: var(--paper); border-radius: 14px; padding: 12px 16px; box-shadow: 0 1px 0 var(--edge); transition: opacity 0.15s; }
    .card.compact { padding: 10px 12px; gap: 7px; }
    .stale { opacity: 0.6; }
    .top { display: flex; align-items: baseline; flex-wrap: wrap; column-gap: 10px; row-gap: 2px; }
    h2 { margin: 0; font-family: var(--cond); font-size: 20px; font-weight: 700; }
    .muted { font-size: 13px; color: var(--muted); }
    .empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; padding: 24px 16px; text-align: center; color: var(--muted); font-size: 14px; }
    .empty b { color: var(--soft); font-size: 16px; }
    ol { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
    li { position: relative; display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: 10px; background: var(--log-bg); }
    .rank { width: 18px; font-family: var(--cond); font-weight: 700; font-size: 17px; color: var(--muted); }
    .what { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 5px; }
    .chips, .tags { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; }
    .txt { font-size: 14px; font-weight: 600; }
    .tag { height: 22px; padding: 0 7px; border-radius: 11px; display: inline-flex; align-items: center; font-size: 12px; font-weight: 700; box-shadow: inset 0 0 0 1.5px var(--muted); color: var(--soft); box-sizing: border-box; }
    .tag.plus { background: var(--ink); color: var(--ground); box-shadow: none; }
    /* The tip spans its row (not the icon), so it never runs off the card. */
    .info { display: inline-flex; }
    .info button { width: 28px; height: 28px; margin: -3px 0; padding: 0; border: none; border-radius: 50%; background: none; color: var(--muted); display: flex; align-items: center; justify-content: center; cursor: pointer; }
    .tip { display: none; position: absolute; z-index: 4; bottom: calc(100% - 4px); left: 30px; right: 8px; padding: 8px 10px; border-radius: 10px; background: var(--ink); color: var(--ground); font-size: 13px; font-weight: 500; line-height: 1.35; box-shadow: 0 6px 20px var(--shadow); }
    .info:hover .tip, .info:focus-within .tip, .info.open .tip { display: block; }
    .pct { width: 84px; display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }
    .pct b { font-family: var(--cond); font-weight: 800; font-size: 22px; line-height: 1; font-variant-numeric: tabular-nums; }
    .bar { width: 100%; height: 5px; border-radius: 3px; background: var(--btn-muted-bg); display: flex; overflow: hidden; }
    .bar span { background: var(--faint); }
    .all { align-self: flex-start; height: 36px; padding: 0 4px; border: none; background: none; color: var(--soft); font-weight: 600; font-size: 14px; text-decoration: underline; text-underline-offset: 3px; cursor: pointer; }
</style>
