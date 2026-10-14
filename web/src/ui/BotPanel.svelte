<script lang="ts">
    import { BOT_NAMES, type BotKind } from '../engine/protocol';
    import type { GameController } from '../game/controller.svelte';
    import { sheetModel } from '../game/rules';
    import SheetRows from './SheetRows.svelte';
    let { c, size = 25, onNewGame }: { c: GameController; size?: number; onNewGame?: () => void } = $props();

    // New game abandons the current one, so mid-game it takes two clicks:
    // the first arms it ("Abandon game?") for 3 s, the second starts.
    let confirming = $state(false);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const midGame = $derived(!!c.view && c.view.phase !== 'over');
    function newGame() {
        if (c.busy) return;
        if (midGame && !confirming) {
            confirming = true;
            clearTimeout(timer);
            timer = setTimeout(() => (confirming = false), 3000);
            return;
        }
        confirming = false;
        clearTimeout(timer);
        onNewGame?.();
        c.newGame();
    }
    $effect(() => () => clearTimeout(timer));
    const rows = $derived(c.view ? sheetModel(c.view, 'bot', { sel: [], strikeSel: false, committing: false, fresh: c.fresh, hidden: c.hidden }) : []);
    const strikes = $derived((c.view?.bot.strikes ?? 0) - (c.hidden.includes('bot:strike') ? 1 : 0));
</script>

<section aria-label="Bot's sheet" class="panel">
    <div class="head">
        <div class="pick">
            <label for="bot-picker">Bot</label>
            <select id="bot-picker" value={c.nextBot} onchange={(e) => c.setBot((e.currentTarget as HTMLSelectElement).value as BotKind)}>
                {#each Object.entries(BOT_NAMES) as [id, name]}<option value={id}>{name}</option>{/each}
            </select>
        </div>
        <span class="meta">
            {#if c.thinking}<i>thinking…</i>{/if}
            {c.view?.bot.score ?? 0} pts · {strikes} strikes
        </span>
    </div>
    <div class="row">
        <p class="note">{#if c.nextBot !== c.bot}{BOT_NAMES[c.nextBot]} plays from the next game.{/if}</p>
        <button type="button" class="new" class:armed={confirming} aria-disabled={c.busy} onclick={newGame}>
            {confirming ? 'Abandon game?' : 'New game'}
        </button>
    </div>
    <SheetRows {rows} {size} />
</section>

<style>
    .panel { display: flex; flex-direction: column; gap: 6px; background: var(--paper); border-radius: 14px; padding: 12px; box-shadow: 0 1px 0 var(--edge); }
    .head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .pick { display: flex; align-items: center; gap: 8px; }
    label { font-family: var(--cond); font-size: 20px; font-weight: 700; }
    select { height: 32px; border-radius: 8px; border: 2px solid var(--ink); background: var(--paper); color: var(--ink); font-weight: 600; font-size: 14px; padding: 0 8px; }
    .meta { font-size: 13px; color: var(--muted); }
    .meta i { margin-right: 6px; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 8px; min-height: 32px; }
    .note { margin: 0; font-size: 12px; color: var(--muted); }
    .new { height: 32px; padding: 0 12px; border-radius: 8px; border: 2px solid var(--ink); background: var(--paper); color: var(--ink); font-weight: 600; font-size: 14px; cursor: pointer; flex-shrink: 0; }
    .new.armed { background: var(--ink); color: var(--ground); }
    .new[aria-disabled='true'] { opacity: 0.5; cursor: default; }
</style>
