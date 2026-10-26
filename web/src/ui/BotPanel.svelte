<script lang="ts">
    import { BOT_NAMES, type BotKind } from '../engine/protocol';
    import type { GameController } from '../game/controller.svelte';
    import { sheetModel } from '../game/rules';
    import SheetRows from './SheetRows.svelte';
    import StrikeBoxes from './StrikeBoxes.svelte';
    let { c, size = 25 }: { c: GameController; size?: number } = $props();

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
            {c.view?.bot.score ?? 0} pts
            <StrikeBoxes count={strikes} size={Math.round(size * 0.8)} label="Bot's" />
        </span>
    </div>
    {#if c.nextBot !== c.bot}<p class="note">{BOT_NAMES[c.nextBot]} plays from the next game (or start one from the ⋯ menu).</p>{/if}
    <SheetRows {rows} {size} />
</section>

<style>
    .panel { display: flex; flex-direction: column; gap: 6px; background: var(--paper); border-radius: 14px; padding: 12px; box-shadow: 0 1px 0 var(--edge); }
    .head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .pick { display: flex; align-items: center; gap: 8px; }
    label { font-family: var(--cond); font-size: 20px; font-weight: 700; }
    select { height: 32px; border-radius: 8px; border: 2px solid var(--ink); background: var(--paper); color: var(--ink); font-weight: 600; font-size: 14px; padding: 0 8px; }
    .meta { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--muted); }
    .note { margin: 0; font-size: 12px; color: var(--muted); }
</style>
