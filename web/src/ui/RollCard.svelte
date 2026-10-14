<script lang="ts">
    import type { ExplorerController } from '../game/explorer.svelte';
    import { closedRows } from '../game/position';
    import Die from './Die.svelte';
    let { x, compact = false }: { x: ExplorerController; compact?: boolean } = $props();
    const FACES = [['var(--die-face)', 'var(--die-pip)', 'White'], ['var(--die-face)', 'var(--die-pip)', 'White'],
        ['var(--red-band)', 'var(--on-band)', 'Red'], ['var(--yellow-band)', 'var(--on-yellow)', 'Yellow'],
        ['var(--green-band)', 'var(--on-band)', 'Green'], ['var(--blue-band)', 'var(--on-band)', 'Blue']];
    const size = $derived(compact ? 40 : 52);
    const closed = $derived(closedRows(x.pos));
    // Colour dice are ignored on the opponent's roll and for closed rows: greyed.
    const out = (i: number) => i >= 2 && (x.pos.roller === 'opponent' || closed[i - 2]);
    const d = $derived(x.pos.dice);
    const white = $derived(d[0] !== null && d[1] !== null ? String(d[0] + d[1]) : '–');
</script>

<section class="card" class:compact aria-label="Roll">
    <div class="top">
        {#if !compact}<h2>Roll</h2>{/if}
        <div class="seg" role="radiogroup" aria-label="Whose roll">
            <button type="button" role="radio" aria-checked={x.pos.roller === 'you'} onclick={() => x.setRoller('you')}>Your roll</button>
            <button type="button" role="radio" aria-checked={x.pos.roller === 'opponent'} onclick={() => x.setRoller('opponent')}>{compact ? 'Opponent’s' : 'Opponent’s roll'}</button>
        </div>
        <button type="button" class="tool" aria-label="Random roll" onclick={() => x.random()}>
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="4"></rect><circle cx="8.5" cy="8.5" r="1.2" fill="currentColor"></circle><circle cx="15.5" cy="15.5" r="1.2" fill="currentColor"></circle><circle cx="15.5" cy="8.5" r="1.2" fill="currentColor"></circle><circle cx="8.5" cy="15.5" r="1.2" fill="currentColor"></circle></svg>
            {#if !compact}Random{/if}
        </button>
        <button type="button" class="tool muted" aria-label="Clear dice" onclick={() => x.clearDice()}>{compact ? '✕' : 'Clear'}</button>
    </div>
    <div class="dice">
        {#if compact}<div class="sum" role="status" aria-label="White sum {white}"><span>Sum</span><b>{white}</b></div>{/if}
        {#each FACES as f, i}
            <div class="slot">
                <button type="button" class="diebtn" class:open={x.picker === i} aria-expanded={x.picker === i}
                    aria-label="{f[2]} die, {d[i] ?? 'not set'}. Choose a face" onclick={() => x.openPicker(i)}>
                    {#if d[i] === null}
                        <span class="unset" style="--s: {size}px; --c: {i < 2 ? 'var(--faint)' : f[0]}">?</span>
                    {:else}
                        <Die value={d[i] ?? 0} face={f[0]} pip={f[1]} {size} out={out(i)} label="{f[2]} die, {d[i]}" />
                    {/if}
                </button>
                {#if x.picker === i}
                    <div class="picker" role="group" aria-label="Set the {f[2].toLowerCase()} die">
                        {#each [1, 2, 3, 4, 5, 6] as v}
                            <button type="button" aria-label="Face {v}" onclick={() => x.pickFace(v)}>{v}</button>
                        {/each}
                    </div>
                {/if}
            </div>
        {/each}
        {#if !compact}<div class="wsum"><span>White sum</span><b>{white}</b></div>{/if}
    </div>
</section>

<style>
    .card { display: flex; flex-direction: column; gap: 10px; background: var(--paper); border-radius: 14px; padding: 12px 16px; box-shadow: 0 1px 0 var(--edge); }
    .card.compact { padding: 10px 12px; }
    .top { display: flex; align-items: center; gap: 10px; }
    h2 { margin: 0; font-family: var(--cond); font-size: 20px; font-weight: 700; }
    .seg { display: flex; gap: 2px; padding: 3px; border-radius: 10px; background: var(--btn-muted-bg); }
    .seg button { height: 32px; padding: 0 12px; border-radius: 7px; border: none; background: transparent; color: var(--muted); font-weight: 600; font-size: 14px; cursor: pointer; }
    .seg button[aria-checked='true'] { background: var(--paper); color: var(--ink); font-weight: 700; box-shadow: 0 1px 0 var(--edge); }
    .tool { height: 36px; min-width: 40px; padding: 0 12px; border-radius: 9px; border: none; background: var(--ground); color: var(--ink); display: flex; align-items: center; justify-content: center; gap: 6px; font-weight: 600; font-size: 14px; cursor: pointer; }
    .top .tool:first-of-type { margin-left: auto; }
    .tool.muted { color: var(--muted); }
    .dice { display: flex; align-items: center; gap: 10px; }
    .compact .dice { gap: 6px; }
    .slot { position: relative; }
    .diebtn { display: block; padding: 0; border: none; background: none; border-radius: 12px; cursor: pointer; }
    .diebtn.open { outline: 3px solid var(--ink); outline-offset: 2px; }
    .unset { box-sizing: border-box; width: var(--s); height: var(--s); border-radius: calc(var(--s) * 0.21); border: 2px dashed var(--c); display: flex; align-items: center; justify-content: center; font-family: var(--cond); font-weight: 700; font-size: 22px; color: var(--muted); background: var(--paper); }
    .picker { position: absolute; z-index: 3; top: calc(100% + 8px); left: 0; display: flex; gap: 4px; padding: 6px; border-radius: 12px; background: var(--paper); box-shadow: 0 0 0 1px var(--edge), 0 10px 30px var(--shadow); }
    .slot:nth-last-child(-n + 4) .picker { left: auto; right: 0; }
    .picker button { width: 44px; height: 44px; border-radius: 9px; border: none; background: var(--ground); color: var(--ink); font-family: var(--cond); font-weight: 700; font-size: 22px; cursor: pointer; }
    .wsum, .sum { display: flex; flex-direction: column; gap: 2px; margin-left: 6px; }
    .wsum span, .sum span { font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .wsum b { font-family: var(--cond); font-weight: 800; font-size: 30px; line-height: 1; }
    .sum { min-height: 40px; min-width: 56px; margin: 0; flex-direction: row; align-items: center; justify-content: center; gap: 5px; border-radius: 10px; background: var(--ground); }
    .sum b { font-family: var(--cond); font-weight: 800; font-size: 24px; line-height: 1; }
</style>
