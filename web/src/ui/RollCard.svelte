<script lang="ts">
    import { luckKey, type ExplorerController } from '../game/explorer.svelte';
    import { closedRows, isOver, luckText } from '../game/position';
    import { chipsFor, predictedTurn } from '../game/rules';
    import ChipView from './Chip.svelte';
    import Die from './Die.svelte';
    let { x, compact = false }: { x: ExplorerController; compact?: boolean } = $props();
    const FACES = [['var(--die-face)', 'var(--die-pip)', 'White'], ['var(--die-face)', 'var(--die-pip)', 'White'],
        ['var(--red-band)', 'var(--on-band)', 'Red'], ['var(--yellow-band)', 'var(--on-yellow)', 'Yellow'],
        ['var(--green-band)', 'var(--on-band)', 'Green'], ['var(--blue-band)', 'var(--on-band)', 'Blue']];
    // Compact (phone) row: Sum 46 + 6 dice × 38 + 6 gaps × 5 = 304 px, within
    // the 312 px a 360 px phone leaves inside the card (2×12 page, 2×12 card).
    const size = $derived(compact ? 38 : 52);
    const gap = $derived(compact ? 5 : 10);
    const lead = $derived(compact ? 46 + gap : 0); // the Sum box before the dice
    const rowEnd = $derived(lead + 6 * size + 5 * gap);
    // The 1–6 picker (6 × 44 + 5 × 4 + 2 × 6 = 296 px) sits under its die,
    // centred but clamped to the dice row so it never leaves the card.
    const PICKER_W = 296;
    const pickerLeft = (i: number) => Math.max(0, Math.min(lead + i * (size + gap) + size / 2 - PICKER_W / 2, rowEnd - PICKER_W));
    const closed = $derived(closedRows(x.pos));
    // Colour dice are ignored for closed rows: greyed. (On the opponent's roll
    // they set the opponent's predicted turn, which ranks your moves.)
    const out = (i: number) => i >= 2 && closed[i - 2];
    const d = $derived(x.pos.dice);
    const white = $derived(d[0] !== null && d[1] !== null ? String(d[0] + d[1]) : '–');
    // Spec "UI" (roll luck): a chip after the roll toggle, there from the
    // moment all six dice are set. Random and Clear sit at the row's end, so
    // nothing moves when the number arrives.
    const luckShown = $derived(d.every((v) => v !== null) && !isOver(x.pos));
    const luck = $derived(x.luck ? luckText(x.luck.value) : null);
    const luckStale = $derived(!!x.luck && x.luck.key !== luckKey(x.pos));
    // A click outside the picker closes it; a die button toggles it itself.
    // (closest, not contains: a picked face may already have left the DOM.)
    function outside(e: MouseEvent) {
        if (x.picker !== null && !(e.target as Element | null)?.closest?.('.diebtn, .picker')) x.picker = null;
    }
</script>

<svelte:document onclick={outside} />

<section class="card" class:compact aria-label="Roll">
    <div class="top">
        {#if !compact}<h2>Roll</h2>{/if}
        <div class="seg" role="radiogroup" aria-label="Whose roll">
            <button type="button" role="radio" aria-checked={x.pos.roller === 'you'} aria-label="Your roll" onclick={() => x.setRoller('you')}>{compact ? 'You' : 'Your roll'}</button>
            <button type="button" role="radio" aria-checked={x.pos.roller === 'opponent'} aria-label="Opponent’s roll" onclick={() => x.setRoller('opponent')}>{compact ? 'Them' : 'Opponent’s roll'}</button>
        </div>
        {#if luckShown}
            <span class="luck {luck?.tone ?? 'pending'}" class:stale={luckStale} aria-live="polite" title={luck?.detail}>
                {#if luck}
                    {luck.word}{#if luck.value}<b aria-label="{luck.value} points">{luck.value}</b>{/if}
                    <span class="sr">{luck.detail}</span>
                {:else}
                    Luck …
                {/if}
            </span>
        {/if}
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
                    <div class="picker" role="group" style="left: {pickerLeft(i)}px" aria-label="Set the {f[2].toLowerCase()} die">
                        {#each [1, 2, 3, 4, 5, 6] as v}
                            <button type="button" aria-label="Face {v}" onclick={() => x.pickFace(v)}>{v}</button>
                        {/each}
                    </div>
                {/if}
            </div>
        {/each}
        {#if !compact}<div class="wsum"><span>White sum</span><b>{white}</b></div>{/if}
    </div>
    <!-- Spec "Moves": on your roll the ranking assumes the opponent's predicted white-sum mark. -->
    {#if x.pos.roller === 'you' && x.answer?.opponent_white}
        {@const ow = x.answer.opponent_white}
        <p class="opp" class:stale={x.pending}>
            {#if ow.mark}Opponent likely marks {#each chipsFor([ow.mark], false) as ch}<ChipView chip={ch} size={22} />{/each}{:else}Opponent likely passes{/if}
        </p>
    {/if}
    <!-- On their roll the ranking plays out their predicted turn (Master's lookahead); shown after your top move. -->
    {#if x.pos.roller === 'opponent' && x.answer?.opponent_turn}
        {@const t = predictedTurn(x.answer.opponent_turn)}
        <p class="opp" class:stale={x.pending}>
            Opponent likely {t.verb} {#each t.chips as ch}<ChipView chip={ch} size={22} />{/each}<span class="after">after your top move</span>
        </p>
    {/if}
</section>

<style>
    .luck { flex: none; height: 28px; padding: 0 11px; border-radius: 14px; display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; font-weight: 700; font-size: 14px; transition: opacity 0.15s; }
    .luck b { font-family: var(--cond); font-weight: 800; font-size: 18px; }
    .luck.lucky { color: var(--lucky-ink); background: color-mix(in srgb, var(--lucky-ink) 14%, var(--paper)); }
    .luck.unlucky { color: var(--unlucky-ink); background: color-mix(in srgb, var(--unlucky-ink) 14%, var(--paper)); }
    .luck.average { color: var(--ink); background: var(--btn-muted-bg); }
    .luck.pending { color: var(--muted); font-weight: 600; background: var(--btn-muted-bg); }
    .luck.stale { opacity: 0.6; }
    .sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
    .compact .luck { height: 26px; padding: 0 9px; font-size: 13px; gap: 5px; }
    .compact .luck b { font-size: 17px; }
    .opp { margin: 0; display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--soft); min-height: 22px; }
    .opp.stale { opacity: 0.6; }
    .opp .after { color: var(--muted); }
    .card { display: flex; flex-direction: column; gap: 10px; background: var(--paper); border-radius: 14px; padding: 12px 16px; box-shadow: 0 1px 0 var(--edge); }
    .card.compact { padding: 10px 12px; }
    .top { display: flex; align-items: center; gap: 10px; }
    h2 { margin: 0; font-family: var(--cond); font-size: 20px; font-weight: 700; }
    .seg { display: flex; gap: 2px; padding: 3px; border-radius: 10px; background: var(--btn-muted-bg); }
    .seg button { height: 32px; padding: 0 12px; border-radius: 7px; border: none; background: transparent; color: var(--muted); font-weight: 600; font-size: 14px; cursor: pointer; white-space: nowrap; }
    .seg button[aria-checked='true'] { background: var(--paper); color: var(--ink); font-weight: 700; box-shadow: 0 1px 0 var(--edge); }
    .tool { height: 36px; min-width: 40px; padding: 0 12px; border-radius: 9px; border: none; background: var(--ground); color: var(--ink); display: flex; align-items: center; justify-content: center; gap: 6px; font-weight: 600; font-size: 14px; cursor: pointer; }
    .top .tool:first-of-type { margin-left: auto; }
    .tool.muted { color: var(--muted); }
    .compact .top { gap: 6px; }
    .compact .seg button { padding: 0 8px; font-size: 13px; }
    .compact .tool { min-width: 36px; padding: 0 8px; }
    /* The picker is positioned against the dice row (see pickerLeft). */
    .dice { position: relative; display: flex; align-items: center; gap: 10px; }
    .compact .dice { gap: 5px; }
    .diebtn { display: block; padding: 0; border: none; background: none; border-radius: 12px; cursor: pointer; }
    .diebtn.open { outline: 3px solid var(--ink); outline-offset: 2px; }
    .unset { box-sizing: border-box; width: var(--s); height: var(--s); border-radius: calc(var(--s) * 0.21); border: 2px dashed var(--c); display: flex; align-items: center; justify-content: center; font-family: var(--cond); font-weight: 700; font-size: 22px; color: var(--muted); background: var(--paper); }
    .picker { position: absolute; z-index: 3; top: calc(100% + 8px); display: flex; gap: 4px; padding: 6px; border-radius: 12px; background: var(--paper); box-shadow: 0 0 0 1px var(--edge), 0 10px 30px var(--shadow); }
    .picker button { width: 44px; height: 44px; border-radius: 9px; border: none; background: var(--ground); color: var(--ink); font-family: var(--cond); font-weight: 700; font-size: 22px; cursor: pointer; }
    .wsum, .sum { display: flex; flex-direction: column; gap: 2px; margin-left: 6px; }
    .wsum span, .sum span { font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .wsum b { font-family: var(--cond); font-weight: 800; font-size: 30px; line-height: 1; }
    .sum { box-sizing: border-box; flex: none; width: 46px; height: 38px; margin: 0; align-items: center; justify-content: center; gap: 1px; border-radius: 10px; background: var(--ground); }
    .sum span { font-size: 10px; }
    .sum b { font-family: var(--cond); font-weight: 800; font-size: 20px; line-height: 1; }
</style>
