<script lang="ts">
    import { luckText } from '../game/position';
    import { bestShown, explorerHref, rollLuck, signed, type Side } from '../game/review';
    import type { ReviewController } from '../game/review.svelte';
    import { chipsFor } from '../game/rules';
    import ChipView from './Chip.svelte';
    import Die from './Die.svelte';
    import LuckPill from './LuckPill.svelte';
    // Spec "The page": the selected roll — its luck, dice, and your choice
    // against the best one. The Played / Best rows pick which outcome the
    // sheets show; desktop adds the step buttons.
    let { r, compact = false }: { r: ReviewController; compact?: boolean } = $props();
    const FACES = [['var(--die-face)', 'var(--die-pip)', 'White'], ['var(--die-face)', 'var(--die-pip)', 'White'],
        ['var(--red-band)', 'var(--on-band)', 'Red'], ['var(--yellow-band)', 'var(--on-yellow)', 'Yellow'],
        ['var(--green-band)', 'var(--on-band)', 'Green'], ['var(--blue-band)', 'var(--on-band)', 'Blue']];
    const k = $derived(r.sel === 0 ? -1 : Math.floor((r.sel - 1) / 2));
    const roll = $derived(k >= 0 ? r.game?.rolls[k] : undefined);
    const a = $derived(k >= 0 ? r.an[k] : undefined);
    const luck = $derived.by(() => {
        const l = k >= 0 ? rollLuck(r.an, k) : null;
        return a && l !== null ? luckText({ this_roll: a.this_roll, average: a.this_roll - l }) : null;
    });
    const played = $derived(roll ? chipsFor(roll.record.you, roll.record.roller === 'you' && roll.record.you_strike) : []);
    const shown = $derived(r.game && k >= 0 ? bestShown(r.game, r.an, k) : null);
    const best = $derived(shown ? chipsFor(shown.marks, shown.strike) : null);
    const cost = $derived(!a || !shown ? '' : shown.grade === 'best' ? 'Best' : shown.grade === 'excellent' ? 'Excellent' : `${signed(-a.you_cost)} pts`);
    const side = $derived(r.sel % 2 === 0 ? r.side : null);
    const pick = (s: Side) => r.show(s);
    const STEPS = [
        { label: 'First', d: 'M6 5v14M18 6l-7 6 7 6', go: () => r.first() }, { label: 'Previous', d: 'M15 6l-6 6 6 6', go: () => r.step(-1) },
        { label: 'Next', d: 'M9 6l6 6-6 6', go: () => r.step(1) }, { label: 'Last', d: 'M18 5v14M6 6l7 6-7 6', go: () => r.last() },
    ];
</script>

<section class="strip" class:compact aria-label="Selected roll">
    <div class="roll">
        <div class="title">
            {#if roll}
                <b>Roll {k + 1}</b>
                <span class="whose">{roll.record.roller === 'you' ? 'your roll' : 'bot’s roll'}</span>
                <LuckPill {luck} size="sm" />
                {#if compact && r.game}<a class="open explore" href={explorerHref(r.game, k)}>Explore</a>{/if}
            {:else}
                <b>Start</b><span class="whose">before the first roll</span>
            {/if}
        </div>
        <!-- On desktop, at the start the dice and the switch keep their space (hidden), so stepping never moves the buttons. -->
        {#if roll?.before.dice}
            <div class="dice">
                {#each FACES as f, i}<Die value={roll.before.dice[i]} face={f[0]} pip={f[1]} size={compact ? 38 : 34} label="{f[2]} die, {roll.before.dice[i]}" />{/each}
            </div>
        {:else if !compact}
            <div class="dice ghost" aria-hidden="true">
                {#each FACES as f}<Die value={1} face={f[0]} pip={f[1]} size={compact ? 38 : 34} label="" />{/each}
            </div>
        {/if}
    </div>
    {#if !roll}
        {#if !compact}<div class="switch ghost" aria-hidden="true">
            <button type="button" tabindex="-1" disabled><span class="lab">Played</span></button>
            <button type="button" tabindex="-1" disabled><span class="lab">Best</span></button>
        </div>{/if}
    {:else}
        <div class="switch" role="radiogroup" aria-label="Show on the sheets">
            <button type="button" role="radio" aria-checked={side === 'played'} class:on={side === 'played'} onclick={() => pick('played')}>
                <span class="lab">Played</span>
                {#each played as ch}<ChipView chip={ch} size={24} />{/each}
                {#if played.length === 0}<span class="none">nothing</span>{/if}
                <b class="cost" class:best={shown && shown.grade !== 'cost'}>{cost}</b>
            </button>
            <button type="button" role="radio" aria-checked={side === 'best'} class:on={side === 'best'} onclick={() => pick('best')}>
                <span class="lab">Best</span>
                {#if best}
                    {#each best as ch}<ChipView chip={ch} size={24} />{/each}
                    {#if best.length === 0}<span class="none">nothing</span>{/if}
                {:else}<span class="none">working it out…</span>{/if}
            </button>
        </div>
    {/if}
    {#if !compact}
        <div class="nav">
            <div class="steps" role="group" aria-label="Step through the game">
                {#each STEPS as s}
                    <button type="button" aria-label={s.label} onclick={s.go}>
                        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d={s.d}></path></svg>
                    </button>
                {/each}
            </div>
            {#if roll && r.game}<a class="open" href={explorerHref(r.game, k)}>Open in explorer</a>
            {:else}<span class="open ghost" aria-hidden="true">Open in explorer</span>{/if}
        </div>
    {/if}
</section>

<style>
    .strip { display: flex; align-items: center; gap: 14px; background: var(--paper); border-radius: 14px; padding: 10px 14px; box-shadow: 0 1px 0 var(--edge); }
    .strip.compact { flex-direction: column; align-items: stretch; gap: 8px; padding: 10px 12px; }
    .roll { display: flex; flex-direction: column; gap: 6px; }
    .title { display: flex; align-items: center; gap: 8px; min-height: 26px; }
    .title b { font-family: var(--cond); font-weight: 700; font-size: 19px; white-space: nowrap; }
    .whose { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; color: var(--muted); }
    .dice { display: flex; gap: 5px; }
    .compact .dice { gap: 6px; }
    .switch { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 4px; padding-left: 14px; border-left: 1px solid var(--edge); }
    .compact .switch { padding-left: 0; border-left: none; }
    .switch button { display: flex; align-items: center; gap: 6px; height: 34px; padding: 0 10px; border-radius: 9px; border: none; background: var(--log-bg); color: var(--ink); cursor: pointer; text-align: left; }
    .switch button.on { background: var(--paper); box-shadow: inset 0 0 0 2px var(--ink); }
    .lab { width: 52px; flex: none; font-size: 13px; font-weight: 600; color: var(--muted); }
    .on .lab { font-weight: 700; color: var(--ink); }
    .none { font-size: 13px; color: var(--muted); }
    .cost { margin-left: auto; font-family: var(--cond); font-weight: 800; font-size: 18px; color: var(--unlucky-ink); }
    .cost.best { color: var(--lucky-ink); font-size: 15px; }
    .ghost { visibility: hidden; }
    /* Desktop: the three columns share the step column's height, so their tops and bottoms line up. */
    .strip:not(.compact) > :is(.roll, .switch, .nav) { height: 76px; box-sizing: border-box; justify-content: space-between; }
    .nav { margin-left: auto; display: flex; flex-direction: column; align-items: flex-end; gap: 6px; }
    .steps { display: flex; gap: 4px; }
    .steps button { width: 38px; height: 38px; border-radius: 9px; border: none; background: var(--ground); color: var(--ink); display: flex; align-items: center; justify-content: center; cursor: pointer; padding: 0; }
    .open { height: 32px; padding: 0 12px; border-radius: 9px; background: var(--ground); color: var(--ink); display: inline-flex; align-items: center; font-size: 13px; font-weight: 600; text-decoration: none; white-space: nowrap; }
    .open:hover { background: color-mix(in srgb, var(--ink) 10%, var(--ground)); }
    .open:focus-visible { outline: 2px solid var(--ink); outline-offset: 1px; }
    /* Phone: the explorer button ends the title row. */
    .explore { flex: none; margin-left: auto; height: 28px; padding: 0 10px; }
</style>
