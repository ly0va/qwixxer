<script lang="ts">
    import type { GameController } from '../game/controller.svelte';
    import { ROWS } from '../game/rules';
    import Die from './Die.svelte';
    let { c, size = 56, compact = false }: { c: GameController; size?: number; compact?: boolean } = $props();
    const v = $derived(c.view);
    // Controller ruling: during the bot's think after your tumble, the rolled
    // dice (pendingDice) are shown while the view is still 'you_to_roll'.
    const idle = $derived(v?.phase === 'you_to_roll' && !c.rolling && !c.pendingDice);
    const values = $derived(c.rolling ? c.shownDice : (c.pendingDice ?? v?.dice));
    const faces = [['var(--die-face)', 'var(--die-pip)', 'White'], ['var(--die-face)', 'var(--die-pip)', 'White'],
        ['var(--red-band)', 'var(--on-band)', 'Red'], ['var(--yellow-band)', 'var(--on-yellow)', 'Yellow'],
        ['var(--green-band)', 'var(--on-band)', 'Green'], ['var(--blue-band)', 'var(--on-band)', 'Blue']];
    const canRoll = $derived(v?.phase === 'you_to_roll' && !c.busy);
    const rollLabel = $derived(c.rolling ? 'Rolling…' : canRoll ? 'Roll' : v?.phase === 'passive' || v?.phase === 'bot_to_roll' ? 'Bot rolled' : 'Rolled');
    const shown = $derived(c.pendingDice ?? v?.dice);
    const white = $derived(shown && !idle && !c.rolling ? String(shown[0] + shown[1]) : '–');
</script>

<section aria-label="Dice" class="tray">
    {#if !compact || canRoll}
        <button type="button" class="roll" class:primary={canRoll} aria-disabled={!canRoll} onclick={() => c.roll()} style="width: {compact ? 110 : 148}px">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="4"></rect><circle cx="8.5" cy="8.5" r="1.2" fill="currentColor"></circle><circle cx="15.5" cy="15.5" r="1.2" fill="currentColor"></circle><circle cx="15.5" cy="8.5" r="1.2" fill="currentColor"></circle><circle cx="8.5" cy="15.5" r="1.2" fill="currentColor"></circle></svg>
            <span>{compact ? 'Roll' : rollLabel}</span>
        </button>
    {:else}
        <div class="white-badge" role="status" aria-label="White sum {white}"><span>White</span><b>{white}</b></div>
    {/if}
    {#each faces as f, i}
        <Die value={values?.[i] ?? 0} face={f[0]} pip={f[1]} {size} rolling={c.rolling} idle={idle}
            out={i >= 2 && !!v?.locked[i - 2]} label={i >= 2 && v?.locked[i - 2] ? `${f[2]} die, out of the game` : values?.[i] ? `${f[2]} die, ${values[i]}` : `${f[2]} die, not rolled`} />
    {/each}
    {#if !compact}
        <div class="sum"><span>White sum</span><b>{white}</b></div>
    {/if}
</section>

<style>
    .tray { display: flex; align-items: center; gap: 14px; }
    .roll { min-height: 48px; padding: 0; border: none; border-radius: 12px; display: inline-flex; align-items: center; justify-content: center; gap: 8px; font-weight: 700; font-size: 17px; flex-shrink: 0; cursor: not-allowed; background: var(--btn-muted-bg); color: var(--btn-muted-fg); }
    .roll.primary { cursor: pointer; background: var(--ink); color: var(--ground); box-shadow: 0 0 0 4px var(--ground), 0 0 0 6px var(--ink); }
    .sum { margin-left: 8px; display: flex; flex-direction: column; gap: 2px; }
    .sum span, .white-badge span { font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .sum b { font-family: var(--cond); font-weight: 800; font-size: 34px; line-height: 1; }
    .white-badge { min-height: 44px; width: 110px; box-sizing: border-box; border-radius: 12px; background: var(--paper); box-shadow: 0 1px 0 var(--edge); display: flex; align-items: center; justify-content: center; gap: 6px; flex-shrink: 0; }
    .white-badge b { font-family: var(--cond); font-weight: 800; font-size: 26px; line-height: 1; }
</style>
