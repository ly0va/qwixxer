<script lang="ts">
    import type { Chip } from '../game/rules';
    import { NAME, ROWS } from '../game/rules';
    let { chip, size = 26 }: { chip: Chip; size?: number } = $props();
</script>

{#if chip.kind === 'num'}
    <span class="chip" role="img" aria-label="{NAME[chip.row]} {chip.n}" style="--s: {size}px; background: var(--{ROWS[chip.row]}-band); color: {chip.row === 1 ? 'var(--on-yellow)' : 'var(--on-band)'}">{chip.n}</span>
{:else if chip.kind === 'strike'}
    <span class="chip strike" role="img" aria-label="strike" style="--s: {size}px">
        <svg viewBox="0 0 24 24" width={size * 0.54} height={size * 0.54} fill="none" stroke="var(--strike-red)" stroke-width="3" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"></path></svg>
    </span>
{:else}
    <span class="chip lock" role="img" aria-label="locked the row" style="--s: {size}px">
        <svg viewBox="0 0 24 24" width={size * 0.54} height={size * 0.54} fill="none" stroke="var(--paper)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="5" y="11" width="14" height="10" rx="2"></rect><path d="M8 11V8a4 4 0 0 1 8 0v3"></path></svg>
    </span>
{/if}

<style>
    .chip { box-sizing: border-box; min-width: var(--s); height: var(--s); padding: 0 4px; border-radius: calc(var(--s) * 0.27); display: inline-flex; align-items: center; justify-content: center; font-family: var(--cond); font-weight: 700; font-size: calc(var(--s) * 0.6); }
    .strike { background: var(--paper); box-shadow: inset 0 0 0 2px var(--strike-red); }
    .lock { background: var(--ink); }
</style>
