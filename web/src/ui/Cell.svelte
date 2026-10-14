<script lang="ts">
    import type { CellModel } from '../game/rules';
    import MarkX from './MarkX.svelte';
    let { cell, color, size, onTap }: { cell: CellModel; color: string; size: number; onTap?: (c: CellModel) => void } = $props();
</script>

{#if onTap}
    <button type="button" class="cell look-{cell.look}" style="--c: var(--{color}-ink); --s: {size}px"
        aria-label={cell.label} aria-pressed={cell.mark === 'pencil' || (cell.clickable && cell.mark === 'ink')} aria-disabled={!cell.clickable && cell.mark !== 'pencil'}
        onclick={() => onTap(cell)}>
        <span>{cell.n}</span>
        {#if cell.mark !== 'none'}<MarkX pencil={cell.mark === 'pencil'} anim={cell.anim} />{/if}
    </button>
{:else}
    <div class="cell look-{cell.look}" role="img" aria-label={cell.label} style="--c: var(--{color}-ink); --s: {size}px">
        <span>{cell.n}</span>
        {#if cell.mark !== 'none'}<MarkX pencil={cell.mark === 'pencil'} anim={cell.anim} />{/if}
    </div>
{/if}

<style>
    .cell {
        position: relative; box-sizing: border-box; width: var(--s); height: var(--s); border: none; padding: 0;
        border-radius: calc(var(--s) * 0.15); display: flex; align-items: center; justify-content: center;
        background: var(--paper); color: var(--c); font-family: var(--cond); font-weight: 700; font-size: calc(var(--s) * 0.5);
    }
    button.cell { cursor: default; }
    .look-legal { font-weight: 800; cursor: pointer !important; }
    .look-soft { font-weight: 500; color: color-mix(in srgb, var(--c) var(--soft-mix), transparent); }
    .look-skipped { opacity: 0.35; }
    .look-out { opacity: 0.3; color: var(--muted); }
    [aria-pressed='true'] { cursor: pointer !important; }
</style>
