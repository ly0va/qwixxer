<script lang="ts">
    import type { Snippet } from 'svelte';
    import type { CellModel, RowModel } from '../game/rules';
    import Cell from './Cell.svelte';
    import LockCell from './LockCell.svelte';
    // `after`: something to show at the end of each row (the explorer's expected marks).
    let { rows, size, onTap, onLock, after }: { rows: RowModel[]; size: number; onTap?: (c: CellModel) => void; onLock?: (row: number) => void; after?: Snippet<[number]> } = $props();
    const gap = $derived(size >= 40 ? 6 : 3);
</script>

<div class="rows" style="gap: {gap}px">
    {#snippet band(row: RowModel)}
        <div class="band {row.band}" aria-label={row.label} style="--band: var(--{row.color}-band); --dark: var(--{row.color}-dark); gap: {gap}px; padding: {gap}px; border-radius: {size >= 40 ? 10 : 7}px">
            {#each row.cells as cell (cell.n)}
                <Cell {cell} color={row.color} {size} {onTap} />
            {/each}
            <span style="width: {size >= 40 ? 4 : 2}px"></span>
            <LockCell look={row.lock} color={row.color} {size} label="{row.label} lock" onTap={onLock ? () => onLock(row.row) : undefined} />
        </div>
    {/snippet}
    {#each rows as row (row.row)}
        {#if after}<div class="line">{@render band(row)}{@render after(row.row)}</div>{:else}{@render band(row)}{/if}
    {/each}
</div>

<style>
    .rows { display: flex; flex-direction: column; }
    .line { display: flex; align-items: center; gap: 6px; }
    .band { display: flex; align-items: center; width: fit-content; background: var(--band); }
    .locking { background: repeating-linear-gradient(135deg, var(--band) 0 10px, var(--dark) 10px 20px); }
    .locked { background: var(--locked-band); }
</style>
