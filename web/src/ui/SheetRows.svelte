<script lang="ts">
    import type { CellModel, RowModel } from '../game/rules';
    import Cell from './Cell.svelte';
    import LockCell from './LockCell.svelte';
    let { rows, size, onTap, onLock }: { rows: RowModel[]; size: number; onTap?: (c: CellModel) => void; onLock?: (row: number) => void } = $props();
    const gap = $derived(size >= 40 ? 6 : 3);
</script>

<div class="rows" style="gap: {gap}px">
    {#each rows as row (row.row)}
        <div class="band {row.band}" aria-label={row.label} style="--band: var(--{row.color}-band); --dark: var(--{row.color}-dark); gap: {gap}px; padding: {gap}px; border-radius: {size >= 40 ? 10 : 7}px">
            {#each row.cells as cell (cell.n)}
                <Cell {cell} color={row.color} {size} {onTap} />
            {/each}
            <span style="width: {size >= 40 ? 4 : 2}px"></span>
            <LockCell look={row.lock} color={row.color} {size} label="{row.label} lock" onTap={onLock ? () => onLock(row.row) : undefined} />
        </div>
    {/each}
</div>

<style>
    .rows { display: flex; flex-direction: column; }
    .band { display: flex; align-items: center; width: fit-content; background: var(--band); }
    .locking { background: repeating-linear-gradient(135deg, var(--band) 0 10px, var(--dark) 10px 20px); }
    .locked { background: var(--locked-band); }
</style>
