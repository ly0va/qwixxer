<script lang="ts">
    import type { CellModel, RowModel } from '../game/rules';
    import Cell from './Cell.svelte';
    import LockCell from './LockCell.svelte';
    let { rows, size = 48, onTap }: { rows: RowModel[]; size?: number; onTap: (c: CellModel) => void } = $props();
</script>

<div class="cols">
    {#each rows as row (row.row)}
        <div class="band {row.band}" aria-label={row.label} style="--band: var(--{row.color}-band); --dark: var(--{row.color}-dark)">
            {#each row.cells as cell (cell.n)}<Cell {cell} color={row.color} {size} {onTap} />{/each}
            <span style="height: 3px"></span>
            <LockCell look={row.lock} color={row.color} {size} label="{row.label} lock" />
        </div>
    {/each}
</div>

<style>
    .cols { display: flex; gap: 6px; background: var(--paper); border-radius: 14px; padding: 6px; box-shadow: 0 1px 0 var(--edge), 0 8px 24px var(--shadow); }
    .band { display: flex; flex-direction: column; gap: 3px; padding: 4px; border-radius: 10px; background: var(--band); }
    .locking { background: repeating-linear-gradient(135deg, var(--band) 0 10px, var(--dark) 10px 20px); }
    .locked { background: var(--locked-band); }
</style>
