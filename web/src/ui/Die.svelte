<script lang="ts">
    let { value, face, pip, out = false, idle = false, rolling = false, size = 56, label }: { value: number; face: string; pip: string; out?: boolean; idle?: boolean; rolling?: boolean; size?: number; label: string } = $props();
    const MAP: Record<number, number[]> = { 1: [4], 2: [0, 8], 3: [0, 4, 8], 4: [0, 2, 6, 8], 5: [0, 2, 4, 6, 8], 6: [0, 2, 3, 5, 6, 8] };
    const on = $derived(out || idle || !value ? [] : MAP[value]);
</script>

<div class="die" class:qx-roll={rolling && !out} class:out class:idle role="img" aria-label={label}
    style="--s: {size}px; background: {out ? 'var(--locked-band)' : face}">
    {#each [0, 1, 2, 3, 4, 5, 6, 7, 8] as i}
        <span style="background: {on.includes(i) ? pip : 'transparent'}"></span>
    {/each}
</div>

<style>
    .die {
        box-sizing: border-box; width: var(--s); height: var(--s); border-radius: calc(var(--s) * 0.21); padding: calc(var(--s) * 0.14);
        display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 3px;
        box-shadow: inset 0 -3px 0 rgba(0, 0, 0, 0.18), 0 2px 6px var(--shadow);
    }
    .die span { width: calc(var(--s) * 0.18); height: calc(var(--s) * 0.18); border-radius: 50%; place-self: center; }
    .out { opacity: 0.5; box-shadow: none; }
    .idle { opacity: 0.4; }
</style>
