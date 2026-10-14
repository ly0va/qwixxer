<script lang="ts">
    import MarkX from './MarkX.svelte';
    // Spec "Editing · Strike boxes": tap box k → k+1 strikes, or k if it was the last taken.
    let { count, size = 32, label, onTap }: { count: number; size?: number; label: string; onTap: (k: number) => void } = $props();
</script>

<div class="boxes" role="group" aria-label="{label} strikes">
    {#each [0, 1, 2, 3] as k}
        <button type="button" class="box" class:faded={k >= count} style="--s: {size}px" aria-label="{label}, strike {k + 1}{k < count ? ', taken' : ''}" aria-pressed={k < count} onclick={() => onTap(k)}>
            {#if k < count}<MarkX />{/if}
        </button>
    {/each}
</div>

<style>
    .boxes { display: flex; gap: 5px; }
    .box { position: relative; box-sizing: border-box; width: var(--s); height: var(--s); border-radius: 7px; padding: 0; border: 2px solid var(--ink); background: var(--paper); cursor: pointer; }
    .box.faded { opacity: 0.45; }
</style>
