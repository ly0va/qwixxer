<script lang="ts">
    let { pencil = false, anim = '' }: { pencil?: boolean; anim?: string } = $props();
</script>

<svg viewBox="0 0 40 40" class="x {anim ? `anim-${anim}` : ''}" aria-hidden="true">
    <path d="M10 9 C 17 16, 24 24, 31 32 M31 9 C 24 16, 17 24, 9 31" stroke={pencil ? 'var(--pencil)' : 'var(--ink)'} stroke-width={pencil ? 3 : 4} stroke-linecap="round" fill="none" />
    {#if pencil}
        <path d="M11 11 C 18 17, 23 24, 30 30 M29 11 C 23 17, 18 23, 11 29" stroke="var(--pencil)" stroke-width="1.2" stroke-linecap="round" fill="none" opacity="0.6" />
    {/if}
</svg>

<style>
    .x { position: absolute; inset: 3px; width: calc(100% - 6px); height: calc(100% - 6px); }
</style>
