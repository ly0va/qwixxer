<script lang="ts">
    import type { LockLook } from '../game/rules';
    let { look, color, size, label }: { look: LockLook; color: string; size: number; label: string } = $props();
    const stroke = $derived(look === 'locked_by' ? 'var(--paper)' : look === 'grey' || look === 'open' ? `var(--${color}-ink)` : 'var(--ink)');
</script>

<div class="lock {look}" role="img" aria-label={label} style="--s: {size}px">
    <svg viewBox="0 0 24 24" width={size * 0.47} height={size * 0.47} fill="none" stroke={stroke} stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="5" y="11" width="14" height="10" rx="2"></rect><path d="M8 11V8a4 4 0 0 1 8 0v3"></path>
    </svg>
</div>

<style>
    .lock { box-sizing: border-box; width: var(--s); height: var(--s); border-radius: 50%; display: flex; align-items: center; justify-content: center; background: var(--paper); flex-shrink: 0; }
    .grey { opacity: 0.35; }
    .locking { box-shadow: 0 0 0 3px var(--ink); }
    .locked_by { background: var(--ink); }
    .locked_other { box-shadow: inset 0 0 0 2px var(--ink); }
</style>
