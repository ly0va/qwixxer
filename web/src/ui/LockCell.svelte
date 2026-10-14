<script lang="ts">
    import type { LockLook } from '../game/rules';
    let { look, color, size, label }: { look: LockLook; color: string; size: number; label: string } = $props();
    const stroke = $derived(look === 'locked_by' ? 'var(--paper)' : look === 'locked_other' ? 'var(--muted)' : look === 'grey' || look === 'open' || look === 'pencil' ? `var(--${color}-ink)` : 'var(--ink)');
</script>

<!-- Rings are drawn inside the circle (border), on the paper colour: an
     outer ring over the hatched "locking" band looked uneven where it met
     the dark and light stripes. -->
<div class="lock {look}" role="img" aria-label={label} style="--s: {size}px; --ring: {size >= 40 ? 3 : 2}px">
    <svg viewBox="0 0 24 24" width={size * 0.47} height={size * 0.47} fill="none" stroke={stroke} stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="5" y="11" width="14" height="10" rx="2"></rect><path d="M8 11V8a4 4 0 0 1 8 0v3"></path>
    </svg>
</div>

<style>
    .lock { box-sizing: border-box; width: var(--s); height: var(--s); border-radius: 50%; display: flex; align-items: center; justify-content: center; background: var(--paper); flex-shrink: 0; }
    .grey { opacity: 0.35; }
    .locking { border: var(--ring) solid var(--ink); }
    /* The "locking" band's diagonal hatch, in pencil grey, scaled to the circle,
       on the rim only: a paper disc behind the icon keeps the row colour clean. */
    .pencil { --w: calc(var(--s) / 9); background:
        radial-gradient(circle, var(--paper) calc(var(--s) * 0.3), transparent calc(var(--s) * 0.3 + 0.5px)),
        repeating-linear-gradient(135deg, color-mix(in srgb, var(--pencil) 45%, var(--paper)) 0 var(--w), var(--paper) var(--w) calc(2 * var(--w))); }
    .locked_by { background: var(--ink); }
    .locked_other { opacity: 0.3; }
</style>
