<script lang="ts">
    import type { Luck } from '../engine/protocol';
    import { luckHistogram } from '../game/position';
    // The luck chip's panel: how lucky every possible roll would have been, on
    // the chip's own axis (points of win chance against the average roll),
    // this roll's bin solid, and where it ranks among the others.
    let { luck, stale = false }: { luck: Luck; stale?: boolean } = $props();
    const h = $derived(luckHistogram(luck));
    const x = (v: number) => `${((v - h.min) / (h.max - h.min)) * 100}%`;
    const width = $derived(`calc(${100 / (h.max - h.min)}% - 1px)`);
    const label = (t: number) => (t > 0 ? `+${t}` : t < 0 ? `−${-t}` : '0');
    const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}`;
</script>

<section class="dist" class:stale aria-label="Luck of every possible roll">
    <div class="head"><span class="title">Luck of every roll</span><span class="rank">{h.rank}</span></div>
    <div class="plot" role="img" aria-label="Histogram of every roll's luck, from {label(Math.round(h.min))} to {label(Math.round(h.max))} points; this roll {signed(h.thisLuck)}">
        {#each h.bars as b (b.lo)}
            <span class="bar {b.tone}" class:current={b.current} style:left={x(b.lo)} style:width style:height="{(b.share / h.tallest) * 100}%"></span>
        {/each}
    </div>
    <div class="axis" aria-hidden="true">
        {#each h.ticks as t (t)}<span style:left={x(t)}>{label(t)}</span>{/each}
    </div>
</section>

<style>
    .dist { display: flex; flex-direction: column; gap: 6px; padding: 10px 12px 6px; border-radius: 12px; background: var(--log-bg); transition: opacity 0.15s; }
    .stale { opacity: 0.6; }
    .head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
    .title { font-weight: 700; font-size: 14px; }
    .rank { font-size: 13px; color: var(--muted); white-space: nowrap; }
    .plot { position: relative; height: 96px; border-bottom: 1.5px solid var(--ink); }
    .bar { position: absolute; bottom: 0; border-radius: 2px 2px 0 0; }
    .bar.lucky { background: color-mix(in srgb, var(--lucky-ink) 35%, var(--paper)); }
    .bar.unlucky { background: color-mix(in srgb, var(--unlucky-ink) 35%, var(--paper)); }
    .bar.current { min-height: 2px; }
    .bar.lucky.current { background: var(--lucky-ink); }
    .bar.unlucky.current { background: var(--unlucky-ink); }
    .axis { position: relative; height: 16px; }
    .axis span { position: absolute; transform: translateX(-50%); font-size: 11px; font-weight: 600; color: var(--muted); }
</style>
