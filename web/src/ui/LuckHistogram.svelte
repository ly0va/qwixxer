<script lang="ts">
    import type { Luck } from '../engine/protocol';
    import { luckBarTip, luckHistogram } from '../game/position';
    // The luck chip's panel: how lucky every possible roll would have been, on
    // the chip's own axis (points of win chance against the average roll),
    // this roll's bin solid, and where it ranks among the others.
    let { luck, stale = false }: { luck: Luck; stale?: boolean } = $props();
    const title = 'Luck of this roll vs others';
    const h = $derived(luckHistogram(luck));
    const x = (v: number) => `${((v - h.min) / (h.max - h.min)) * 100}%`;
    const width = $derived(`calc(${100 / (h.max - h.min)}% - 1px)`);
    const label = (t: number) => (t > 0 ? `+${t}` : t < 0 ? `−${-t}` : '0');
    const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}`;

    // The bar under the pointer (hover, or a touch held on the plot) and its tooltip.
    let hover = $state<number | null>(null);
    function point(e: PointerEvent) {
        const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
        const v = h.min + ((e.clientX - r.left) / r.width) * (h.max - h.min);
        const i = h.bars.findIndex((b) => v >= b.lo && v < b.lo + 1);
        hover = i >= 0 && h.bars[i].share > 0 ? i : null;
    }
    // A new histogram can arrive under a resting pointer: an index past its bars is no bar.
    const tip = $derived.by(() => {
        const bar = hover === null ? undefined : h.bars[hover];
        return bar ? { bar, text: luckBarTip(bar) } : null;
    });
    // The tooltip sits at the plot's top beside its bar, on the side with more
    // room, so it never covers the bar or leaves the plot.
    const tipAt = $derived.by(() => {
        if (!tip) return { left: '0', shift: '0' };
        const right = (tip.bar.lo + 0.5 - h.min) / (h.max - h.min) < 0.5;
        return right ? { left: x(tip.bar.lo + 1), shift: '6px' } : { left: x(tip.bar.lo), shift: 'calc(-100% - 6px)' };
    });
</script>

<section class="dist" class:stale aria-label={title}>
    <div class="head"><span class="title">{title}</span><span class="rank">{h.rank}</span></div>
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="plot" role="img" onpointermove={point} onpointerdown={point} onpointerleave={() => (hover = null)} aria-label="Histogram of every roll's luck, from {label(Math.round(h.min))} to {label(Math.round(h.max))} points; this roll {signed(h.thisLuck)}">
        {#each h.bars as b (b.lo)}
            <span class="bar {b.tone}" class:current={b.current} class:hover={h.bars[hover ?? -1] === b} style:left={x(b.lo)} style:width style:height="{(b.share / h.tallest) * 100}%"></span>
        {/each}
        {#if tip?.text}
            <span class="tip" role="tooltip" style:left={tipAt.left} style:transform="translateX({tipAt.shift})">
                <b>{tip.text.share}</b><span>{tip.text.luck}</span>
            </span>
        {/if}
    </div>
    <div class="axis" aria-hidden="true">
        {#each h.ticks as t (t)}<span style:left={x(t)}>{label(t)}</span>{/each}
    </div>
</section>

<style>
    .dist { display: flex; flex-direction: column; gap: 6px; padding: 10px 12px 6px; border-radius: 12px; background: var(--log-bg); transition: opacity 0.15s; }
    .stale { opacity: 0.6; }
    .head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; column-gap: 12px; row-gap: 2px; }
    .title { font-weight: 700; font-size: 14px; }
    .rank { font-size: 13px; color: var(--muted); white-space: nowrap; }
    .plot { position: relative; height: 96px; border-bottom: 1.5px solid var(--ink); touch-action: pan-y; }
    .bar { position: absolute; bottom: 0; border-radius: 2px 2px 0 0; }
    .bar.lucky { background: color-mix(in srgb, var(--lucky-ink) 35%, var(--paper)); }
    .bar.unlucky { background: color-mix(in srgb, var(--unlucky-ink) 35%, var(--paper)); }
    .bar.current { min-height: 2px; }
    .bar.lucky.current { background: var(--lucky-ink); }
    .bar.unlucky.current { background: var(--unlucky-ink); }
    .bar.hover { filter: brightness(0.85); }
    .tip { position: absolute; top: 0; z-index: 4; display: flex; flex-direction: column; gap: 1px; width: max-content; padding: 6px 9px; border-radius: 9px; background: var(--ink); color: var(--ground); font-size: 12px; font-weight: 500; line-height: 1.3; box-shadow: 0 6px 20px var(--shadow); pointer-events: none; }
    .tip b { font-size: 13px; font-weight: 700; }
    .axis { position: relative; height: 16px; }
    .axis span { position: absolute; transform: translateX(-50%); font-size: 11px; font-weight: 600; color: var(--muted); }
</style>
