<script lang="ts" module>
    let ids = 0;
</script>

<script lang="ts">
    import { monotone } from '../game/curve';
    import { stepLabel, type GraphModel } from '../game/review';
    // Spec "The graph": one view of the game, a smooth curve through the
    // model's knots, the selected point with its value, and a hit target per
    // point. Points past the analysis are hatched.
    let { m, selected, pending, width, height, onSelect }: {
        m: GraphModel; selected: number; pending: string | null; width: number; height: number; onSelect: (j: number) => void;
    } = $props();
    const id = ++ids;
    const last = $derived(Math.max(1, m.values.length - 1));
    const X = (j: number) => (j / last) * width;
    const Y = (v: number) => height * (m.hi - Math.max(m.lo, Math.min(m.hi, v))) / (m.hi - m.lo);
    const done = $derived(m.knots.length ? m.knots[m.knots.length - 1] : -1);
    const curve = $derived(monotone(m.knots.map(X), m.knots.map((j) => Y(m.values[j]!))));
    const base = $derived(m.area ? height : Y(0));
    const fill = $derived(done < 0 ? '' : `${curve.d} L${X(done).toFixed(1)},${base} L0,${base} Z`);
    const sv = $derived(m.values[selected] ?? null);
    const sx = $derived(X(selected));
    const sy = $derived(sv === null || selected > done ? height / 2 : curve.at(sx));
    const tagLeft = $derived(sx + 9 + 64 > width ? sx - 9 - 64 : sx + 9);
    const n = $derived((m.values.length - 1) / 2);
    const label = (j: number) => {
        const s = stepLabel(j, n);
        return `${s.title}, ${s.sub}${m.values[j] === null ? ', not analysed yet' : `: ${m.tag(m.values[j]!)}`}`;
    };
</script>

<div class="g" class:area={m.area} style="width: {width}px; height: {height}px">
    <svg {width} {height} viewBox="0 0 {width} {height}" aria-hidden="true">
        <defs>
            <clipPath id="g{id}u"><rect x="0" y="0" {width} height={base}></rect></clipPath>
            <clipPath id="g{id}d"><rect x="0" y={base} {width} height={Math.max(0, height - base)}></rect></clipPath>
        </defs>
        {#each m.bands as b}<rect x="0" y={Y(b.to)} {width} height={Y(b.from) - Y(b.to)} class="band"></rect>{/each}
        {#if m.area}
            <path d={fill} class="fill"></path>
        {:else}
            <path d={fill} class="up" clip-path="url(#g{id}u)"></path>
            <path d={fill} class="down" clip-path="url(#g{id}d)"></path>
        {/if}
        {#each m.guides as gd}<line x1="0" y1={Y(gd.v)} x2={width} y2={Y(gd.v)} class="guide" class:dash={gd.dash}></line>{/each}
        {#if !m.area}<path d={curve.d} class="line"></path>{/if}
    </svg>
    {#each m.labels as lb}<span class="lab" class:right={lb.right} style="top: {Math.max(1, Math.min(height - 17, Y(lb.v) - 8))}px">{lb.text}</span>{/each}
    {#if pending}<div class="pending" style="left: {X(Math.max(0, done))}px">{pending}</div>{/if}
    <span class="sel" style="left: {sx - 1}px"></span>
    {#each m.dots as d}<span class="dot {d.tone}" style="left: {X(d.j) - 4}px; top: {curve.at(X(d.j)) - 4}px"></span>{/each}
    <span class="ring" style="left: {sx - 6}px; top: {sy - 6}px"></span>
    <span class="tag" style="left: {tagLeft}px; top: {Math.max(2, Math.min(height - 24, sy - 30))}px">{sv === null || selected > done ? '…' : m.tag(sv)}</span>
    {#each m.values as _, j}
        <button type="button" class="hit" aria-label={label(j)} aria-current={j === selected ? 'step' : undefined}
            style="left: {X(j) - width / (2 * last)}px; width: {width / last}px" onclick={() => onSelect(j)}></button>
    {/each}
</div>

<style>
    .g { position: relative; }
    svg { display: block; border-radius: 6px; background: var(--log-bg); }
    .area svg { background: var(--btn-muted-bg); }
    .fill { fill: var(--ink); opacity: 0.88; }
    .up { fill: var(--lucky-ink); opacity: 0.28; }
    .down { fill: var(--unlucky-ink); opacity: 0.28; }
    .band { fill: color-mix(in srgb, var(--ink) 4%, transparent); }
    .guide { stroke: var(--faint); stroke-width: 1; }
    .guide.dash { stroke-dasharray: 3 4; }
    .line { fill: none; stroke: var(--soft); stroke-width: 1.6; stroke-linejoin: round; }
    .lab { position: absolute; left: 6px; font-size: 11px; line-height: 16px; font-weight: 600; color: var(--muted); pointer-events: none; }
    .lab.right { left: auto; right: 6px; }
    .pending { position: absolute; top: 0; bottom: 0; right: 0; box-sizing: border-box; padding-top: 8px; display: flex; justify-content: center; font-size: 13px; font-weight: 600; color: var(--muted); pointer-events: none;
        background: repeating-linear-gradient(135deg, transparent 0 6px, color-mix(in srgb, var(--ink) 5%, transparent) 6px 12px); }
    .sel { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--lucky-ink); opacity: 0.9; pointer-events: none; }
    .dot { position: absolute; width: 8px; height: 8px; border-radius: 50%; box-shadow: 0 0 0 2px var(--paper); pointer-events: none; }
    .dot.you { background: var(--unlucky-ink); }
    .dot.bot { background: var(--soft); }
    .ring { position: absolute; width: 12px; height: 12px; border-radius: 50%; background: var(--paper); box-shadow: 0 0 0 3px var(--lucky-ink); pointer-events: none; }
    .tag { position: absolute; padding: 1px 6px; border-radius: 6px; background: var(--paper); color: var(--ink); font-family: var(--cond); font-weight: 800; font-size: 15px; white-space: nowrap; box-shadow: 0 1px 4px var(--shadow); pointer-events: none; }
    .hit { position: absolute; top: 0; bottom: 0; border: none; padding: 0; background: transparent; cursor: pointer; border-radius: 3px; }
    .hit:hover { background: color-mix(in srgb, var(--ink) 8%, transparent); }
    .hit:focus-visible { outline: 2px solid var(--ink); }
</style>
