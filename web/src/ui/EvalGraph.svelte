<script lang="ts" module>
    let ids = 0;
</script>

<script lang="ts">
    import { stepLabel } from '../game/review';
    // Spec "The page": your win chance at every point (2 per roll), the
    // running dice luck under it, the selected point, and a hit target per
    // point. `values`/`luck` are per point (null: not analysed yet).
    let { values, luck, dots, selected, pending, width, height, luckHeight, onSelect }: {
        values: (number | null)[]; luck: (number | null)[]; dots: number[]; selected: number; pending: string | null;
        width: number; height: number; luckHeight: number; onSelect: (j: number) => void;
    } = $props();
    const id = ++ids;
    const last = $derived(Math.max(1, values.length - 1));
    const X = (j: number) => (j / last) * width;
    const Y = (v: number) => height * (1 - v);
    const LMAX = 0.4;
    const LY = (v: number) => luckHeight / 2 - (Math.max(-LMAX, Math.min(LMAX, v)) / LMAX) * (luckHeight / 2);
    /** The analysed prefix: points up to the first missing one. */
    const firstMissing = $derived(values.findIndex((v) => v === null));
    const done = $derived(firstMissing < 0 ? values.length - 1 : firstMissing - 1);
    const area = $derived(done < 0 ? '' : values.slice(0, done + 1).reduce((d, v, j) => `${d} L${X(j).toFixed(1)},${Y(v!).toFixed(1)}`, `M0,${height}`) + ` L${X(done).toFixed(1)},${height} Z`);
    const luckLine = $derived(luck.slice(0, done + 1).reduce((d, v, j) => `${d} L${X(j).toFixed(1)},${LY(v ?? 0).toFixed(1)}`, `M0,${LY(0)}`));
    const luckArea = $derived(`${luckLine} L${X(Math.max(0, done)).toFixed(1)},${luckHeight / 2} Z`);
    const sv = $derived(values[selected] ?? null);
    const sx = $derived(X(selected));
    const sy = $derived(sv === null ? height / 2 : Y(sv));
    const n = $derived((values.length - 1) / 2);
    const label = (j: number) => {
        const s = stepLabel(j, n);
        return `${s.title}, ${s.sub}${values[j] === null ? ', not analysed yet' : `: ${Math.round(values[j]! * 100)}%`}`;
    };
</script>

<div class="wrap" style="width: {width}px">
    <div class="g main" style="height: {height}px">
        <svg {width} {height} viewBox="0 0 {width} {height}" aria-hidden="true">
            <path d={area} class="area"></path>
            <line x1="0" y1={height / 2} x2={width} y2={height / 2} class="mid"></line>
        </svg>
        {#if pending}<div class="pending" style="left: {X(Math.max(0, done))}px">{pending}</div>{/if}
        <span class="sel" style="left: {sx - 1}px"></span>
        {#each dots as j}
            {#if values[j] !== null}<span class="dot" style="left: {X(j) - 4}px; top: {Y(values[j]!) - 4}px"></span>{/if}
        {/each}
        <span class="ring" style="left: {sx - 6}px; top: {sy - 6}px"></span>
        <span class="tag" style="left: {sx + 9}px; top: {Math.max(2, sy - 30)}px">{sv === null ? '…' : `${Math.round(sv * 100)}%`}</span>
        {#each values as _, j}
            <button type="button" class="hit" aria-label={label(j)} aria-current={j === selected ? 'step' : undefined}
                style="left: {X(j) - width / (2 * last)}px; width: {width / last}px" onclick={() => onSelect(j)}></button>
        {/each}
    </div>
    {#if luckHeight > 0}
        <span class="lab">Dice luck</span>
        <div class="g luck" style="height: {luckHeight}px">
            <svg {width} height={luckHeight} viewBox="0 0 {width} {luckHeight}" aria-hidden="true">
                <defs>
                    <clipPath id="lk{id}u"><rect x="0" y="0" {width} height={luckHeight / 2}></rect></clipPath>
                    <clipPath id="lk{id}d"><rect x="0" y={luckHeight / 2} {width} height={luckHeight / 2}></rect></clipPath>
                </defs>
                <path d={luckArea} class="up" clip-path="url(#lk{id}u)"></path>
                <path d={luckArea} class="down" clip-path="url(#lk{id}d)"></path>
                <line x1="0" y1={luckHeight / 2} x2={width} y2={luckHeight / 2} class="zero"></line>
                <path d={luckLine} class="line"></path>
            </svg>
            <span class="sel" style="left: {sx - 1}px"></span>
        </div>
    {/if}
</div>

<style>
    .wrap { display: flex; flex-direction: column; gap: 6px; }
    .g { position: relative; }
    svg { display: block; }
    .main svg { border-radius: 6px; background: var(--btn-muted-bg); }
    .area { fill: var(--ink); opacity: 0.88; }
    .mid { stroke: var(--faint); stroke-width: 1; stroke-dasharray: 3 4; }
    .pending { position: absolute; top: 0; bottom: 0; right: 0; box-sizing: border-box; padding-top: 8px; display: flex; justify-content: center; font-size: 13px; font-weight: 600; color: var(--muted); pointer-events: none;
        background: repeating-linear-gradient(135deg, transparent 0 6px, color-mix(in srgb, var(--ink) 5%, transparent) 6px 12px); }
    .sel { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--lucky-ink); opacity: 0.9; pointer-events: none; }
    .dot { position: absolute; width: 8px; height: 8px; border-radius: 50%; background: var(--unlucky-ink); box-shadow: 0 0 0 2px var(--paper); pointer-events: none; }
    .ring { position: absolute; width: 12px; height: 12px; border-radius: 50%; background: var(--paper); box-shadow: 0 0 0 3px var(--lucky-ink); pointer-events: none; }
    .tag { position: absolute; padding: 1px 6px; border-radius: 6px; background: var(--paper); color: var(--ink); font-family: var(--cond); font-weight: 800; font-size: 15px; box-shadow: 0 1px 4px var(--shadow); pointer-events: none; }
    .hit { position: absolute; top: 0; bottom: 0; border: none; padding: 0; background: transparent; cursor: pointer; border-radius: 3px; }
    .hit:hover { background: color-mix(in srgb, var(--ink) 8%, transparent); }
    .hit:focus-visible { outline: 2px solid var(--ink); }
    .lab { font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .luck svg { border-radius: 6px; background: var(--log-bg); }
    .up { fill: var(--lucky-ink); opacity: 0.28; }
    .down { fill: var(--unlucky-ink); opacity: 0.28; }
    .zero { stroke: var(--faint); stroke-width: 1; }
    .line { fill: none; stroke: var(--soft); stroke-width: 1.6; stroke-linejoin: round; }
</style>
