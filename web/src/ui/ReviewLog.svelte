<script lang="ts">
    import { reviewLog } from '../game/review';
    import type { ReviewController } from '../game/review.svelte';
    import ChipView from './Chip.svelte';
    // Spec "The page": the game log, one entry per roll with its luck; your
    // choices show their cost. Clicking an entry selects it (after the move).
    let { r }: { r: ReviewController } = $props();
    const entries = $derived(r.game ? reviewLog(r.game, r.an) : []);
    const k = $derived(r.sel === 0 ? -1 : Math.floor((r.sel - 1) / 2));
    let list = $state<HTMLOListElement>();
    // Keep the selected entry in view by scrolling the list only (scrollIntoView
    // would scroll the whole page on the phone).
    $effect(() => {
        const el = list?.querySelector<HTMLElement>(`[data-k="${k}"]`);
        if (!list || !el) return;
        const top = el.offsetTop;
        if (top < list.scrollTop) list.scrollTop = top;
        else if (top + el.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = top + el.offsetHeight - list.clientHeight;
    });
    const tone = (luck: string) => (luck.includes('…') || luck === 'luck 0.0' ? 'pending' : luck.includes('+') ? 'lucky' : 'unlucky');
</script>

<section class="log" aria-label="Moves">
    <h2>Moves</h2>
    <ol bind:this={list}>
        {#each entries as e (e.k)}
            <li data-k={e.k}>
                <button type="button" class:sel={e.k === k} aria-current={e.k === k ? 'step' : undefined} onclick={() => r.select(2 * e.k + 2)}>
                    <span class="n">{e.n}</span>
                    <span class="lines">
                        {#each e.lines as line}
                            <span class="line">
                                <span class="who" class:you={line.you}>{line.who}</span>
                                {#each line.chips as ch}<ChipView chip={ch} size={24} />{/each}
                                {#if line.chips.length === 0}<span class="empty">nothing</span>{/if}
                                {#if line.cost}<b class="cost" class:you={line.you}>{line.cost}</b>{/if}
                            </span>
                        {/each}
                    </span>
                    <span class="pill {tone(e.luck)}">{e.luck}</span>
                </button>
            </li>
        {/each}
    </ol>
</section>

<style>
    .log { flex: 1; min-height: 0; display: flex; flex-direction: column; gap: 6px; background: var(--paper); border-radius: 14px; padding: 12px 16px; box-shadow: 0 1px 0 var(--edge); }
    h2 { margin: 0; font-family: var(--cond); font-size: 18px; font-weight: 700; }
    ol { position: relative; list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; overflow-y: auto; min-height: 0; scrollbar-width: thin; scrollbar-color: var(--faint) transparent; }
    button { width: 100%; display: flex; align-items: flex-start; gap: 8px; padding: 5px 10px; border-radius: 10px; border: none; background: var(--log-bg); color: var(--ink); cursor: pointer; text-align: left; font: inherit; }
    button.sel { background: var(--paper); box-shadow: inset 0 0 0 2px var(--ink); }
    .n { width: 22px; flex: none; font-family: var(--cond); font-weight: 700; font-size: 16px; color: var(--muted); padding-top: 2px; }
    .lines { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 3px; }
    .line { display: flex; align-items: center; gap: 5px; min-height: 24px; }
    .who { width: 46px; flex: none; font-size: 13px; font-weight: 700; color: var(--muted); }
    .who.you { color: var(--ink); }
    .empty { font-size: 13px; color: var(--faint); }
    .cost { margin-left: auto; font-family: var(--cond); font-weight: 800; font-size: 16px; color: var(--soft); }
    .cost.you { color: var(--unlucky-ink); }
    .pill { flex: none; margin-top: 3px; height: 20px; padding: 0 7px; border-radius: 10px; display: inline-flex; align-items: center; font-size: 12px; font-weight: 700; }
    .pill.lucky { color: var(--lucky-ink); background: color-mix(in srgb, var(--lucky-ink) 14%, var(--paper)); }
    .pill.unlucky { color: var(--unlucky-ink); background: color-mix(in srgb, var(--unlucky-ink) 14%, var(--paper)); }
    .pill.pending { color: var(--muted); background: var(--btn-muted-bg); }
</style>
