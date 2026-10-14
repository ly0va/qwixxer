<script lang="ts">
    import type { LogEntry } from '../game/rules';
    import ChipView from './Chip.svelte';
    // Spec "Implementation notes · Log": the log scrolls instead of truncating.
    let { entries, current = null, chip = 26 }: { entries: LogEntry[]; current?: LogEntry | null; chip?: number } = $props();
    const shown = $derived([...(current ? [current] : []), ...entries]);
</script>

<section aria-label="Turn log" class="log">
    <h2>Log</h2>
    {#if shown.length === 0}<p class="empty">Nothing yet.</p>{/if}
    {#each shown as e, i (i + ':' + e.n)}
        <div class="roll">
            <span class="n">{e.n}</span>
            <div class="lines">
                {#each e.lines as line}
                    <div class="line">
                        <span class="who" class:you={line.who === 'You'}>{line.who}</span>
                        {#each line.chips as c}<ChipView chip={c} size={chip} />{/each}
                        {#if line.chips.length === 0}<span class="empty">{line.empty}</span>{/if}
                    </div>
                {/each}
            </div>
        </div>
    {/each}
</section>

<style>
    .log { display: flex; flex-direction: column; gap: 8px; min-height: 0; max-height: 100%; overflow-y: auto; }
    h2 { margin: 0; font-family: var(--cond); font-size: 20px; font-weight: 700; }
    .roll { display: flex; align-items: flex-start; gap: 6px; padding: 6px 8px; border-radius: 10px; background: var(--log-bg); }
    .n { width: 26px; font-family: var(--cond); font-weight: 700; font-size: 16px; color: var(--muted); }
    .lines { display: flex; flex-direction: column; gap: 4px; flex: 1; }
    .line { display: flex; align-items: center; gap: 5px; min-height: 26px; }
    .who { width: 30px; font-size: 13px; font-weight: 700; color: var(--muted); }
    .who.you { color: var(--ink); }
    .empty { color: var(--faint); font-size: 14px; }
</style>
