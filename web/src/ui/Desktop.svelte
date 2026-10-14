<script lang="ts">
    import type { GameController } from '../game/controller.svelte';
    import { chipsFor, ROWS, sheetModel, type LogEntry } from '../game/rules';
    import ActionBar from './ActionBar.svelte';
    import BotPanel from './BotPanel.svelte';
    import DiceTray from './DiceTray.svelte';
    import Header from './Header.svelte';
    import Log from './Log.svelte';
    import SheetRows from './SheetRows.svelte';
    let { c }: { c: GameController } = $props();
    const rows = $derived(c.view ? sheetModel(c.view, 'you', { sel: c.sel, strikeSel: c.strikeSel, committing: c.committing, fresh: c.fresh, hidden: [] }) : []);
    const current = $derived<LogEntry | null>(c.view?.phase === 'active'
        ? { n: c.rollNo, lines: [{ who: 'You', chips: [], empty: 'choosing…' }, { who: 'Bot', chips: chipsFor(c.view.bot_white ? [c.view.bot_white] : [], false), empty: '–' }] }
        : c.view?.phase === 'passive' ? { n: c.rollNo, lines: [{ who: 'Bot', chips: [], empty: '…' }, { who: 'You', chips: [], empty: 'choosing…' }] } : null);
    const scoreParts = $derived(c.view ? ROWS.map((r, i) => ({ r, text: `${['Red', 'Yellow', 'Green', 'Blue'][i]} ${c.view!.you.row_points[i]}` })) : []);
</script>

<div class="desk">
    <Header {c} />
    <div class="body">
        <main>
            <DiceTray {c} />
            <section aria-label="Your sheet" class="you">
                <h2>You</h2>
                <div class="sheet"><SheetRows {rows} size={58} onTap={(cell) => c.tapSquare(cell.sq)} /></div>
                <div class="score">
                    {#each scoreParts as p}<span class="part" style="background: {p.r === 'yellow' ? 'var(--score-yellow)' : `var(--${p.r}-band)`}; color: {p.r === 'yellow' ? 'var(--on-yellow)' : 'var(--on-band)'}">{p.text}</span>{/each}
                    <span class="part strikes">−{5 * (c.view?.you.strikes ?? 0)} strikes</span>
                    <b>= {c.view?.you.score ?? 0}</b>
                </div>
            </section>
            <ActionBar {c} />
        </main>
        <aside>
            <BotPanel {c} />
            <div class="logbox"><Log entries={c.log} {current} /></div>
        </aside>
    </div>
</div>

<style>
    .desk { width: 1280px; height: 800px; box-sizing: border-box; padding: 22px 32px; display: flex; flex-direction: column; gap: 16px; overflow: hidden; margin: 0 auto; }
    .body { display: flex; gap: 24px; flex: 1; min-height: 0; }
    main { flex: 1; display: flex; flex-direction: column; gap: 14px; min-width: 0; }
    .you { display: flex; flex-direction: column; gap: 8px; }
    h2 { margin: 0; font-family: var(--cond); font-size: 22px; font-weight: 700; }
    .sheet { background: var(--paper); border-radius: 14px; padding: 10px; width: fit-content; box-shadow: 0 1px 0 var(--edge), 0 8px 24px var(--shadow); }
    .score { display: flex; gap: 8px; font-size: 14px; color: var(--soft); align-items: center; }
    .part { padding: 2px 8px; border-radius: 6px; font-weight: 600; }
    .part.strikes { background: var(--btn-muted-bg); color: var(--ink); }
    main :global(.bar) { margin-top: auto; }
    aside { width: 392px; display: flex; flex-direction: column; gap: 14px; min-height: 0; }
    .logbox { flex: 1; min-height: 0; background: var(--paper); border-radius: 14px; padding: 12px 14px; box-shadow: 0 1px 0 var(--edge); overflow: hidden; }
</style>
