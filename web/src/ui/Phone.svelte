<script lang="ts">
    import { gameHref } from '../game/review';
    import type { GameController } from '../game/controller.svelte';
    import { phoneLayout } from '../game/layout';
    import { chipsFor, pendingGain, confirmState, ROWS, sheetModel, strikeOpen, type Chip, type LogEntry } from '../game/rules';
    import BotPanel from './BotPanel.svelte';
    import ChipView from './Chip.svelte';
    import DiceTray from './DiceTray.svelte';
    import Log from './Log.svelte';
    import Icon from './Icon.svelte';
    import MarkX from './MarkX.svelte';
    import Menu from './Menu.svelte';
    import type { Nav } from './nav';
    import SheetColumns from './SheetColumns.svelte';
    // `drawer` is bound by App so its page-wide key handler knows the drawer is open (keys.ts).
    let { c, nav, drawer = $bindable(false) }: { c: GameController; nav: Nav; drawer?: boolean } = $props();
    const v = $derived(c.view);
    let h = $state(window.innerHeight);
    let w = $state(window.innerWidth);
    const L = $derived(phoneLayout(w, h));
    // Focus moves to Close when the drawer opens and back to the Bot button when it closes.
    let botBtn = $state<HTMLButtonElement>();
    let closeBtn = $state<HTMLButtonElement>();
    let wasOpen = false;
    $effect(() => {
        if (drawer) { closeBtn?.focus(); wasOpen = true; }
        else if (wasOpen) { wasOpen = false; botBtn?.focus(); }
    });
    const rows = $derived(v ? sheetModel(v, 'you', { sel: c.sel, strikeSel: c.strikeSel, committing: c.committing, fresh: c.fresh, hidden: [] }) : []);
    const conf = $derived(v ? confirmState(v, c.sel, c.strikeSel) : { label: 'Mark', ok: false });
    const open = $derived(v ? strikeOpen(v, c.sel) : false);
    // The bot's most recent marks ride on the header button (its revealed white sum on your turn).
    const botChips = $derived<Chip[]>(v?.phase === 'active' && v.bot_white ? chipsFor([v.bot_white], false)
        : (c.log[0]?.lines.find((l) => l.who === 'Bot')?.chips ?? []));
    const current = $derived<LogEntry | null>(v?.phase === 'active'
        ? { n: c.rollNo, lines: [{ who: 'You', chips: [], empty: 'choosing…' }, { who: 'Bot', chips: chipsFor(v.bot_white ? [v.bot_white] : [], false), empty: '–' }] }
        : v?.phase === 'passive' ? { n: c.rollNo, lines: [{ who: 'Bot', chips: [], empty: '…' }, { who: 'You', chips: [], empty: 'choosing…' }] } : null);
    // Whose turn it is: the active player's score in the header is inked, like
    // the desktop's "Your turn" pill (none once the game is over).
    const yourTurn = $derived(v?.phase === 'you_to_roll' || v?.phase === 'active');
    const botTurn = $derived(v?.phase === 'bot_to_roll' || v?.phase === 'passive');
    // Spec "Game over": the result, and Analyze for the review. On the phone it replaces
    // the dice row (the dice no longer matter), so it reads as a headline.
    const result = $derived(v?.over ? {
        title: v.you.score > v.bot.score ? 'You win' : v.you.score < v.bot.score ? 'Bot wins' : 'Draw',
        score: `${v.you.score} – ${v.bot.score}`,
    } : null);
    const gain = $derived(c.view && !c.committing ? pendingGain(c.view, c.sel, c.strikeSel) : null);
</script>

<svelte:window bind:innerWidth={w} bind:innerHeight={h} />

<div class="phone" class:short={L.short} class:scroll={L.scroll} style:--vh="{h}px">
    <div class="main" inert={drawer}>
    <header class="head">
        <div class="word">Qwixxer</div>
        <div class="right">
            <span class="you" class:turn={yourTurn}>You <b>{v?.you.score ?? 0}</b></span>
            <button type="button" class="botbtn" class:turn={botTurn} bind:this={botBtn} aria-label="Bot, {v?.bot.score ?? 0} points. Open bot sheet and log" onclick={() => (drawer = true)}>
                Bot <b>{v?.bot.score ?? 0}</b>
                {#each botChips as ch}<ChipView chip={ch} size={20} />{/each}
            </button>
            <Menu {c} {nav} />
        </div>
    </header>
    {#if result}
        <div class="result" role="status">
            <div class="headline"><b>{result.title}</b><span>{result.score}</span></div>
            {#if c.lastCode}{@const code = c.lastCode}<button type="button" class="analyze" onclick={() => (location.hash = gameHref(code))}><Icon name="chart" size={16} />Analyze</button>{/if}
        </div>
    {:else}
        <DiceTray {c} size={L.die} compact />
    {/if}
    <div class="body">
        <SheetColumns {rows} size={L.size} onTap={(cell) => c.tapSquare(cell.sq)} />
        <aside class="side">
            <div class="card">
                <span class="label">Score</span>
                {#each ROWS as r, i}<span class="part" style="background: {r === 'yellow' ? 'var(--score-yellow)' : `var(--${r}-band)`}; color: {r === 'yellow' ? 'var(--on-yellow)' : 'var(--on-band)'}">{['Red', 'Yellow', 'Green', 'Blue'][i]} {v?.you.row_points[i] ?? 0}</span>{/each}
                <span class="part strikes">Strikes −{5 * (v?.you.strikes ?? 0)}</span>
                <b class="total">= {v?.you.score ?? 0}{#if gain !== null}<span class="gain" aria-label="{gain >= 0 ? 'plus' : 'minus'} {Math.abs(gain)} if confirmed">{gain >= 0 ? '+' : '−'}{Math.abs(gain)}</span>{/if}</b>
            </div>
            <div class="card">
                <span class="label">Strikes</span>
                <div class="grid">
                    {#each [0, 1, 2, 3] as i}
                        {@const strikes = v?.you.strikes ?? 0}
                        {@const pencil = i === strikes && c.strikeSel}
                        <button type="button" class="box" class:faded={!(i < strikes) && !pencil && !(i === strikes && open)}
                            aria-label={i < strikes ? 'Strike taken' : 'Strike box'} aria-pressed={pencil} aria-disabled={!(i === strikes && open)}
                            onclick={() => i === strikes && c.tapStrike()}>
                            {#if i < strikes || pencil}<MarkX pencil={pencil} anim={pencil ? 'draw' : (c.fresh[`you:strike:${i}`] ?? '')} />{/if}
                        </button>
                    {/each}
                </div>
            </div>
            <button type="button" class="confirm" class:primary={conf.ok || !!c.error} aria-disabled={!conf.ok && !c.error}
                onclick={() => (c.error ? c.newGame() : c.confirm())}>{#if c.error || v?.over}<Icon name="plus" size={20} />{/if}{c.error ? 'New game' : conf.label}</button>
        </aside>
    </div>
    <div class="msg" role="status">{c.error ? `Something went wrong: ${c.error}` : c.tip}</div>
    </div>

    {#if drawer}
        <button type="button" class="scrim" aria-label="Close" onclick={() => (drawer = false)}></button>
        <div class="drawer" role="dialog" aria-modal="true" aria-label="Bot and log">
            <div class="handle"></div>
            <BotPanel {c} size={L.botSize} />
            <Log entries={c.log} {current} chip={24} />
            <button type="button" class="close" bind:this={closeBtn} onclick={() => (drawer = false)}>Close</button>
        </div>
    {/if}
</div>

<style>
    /* --vh is window.innerHeight, the height the layout sizes the squares for: an
       installed app can report 100dvh taller than that, leaving blank space to scroll. */
    .phone { max-width: 430px; height: var(--vh, 100dvh); margin: 0 auto; overflow: hidden; }
    /* Too short even for 28 px squares: grow and let the page scroll rather than clip the sheet. */
    .phone.scroll { height: auto; min-height: var(--vh, 100dvh); overflow: visible; }
    .main { height: 100%; box-sizing: border-box; padding: 12px 12px 16px; display: flex; flex-direction: column; gap: 10px; }
    .short .main { gap: 8px; padding-bottom: 12px; }
    .head { display: flex; align-items: center; justify-content: space-between; height: 44px; }
    .word { font-family: var(--cond); font-weight: 800; font-size: 26px; }
    .right { display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 14px; }
    .right b { font-family: var(--cond); font-size: 20px; }
    .you { height: 44px; border-radius: 12px; display: flex; align-items: center; gap: 5px; padding: 0 10px; }
    .you.turn, .botbtn.turn { background: var(--ink); color: var(--ground); box-shadow: none; }
    .botbtn { height: 44px; border-radius: 12px; border: none; background: var(--paper); box-shadow: 0 1px 0 var(--edge); color: var(--muted); display: flex; align-items: center; gap: 5px; padding: 0 10px; cursor: pointer; }
    .body { display: flex; gap: 10px; align-items: stretch; }
    .side { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 10px; }
    .card { display: flex; flex-direction: column; gap: 5px; background: var(--paper); border-radius: 14px; padding: 8px 6px; box-shadow: 0 1px 0 var(--edge); }
    .label { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; padding-left: 4px; }
    .part { padding: 2px 6px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    .part.strikes { background: var(--btn-muted-bg); color: var(--ink); }
    /* What the pencilled marks would add, in pencil, until they are inked. */
    .gain { margin-left: 6px; font-family: var(--cond); font-weight: 700; font-size: 22px; color: var(--pencil); }
    .total { font-family: var(--cond); font-size: 22px; text-align: right; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; justify-items: center; }
    .box { position: relative; box-sizing: border-box; width: 40px; height: 40px; border-radius: 8px; padding: 0; border: 2px solid var(--ink); background: var(--paper); }
    .box.faded { opacity: 0.35; }
    .confirm { display: inline-flex; align-items: center; justify-content: center; gap: 8px; margin-top: auto; min-height: 64px; border: none; border-radius: 14px; font-weight: 700; font-size: 18px; background: var(--btn-muted-bg); color: var(--btn-muted-fg); }
    .confirm.primary { background: var(--ink); color: var(--ground); }
    .result { height: 48px; box-sizing: border-box; padding: 0 12px; border-radius: 12px; background: var(--paper); box-shadow: 0 1px 0 var(--edge); display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
    .headline { display: flex; align-items: baseline; gap: 10px; line-height: 1; }
    .headline b { font-family: var(--cond); font-weight: 800; font-size: 26px; }
    .headline span { font-family: var(--cond); font-weight: 700; font-size: 22px; }
    .msg { font-size: 13px; line-height: 1.3; color: var(--soft); min-height: 34px; }
    .scrim { position: fixed; inset: 0; z-index: 1; border: none; background: var(--scrim); }
    .drawer { position: fixed; z-index: 2; bottom: 0; left: 50%; transform: translateX(-50%); width: 100%; max-width: 430px; max-height: 75dvh; box-sizing: border-box; background: var(--paper); border-radius: 20px 20px 0 0; padding: 8px 14px 16px; display: flex; flex-direction: column; gap: 12px; overflow: hidden; }
    .handle { align-self: center; width: 40px; height: 5px; border-radius: 3px; background: var(--edge); }
    .close { align-self: flex-end; min-height: 44px; padding: 0 16px; border-radius: 12px; border: none; background: var(--ground); color: var(--ink); font-weight: 600; }
    .analyze { display: inline-flex; align-items: center; justify-content: center; gap: 6px; height: 34px; padding: 0 14px; border: none; border-radius: 9px; background: var(--ink); color: var(--ground); font-weight: 700; font-size: 14px; cursor: pointer; }
</style>
