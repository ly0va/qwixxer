<script lang="ts">
    import type { GameController } from '../game/controller.svelte';
    import { confirmState, strikeOpen } from '../game/rules';
    import { gameHref } from '../game/review';
    import MarkX from './MarkX.svelte';
    let { c }: { c: GameController } = $props();
    const v = $derived(c.view);
    const conf = $derived(v ? confirmState(v, c.sel, c.strikeSel) : { label: 'Mark', ok: false });
    const open = $derived(v ? strikeOpen(v, c.sel) : false);
    const strikes = $derived(v?.you.strikes ?? 0);
    const result = $derived(v?.over ? { title: v.you.score > v.bot.score ? 'You win' : v.you.score < v.bot.score ? 'Bot wins' : 'Draw',
        reason: v.over === 'locks' ? 'Two rows are locked.' : v.over === 'you_strikes' ? 'You took a fourth strike.' : 'The bot took a fourth strike.' } : null);
</script>

<section aria-label="Your move" class="bar">
    <div class="strikes">
        <span class="label">Strikes</span>
        {#each [0, 1, 2, 3] as i}
            {@const taken = i < strikes}
            {@const pencil = i === strikes && c.strikeSel}
            <button type="button" class="box" class:faded={!taken && !pencil && !(i === strikes && open)}
                aria-label={taken ? 'Strike taken' : i === strikes ? 'Strike box' : 'Strike box, unavailable'}
                aria-pressed={pencil} aria-disabled={!(i === strikes && open)} onclick={() => i === strikes && c.tapStrike()}>
                {#if taken || pencil}<MarkX pencil={pencil} anim={pencil ? 'draw' : (c.fresh[`you:strike:${i}`] ?? '')} />{/if}
            </button>
        {/each}
    </div>
    <div class="msg">
        {#if c.error}
            <div class="tip" role="alert">Something went wrong: {c.error}</div>
        {:else if result}
            <div class="result" role="status"><b>{result.title}</b><span class="score">{v?.you.score} – {v?.bot.score}</span><span class="reason">{result.reason}</span>
                {#if c.lastCode}<a class="analyse" href={gameHref(c.lastCode)}>Analyse game</a>{/if}</div>
        {:else if c.tip}
            <div class="tip" role="alert">{c.tip}</div>
        {/if}
    </div>
    <button type="button" class="confirm" class:primary={conf.ok || !!c.error} aria-disabled={!conf.ok && !c.error}
        onclick={() => (c.error ? c.newGame() : c.confirm())}>{c.error ? 'New game' : conf.label}</button>
</section>

<style>
    .bar { display: flex; align-items: center; gap: 20px; background: var(--paper); border-radius: 14px; padding: 0 16px; height: 72px; box-sizing: border-box; flex-shrink: 0; box-shadow: 0 1px 0 var(--edge); }
    .strikes { display: flex; align-items: center; gap: 8px; }
    .label { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
    .box { position: relative; box-sizing: border-box; width: 40px; height: 40px; border-radius: 8px; padding: 0; border: 2px solid var(--ink); background: var(--paper); cursor: pointer; }
    .box.faded { opacity: 0.35; cursor: default; }
    .msg { flex: 1; display: flex; justify-content: flex-end; }
    .tip { display: flex; align-items: center; gap: 8px; background: var(--paper); border-radius: 10px; padding: 6px 12px; font-size: 13px; line-height: 1.3; max-height: 64px; overflow: hidden; box-shadow: inset 0 0 0 2px var(--ink); max-width: 460px; }
    .result { display: flex; align-items: baseline; gap: 12px; }
    .result b { font-family: var(--cond); font-weight: 800; font-size: 28px; }
    .score { font-family: var(--cond); font-weight: 700; font-size: 22px; }
    .reason { font-size: 14px; color: var(--muted); }
    /* Fixed width so the bar doesn't shift as the label changes (Mark 1, Strike, New game…). */
    .confirm { width: 136px; min-height: 48px; padding: 0; border: none; border-radius: 12px; font-weight: 700; font-size: 17px; cursor: not-allowed; background: var(--btn-muted-bg); color: var(--btn-muted-fg); }
    .confirm.primary { cursor: pointer; background: var(--ink); color: var(--ground); }
    .analyse { margin-left: 10px; font-weight: 700; font-size: 15px; color: var(--ink); text-decoration: underline; text-underline-offset: 3px; }
</style>
