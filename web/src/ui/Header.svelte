<script lang="ts">
    import type { GameController } from '../game/controller.svelte';
    let { c }: { c: GameController } = $props();
    const p = $derived(c.view?.phase);
    const pill = $derived(p === 'over' ? 'Game over' : p === 'you_to_roll' || p === 'active' ? 'Your turn' : 'Bot’s turn');
</script>

<header class="head">
    <div class="word">Qwixxer</div>
    <div class="pill"><span class="dot"></span>{pill}</div>
    <div class="right">
        <span>You <b>{c.view?.you.score ?? 0}</b></span>
        <span class="muted">Bot <b>{c.view?.bot.score ?? 0}</b></span>
        <button type="button" class="theme" aria-label="Dark mode" aria-pressed={c.theme === 'dark'} onclick={() => c.toggleTheme()}>
            {#if c.theme === 'dark'}
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="var(--ink)" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path></svg>
            {:else}
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="var(--ink)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"></path></svg>
            {/if}
        </button>
    </div>
</header>

<style>
    .head { display: flex; align-items: center; justify-content: space-between; }
    .word { font-family: var(--cond); font-weight: 800; font-size: 30px; letter-spacing: 0.5px; }
    .pill { display: flex; align-items: center; gap: 10px; background: var(--ink); color: var(--ground); padding: 8px 16px; border-radius: 999px; font-weight: 600; font-size: 15px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--dot); }
    .right { display: flex; align-items: center; gap: 18px; font-weight: 600; font-size: 15px; }
    .right b { font-family: var(--cond); font-size: 22px; color: var(--ink); }
    .muted { color: var(--muted); }
    .theme { width: 44px; height: 44px; border-radius: 12px; border: none; background: var(--paper); box-shadow: 0 1px 0 var(--edge); display: flex; align-items: center; justify-content: center; cursor: pointer; padding: 0; }
</style>
