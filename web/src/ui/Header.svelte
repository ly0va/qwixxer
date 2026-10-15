<script lang="ts">
    import type { GameController } from '../game/controller.svelte';
    import Menu from './Menu.svelte';
    import type { Nav } from './nav';
    import Tabs from './Tabs.svelte';
    let { c, nav }: { c: GameController; nav: Nav } = $props();
    const p = $derived(c.view?.phase);
    const pill = $derived(p === 'over' ? 'Game over' : p === 'you_to_roll' || p === 'active' ? 'Your turn' : 'Bot’s turn');
</script>

<header class="head">
    <div class="left"><div class="word">Qwixxer</div><Tabs current="play" onPlay={nav.play} onExplore={nav.explorer} /></div>
    <div class="pill"><span class="dot"></span>{pill}</div>
    <div class="right">
        <span>You <b>{c.view?.you.score ?? 0}</b></span>
        <span class="muted">Bot <b>{c.view?.bot.score ?? 0}</b></span>
        <Menu {c} {nav} tabs />
    </div>
</header>

<style>
    .head { display: flex; align-items: center; justify-content: space-between; height: 44px; }
    .left { display: flex; align-items: center; gap: 20px; }
    .word { font-family: var(--cond); font-weight: 800; font-size: 30px; letter-spacing: 0.5px; }
    .pill { display: flex; align-items: center; gap: 10px; background: var(--ink); color: var(--ground); padding: 8px 16px; border-radius: 999px; font-weight: 600; font-size: 15px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--dot); }
    .right { display: flex; align-items: center; gap: 18px; font-weight: 600; font-size: 15px; }
    .right b { font-family: var(--cond); font-size: 22px; color: var(--ink); }
    .muted { color: var(--muted); }
</style>
