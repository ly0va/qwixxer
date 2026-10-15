<script lang="ts">
    import type { ExplorerController } from '../game/explorer.svelte';
    import type { GameController } from '../game/controller.svelte';
    import { DESK_H, DESK_W, deskScale } from '../game/layout';
    import ExSheet from './ExSheet.svelte';
    import Menu from './Menu.svelte';
    import MovesList from './MovesList.svelte';
    import type { Nav } from './nav';
    import PositionCard from './PositionCard.svelte';
    import RollCard from './RollCard.svelte';
    import Tabs from './Tabs.svelte';
    let { c, x, nav }: { c: GameController; x: ExplorerController; nav: Nav } = $props();
    let w = $state(window.innerWidth);
    let h = $state(window.innerHeight);
    const scale = $derived(deskScale(w, h));
    const base = () => location.origin + location.pathname;
</script>

<svelte:window bind:innerWidth={w} bind:innerHeight={h} />
<div class="fit" style="width: {Math.floor(DESK_W * scale)}px; height: {Math.floor(DESK_H * scale)}px">
<div class="desk" style="transform: scale({scale})">
    <header class="head">
        <div class="left"><div class="word">Qwixxer</div><Tabs current="explore" onPlay={nav.play} onExplore={nav.explorer} /></div>
        <div class="right">
            {#if x.notice}<span class="notice" role="status">{x.notice}</span>{/if}
            <button type="button" class="btn" onclick={() => x.copyLink(base())}>
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"></path><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"></path></svg>
                Copy link
            </button>
            <Menu {c} {nav} tabs />
        </div>
    </header>
    <div class="body">
        <main>
            <ExSheet {x} who="you" />
            <div class="mid">
                <button type="button" class="btn small" onclick={() => x.swap()}>
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 4v16M7 20l-3-3M7 20l3-3M17 20V4M17 4l-3 3M17 4l3 3"></path></svg>
                    Swap sides
                </button>
                <span class="hint">Tap squares, locks and strike boxes to set up any position.</span>
            </div>
            <ExSheet {x} who="opponent" />
        </main>
        <aside>
            <PositionCard {x} />
            <RollCard {x} />
            <div class="moves"><MovesList {x} /></div>
        </aside>
    </div>
</div>
</div>

<style>
    .fit { margin: 0 auto; overflow: hidden; }
    .desk { width: 1280px; height: 800px; box-sizing: border-box; padding: 22px 32px; display: flex; flex-direction: column; gap: 16px; overflow: hidden; transform-origin: top left; }
    .head { display: flex; align-items: center; justify-content: space-between; height: 44px; }
    .left, .right { display: flex; align-items: center; gap: 20px; }
    .right { gap: 10px; }
    .word { font-family: var(--cond); font-weight: 800; font-size: 30px; letter-spacing: 0.5px; }
    .notice { font-size: 14px; color: var(--soft); }
    .btn { height: 44px; padding: 0 14px; border-radius: 12px; border: none; background: var(--paper); box-shadow: 0 1px 0 var(--edge); color: var(--ink); display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 15px; cursor: pointer; }
    .btn.small { height: 34px; padding: 0 12px; border-radius: 9px; font-size: 14px; gap: 6px; }
    .body { display: flex; gap: 24px; flex: 1; min-height: 0; }
    main { width: 640px; display: flex; flex-direction: column; gap: 10px; }
    .mid { display: flex; align-items: center; gap: 10px; }
    .hint { font-size: 13px; color: var(--muted); }
    aside { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 12px; min-height: 0; }
    .moves { flex: 1; min-height: 0; overflow-y: auto; border-radius: 14px; }
</style>
