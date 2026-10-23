<script lang="ts">
    import type { GameController } from '../game/controller.svelte';
    import Icon from './Icon.svelte';
    import type { Nav } from './nav';
    // `tabs`: the desktop header's Play / Explore tabs already switch views.
    let { c, nav, tabs = false }: { c: GameController; nav: Nav; tabs?: boolean } = $props();

    // The "⋯" header menu (all layouts). In the game: New game and (phone)
    // Explorer; in the explorer (phone): Back to game. Then, in both, dark
    // mode, the rules PDF, the source repo and a trademark notice. Open state lives on the
    // controller so App's page-wide keys treat it as a modal.
    let trigger = $state<HTMLButtonElement>();
    let root = $state<HTMLDivElement>();

    // New game abandons the current one, so mid-game it takes two clicks:
    // the first arms it ("Abandon game?") for 3 s, the second starts.
    let armed = $state(false);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const midGame = $derived(!!c.view && c.view.phase !== 'over');

    // Focus the first item on open; on close disarm and return focus to "⋯".
    let wasOpen = false;
    $effect(() => {
        if (c.menuOpen) {
            root?.querySelector<HTMLElement>('[role^="menuitem"]')?.focus();
            wasOpen = true;
        } else if (wasOpen) {
            wasOpen = false;
            armed = false;
            clearTimeout(timer);
            trigger?.focus();
        }
    });
    $effect(() => () => clearTimeout(timer));

    function newGame() {
        if (c.busy) return;
        if (midGame && !armed) {
            armed = true;
            clearTimeout(timer);
            timer = setTimeout(() => (armed = false), 3000);
            return;
        }
        c.menuOpen = false;
        c.newGame();
    }

    function outside(e: MouseEvent) {
        if (c.menuOpen && root && !root.contains(e.target as Node)) c.menuOpen = false;
    }
</script>

<svelte:document onclick={outside} />

<div class="menu" bind:this={root}>
    <button type="button" class="trigger" bind:this={trigger} aria-label="Menu" aria-haspopup="menu" aria-expanded={c.menuOpen}
        onclick={() => (c.menuOpen = !c.menuOpen)}>
        <svg viewBox="0 0 24 24" width="20" height="20" fill="var(--ink)" aria-hidden="true"><circle cx="5" cy="12" r="2"></circle><circle cx="12" cy="12" r="2"></circle><circle cx="19" cy="12" r="2"></circle></svg>
    </button>
    {#if c.menuOpen}
        <div class="pop" role="menu" aria-label="Menu">
            {#if nav.current === 'play'}
                <button type="button" role="menuitem" class="item" class:armed aria-disabled={c.busy} onclick={newGame}>
                    <span class="lab"><Icon name="plus" />{armed ? 'Abandon game?' : 'New game'}</span>
                </button>
                {#if !tabs}<button type="button" role="menuitem" class="item" onclick={() => { c.menuOpen = false; nav.explorer(); }}><span class="lab"><Icon name="search" />Explorer</span></button>{/if}
                {#if !tabs && c.lastCode}<button type="button" role="menuitem" class="item" onclick={() => { c.menuOpen = false; nav.review(); }}><span class="lab"><Icon name="chart" />Review last game</span></button>{/if}
            {:else if !tabs}
                <button type="button" role="menuitem" class="item" onclick={() => { c.menuOpen = false; nav.play(); }}><span class="lab"><Icon name="die" />Back to game</span></button>
            {/if}
            <button type="button" role="menuitemcheckbox" class="item" aria-checked={c.theme === 'dark'} onclick={() => c.toggleTheme()}>
                <span class="lab"><Icon name="moon" />Dark mode</span>
                <span class="switch" class:on={c.theme === 'dark'} aria-hidden="true"><span></span></span>
            </button>
            <a role="menuitem" class="item" href="https://m.media-amazon.com/images/I/A1Y4fDIZZmL.pdf" target="_blank" rel="noopener noreferrer"
                onclick={() => (c.menuOpen = false)}>
                <span class="lab"><Icon name="book" />Rules</span>
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--muted)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"></path></svg>
            </a>
            <a role="menuitem" class="item" href="https://github.com/ly0va/qwixxer" target="_blank" rel="noopener noreferrer"
                onclick={() => (c.menuOpen = false)}>
                <span class="lab"><Icon name="github" />Source code</span>
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--muted)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"></path></svg>
            </a>
            <p class="legal">Unofficial fan project. Qwixx is a trademark of Nürnberger-Spielkarten-Verlag, published in North America by Gamewright. Not affiliated with or endorsed by either.</p>
        </div>
    {/if}
</div>

<style>
    .menu { position: relative; }
    .trigger { width: 44px; height: 44px; border-radius: 12px; border: none; background: var(--paper); box-shadow: 0 1px 0 var(--edge); display: flex; align-items: center; justify-content: center; cursor: pointer; padding: 0; }
    .pop {
        position: absolute; right: 0; top: calc(100% + 6px); z-index: 5; min-width: 200px; box-sizing: border-box;
        display: flex; flex-direction: column; gap: 2px; padding: 6px; border-radius: 12px;
        background: var(--paper); box-shadow: 0 0 0 1px var(--edge), 0 10px 30px var(--shadow);
    }
    .item {
        min-height: 44px; padding: 0 12px; border: none; border-radius: 8px; background: transparent; color: var(--ink);
        display: flex; align-items: center; justify-content: space-between; gap: 12px;
        font-family: var(--text); font-weight: 600; font-size: 15px; text-align: left; text-decoration: none; cursor: pointer;
    }
    .lab { display: flex; align-items: center; gap: 10px; }
    .legal { margin: 4px 12px 6px; max-width: 220px; font-size: 11px; line-height: 1.35; color: var(--muted); }
    .item:hover { background: var(--log-bg); }
    .item.armed { background: var(--ink); color: var(--ground); }
    .item[aria-disabled='true'] { opacity: 0.5; cursor: default; }
    .switch { width: 34px; height: 20px; border-radius: 10px; background: var(--btn-muted-bg); position: relative; flex-shrink: 0; }
    .switch span { position: absolute; top: 2px; left: 2px; width: 16px; height: 16px; border-radius: 50%; background: var(--paper); box-shadow: 0 1px 2px var(--shadow); transition: left 0.15s; }
    .switch.on { background: var(--ink); }
    .switch.on span { left: 16px; }
</style>
