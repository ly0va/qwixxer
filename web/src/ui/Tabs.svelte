<script lang="ts">
    // Desktop Play / Explore / Review switch (spec "Navigation"); the phone uses the ⋯ menu.
    // On the about page no tab is current.
    let { current, onPlay, onExplore, onReview }: { current: 'play' | 'explore' | 'review' | 'about'; onPlay: () => void; onExplore: () => void; onReview: () => void } = $props();
</script>

<nav aria-label="Main" class="tabs">
    <a href="#/" aria-current={current === 'play' ? 'page' : undefined} onclick={(e) => { e.preventDefault(); onPlay(); }}>Play</a>
    <a href="#/explore" aria-current={current === 'explore' ? 'page' : undefined} onclick={(e) => { e.preventDefault(); onExplore(); }}>Explore</a>
    <a href="#/review" aria-current={current === 'review' ? 'page' : undefined} onclick={(e) => { e.preventDefault(); onReview(); }}>Review</a>
</nav>

<style>
    .tabs { display: flex; gap: 2px; padding: 3px; border-radius: 12px; background: var(--btn-muted-bg); }
    a { min-height: 38px; padding: 0 18px; border-radius: 9px; display: flex; align-items: center; font-weight: 600; font-size: 15px; text-decoration: none; color: var(--muted); }
    a[aria-current='page'] { background: var(--paper); color: var(--ink); font-weight: 700; box-shadow: 0 1px 0 var(--edge); }
</style>
