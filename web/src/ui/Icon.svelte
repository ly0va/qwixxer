<script lang="ts" module>
    export type IconName = 'plus' | 'search' | 'die' | 'moon' | 'book' | 'github' | 'chart' | 'link' | 'info';
</script>

<script lang="ts">
    // Line icons for menu items and buttons (24-unit grid, drawn in the text colour).
    let { name, size = 18 }: { name: IconName; size?: number } = $props();
</script>

{#if name === 'github'}
    <svg viewBox="0 0 16 16" width={size} height={size} fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>
{:else}
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        {#if name === 'plus'}<path d="M12 5v14M5 12h14"></path>
        {:else if name === 'search'}<circle cx="11" cy="11" r="6.5"></circle><path d="M20 20l-4.4-4.4"></path>
        {:else if name === 'die'}<rect x="4" y="4" width="16" height="16" rx="3.5"></rect><g fill="currentColor" stroke="none"><circle cx="8.6" cy="8.6" r="1.4"></circle><circle cx="12" cy="12" r="1.4"></circle><circle cx="15.4" cy="15.4" r="1.4"></circle></g>
        {:else if name === 'chart'}<path d="M4 19h16M5 15l4-5 4 3 6-7"></path>
        {:else if name === 'link'}<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"></path><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"></path>
        {:else if name === 'moon'}<path d="M20 12.8A8 8 0 1 1 11.2 4a6.2 6.2 0 0 0 8.8 8.8z"></path>
        {:else if name === 'book'}<path d="M5 4.5A1.5 1.5 0 0 1 6.5 3H19v15H6.5A1.5 1.5 0 0 0 5 19.5zM5 19.5A1.5 1.5 0 0 0 6.5 21H19v-3"></path>
        {:else if name === 'info'}<circle cx="12" cy="12" r="8.5"></circle><path d="M12 11v5.5"></path><circle cx="12" cy="7.8" r="0.6" fill="currentColor"></circle>
        {/if}
    </svg>
{/if}
