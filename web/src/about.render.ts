import { render } from 'svelte/server';
import About from './ui/About.svelte';

/** The about page's HTML, rendered at build time into about/index.html. */
export const renderAbout = (): string => render(About).body;
