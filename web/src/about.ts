import './styles/theme.css';
import { hydrate } from 'svelte';
import About from './ui/About.svelte';

// about/index.html arrives with the page already rendered (see aboutPage in
// vite.config.ts); hydrating only attaches Svelte to it and brings the styles.
hydrate(About, { target: document.getElementById('app')! });
