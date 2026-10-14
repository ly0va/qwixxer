import './styles/theme.css';
import { mount } from 'svelte';
import App from './ui/App.svelte';

mount(App, { target: document.getElementById('app')! });

// The service worker (offline play) only runs on the deployed build. On the
// dev server it would serve cached modules across restarts, so any leftover
// registration is removed there.
if ('serviceWorker' in navigator) {
    if (import.meta.env.PROD) {
        navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`);
    } else {
        navigator.serviceWorker.getRegistrations().then((rs) => rs.forEach((r) => r.unregister()));
        if ('caches' in window) caches.keys().then((ks) => ks.forEach((k) => caches.delete(k)));
    }
}
