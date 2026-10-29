// Network-first: always serve the latest deploy when online, fall back to the
// last cached copy offline. (The previous cache-first strategy served stale
// code after a deploy and could mix old and new modules.)
const CACHE_NAME = 'qwixxer-v3';

// Pages the app links to but never loads itself (the about page): fetched
// with the files they reference when the worker installs, so they open
// offline even if never visited. Visits refresh them like everything else.
const PRECACHE = ['about/'];

async function precache() {
  const cache = await caches.open(CACHE_NAME);
  for (const path of PRECACHE) {
    const url = new URL(path, self.registration.scope);
    const response = await fetch(url);
    if (!response.ok) continue;
    const html = await response.clone().text();
    await cache.put(url, response);
    const files = [...html.matchAll(/(?:href|src)="([^"]+\.(?:js|css))"/g)]
      .map((m) => new URL(m[1], url))
      .filter((u) => u.origin === self.location.origin);
    await cache.addAll(files);
  }
}

self.addEventListener('install', (event) => {
  self.skipWaiting();
  // Offline extras: a failure here must not stop the worker installing.
  event.waitUntil(precache().catch(() => {}));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return;
  event.respondWith(
    fetch(req)
      .then((response) => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(req, clone));
        }
        return response;
      })
      .catch(() => caches.match(req).then((cached) => cached || Response.error()))
  );
});
