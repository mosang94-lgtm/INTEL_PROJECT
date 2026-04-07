// ── CarInspect Service Worker ──
// Cache-first for static assets, network-first for API calls.

const CACHE_NAME = "carinspect-v1";

const STATIC_ASSETS = [
  "/index.html",
  "/style.css",
  "/data.js",
  "/api.js",
  "/camera.js",
  "/app.js",
  "/manifest.json"
];

// ── Install: pre-cache static assets ──
self.addEventListener("install", function (event) {
  event.waitUntil(
    caches.open(CACHE_NAME).then(function (cache) {
      return cache.addAll(STATIC_ASSETS);
    }).then(function () {
      return self.skipWaiting();
    })
  );
});

// ── Activate: clean old caches ──
self.addEventListener("activate", function (event) {
  event.waitUntil(
    caches.keys().then(function (names) {
      return Promise.all(
        names.filter(function (name) {
          return name !== CACHE_NAME;
        }).map(function (name) {
          return caches.delete(name);
        })
      );
    }).then(function () {
      return self.clients.claim();
    })
  );
});

// ── Fetch: cache-first for static, network-first for API ──
self.addEventListener("fetch", function (event) {
  var url = new URL(event.request.url);

  // Skip non-GET requests (e.g., POST /inspect)
  if (event.request.method !== "GET") {
    return;
  }

  // Skip cross-origin requests
  if (url.origin !== self.location.origin) {
    return;
  }

  // For navigation requests, serve index.html (SPA fallback)
  if (event.request.mode === "navigate") {
    event.respondWith(
      caches.match("/index.html").then(function (cached) {
        return cached || fetch(event.request);
      })
    );
    return;
  }

  // Cache-first strategy for static assets
  event.respondWith(
    caches.match(event.request).then(function (cached) {
      if (cached) {
        // Return cache hit, but also update cache in background
        var fetchPromise = fetch(event.request).then(function (response) {
          if (response && response.ok) {
            var clone = response.clone();
            caches.open(CACHE_NAME).then(function (cache) {
              cache.put(event.request, clone);
            });
          }
          return response;
        }).catch(function () {
          // Network failed, cache is already returned
        });
        return cached;
      }

      // No cache hit — go to network
      return fetch(event.request).then(function (response) {
        if (!response || !response.ok) {
          return response;
        }
        var clone = response.clone();
        caches.open(CACHE_NAME).then(function (cache) {
          cache.put(event.request, clone);
        });
        return response;
      }).catch(function () {
        // Both cache and network failed — return offline fallback for HTML
        if (event.request.headers.get("accept") && event.request.headers.get("accept").includes("text/html")) {
          return caches.match("/index.html");
        }
      });
    })
  );
});