// Memories — Service Worker
// Handles background push notifications and PWA install support.

const CACHE_NAME = 'memories-v1';

// --- Push notifications ---
self.addEventListener('push', event => {
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch (e) {
    data = { title: 'Memories', body: event.data ? event.data.text() : '' };
  }

  const title   = data.title || 'Memories';
  const options = {
    body:  data.body  || 'You have a new memory prompt waiting.',
    icon:  data.icon  || '/static/memories.png',
    badge: data.badge || '/static/memories.png',
    data:  { url: data.url || '/' },
    // Keep the notification visible until the user interacts
    requireInteraction: false,
    tag: data.tag || 'memories-prompt',
    // Replace earlier notifications with the same tag so only one shows
    renotify: false,
  };

  event.waitUntil(self.registration.showNotification(title, options));
});

// Open (or focus) the app when the user taps a notification
self.addEventListener('notificationclick', event => {
  event.notification.close();
  const target = (event.notification.data && event.notification.data.url) || '/';
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then(list => {
      // If the app is already open, just focus it and navigate
      for (const client of list) {
        if ('focus' in client) {
          client.focus();
          if ('navigate' in client) client.navigate(target);
          return;
        }
      }
      // Otherwise open a new window
      return clients.openWindow(target);
    })
  );
});
