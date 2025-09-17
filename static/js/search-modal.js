// Minimal placeholder for nav search to avoid 404s on preload.
// If you plan to add a real search modal later, hook it up here.
(function(){
  const input = document.getElementById('nav-search');
  if (!input) return;
  // Optional: small UX nicety — focus with '/'
  document.addEventListener('keydown', (e) => {
    if (e.key === '/' && document.activeElement?.tagName !== 'INPUT' && document.activeElement?.tagName !== 'TEXTAREA') {
      e.preventDefault();
      input.focus();
    }
  });
})();

