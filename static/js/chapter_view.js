(() => {
  const $ = (sel, el=document) => el.querySelector(sel);

  // ── Markdown renderer ────────────────────────────────────────────────────
  function renderMd(text) {
    if (!text) return '<p class="opacity-70">No draft yet. Click "Compile Chapter".</p>';
    if (typeof marked !== 'undefined') return marked.parse(text, { breaks: true });
    // Minimal fallback
    return text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/^### (.+)$/gm, '<h3>$1</h3>')
      .replace(/^## (.+)$/gm, '<h2>$1</h2>')
      .replace(/^# (.+)$/gm, '<h1>$1</h1>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/^/, '<p>') + '</p>';
  }

  // ── Progress cycling ──────────────────────────────────────────────────────
  const COMPILE_MSGS = [
    'Pass 1 of 4 — Organizing your responses into themes…',
    'Pass 2 of 4 — Drafting each section as memoir prose…',
    'Pass 3 of 4 — Assembling and polishing the full chapter…',
    'Pass 4 of 4 — Identifying follow-up questions…',
    'Still working — LLM prose generation takes a few minutes…',
    'Almost there — finalizing your chapter…',
  ];
  let _msgIdx = 0;
  let _progressTimer = null;
  let _elapsedTimer = null;
  let _compileStart = 0;

  function startProgress() {
    const sp = $('#spinner');
    const card = $('#compileProgress');
    if (!sp || !card) return;
    _compileStart = Date.now();
    _msgIdx = 0;
    sp.textContent = COMPILE_MSGS[0];
    card.classList.remove('hidden');
    $('#draftArea')?.classList.add('hidden');
    _progressTimer = setInterval(() => {
      _msgIdx = Math.min(_msgIdx + 1, COMPILE_MSGS.length - 1);
      _stampElapsed();
    }, 90_000);
    _elapsedTimer = setInterval(_stampElapsed, 5_000);
  }

  function _stampElapsed() {
    const sp = $('#spinner');
    if (!sp) return;
    const elapsed = Math.round((Date.now() - _compileStart) / 1000);
    sp.textContent = `${COMPILE_MSGS[_msgIdx]} (${elapsed}s elapsed)`;
  }

  function stopProgress() {
    clearInterval(_progressTimer);
    clearInterval(_elapsedTimer);
    $('#compileProgress')?.classList.add('hidden');
    $('#draftArea')?.classList.remove('hidden');
  }

  // ── Gap chips ─────────────────────────────────────────────────────────────
  function renderGaps(gaps) {
    const box = $('#gapChips');
    if (!box) return;
    box.innerHTML = '';
    if (!gaps || !gaps.length) {
      box.innerHTML = '<div class="text-sm opacity-70">None yet.</div>';
      return;
    }
    for (const g of gaps) {
      const card = document.createElement('div');
      card.className = 'border border-black/20 rounded-xl bg-white/60 backdrop-blur-md p-3 flex items-start gap-3';
      const icon = document.createElement('span');
      icon.className = 'material-symbols-outlined text-base mt-0.5 opacity-50';
      icon.textContent = 'lightbulb';
      const text = document.createElement('div');
      const q = document.createElement('div');
      q.className = 'font-medium text-sm';
      q.textContent = g.question;
      text.appendChild(q);
      if (g.why) {
        const why = document.createElement('div');
        why.className = 'text-xs opacity-60 mt-0.5';
        why.textContent = g.why;
        text.appendChild(why);
      }
      card.appendChild(icon);
      card.appendChild(text);
      box.appendChild(card);
    }
  }

  // ── Render a completed compilation ────────────────────────────────────────
  // Reload the page so the server renders the markdown cleanly.
  function renderCompilation(_comp) {
    stopProgress();
    window.location.reload();
  }

  // ── Poll for background compile result ────────────────────────────────────
  let _pollTimer = null;
  let _knownVersion = null;

  async function pollForResult() {
    try {
      const data = await fetch(
        `/api/chapter/${encodeURIComponent(window.CHAPTER_KEY)}/status`,
        { credentials: 'include' }
      ).then(r => r.json());

      const comp = data.latest_compilation;
      const newVersion = comp && (comp.version !== _knownVersion);
      const stillRunning = data.is_compiling;

      if (newVersion && !stillRunning) {
        clearInterval(_pollTimer);
        renderCompilation(comp);
      } else if (!stillRunning) {
        // compile finished but no new version (something went wrong)
        clearInterval(_pollTimer);
        stopProgress();
      }
    } catch (e) {
      console.warn('poll error', e);
    }
  }

  // ── Compile button ────────────────────────────────────────────────────────
  async function compileNow() {
    startProgress();

    // capture current version before firing
    try {
      const cur = await fetch(
        `/api/chapter/${encodeURIComponent(window.CHAPTER_KEY)}/status`,
        { credentials: 'include' }
      ).then(r => r.json());
      _knownVersion = cur.latest_compilation?.version ?? null;
    } catch (_) { _knownVersion = null; }

    try {
      await fetch(
        `/api/chapter/${encodeURIComponent(window.CHAPTER_KEY)}/compile`,
        { method: 'POST', credentials: 'include' }
      );
    } catch (e) {
      console.warn('compile kick-off error (still polling):', e);
    }

    clearInterval(_pollTimer);
    _pollTimer = setInterval(pollForResult, 12_000);
  }

  // ── Status refresh (page load only) ──────────────────────────────────────
  async function refreshStatus() {
    try {
      const data = await fetch(
        `/api/chapter/${encodeURIComponent(window.CHAPTER_KEY)}/status`,
        { credentials: 'include' }
      ).then(r => r.json());

      const badge = $('#chapterStatusBadge');
      if (badge) badge.textContent = data.ready ? 'Ready' : 'Locked';

      const comp = data.latest_compilation;
      // Server already rendered the initial markdown — just track the version
      _knownVersion = window.INITIAL_VERSION ?? comp?.version ?? null;

      if (data.is_compiling) {
        _knownVersion = comp?.version ?? null;
        // Show progress card but keep buttons enabled — user can re-trigger if stuck
        startProgress();
        clearInterval(_pollTimer);
        _pollTimer = setInterval(pollForResult, 12_000);
      }
    } catch (e) {
      console.warn('status err', e);
    }
  }

  // ── Publish ───────────────────────────────────────────────────────────────
  async function publishNow() {
    try {
      const r = await fetch(
        `/api/chapter/${encodeURIComponent(window.CHAPTER_KEY)}/publish`,
        { method: 'POST', credentials: 'include' }
      ).then(r => r.json());
      if (r?.ok) {
        const badge = $('#chapterStatusBadge');
        if (badge) badge.textContent = 'Published';
      }
    } catch (e) {
      alert('Publish failed: ' + (e.message || e));
    }
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  // Chapter HTML is rendered server-side; just show the action buttons if a
  // compilation already exists.
  if (window.INITIAL_VERSION != null) {
    document.addEventListener('DOMContentLoaded', () => {
      $('#btnPublish')?.classList.remove('hidden');
      $('#btnRecompile')?.classList.remove('hidden');
    });
  }

  $('#btnCompile')?.addEventListener('click', compileNow);
  $('#btnRecompile')?.addEventListener('click', compileNow);
  $('#btnPublish')?.addEventListener('click', publishNow);

  refreshStatus();

  // ── Gap follow-up: create prompt + redirect ───────────────────────────────
  document.addEventListener('click', async (e) => {
    const card = e.target.closest('.gap-card');
    if (!card) return;
    const question = card.dataset.question;
    if (!question) return;

    card.disabled = true;
    card.style.opacity = '0.6';

    try {
      const data = await fetch(
        `/api/chapter/${encodeURIComponent(window.CHAPTER_KEY)}/gap/create`,
        {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question }),
        }
      ).then(r => r.json());

      if (data?.url) {
        window.location.assign(data.url);
      } else {
        card.disabled = false;
        card.style.opacity = '';
        alert('Could not create prompt. Please try again.');
      }
    } catch (err) {
      card.disabled = false;
      card.style.opacity = '';
      alert('Error: ' + (err.message || err));
    }
  });
})();
