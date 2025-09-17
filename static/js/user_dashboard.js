// User Dashboard extracted scripts

// Prompt band interactions (banner, skip, CTA pulse)
(function () {
  const wrap = document.getElementById('promptWrap');
  const band = document.getElementById('promptBand');
  if (!wrap || !band) return;

  const titleEl = document.getElementById('bannerTitle');
  const chapterRow = document.getElementById('bannerChapterRow');
  const chapterSpan = document.getElementById('bannerChapter');
  const skipBtn = document.getElementById('skipBtn');

  band.addEventListener('click', (e) => {
    if (e.target.closest('.js-action')) return;
    const href = band.dataset.href;
    if (href) window.location.href = href;
  });
  band.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      const href = band.dataset.href;
      if (href) window.location.href = href;
    }
  });

  function setFreeformBanner() {
    titleEl.textContent = 'What stories do you have?';
    if (chapterRow) chapterRow.style.display = 'none';
    band.dataset.href = '/user_record/freeform';
    wrap.dataset.currentId = '';
    band.querySelector('.hover-icons')?.classList.add('hidden');
  }
  function setPromptBanner(p) {
    if (!p) return setFreeformBanner();
    titleEl.textContent = p.text || '';
    if (p.chapter) { if (chapterSpan) chapterSpan.textContent = p.chapter; if (chapterRow) chapterRow.style.display = ''; }
    else if (chapterRow) { chapterRow.style.display = 'none'; }
    band.dataset.href = `/user_record/${p.id}`;
    wrap.dataset.currentId = String(p.id);
    band.querySelector('.hover-icons')?.classList.remove('hidden');
  }

  skipBtn?.addEventListener('click', async (e) => {
    e.preventDefault(); e.stopPropagation();
    skipBtn.disabled = true; skipBtn.style.opacity = '0.6';
    try {
      const r = await fetch('/api/skip_prompt', { method:'POST', credentials:'include' });
      const data = await r.json().catch(() => ({}));
      if (r.ok && data && data.next_id) {
        const pr = await fetch(`/api/prompt/${data.next_id}`, { credentials:'include' });
        if (pr.ok) { const p = await pr.json(); setPromptBanner(p); }
        else { window.location.href = `/user_dashboard?prompt_id=${data.next_id}`; }
      } else { setFreeformBanner(); }
    } catch { setFreeformBanner(); }
    finally { skipBtn.disabled = false; skipBtn.style.opacity = ''; }
  });

  // CTA pulse (one-time)
  const pill = band.querySelector('.prompt-cta');
  const KEY = 'memories_prompt_cta_seen';
  const prefersReduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (pill && !localStorage.getItem(KEY) && !prefersReduced) { pill.classList.add('pulse'); }
  const markSeen = () => { band.classList.add('seen'); pill?.classList.remove('pulse'); try { localStorage.setItem(KEY, '1'); } catch {} };
  band.addEventListener('pointerenter', markSeen, { once:true });
  band.addEventListener('focus',        markSeen, { once:true });
  band.addEventListener('click',        markSeen, { once:true });
})();

// Chapter progress + legend + CTA hover glow
(() => {
  const injected = (window.USER_CHAPTER_PROGRESS && Array.isArray(window.USER_CHAPTER_PROGRESS)) ? window.USER_CHAPTER_PROGRESS : null;
  const clamp = (v, lo=0, hi=100) => Math.max(lo, Math.min(hi, v));
  const pct    = (a,b) => (b > 0 ? (a / b) * 100 : 0);
  const hashHue = (s) => { let h=0; for (let i=0;i<s.length;i++) h=(h*31 + s.charCodeAt(i))&0xfffffff; return h%360; };
  const fallbackColor = (key,name) => `hsl(${hashHue(key||name||"chapter")} 90% 60%)`;

  async function fetchMeta() {
    try { const r = await fetch('/api/chapters_meta', { credentials:'include' }); if (!r.ok) return []; return await r.json(); } catch { return []; }
  }
  async function load() {
    if (injected) return { items: injected, meta: await fetchMeta() };
    const [prog, meta] = await Promise.all([
      fetch('/api/chapters_progress', { credentials:'include' }).then(r => r.ok ? r.json() : []), fetchMeta()
    ]);
    const items = (prog || []).map(p => ({
      key: p.slug || p.key || p.name,
      name: p.display_name || p.name || p.slug,
      completed: p.completed ?? p.done ?? 0,
      total: p.total ?? p.count ?? 0,
      color: p.tint || p.color
    }));
    return { items, meta };
  }

  function render({ items, meta }){
    const orderMap = new Map((meta || []).map(m => [m.name || m.slug || 'Misc', m.order ?? 999999]));
    const tintMap  = new Map((meta || []).map(m => [m.name || m.slug || 'Misc', m.tint || null]));
    const normalized = (items || []).map(ch => {
      const name = ch.name || ch.key || 'Misc';
      const key  = ch.key || name;
      const done = ch.completed || 0;
      const tot  = ch.total || 0;
      const p    = clamp(pct(done, tot)); // within-chapter percent (for tooltip)
      const tint = tintMap.get(name);
      return { key, name, pctInChapter: p, completed: done, total: tot, color: ch.color || tint || fallbackColor(key, name) };
    }).sort((a,b) => {
      const oa = orderMap.get(a.name) ?? 999999; const ob = orderMap.get(b.name) ?? 999999; return oa === ob ? a.name.localeCompare(b.name) : oa - ob;
    });

    const totals = normalized.reduce((acc, it) => { acc.c += it.completed; acc.t += it.total; return acc; }, {c:0,t:0});
    const overall = clamp(pct(totals.c, totals.t));
    const overallEl = document.getElementById('udOverallPct');
    if (overallEl) overallEl.textContent = isFinite(overall) ? `${overall.toFixed(0)}%` : '--';

    const meter = document.getElementById('udMeter');
    if (meter) {
      meter.innerHTML = '';
      meter.setAttribute('aria-valuenow', String(overall.toFixed ? overall.toFixed(0) : overall));
      const frag = document.createDocumentFragment();
      let used = 0;
      normalized.forEach((it) => {
        // Width is share of ALL assigned prompts that are completed in this chapter
        const w = Math.max(0, Math.min(100, pct(it.completed, totals.t)));
        const seg = document.createElement('div');
        seg.className = 'ud-seg ud-seg-glass';
        seg.style.width = `${w.toFixed(4)}%`;
        seg.style.setProperty('--seg-color', it.color);
        seg.dataset.label = it.name;
        seg.dataset.pct = Math.round(w).toString(); // percent of overall
        seg.dataset.within = Math.round(it.pctInChapter).toString(); // within-chapter percent
        seg.dataset.done = String(it.completed);
        seg.dataset.total = String(it.total);
        frag.appendChild(seg); used += w;
      });
      const remainder = Math.max(0, Math.min(100, 100 - used));
      if (remainder > 0.01) { const empty = document.createElement('div'); empty.className='ud-seg ud-seg-empty'; empty.style.width = `${remainder.toFixed(4)}%`; frag.appendChild(empty); }
      meter.appendChild(frag);

      const tip = meter.querySelector('#udTip') || (() => { const t = document.createElement('div'); t.id='udTip'; t.className='ud-tip'; meter.appendChild(t); return t; })();
      function show(text, midX){ tip.textContent = text || ''; tip.style.left = text ? `${midX}px` : ''; tip.style.opacity = text ? '1' : '0'; }
      function hide(){ tip.style.opacity = '0'; }
      function midXOf(seg){ const rBar = meter.getBoundingClientRect(); const rSeg = seg.getBoundingClientRect(); return (rSeg.left - rBar.left) + (rSeg.width / 2); }
      meter.addEventListener('pointerover', (e) => { const seg = e.target.closest('.ud-seg'); if (!seg || !seg.dataset.label){ hide(); return;} show(`${seg.dataset.label} - ${seg.dataset.pct}% of total (${seg.dataset.done}/${seg.dataset.total}, ${seg.dataset.within}% in chapter)`, midXOf(seg)); });
      meter.addEventListener('pointermove', (e) => { const seg = e.target.closest('.ud-seg'); if (!seg || !seg.dataset.label){ hide(); return;} show(`${seg.dataset.label} - ${seg.dataset.pct}% of total (${seg.dataset.done}/${seg.dataset.total}, ${seg.dataset.within}% in chapter)`, midXOf(seg)); });
      meter.addEventListener('pointerout', (e) => { if (!meter.contains(e.relatedTarget)) hide(); });

      meter.addEventListener('click', (e) => { const seg = e.target.closest('.ud-seg'); if (!seg || seg.classList.contains('ud-seg-empty')) return; const key = seg.dataset.label || ''; if (key) window.location.href = `/chapter/${encodeURIComponent(key)}`; });
      meter.addEventListener('keydown', (e) => { if (e.key !== 'Enter' && e.key !== ' ') return; const seg = e.target.closest('.ud-seg'); if (!seg || seg.classList.contains('ud-seg-empty')) return; const key = seg.dataset.label || ''; if (key){ e.preventDefault(); window.location.href = `/chapter/${encodeURIComponent(key)}`; } });
    }
  }

  // CTA hover glow
  const cta = document.querySelector('.ud-glass-btn');
  if (cta) { cta.addEventListener('pointermove', (e) => { const r = cta.getBoundingClientRect(); const mx = ((e.clientX - r.left) / r.width) * 100; const my = ((e.clientY - r.top) / r.height) * 100; cta.style.setProperty('--mx', mx + '%'); cta.style.setProperty('--my', my + '%'); }); }

  load().then(render).catch(() => { render({ items:[ { key:'our-family-history', name:'Our Family History', completed:7, total:10, color:'#F472B6' }, { key:'childhood', name:'Childhood', completed:4, total:8 }, { key:'school-years', name:'School Years', completed:3, total:7 }, { key:'early-adulthood', name:'Early Adulthood', completed:6, total:12 }, { key:'parenthood', name:'Parenthood', completed:5, total:9 } ], meta: [] }); });
})();
