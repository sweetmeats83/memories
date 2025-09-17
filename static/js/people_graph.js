// static/js/people_graph.js
// People Graph — full build with sticky click-path, hover path-to-you,
// spouse edge styling (incl. user↔spouse), editor panels, roles, etc.
console.info("people_graph.js — click path & hover path build loaded");

(function () {
  const canvas = document.getElementById('graph');
  if (!canvas) return;

  // --- DOM refs (top toolbar) ---
  const tip = document.getElementById('tip');
  const inf = document.getElementById('inferred');
  const infList = document.getElementById('inf_list');
  const infClose = document.getElementById('inf_close');
  const infDock = document.getElementById('inf_dock');
  const npName = document.getElementById('np_name');
  const npRole = document.getElementById('np_role');
  const npAdd  = document.getElementById('np_add');

  // Hover mini-card removed (no corresponding HTML elements)

  // View panel
  const panelView = document.getElementById('panel_view');
  const pvName = document.getElementById('pv_name');
  const pvPhoto = document.getElementById('pv_photo');
  const pvRole = document.getElementById('pv_role');
  const pvYears = document.getElementById('pv_years');
  const pvConnect = document.getElementById('pv_connect');
  const pvDeceased = document.getElementById('pv_deceased');
  const pvBio = document.getElementById('pv_bio');
  const pvEdges = document.getElementById('pv_edges');
  const pvClose = document.getElementById('pv_close');
  const pvEdit = document.getElementById('pv_edit');
  const pgSearch = document.getElementById('pg_search');
  const pgDD = document.getElementById('pgSearchDD');

  // Edit panel
  const panel = document.getElementById('panel');
  const pName = document.getElementById('p_name');
  const pRole = document.getElementById('p_role');
  const pDisplayName = document.getElementById('p_display_name');
  const pEdges = document.getElementById('p_edges');
  const pTarget = document.getElementById('p_target');
  const pRel = document.getElementById('p_rel');
  const pSubjectLabel = document.getElementById('p_subject_label');
  const pAdd = document.getElementById('p_add');
  const pDeceased = document.getElementById('p_deceased');
  const pBirth = document.getElementById('p_birth');
  const pDeath = document.getElementById('p_death');
  const pBio = document.getElementById('p_bio');
  const pPhotoImg = document.getElementById('p_photo_img');
  const pPhotoFile = document.getElementById('p_photo_file');
  const pPhotoRemove = document.getElementById('p_photo_remove');
  const pDeletePerson = document.getElementById('p_delete_person');

  const ctx = canvas.getContext('2d');

  // Reset all edit-panel fields to a clean state (pre-hydration)
  function resetEditPanel() {
    try {
      if (pDisplayName) pDisplayName.value = '';
      if (pRole && pRole.options && pRole.options.length) pRole.selectedIndex = 0;
      if (pBirth) pBirth.value = '';
      const pBirthFull = document.getElementById('p_birth_full');
      if (pBirthFull) pBirthFull.value = '';
      if (pDeceased) pDeceased.checked = false;
      if (pDeath) { pDeath.value = ''; pDeath.classList.add('hidden'); pDeath.style.display = 'none'; }
      if (pBio) { pBio.value = ''; try { pBio.style.height = 'auto'; } catch {} }
      if (pPhotoImg) { pPhotoImg.removeAttribute('src'); pPhotoImg.style.display = 'none'; }
      if (pPhotoRemove) pPhotoRemove.style.display = 'none';
      if (pPhotoFile) pPhotoFile.style.display = 'block';
      if (pEdges) pEdges.innerHTML = '';
    } catch {}
  }

  // DPR & resize
  let DPR = window.devicePixelRatio || 1;
  function resize() {
    const w = window.innerWidth, h = window.innerHeight;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    canvas.width = Math.floor(w * DPR);
    canvas.height = Math.floor(h * DPR);
  }
  window.addEventListener('resize', resize);
  resize();

  // Visual/physics cfg
  const CFG = {
    nodeRadius: 6,
    hoverRadius: 10,
    userScale: 1.4,
    linkColor: 'rgba(0,0,0,0.25)',
    nodeColor: '#374151',
    deadColor: '#9ca3af',
    userColor: '#111111',
    nodeHalo: 'rgba(0,0,0,0.10)',
    focusColor: '#eb9525ff',
    labelColor: '#111111',
    repel: 3750,
    spring: 0.045,
    friction: 0.92,
    centerK: 0.0012,
    centerKSelected: 0.02,
    wallMargin: 40,
    wallK: 0.025,
    noise: 0.06,
    fadeAlpha: 0.18,
    maxSpeed: 120,
    // orbit/jitter
    degreeCenterBoost: 0.015,
    jitterAmp: 0.35,
    jitterFreqMin: 0.3,
    jitterFreqMax: 0.8,
    driftK: 0.0020,
    driftRadiusBase: 325,
    driftRadiusVar: 175,
    driftFreqMin: 0.010,
    driftFreqMax: 0.035,
    // Pet relationship tuning
    petSpringScale: 0.28,   // how strongly pets are pulled vs default spring
    petPathWeight: 3.0,     // path cost for pet edges (higher = less preferred)
    // Idle wandering (waypoint steering)
    idleTimeoutMs: 5000,
    wanderK: 0.0030,
    wanderIntervalMin: 6.0,   // seconds
    wanderIntervalMax: 12.0,  // seconds
    // Extra pull for user's immediate family
    meSpouseBoost: 1.20,
    meChildBoost: 1.15,
    // Cursor attraction
    cursorAttractK: 0.0025,
    cursorInfluence: 320,
  };
  const PET_COLOR = '#cc5500'; // burnt orange for pets

  // App state
  const state = {
    nodes: [],
    edges: [],
    edgesDraw: [],
    edgesPC: [],
    edgesUserCore: [],
    adj: new Map(),
    adjPath: new Map(),
    adjBio: new Map(),
    adjW: new Map(),
    hoverId: null,
    dragId: null,
    selectedId: null,
    cam: { s: 1.0 },

    // Sticky (from clicks) and Hover (ephemeral) path layers
    pathEdgesSticky: [],
    pathNodesSticky: new Set(),
    pathEdgesHover: [],
    pathNodesHover: new Set(),

    // Click path anchor rotation (A -> B -> C ...)
    anchorId: null,
    cursor: { x: 0, y: 0, active: false }
  };
  let lastInteract = performance.now();
  function markInteract(){ lastInteract = performance.now(); }
  const SID = (x) => String(x);
  state.meId = canvas.getAttribute('data-me-id') || (window.state && window.state.meId) || '';

  // Helpers
  function relLabel(rt) {
    let s = String(rt || '');
    s = s.replace(/^(relationship:|role:)/i, '').replace(/-of$/i, '').replace(/-/g, ' ');
    return s;
  }

  // Kinship label cache and fetcher (between two person ids)
  const kinCache = new Map(); // key: "ego|alter" (directional), val: string label
  async function fetchKinshipLabel(aId, bId) {
    const a = parseInt(aId, 10), b = parseInt(bId, 10);
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    const key = `${a}|${b}`; // cache is directional: label of alter (b) relative to ego (a)
    if (kinCache.has(key)) return kinCache.get(key);
    try {
      const r = await fetch(`/api/people/kinship2/${a}/${b}`, { credentials: 'include', cache: 'no-store' });
      const j = await r.json();
      const ln = (j && j.label_neutral) || null;
      const lg = (j && j.label_gendered) || null;
      // Prefer neutral for aunt/uncle and niece/nephew families (incl. grand/great forms)
      const nlc = String(ln || '').toLowerCase();
      const preferNeutral = (
        nlc.includes('aunt/uncle') || nlc.includes('niece/nephew')
      );
      const lbl = preferNeutral ? (ln || lg) : (lg || ln);
      if (lbl) kinCache.set(key, lbl);
      return lbl;
    } catch { return null; }
  }

  async function ensureRoles() {
    if (!Array.isArray(window.__ROLES__) || !window.__ROLES__.length) {
      try {
        const r = await fetch('/api/roles', { credentials: 'include' });
        const j = await r.json();
        window.__ROLES__ = j.roles || [];
      } catch {
        window.__ROLES__ = [];
      }
    }
    // Ensure a Pet role exists even if backend doesn't provide it yet
    try {
      const hasPet = (window.__ROLES__ || []).some(rr => String(rr?.slug||'').toLowerCase() === 'role:pet');
      if (!hasPet) {
        window.__ROLES__.push({ slug: 'role:pet', label: 'Pet' });
      }
    } catch {}
    return window.__ROLES__;
  }

  // Populate the toolbar datalist for quick-add role typing
  async function renderToolbarRoles() {
    try {
      const roles = await ensureRoles();
      const dl = document.getElementById('np_roles_list');
      if (!dl) return;
      dl.innerHTML = '';
      roles.forEach(r => {
        if (r && r.slug && r.slug.startsWith('role:')) {
          const opt = document.createElement('option');
          opt.value = r.slug.split(':',1)[0] === 'role' ? r.slug.split(':',2)[1] : r.label || r.slug;
          // Prefer the human label when present
          if (r.label) opt.label = r.label;
          dl.appendChild(opt);
        }
      });
      // Add common helpful entries
      ['you','mother','father','parent','spouse','partner','sibling','friend','neighbor','pet'].forEach(v => {
        const opt = document.createElement('option'); opt.value = v; dl.appendChild(opt);
      });
    } catch {}
  }

  // Category/channel
  function isParentChild(r) {
    const x = String(r || '').toLowerCase();
    return ['parent-of', 'child-of', 'mother-of', 'father-of', 'son-of', 'daughter-of'].includes(x);
  }
  function isSpouse(r) {
    const x = String(r || '').toLowerCase();
    return ['spouse-of', 'partner-of', 'ex-partner-of'].includes(x);
  }
  function isSibling(r) {
    const x = String(r || '').toLowerCase();
    return ['sibling-of','brother-of','sister-of','half-sibling-of','step-sibling-of','sibling','brother','sister'].includes(x);
  }
  function roleToCat(role) {
    const r = String(role || '').toLowerCase();
    if (['mother','father','parent','son','daughter','child'].includes(r)) return 'family';
    if (['spouse','partner','husband','wife'].includes(r)) return 'spouse';
    if (['pet'].includes(r)) return 'pet';
    if (['friend','neighbor','coworker','colleague','mentor','teacher','student'].includes(r)) return 'social';
    return 'other';
  }
  function relToCat(rel) {
    const r = String(rel || '').toLowerCase();
    if (isParentChild(r)) return 'family';
    if (isSpouse(r)) return 'spouse';
    if (r.endsWith('-of')) return roleToCat(r.replace(/-of$/, ''));
    return roleToCat(r);
  }
  function channelOf(rel) {
    const r = String(rel || '').toLowerCase();
    if (isParentChild(r)) return 'bio';
    if (isSpouse(r)) return 'spouse';
    if (isSibling(r)) return 'sibling';
    if (r === 'pet-of' || r === 'owner-of' || r === 'pet') return 'pet';
    if (r.endsWith('-in-law-of') || r === 'parent-in-law-of' || r === 'child-in-law-of') return 'affinal';
    if (['grandparent-of','grandchild-of','aunt-of','uncle-of','niece-of','nephew-of','cousin-of'].includes(r)) return 'extended';
    if (['friend-of','neighbor-of','coworker-of','mentor-of','teacher-of','student-of'].includes(r)) return 'social';
    return 'other';
  }
  function weightForChannel(chan) {
    switch (chan) {
      case 'bio': return 1.0;
      case 'spouse': return 0.4;
      case 'sibling': return 2.0;
      case 'affinal': return 1.6;
      case 'extended': return 2.0;
      case 'pet': return (CFG.petPathWeight || 1.0);
      case 'social': return 2.6;
      default: return 1.8;
    }
  }

  function randPos() {
    const w = canvas.width, h = canvas.height;
    const cx = w / 2, cy = h / 2;
    return [cx + (Math.random() - 0.5) * w * 0.35, cy + (Math.random() - 0.5) * h * 0.35];
  }

  function normalizeRel(raw){
    let r = String(raw || '').toLowerCase().trim();
    r = r.replace(/^relationship:/, '').replace(/^role:/, '');
    const base = new Set([
      'mother','father','parent','son','daughter','child',
      'spouse','partner','husband','wife',
      'brother','sister','sibling','half-sibling','step-sibling',
      'ex-partner','pet'
    ]);
    if (base.has(r) && !r.endsWith('-of')) r = r + '-of';
    if (r === 'wife-of' || r === 'husband-of') r = 'spouse-of';
    if (r === 'ex-partner-of' && !r.endsWith('-of')) r = 'ex-partner-of';
    return r;
  }

  function build(nodes, edges) {
    const map = new Map();
    nodes.forEach(n => {
      const [x, y] = randPos();
      const jphi = Math.random() * Math.PI * 2;
      const jw = (CFG.jitterFreqMin + Math.random() * (CFG.jitterFreqMax - CFG.jitterFreqMin)) * 2 * Math.PI;
      const dw = (CFG.driftFreqMin + Math.random() * (CFG.driftFreqMax - CFG.driftFreqMin)) * 2 * Math.PI;
      const rbase = CFG.driftRadiusBase + (Math.random() - 0.5) * 2 * CFG.driftRadiusVar;
      map.set(n.id, {
        id: n.id,
        label: n.name || n.display_name || ('#' + n.id),
        kind: String(n.kind || 'person').toLowerCase(),
        dead: !!(n.dead || n.deceased),
        color: n.color || n.dot_color || null,
        role: (n.role_hint || n.role || '').toString().toLowerCase(),
        x, y, vx: (Math.random()-0.5)*20, vy: (Math.random()-0.5)*20, fixed: false,
        jphi, jw, dw, rbase,
        // wandering waypoint (idle mode)
        wx: x, wy: y, nextWanderAt: 0
      });
    });

    // No synthetic user nodes; anchor is a regular person with id == state.meId

    // Build all edges
    const all = [];
    (edges || []).forEach(e => {
      if (map.has(e.src) && map.has(e.dst)) {
        const relNorm = normalizeRel(e.rel);
        const cat  = relToCat(relNorm);
        const chan = channelOf(relNorm);
        let k = CFG.spring;
        if (cat === 'family') k = CFG.spring * 1.15;
        else if (cat === 'spouse') k = CFG.spring * 1.35; // pull spouses closer together
        else if (cat === 'pet') k = CFG.spring * (CFG.petSpringScale || 0.88);
        else if (cat === 'social') k = CFG.spring * 0.35;
        else k = CFG.spring * 0.6;
        const w = weightForChannel(chan);
        // If this edge touches the user's person, add extra pull for spouse/children,
        // and reduce pull by generation for distant kin.
        try {
          const me = state.meId ? String(state.meId) : null;
          const touchesMe = me && ((String(e.src) === me) || (String(e.dst) === me));
          if (touchesMe) {
            if (relNorm === 'spouse-of') k *= (CFG.meSpouseBoost || 1.2);
            if (relNorm === 'parent-of' || relNorm === 'child-of') k *= (CFG.meChildBoost || 1.15);
            const gen = (typeof e.generation === 'number') ? e.generation : (e.generation ? parseInt(e.generation,10) : null);
            if (gen && gen >= 2) {
              const factor = Math.pow(0.5, Math.max(0, gen - 1));
              k = k * factor;
            }
          }
        } catch {}
        all.push({ s: e.src, d: e.dst, rel: relNorm, cat, chan, k, w });
      }
    });

    state.nodes = Array.from(map.values());
    state.edges = all;

    // meId: trust provided data-me-id (person id); only keep it if present in nodes
    let me = state.meId;
    if (!me || !map.has(me)) { me = null; }
    state.meId = me;

    // Unweighted adjacency

    const adj = new Map();
    state.nodes.forEach(n => adj.set(SID(n.id), new Set()));
    all.forEach(e => {
      const s = SID(e.s), d = SID(e.d);
      adj.get(s)?.add(d);
      adj.get(d)?.add(s);
    });
    state.adj = adj;

    // --- Parent/child subset ---
    const pc = all.filter(e => isParentChild(e.rel));
    state.edgesPC = pc;

    // --- Path-to-me adjacency: parent/child + anchor core (parents/children/spouse) ---
    const adjPath = new Map();
    state.nodes.forEach(n => adjPath.set(SID(n.id), new Set()));
    pc.forEach(e => {
      const s = SID(e.s), d = SID(e.d);
      adjPath.get(s)?.add(d);
      adjPath.get(d)?.add(s);
    });
    const meIdStr = state.meId ? SID(state.meId) : null;
    const anchorLinks = meIdStr ? all.filter(e => SID(e.s) === meIdStr || SID(e.d) === meIdStr) : [];
    const anchorCore  = anchorLinks.filter(e => isParentChild(e.rel) || isSpouse(e.rel));
    anchorCore.forEach(e => {
      const s = SID(e.s), d = SID(e.d);
      adjPath.get(s)?.add(d);
      adjPath.get(d)?.add(s);
    });
    state.edgesUserCore = anchorCore;
    state.adjPath = adjPath;

    // --- Weighted adjacency (string keys) ---
    const adjW = new Map();
    state.nodes.forEach(n => adjW.set(SID(n.id), []));
    all.forEach(e => {
      const s = SID(e.s), d = SID(e.d);
      adjW.get(s)?.push({ v: d, w: e.w, chan: e.chan });
      adjW.get(d)?.push({ v: s, w: e.w, chan: e.chan });
    });
    state.adjW = adjW;

    // --- Bio-only adjacency (string keys) ---
    const adjBio = new Map();
    state.nodes.forEach(n => adjBio.set(SID(n.id), new Set()));
    pc.forEach(e => {
      const s = SID(e.s), d = SID(e.d);
      adjBio.get(s)?.add(d);
      adjBio.get(d)?.add(s);
    });
    anchorLinks.forEach(e => {
      if (isParentChild(e.rel)) {
        const s = SID(e.s), d = SID(e.d);
        adjBio.get(s)?.add(d);
        adjBio.get(d)?.add(s);
      }
    });
    state.adjBio = adjBio;

    // Ensure meId is a string
    if (state.meId) state.meId = SID(state.meId);


    // Edge layers to draw (order matters)
    const isUserId = (id) => {
      const sid = String(id);
      if (sid.startsWith('user:')) return true;
      const rec = map.get(id) || map.get(sid);
      return rec && String(rec.kind||'').toLowerCase() === 'user';
    };
    const spousePerson = all.filter(e => isSpouse(e.rel) && !isUserId(e.s) && !isUserId(e.d));
    const socialPerson = all.filter(e => e.cat === 'social' && !isUserId(e.s) && !isUserId(e.d));
    const petPerson    = all.filter(e => e.cat === 'pet'    && !isUserId(e.s) && !isUserId(e.d));
    const userSocial = all.filter(e => (isUserId(e.s) || isUserId(e.d)) && e.cat === 'social');
    const userPet    = all.filter(e => (isUserId(e.s) || isUserId(e.d)) && e.cat === 'pet');
    const userSibling = all.filter(e => (isUserId(e.s) || isUserId(e.d)) && isSibling(e.rel));
    const sibRaw = all.filter(e => isSibling(e.rel) && !isUserId(e.s) && !isUserId(e.d));
    const seenSib = new Set();
    const siblingPerson = [];
    for (const e of sibRaw) {
      const a = String(e.s), b = String(e.d);
      const key = a < b ? `${a}|${b}` : `${b}|${a}`;
      if (seenSib.has(key)) continue;
      seenSib.add(key); siblingPerson.push(e);
    }
    // include userCore AFTER pc so it sits on top, and includes user↔spouse
    state.edgesDraw = pc.concat(spousePerson, socialPerson, petPerson, anchorCore, userSocial, userPet, userSibling, siblingPerson);

    // Precompute pet nodes (by role or by being src of pet-of)
    const petNodes = new Set();
    try {
      for (const e of all) { if (String(e.rel||'').toLowerCase() === 'pet-of') petNodes.add(SID(e.s)); }
      for (const n of state.nodes) { if (String(n.role||'').toLowerCase() === 'pet') petNodes.add(SID(n.id)); }
    } catch {}
    state.petNodes = petNodes;

    const empty = document.getElementById('empty');
    if (empty) empty.style.display = state.nodes.length ? 'none' : 'block';
  }

  // ----- Path helpers (sticky & hover) -----
  function setStickyPath(seq){
    state.pathNodesSticky = new Set(seq || []);
    const pe = [];
    for (let i=0;i<seq.length-1;i++) pe.push({ s: seq[i], d: seq[i+1] });
    state.pathEdgesSticky = pe;
  }
  function clearStickyPath(){
    state.pathNodesSticky = new Set();
    state.pathEdgesSticky = [];
  }
  function setHoverPath(seq){
    state.pathNodesHover = new Set(seq || []);
    const pe = [];
    for (let i=0;i<seq.length-1;i++) pe.push({ s: seq[i], d: seq[i+1] });
    state.pathEdgesHover = pe;
  }
  function clearHoverPath(){
    state.pathNodesHover = new Set();
    state.pathEdgesHover = [];
  }

  function getNode(id) {
    const sid = String(id);
    return state.nodes.find(n => String(n.id) === sid);
  }

  // Physics
  function step() {
    const nodes = state.nodes;
    const edges = state.edges;
    const selectedId = state.selectedId;
    const w = canvas.width, h = canvas.height;
    const cx = w/2, cy = h/2;
    const wall = CFG.wallMargin * DPR;
    const freezeId = state.dragId ? null : (state.hoverId ?? null);
    if (freezeId) {
      const hn = getNode(freezeId);
      if (hn) { hn.vx = 0; hn.vy = 0; }
    }
    const deg = new Map();
    for (const n of nodes) deg.set(n.id, 0);
    for (const e of edges) { deg.set(e.s, (deg.get(e.s)||0)+1); deg.set(e.d, (deg.get(e.d)||0)+1); }
    const now = performance.now();
    const t = now / 1000;
    const idle = (!state.dragId && !state.selectedId && (now - lastInteract) > CFG.idleTimeoutMs);

    // repulsion
    for (let i=0;i<nodes.length;i++) {
      for (let j=i+1;j<nodes.length;j++) {
        const a = nodes[i], b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let d2 = dx*dx + dy*dy; if (d2 < 1) d2 = 1;
        const f = CFG.repel / d2;
        const fx = f * dx, fy = f * dy;
        if (!a.fixed && SID(a.id) !== freezeId) { a.vx += fx; a.vy += fy; }
        if (!b.fixed && SID(b.id) !== freezeId) { b.vx -= fx; b.vy -= fy; }
      }
    }
    // cursor attraction (gentle, distance-attenuated)
    if (state.cursor && state.cursor.active && !state.dragId) {
      const cxw = state.cursor.x, cyw = state.cursor.y;
      const rinfl = (CFG.cursorInfluence || 320) * DPR;
      for (const n of nodes) {
        if (n.fixed || (freezeId && SID(n.id) === freezeId)) continue;
        const dx = cxw - n.x, dy = cyw - n.y;
        const dist = Math.max(1, Math.hypot(dx, dy));
        const fall = 1 / (1 + dist / rinfl);
        const kcur = (CFG.cursorAttractK || 0.0025) * fall;
        n.vx += dx * kcur;
        n.vy += dy * kcur;
      }
    }
    // springs
    for (const e of edges) {
      const a = getNode(e.s), b = getNode(e.d);
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const kk = (e.k != null ? e.k : CFG.spring);
      const fx = dx * kk, fy = dy * kk;
      if (!a.fixed && SID(a.id) !== freezeId) { a.vx += fx; a.vy += fy; }
      if (!b.fixed && SID(b.id) !== freezeId) { b.vx -= fx; b.vy -= fy; }
    }
    // center, rings when selected, drift, walls, jitter
    for (const n of nodes) {
      if (n.fixed || (freezeId && SID(n.id) === freezeId)) continue;
      const kBase = CFG.centerK + (CFG.degreeCenterBoost * (deg.get(n.id)||0));
      let k = kBase;
      if (selectedId && n.id === selectedId) { k = CFG.centerKSelected; }
      n.vx += (cx - n.x) * k;
      n.vy += (cy - n.y) * k;

      if (selectedId && n.id !== selectedId) {
        let r0 = (CFG.driftRadiusBase - (deg.get(n.id)||0) * 26) * DPR;
        if (r0 < 80*DPR) r0 = 80*DPR;
        const dx = n.x - cx, dy = n.y - cy;
        const dist = Math.max(1, Math.hypot(dx, dy));
        const dr = (r0 - dist);
        const ux = dx / dist, uy = dy / dist;
        n.vx += ux * dr * 0.0025;
        n.vy += uy * dr * 0.0025;
      } else if (!selectedId) {
        if (idle) {
          // Idle wandering: steer toward per-node waypoint, pick new target periodically
          if (!n.nextWanderAt || t >= n.nextWanderAt || Math.hypot((n.wx - n.x), (n.wy - n.y)) < 20*DPR) {
            const nx = (Math.random()*0.8 + 0.1) * w;  // avoid edges
            const ny = (Math.random()*0.8 + 0.1) * h;
            n.wx = nx; n.wy = ny;
            const dt = CFG.wanderIntervalMin + Math.random() * (CFG.wanderIntervalMax - CFG.wanderIntervalMin);
            n.nextWanderAt = t + dt;
          }
          n.vx += (n.wx - n.x) * (CFG.wanderK || 0.0030);
          n.vy += (n.wy - n.y) * (CFG.wanderK || 0.0030);
        } else {
          // Default gentle orbit drift about center
          const theta = n.jphi + t * (n.dw || 0);
          const r = ((n.rbase != null ? n.rbase : CFG.driftRadiusBase)) * DPR;
          const tx = cx + r * Math.cos(theta);
          const ty = cy + r * Math.sin(theta);
          n.vx += (tx - n.x) * (CFG.driftK || 0);
          n.vy += (ty - n.y) * (CFG.driftK || 0);
        }
      }

      const j = CFG.jitterAmp * DPR;
      n.vx += Math.cos(n.jphi + t * n.jw) * j;
      n.vy += Math.sin(n.jphi + t * n.jw) * j;

      if (n.x < wall) n.vx += (wall - n.x) * CFG.wallK;
      if (n.x > w - wall) n.vx -= (n.x - (w - wall)) * CFG.wallK;
      if (n.y < wall) n.vy += (wall - n.y) * CFG.wallK;
      if (n.y > h - wall) n.vy -= (n.y - (h - wall)) * CFG.wallK;

      n.vx += (Math.random() - 0.5) * CFG.noise;
      n.vy += (Math.random() - 0.5) * CFG.noise;
      n.vx *= CFG.friction; n.vy *= CFG.friction;
      const sp2 = n.vx*n.vx + n.vy*n.vy;
      const maxV = CFG.maxSpeed * DPR;
      if (sp2 > maxV*maxV) { const s = Math.sqrt(sp2) || 1; n.vx = n.vx / s * maxV; n.vy = n.vy / s * maxV; }
    }
    // integrate
    for (const n of nodes) {
      if (n.fixed || (freezeId && SID(n.id) === freezeId)) continue;
      n.x += n.vx * 0.016; n.y += n.vy * 0.016;
    }
  }

  function loop() { step(); draw(); requestAnimationFrame(loop); }

  // Screen→World mapping (accounts for canvas CSS size & DPR)
  function screenToWorld(px, py) {
    const rect = canvas.getBoundingClientRect();
    const sx = (px - rect.left) * DPR;
    const sy = (py - rect.top)  * DPR;
    const w = canvas.width, h = canvas.height;
    const s = state.cam.s || 1.0;
    const xw = (sx - w/2) / s + w/2;
    const yw = (sy - h/2) / s + h/2;
    return [xw, yw];
  }

  function pickNode(px, py) {
    const [x, y] = screenToWorld(px, py);
    const s = state.cam.s || 1.0;
    const R = 14 * DPR * s; // scale pick radius with zoom
    let best = null, bestD2 = Infinity;
    for (const n of state.nodes) {
      const dx = n.x - x, dy = n.y - y, d2 = dx*dx + dy*dy;
      if (d2 < R*R && d2 < bestD2) { best = n; bestD2 = d2; }
    }
    return best;
  }

  // --- Path search ---
  function bfsPath(adj, s, t) {
    const S = String(s), T = String(t);
    const q = [S], seen = new Set([S]), prev = new Map();
    while (q.length) {
      const u = q.shift(); if (u === T) break;
      const nbrs = adj.get(u) || new Set();
      for (const v of nbrs) {
        const vs = String(v);
        if (seen.has(vs)) continue;
        seen.add(vs); prev.set(vs, u); q.push(vs);
      }
    }
    if (!prev.has(T) && S !== T) return null;
    const path = [T]; let cur = T;
    while (prev.has(cur)) { cur = prev.get(cur); path.push(cur); }
    if (path[path.length-1] !== S) { if (S === T) return [S]; return null; }
    path.reverse(); return path;
  }
  function shortestPathToMe(srcId){
    const me = state.meId && getNode(state.meId) ? state.meId : null;
    if (!me) return null;
    // parent/child + user core (includes spouse ↔ user)
    return bfsPath(state.adjPath || state.adj, srcId, me);
  }
  // Removed legacy alias shortestPathToMeBio (unused)

  function shortestPathBetweenBio(srcId, dstId){
    return bfsPath(state.adjBio || state.adj, srcId, dstId);
  }
  function shortestPathBetweenWeighted(srcId, dstId){
    const adj = state.adjW; if (!adj) return null;
    const s = String(srcId), t = String(dstId);
    const dist = new Map(), prev = new Map();
    const pq = []; function push(n,d){ pq.push({n,d}); pq.sort((a,b)=>a.d-b.d); }
    state.nodes.forEach(n => dist.set(String(n.id), Infinity));
    dist.set(s, 0); push(s, 0);
    const seen = new Set();
    while (pq.length) {
      const {n, d} = pq.shift();
      if (seen.has(n)) continue; seen.add(n);
      if (n === t) break;
      const nbrs = adj.get(n) || [];
      for (const e of nbrs) {
        const v = String(e.v), nd = d + (e.w || 1);
        if (nd < (dist.get(v) ?? Infinity)) { dist.set(v, nd); prev.set(v, n); push(v, nd); }
      }
    }
    if (!prev.has(t) && s !== t) return null;
    const path = [t]; let cur = t;
    while (prev.has(cur)) { cur = prev.get(cur); path.push(cur); }
    if (path[path.length-1] !== s) { if (s === t) return [s]; return null; }
    path.reverse(); return path;
  }

  // ----- Drawing -----
  function draw() {
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const s = state.cam.s || 1.0;
    const hoverId = state.hoverId;
    const selectedId = state.selectedId;
    const neighbors = hoverId && state.adj.get(hoverId) ? state.adj.get(hoverId) : new Set();
    const filtering = state.filterText && state.filterIds && state.filterIds.size > 0;

    function sx(x) { return (x - w/2) * s + w/2; }
    function sy(y) { return (y - h/2) * s + h/2; }

    // Choose which path to glow: sticky (from clicks) overrides hover
    const glowEdges = (state.pathEdgesSticky.length ? state.pathEdgesSticky : state.pathEdgesHover);
    const hasGlow = glowEdges.length > 0;
    const pathKey = new Set();
    for (const pe of glowEdges) {
      pathKey.add(`${pe.s}|${pe.d}`);
      pathKey.add(`${pe.d}|${pe.s}`);
    }

    // Edges (styled)
    for (const e of (state.edgesDraw || state.edges)) {
      const a = getNode(e.s), b = getNode(e.d);
      if (!a || !b) continue;
      const focused = !!hoverId && (String(e.s) === String(hoverId) || String(e.d) === String(hoverId));
      const onPath = pathKey.has(`${e.s}|${e.d}`) || pathKey.has(`${e.d}|${e.s}`);

      ctx.save();
      const rel = String(e.rel || '').toLowerCase();
      if (isSibling(rel)) {
        ctx.strokeStyle = '#ec4899'; // pink
        ctx.lineWidth = 1.2 * DPR;
        ctx.setLineDash([]);
        ctx.globalAlpha = hasGlow ? (onPath ? 0.9 : 0.12)
                        : (hoverId ? (focused ? 0.95 : 0.5) : 0.7);
      } else if (e.cat === 'social') {
        ctx.strokeStyle = '#111';
        ctx.lineWidth = 1.1 * DPR;
        ctx.setLineDash([3*DPR, 2*DPR]);
        ctx.globalAlpha = hasGlow ? (onPath ? 0.8 : 0.08)
                        : (hoverId ? (focused ? 0.8 : 0.4) : 0.45);
      } else if (e.cat === 'spouse') {
        ctx.strokeStyle = 'rgba(37, 99, 235, 0.75)'; // blue
        ctx.lineWidth = 1.6 * DPR;
        ctx.setLineDash([]);
        ctx.globalAlpha = hasGlow ? (onPath ? 0.9 : 0.12)
                        : (hoverId ? (focused ? 0.9 : 0.55) : 0.7);
      } else if (e.cat === 'pet') {
        ctx.strokeStyle = PET_COLOR; // burnt orange
        ctx.lineWidth = 1.1 * DPR;
        ctx.setLineDash([3*DPR, 2*DPR]); // treat like acquaintances
        ctx.globalAlpha = hasGlow ? (onPath ? 0.85 : 0.10)
                        : (hoverId ? (focused ? 0.85 : 0.4) : 0.45);
      } else {
        ctx.strokeStyle = CFG.linkColor;
        ctx.lineWidth = 1.2 * DPR;
        ctx.setLineDash([]);
        ctx.globalAlpha = hasGlow ? (onPath ? 0.9 : 0.12)
                        : (hoverId ? (focused ? 0.9 : CFG.fadeAlpha) : 0.6);
      }
      ctx.beginPath();
      ctx.moveTo(sx(a.x), sy(a.y));
      ctx.lineTo(sx(b.x), sy(b.y));
      // Dim edges not touching a filtered node when filtering
      if (filtering) {
        const touches = state.filterIds.has(String(e.s)) || state.filterIds.has(String(e.d));
        if (!touches) ctx.globalAlpha *= 0.2;
      }
      ctx.stroke();
      ctx.restore();
    }

    // Path glow (gold)
    if (hasGlow) {
      ctx.save();
      ctx.lineWidth = 3 * DPR;
      ctx.strokeStyle = '#f59e0b';
      ctx.shadowColor = '#f59e0b';
      ctx.shadowBlur = 12 * DPR;
      for (const e of glowEdges) {
        const a = getNode(e.s), b = getNode(e.d);
        if (!a || !b) continue;
        ctx.beginPath();
        ctx.moveTo(sx(a.x), sy(a.y));
        ctx.lineTo(sx(b.x), sy(b.y));
        ctx.stroke();
      }
      ctx.restore();
    }

    // Nodes
    for (const n of state.nodes) {
      const isHover = hoverId === String(n.id);
      const isNeighbor = !!hoverId && neighbors.has(String(n.id));
      const isSelected = selectedId === n.id;
      const baseR = CFG.nodeRadius * ((state.meId && SID(n.id) === SID(state.meId)) ? CFG.userScale : 1);
      const r = (isHover ? CFG.hoverRadius : baseR) * DPR * s;

      // Halo for hover OR selected
      const isAnchor = state.anchorId && (state.anchorId === SID(n.id));
      if (isHover || isSelected || isAnchor) {
        ctx.beginPath();
        ctx.fillStyle = isAnchor ? 'rgba(245, 158, 11, 0.25)' : CFG.nodeHalo;
        ctx.arc(sx(n.x), sy(n.y), r*1.8, 0, Math.PI*2);
        ctx.fill();
      }

      // Dot
      ctx.beginPath();
      let fill = CFG.nodeColor;
      if (state.meId && SID(n.id) === SID(state.meId)) fill = CFG.userColor;
      if (n.kind === 'person' && n.dead) fill = CFG.deadColor;
      if ((isHover || isSelected) && !(n.kind === 'person' && n.dead)) fill = CFG.focusColor;
      if (n.kind === 'person' && !n.dead) {
        if (n.color) fill = n.color;
        else if (state.petNodes && state.petNodes.has(SID(n.id))) fill = PET_COLOR;
      }
      ctx.fillStyle = fill;
      ctx.save();
      ctx.globalAlpha = hoverId ? (isHover || isNeighbor ? 1.0 : 0.55) : 1.0;
      if (filtering && !state.filterIds.has(String(n.id))) ctx.globalAlpha *= 0.25;
      ctx.arc(sx(n.x), sy(n.y), r, 0, Math.PI*2);
      ctx.fill();
      ctx.restore();

      // Label
      ctx.save();
      ctx.fillStyle = CFG.labelColor;
      ctx.font = `${Math.max(10, 12*DPR*s)}px system-ui, -apple-system, Segoe UI, Roboto, Arial`;
      ctx.textBaseline = 'bottom';
      const label = (n.label || '').toString();
      if (label) {
        const tx = sx(n.x) + (r + 6*DPR);
        const ty = sy(n.y) - 6*DPR;
        ctx.globalAlpha = (isHover || isSelected) ? 0.95 : 0.6;
        ctx.fillText(label, tx, ty);
      }
      ctx.restore();
    }
  }

  // ----- Events -----

  // Hover: show halo + gold path-to-you ONLY when no sticky path is active
  canvas.addEventListener('mousemove', (ev) => {
    markInteract();
    try {
      const n = pickNode(ev.clientX, ev.clientY);
      state.hoverId = n ? SID(n.id) : null;
      // Track cursor in world coords for attraction
      const [xw, yw] = screenToWorld(ev.clientX, ev.clientY);
      state.cursor.x = xw; state.cursor.y = yw; state.cursor.active = true;

      // Tooltip position no matter what
      if (tip) {
        if (n) {
          tip.style.display = 'block';
          tip.style.left = ev.clientX + 'px';
          tip.style.top = ev.clientY + 'px';
        } else {
          tip.style.display = 'none';
        }
      }

      // Hover mini-card removed

      // Compute hover path only if no sticky path
      if (!state.pathEdgesSticky.length && n && n.kind === 'person') {
        const bio = shortestPathToMe(n.id);
        let seq = bio && bio.length ? bio : (state.meId ? shortestPathBetweenWeighted(n.id, state.meId) : null);
        if (seq && seq.length) {
          setHoverPath(seq);
          // Prefer kinship label when an anchor exists (A vs hovered)
          if (tip) {
            (async () => {
              // If anchor is a person, prefer anchor↔hover kinship
              if (state.anchorId && String(state.anchorId) !== String(state.meId)) {
                const a = getNode(state.anchorId);
                if (a && a.kind === 'person') {
                  const lbl = await fetchKinshipLabel(a.id, n.id);
                  if (lbl) { tip.textContent = lbl; return; }
                }
              }
              // Otherwise, show relationship to You if possible
              try {
                const r = await fetch(`/api/people/you/${n.id}`, { credentials:'include', cache:'no-store' });
                const j = await r.json();
                if (j && j.label_neutral) { tip.textContent = j.label_neutral; return; }
              } catch {}
              // Fallback to path text
              tip.textContent = seq.map(id => id === state.meId ? 'You' : (getNode(id)?.label || ('#'+id))).join(' \u2192 ');
            })();
          }
        } else {
          clearHoverPath();
          if (tip && n) tip.textContent = n.label;
        }
      } else {
        // either no node or sticky active → clear hover path
        clearHoverPath();
        if (n && tip && !state.pathEdgesSticky.length) tip.textContent = n.label;
      }
    } catch (err) {
      // Never let hover crash visuals
      console.warn('hover handler error:', err);
    }
  });

  // When the mouse leaves the canvas, clear hover visuals (not sticky)
  canvas.addEventListener('mouseleave', () => {
    state.hoverId = null;
    clearHoverPath();
    if (tip) tip.style.display = 'none';
    state.cursor.active = false;
  });

  // Drag nodes
  let dragging = false;
  canvas.addEventListener('mousedown', (ev) => {
    markInteract();
    const n = pickNode(ev.clientX, ev.clientY);
    if (n) { state.dragId = n.id; dragging = true; n.fixed = true; }
  });
  window.addEventListener('mouseup', () => {
    markInteract();
    dragging = false;
    const n = getNode(state.dragId);
    if (n) n.fixed = false;
    state.dragId = null;
  });
  window.addEventListener('mousemove', (ev) => {
    if (!dragging) return;
    markInteract();
    const n = getNode(state.dragId);
    if (!n) return;
    const [xw, yw] = screenToWorld(ev.clientX, ev.clientY);
    n.x = xw; n.y = yw; n.vx = 0; n.vy = 0;
  });

  // Click selection & sticky path logic
  canvas.addEventListener('click', (ev) => {
    const n = pickNode(ev.clientX, ev.clientY);

    // Click empty: clear sticky + selection + panels
    if (!n || n.kind !== 'person') {
      state.selectedId = null;
      state.anchorId = null;
      clearStickyPath();
      return;
    }

    state.selectedId = n.id;

    // No anchor yet → A -> You
    const idS = SID(n.id);
    state.selectedId = n.id;  // can stay as-is for label/selection

    if (!state.anchorId) {
      state.anchorId = idS;
      let seq = shortestPathToMe(n.id) || (state.meId ? shortestPathBetweenWeighted(idS, state.meId) : null);
      if (seq && seq.length > 1) setStickyPath(seq); else clearStickyPath();
      // Update tip to relationship to You
      (async () => {
        if (tip) {
          try { const r = await fetch(`/api/people/you/${n.id}`, { credentials:'include' }); const j = await r.json(); if (j && j.label_neutral) tip.textContent = j.label_neutral; } catch {}
        }
      })();
      openDisplay(n.id);
      return;
    }

    if (state.anchorId === idS) {
      let seq = shortestPathToMe(n.id) || (state.meId ? shortestPathBetweenWeighted(idS, state.meId) : null);
      if (seq && seq.length > 1) setStickyPath(seq); else clearStickyPath();
      openDisplay(n.id);
      return;
    }

    // A -> B (then rotate anchor to B)
    const prevAnchor = state.anchorId;
    let seq2 = shortestPathBetweenBio(prevAnchor, idS) || shortestPathBetweenWeighted(prevAnchor, idS);
    if (seq2 && seq2.length > 1) setStickyPath(seq2); else clearStickyPath();
    // Compose relationship sentence before rotating anchor
    (async () => {
      try {
        if (!tip) return;
        const A = getNode(prevAnchor);
        const B = getNode(n.id);
        if (A && B && A.kind === 'person' && B.kind === 'person') {
          // We want: "B is <label> of A" => ego=A.id, alter=B.id
          const lbl = await fetchKinshipLabel(A.id, B.id);
          if (lbl && String(lbl).toLowerCase() !== 'related') {
            tip.textContent = `${B.label || ('#'+B.id)} is ${lbl} of ${A.label || ('#'+A.id)}`;
          } else {
            tip.textContent = `${B.label || ('#'+B.id)} ? ${A.label || ('#'+A.id)}`;
          }
        }
      } catch {}
    })();
    // Now rotate anchor to the clicked node
    state.anchorId = idS;
    openDisplay(n.id);
  });

  // Zoom
  canvas.addEventListener('wheel', (ev) => {
    try { ev.preventDefault(); } catch {}
    markInteract();
    const oldS = state.cam.s;
    const factor = Math.exp(-(ev.deltaY || 0) * 0.001);
    let newS = oldS * factor;
    const MIN_Z = 0.35, MAX_Z = 3.0;
    newS = Math.max(MIN_Z, Math.min(MAX_Z, newS));
    state.cam.s = newS;
  }, { passive: false });

  // Panel drag limiter (avoid going under navbar)
  (function () {
    function navBottom() {
      const nav = document.querySelector('header');
      const r = nav ? nav.getBoundingClientRect() : null;
      return r ? (r.bottom || 56) : 56;
    }
    function clamp(el, nx, ny) {
      const w = window.innerWidth, h = window.innerHeight;
      const rect = el.getBoundingClientRect();
      const minL = 8;
      const maxL = Math.max(minL, w - rect.width - 8);
      const minT = navBottom() + 8;
      const maxT = Math.max(minT, h - rect.height - 8);
      return { left: Math.max(minL, Math.min(maxL, nx)), top: Math.max(minT, Math.min(maxT, ny)) };
    }
    function attachDrag(el, headerEl) {
      if (!el || !headerEl) return;
      let dragging=false, sx=0, sy=0, ox=0, oy=0;
      function onDown(ev){ dragging=true; const r = el.getBoundingClientRect(); ox=r.left; oy=r.top; sx=ev.clientX; sy=ev.clientY; el.style.right='auto'; el.style.left=ox+'px'; el.style.top=oy+'px'; ev.preventDefault(); }
      function onMove(ev){ if(!dragging) return; const nx = ox + (ev.clientX - sx); const ny = oy + (ev.clientY - sy); const p = clamp(el, nx, ny); el.style.left = p.left + 'px'; el.style.top = p.top + 'px'; }
      function onUp(){ dragging=false; }
      headerEl.addEventListener('mousedown', onDown);
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    }
    attachDrag(panel,    document.getElementById('p_header'));
    attachDrag(panelView,document.getElementById('pv_header'));
  })();

  // Persist current edits (auto-save)
  async function savePanel(){
    if (!state.selectedId) return;
    const pBirthFull = document.getElementById('p_birth_full');
    const body = {
      display_name: (pDisplayName && pDisplayName.value ? pDisplayName.value.trim() : undefined),
      role_hint: (pRole && pRole.value) ? String(pRole.value).toLowerCase() : undefined,
      birth_year: Number(pBirth?.value) || null,
      death_year: Number(pDeath?.value) || null,
      birth_date: (pBirthFull?.value || '').trim() || null,
      bio: pBio?.value || '',
      dot_color: selectedColor,
      deceased: !!pDeceased?.checked
    };
    try{
      await fetch(`/api/people/${state.selectedId}`, { method:'PATCH', credentials:'include', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
      await refreshGraph();
    } catch {}
  }

  // Deceased toggle (and death year immediate save)
  (function(){
    function syncDeathUI() {
      if (!pDeath) return;
      pDeath.classList.toggle('hidden', !pDeceased?.checked);
      pDeath.style.display = pDeceased?.checked ? 'block' : 'none';
    }
    async function persistDeceased() {
      if (!state.selectedId) { syncDeathUI(); return; }
      await fetch(`/api/people/${state.selectedId}`, { method:'PATCH', credentials:'include', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ deceased: !!pDeceased?.checked }) });
      await refreshGraph();
    }
    pDeceased?.addEventListener('change', async () => { syncDeathUI(); await persistDeceased(); });
    pDeath?.addEventListener('change', async () => {
      if (!state.selectedId) return;
      const yr = Number(pDeath.value) || null;
      await fetch(`/api/people/${state.selectedId}`, { method:'PATCH', credentials:'include', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ death_year: yr }) });
      await refreshGraph();
    });
    syncDeathUI();
  })();

  // Photo auto-upload / remove
  pPhotoFile?.addEventListener('change', async () => {
    if (!state.selectedId || !pPhotoFile.files || !pPhotoFile.files[0]) return;
    const fd = new FormData();
    // Backend expects the field name 'file' for /api/people/{id}/photo
    fd.append('file', pPhotoFile.files[0]);
    try { await fetch(`/api/people/${state.selectedId}/photo`, { method:'POST', credentials:'include', body: fd }); } catch {}
    pPhotoFile.value = '';
    await refreshGraph();
    await openPanel(state.selectedId);
  });
  pPhotoRemove?.addEventListener('click', async () => {
    if (!state.selectedId) return;
    try { await fetch(`/api/people/${state.selectedId}/photo`, { method:'DELETE', credentials:'include' }); } catch {}
    await refreshGraph();
    await openPanel(state.selectedId);
  });

  // Delete person
  pDeletePerson?.addEventListener('click', async () => {
    if (!state.selectedId) return;
    if (!confirm('Delete this person and all of their connections?')) return;
    await fetch(`/api/people/${state.selectedId}`, { method:'DELETE', credentials:'include' });
    state.selectedId = null; panel.style.display='none';
    await refreshGraph();
  });

  // Add connection
  pAdd?.addEventListener('click', async () => {
    if (!state.selectedId) return;
    const other = parseInt(pTarget.value, 10);
    let rel = pRel.value || 'role:friend';
    if (rel.startsWith('role:')) rel = rel.slice(5) + '-of';
    if (!other || other === state.selectedId) return;
    await fetch('/api/people/edges', {
      method:'POST', credentials:'include', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ src_person_id: state.selectedId, dst_person_id: other, rel_type: rel, confidence: 0.9 })
    });
    await refreshGraph();
    await openPanel(state.selectedId);
  });

  // Quick add person (toolbar)
  npAdd?.addEventListener('click', async () => {
    const name = (npName?.value || '').trim();
    const role = (npRole && npRole.value) ? npRole.value.trim().toLowerCase() : 'friend';
    if (!name) return;
    const r = await fetch('/api/people/add', {
      method:'POST', credentials:'include', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ display_name: name, role_hint: role || 'friend' })
    });
    const j = await r.json().catch(()=>({}));
    const pid = j && (j.person_id ?? j.id);
    npName.value = ''; if (npRole) npRole.value = '';
    await refreshGraph();
    const node = pid ? state.nodes.find(n => String(n.id) === String(pid))
                     : state.nodes.find(n => n.kind==='person' && (n.label||'').toLowerCase() === name.toLowerCase());
    if (node) { state.selectedId = node.id; await openPanel(node.id); }
  });

  // Inferred mentions list
  async function loadInferred(){
    try {
      const r = await fetch('/api/people/inferred/list', { credentials:'include' });
      const j = await r.json();
      const arr = j.people || [];
      infList.innerHTML = '';

      if (infDock) {
        infDock.textContent = `Unconfirmed (${arr.length})`;
        infDock.style.display = (inf.style.display === 'none' && arr.length > 0) ? 'inline-block' : 'none';
      }
      if (!arr.length) { inf.style.display='none'; return; }

      for (const it of arr) {
        const row = document.createElement('div'); row.className = 'flex items-center gap-2 my-1';
        const a = document.createElement('div'); a.className = 'flex-1 text-sm text-black'; a.textContent = `${it.name} (${it.mentions} mention${it.mentions==1?'':'s'})`;
        const confirm = document.createElement('button'); confirm.textContent='Confirm'; confirm.className='graph-btn text-sm';
        confirm.addEventListener('click', async ()=>{
          await fetch(`/api/people/${it.id}`, {
            method:'PATCH', credentials:'include', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ inferred:false, connect_to_owner:true, hidden:false, role_hint: 'friend' })
          });
          await refreshGraph();
          const node = state.nodes.find(n => String(n.id) === String(it.id));
          if (node) { state.selectedId = node.id; await openPanel(node.id); }
          await loadInferred();
        });
        const hide = document.createElement('button'); hide.textContent='Hide'; hide.className='graph-btn text-sm';
        hide.addEventListener('click', async ()=>{
          await fetch(`/api/people/${it.id}`, { method:'PATCH', credentials:'include', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ hidden:true }) });
          await refreshGraph(); await loadInferred();
        });
        row.appendChild(a); row.appendChild(confirm); row.appendChild(hide); infList.appendChild(row);
      }
      inf.style.display='block';
    } catch { inf.style.display='none'; }
  }
  infClose?.addEventListener('click', ()=>{ inf.style.display='none'; if (infDock) infDock.style.display='inline-block'; });
  infDock?.addEventListener('click', ()=>{ inf.style.display='block'; infDock.style.display='none'; });

  // Connections list renderer (used by openPanel)
  async function renderConnections(personId){
    try {
      const res = await fetch(`/api/people/${personId}`, { credentials:'include', cache: 'no-store' });
      if (!res.ok) {
        pEdges.innerHTML = '';
        const m = document.createElement('div');
        m.className = 'text-xs text-stone-600';
        m.textContent = res.status === 404 ? 'Not editable (shared or missing)' : 'Could not load connections';
        pEdges.appendChild(m);
        return;
      }
      const detail = await res.json();
      // Always show a small header so there is visible feedback
      pEdges.innerHTML = '';
      const hdr = document.createElement('div');
      hdr.className = 'text-xs text-stone-500 mb-1';
      const apiCount = Array.isArray(detail.connections) ? detail.connections.length : 0;
      hdr.textContent = `Connections (api=${apiCount}${detail.editable!==undefined?`, editable=${!!detail.editable}`:''})`;
      pEdges.appendChild(hdr);
      const canEdit = !!detail.editable;

      // Accept either `connections` (preferred) or `edges` (legacy)
      let conns = Array.isArray(detail.connections)
        ? detail.connections
        : (Array.isArray(detail.edges) ? detail.edges : []);
      // Show both directions; de-duplicate by (person_id, rel_type)
      conns = (conns || []).filter(c => c);
      const seen = new Set();
      conns = conns.filter(c => {
        const r = String(c.rel_type || c.rel || '').toLowerCase();
        const key = `${c.person_id}|${r}`;
        if (seen.has(key)) return false; seen.add(key); return true;
      });

      // Sort by relationship type, then by neighbor name
      function relOrder(rel) {
        const r = String(rel || '').toLowerCase();
        if (r === 'parent-of' || r === 'mother-of' || r === 'father-of') return 0;
        if (r === 'child-of' || r === 'son-of' || r === 'daughter-of') return 1;
        if (r === 'spouse-of' || r === 'partner-of') return 2;
        if (r === 'sibling-of' || r === 'brother-of' || r === 'sister-of' || r === 'half-sibling-of' || r === 'step-sibling-of') return 3;
        if (r === 'grandparent-of') return 4;
        if (r === 'grandchild-of') return 5;
        if (r === 'aunt-of' || r === 'uncle-of') return 6;
        if (r === 'niece-of' || r === 'nephew-of') return 7;
        if (r === 'cousin-of') return 8;
        if (r === 'friend-of' || r === 'neighbor-of' || r === 'coworker-of' || r === 'colleague-of' || r === 'pet-of') return 9;
        if (r === 'mentor-of' || r === 'teacher-of' || r === 'student-of' || r === 'coach-of') return 10;
        return 99;
      }
      conns.sort((a,b) => {
        const ra = relOrder(a.rel_type || a.rel);
        const rb = relOrder(b.rel_type || b.rel);
        if (ra !== rb) return ra - rb;
        const na = String(a.name||'').toLowerCase();
        const nb = String(b.name||'').toLowerCase();
        return na.localeCompare(nb);
      });

      pEdges.innerHTML = '';
      if (!conns.length) {
        // Fallback: synthesize from current graph state (non-persistent, no remove)
        const allE = state.edges || [];
        const rows = [];
        const seenG = new Set();
        for (const e of allE) {
          let neighborId = null, direction = null;
          if (String(e.s) === String(personId)) { neighborId = e.d; direction = 'out'; }
          else if (String(e.d) === String(personId)) { neighborId = e.s; direction = 'in'; }
          else continue;
          const r = String(e.rel || '').toLowerCase();
          const key = `${neighborId}|${r}`;
          if (seenG.has(key)) continue; seenG.add(key);
          const n = getNode(neighborId);
          rows.push({ name: n ? (n.label || ('#'+neighborId)) : ('#'+neighborId), rel_type: r, person_id: neighborId, direction, injected: true });
        }
        if (!rows.length) {
          const empty = document.createElement('div');
          empty.className = 'text-xs text-stone-600';
          empty.textContent = 'No connections yet';
          pEdges.appendChild(empty);
          return;
        }
        for (const c of rows) {
          const row = document.createElement('div'); row.className='flex items-center gap-2 my-1';
          const a = document.createElement('div'); a.className='flex-1 text-sm text-black';
          const subj = pName.textContent || 'This person';
          const label = relLabel(c.rel_type || c.rel || '');
          a.textContent = `${subj} is ${label} of ${c.name}`;
          row.appendChild(a);
          const note = document.createElement('span'); note.className='text-[11px] text-stone-500'; note.textContent='(not saved)';
          row.appendChild(note);
          pEdges.appendChild(row);
        }
        return;
      }

      conns.forEach(c => {
        const row = document.createElement('div');
        row.className = 'flex items-center gap-2 my-1';

        const a = document.createElement('div');
        a.className = 'flex-1 text-sm text-black';

        const subj = pName.textContent || 'This person';
        let label = relLabel(c.rel_type || c.rel || '');
        // Enhance fallback label using generation metadata for aunt/uncle and niece/nephew
        const gen = Number.isFinite(c.generation) ? Number(c.generation) : (c.generation || null);
        function genAuntLabel(g){ if (!g || g<=1) return 'aunt/uncle'; if (g===2) return 'grandaunt/uncle'; return ('great-'.repeat(g-2)) + 'grandaunt/uncle'; }
        function genNieceLabel(g){ if (!g || g<=1) return 'niece/nephew'; if (g===2) return 'grandniece/nephew'; return ('great-'.repeat(g-2)) + 'grandniece/nephew'; }
        const rt = String(c.rel_type||c.rel||'').toLowerCase();
        if (gen && (rt === 'aunt-of')) label = genAuntLabel(gen);
        if (gen && (rt === 'niece-of')) label = genNieceLabel(gen);

        // ✅ No inverseRel needed: flip the sentence for incoming edges
        a.textContent = `${subj} is ${label} of ${c.name}`;
        if (String(c.direction||'') === 'in') { a.textContent = `${c.name} is ${label} of ${subj}`; }
        // Upgrade to precise kinship term asynchronously, oriented as:
        // subject is <label> of object. classify_kinship returns label of alter relative to ego,
        // so call with ego=objectId, alter=subjectId. For aunt/uncle and niece/nephew, flip the
        // sentence so it reads the younger perspective (requested), i.e., "Object is niece/nephew of Subject".
        (async () => {
          try {
            const subjectId = personId;      // selected person (subject of the sentence)
            const objectId  = c.person_id;   // neighbor (object of the sentence)
            const lbl = await fetchKinshipLabel(objectId, subjectId);
            if (lbl && String(lbl).toLowerCase() !== 'related') {
              a.textContent = `${subj} is ${lbl} of ${c.name}`;
            } else if (gen && (rt==='aunt-of' || rt==='niece-of')) {
              a.textContent = `${subj} is ${label} of ${c.name}`;
            }
          } catch {}
        })();

        const btn = document.createElement('button');
        btn.textContent = 'Remove';
        btn.className = 'graph-btn text-sm';
        btn.disabled = !canEdit;
        btn.style.opacity = canEdit ? '1' : '0.5';
        if (canEdit) {
          btn.addEventListener('click', async () => {
            const edgeId = c.edge_id ?? c.id;     // support either key
            if (!edgeId) return;
            await fetch(`/api/people/edges/${edgeId}`, { method:'DELETE', credentials:'include' });
            await refreshGraph();
            await renderConnections(personId);     // refresh list immediately
          });
        }

        row.appendChild(a);
        row.appendChild(btn);
        pEdges.appendChild(row);
      });
    } catch (err) {
      console.warn('renderConnections failed:', err);
      pEdges.innerHTML = '';
      const m = document.createElement('div');
      m.className = 'text-xs text-stone-600';
      m.textContent = 'Could not load connections';
      pEdges.appendChild(m);
    }
  }

  // Open Edit Panel
  let selectedColor = null;
  const PALETTE = ['#f87171','#fbbf24','#34d399','#60a5fa','#a78bfa','#f472b6','#f59e0b','#10b981','#000000ff'];
  function renderPalette(current){
    const wrapId = 'p_palette';
    let wrap = document.getElementById(wrapId);
    if (!wrap) {
      wrap = document.createElement('div');
      wrap.id = wrapId;
      wrap.className = 'flex flex-wrap gap-2 my-2';
      const label = document.createElement('div');
      label.textContent = 'Dot color';
      label.className = 'text-xs text-stone-600';
      // Insert just before the connections list within the edit panel
      try {
        if (panel && pEdges && panel.contains(pEdges)) {
          panel.insertBefore(label, pEdges);
          panel.insertBefore(wrap, pEdges);
        } else if (panel) {
          // Fallback: append to panel if reference not available
          panel.appendChild(label);
          panel.appendChild(wrap);
        }
      } catch {}
    } else {
      wrap.innerHTML = '';
    }
    PALETTE.forEach(c => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'inline-block rounded-full border-2';
      b.style.width = '20px'; b.style.height = '20px';
      b.style.borderColor = 'var(--accent)';
      b.style.background = c;
      if (current && current.toLowerCase() === c.toLowerCase()) { b.style.outline='2px solid #000'; b.style.outlineOffset='1px'; }
      b.addEventListener('click', () => { selectedColor = c; renderPalette(selectedColor); });
      wrap.appendChild(b);
    });
    const clr = document.createElement('button');
    clr.textContent = 'Clear';
    clr.className = 'graph-btn ml-2 text-sm';
    clr.addEventListener('click', ()=>{ selectedColor = null; renderPalette(null); });
    wrap.appendChild(clr);
  }

  async function openPanel(id){
    state.selectedId = id;
    if (panelView) panelView.style.display = 'none';
    if (pSubjectLabel) pSubjectLabel.textContent = (getNode(id)?.label) || 'This person';
    // Ensure a clean slate before fetching data
    resetEditPanel();
    // Show a placeholder while we load details and connections
    if (pEdges) {
      pEdges.innerHTML = '<div class="text-xs text-stone-500">Loading connections…</div>';
    }

    // Populate target select (alphabetized by last name)
    if (pTarget) {
      pTarget.innerHTML='';
      const people = state.nodes.filter(n => n.kind === 'person' && n.id !== id);
      function nameKey(lbl){
        const s = String(lbl || '').trim();
        if (!s) return { last: '', first: '' };
        if (s.includes(',')) {
          const parts = s.split(',');
          const last = parts[0].trim().toLowerCase();
          const first = parts.slice(1).join(',').trim().toLowerCase();
          return { last, first };
        }
        const parts = s.split(/\s+/);
        const last = (parts[parts.length - 1] || '').toLowerCase();
        const first = parts.slice(0, -1).join(' ').toLowerCase();
        return { last, first };
      }
      people.sort((a,b) => {
        const ka = nameKey(a.label), kb = nameKey(b.label);
        if (ka.last !== kb.last) return ka.last.localeCompare(kb.last);
        if (ka.first !== kb.first) return ka.first.localeCompare(kb.first);
        const la = String(a.label||'').toLowerCase();
        const lb = String(b.label||'').toLowerCase();
        if (la !== lb) return la.localeCompare(lb);
        return String(a.id).localeCompare(String(b.id));
      });
      for (const n of people) {
        const opt = document.createElement('option');
        opt.value = n.id; opt.textContent = n.label; pTarget.appendChild(opt);
      }
    }

    const roleList = await ensureRoles();
    if (pRel) {
      pRel.innerHTML='';
      roleList.forEach(r => { if (r?.slug?.startsWith('role:')) { const o=document.createElement('option'); o.value=r.slug; o.textContent=r.label; pRel.appendChild(o); } });
    }
    if (pRole) {
      pRole.innerHTML='';
      const blank = document.createElement('option'); blank.value=''; blank.textContent='Select role'; pRole.appendChild(blank);
      roleList.forEach(r => { if (r?.slug?.startsWith('role:')) { const base=r.slug.split(':')[1]; const o=document.createElement('option'); o.value=base; o.textContent=r.label; pRole.appendChild(o); } });
      try { if ([...pRole.options].some(opt => String(opt.value).toLowerCase()==='friend')) { pRole.value = 'friend'; } } catch {}
    }

    // Fetch detail and hydrate controls
    try {
      const res = await fetch(`/api/people/${id}`, { credentials:'include' });
      const d = await res.json();
      if (pName) pName.textContent = d.display_name || ('#'+id);
      if (pDisplayName) pDisplayName.value = d.display_name || '';
      if (pPhotoImg) {
        if (d.photo_url) {
          pPhotoImg.src = d.photo_url; pPhotoImg.style.display='block';
          if (pPhotoRemove) pPhotoRemove.style.display='inline-block';
          if (pPhotoFile) pPhotoFile.style.display='none';
        } else {
          pPhotoImg.removeAttribute('src'); pPhotoImg.style.display='none';
          if (pPhotoRemove) pPhotoRemove.style.display='none';
          if (pPhotoFile) pPhotoFile.style.display='block';
        }
      }

      const savedBase = (d.role_hint || '').toString().toLowerCase() || 'friend';
      try {
        if (pRole && pRole.options && pRole.options.length > 0) {
          for (let i=0;i<pRole.options.length;i++) {
            if (String(pRole.options[i].value).toLowerCase() === savedBase) { pRole.selectedIndex = i; break; }
          }
        }
      } catch {}

      if (pBirth) pBirth.value = d.birth_year || '';
      const pBirthFull = document.getElementById('p_birth_full');
      if (pBirthFull) pBirthFull.value = d.birth_date || '';
      if (pDeath) pDeath.value = d.death_year || '';
      if (pBio) { pBio.value = d.bio || ''; try { pBio.style.height='auto'; pBio.style.height=(pBio.scrollHeight)+'px'; } catch {} }
      if (pDeceased) pDeceased.checked = !!d.deceased;
      selectedColor = d.dot_color || null;
      renderPalette(selectedColor);

      if (pDeath) {
        pDeath.classList.toggle('hidden', !pDeceased?.checked);
        pDeath.style.display = pDeceased?.checked ? 'block' : 'none';
      }

    } catch (err) {
      console.warn('openPanel failed:', err);
      if (pEdges) {
        pEdges.innerHTML = '<div class="text-xs text-stone-600">Could not load details; attempting connections…</div>';
      }
    }

    // Always attempt to render connections even if detail hydration failed
    await renderConnections(id);

    if (panel) panel.style.display = 'block';
  }

  async function openDisplay(id){
    state.selectedId = id;
    if (!panelView) return;
    try {
      const res = await fetch(`/api/people/${id}`, { credentials:'include' });
      const d = await res.json();
      if (pvName) pvName.textContent = d.display_name || ('#'+id);
      if (pvPhoto) {
        if (d.photo_url) { pvPhoto.src = d.photo_url; pvPhoto.style.display='block'; }
        else { pvPhoto.removeAttribute('src'); pvPhoto.style.display='none'; }
      }
      if (pvRole)  pvRole.textContent = (d.role_hint ? d.role_hint.replace(/-/g,' ') : '—');
      if (pvYears) {
        const by = d.birth_year ? String(d.birth_year) : '';
        const dy = d.death_year ? String(d.death_year) : (d.deceased ? '—' : '');
        pvYears.textContent = (by || dy) ? `${by || ''}${(by||dy)?'':''}${by && (dy||d.deceased) ? ' – ' : ''}${dy || (d.deceased ? '—' : '')}` : '—';
      }
      if (pvConnect) pvConnect.style.display = d.connect_to_owner ? 'inline-block' : 'none';
      if (pvDeceased) pvDeceased.style.display = d.deceased ? 'inline-block' : 'none';
      if (pvBio) pvBio.textContent = (d.bio || '').trim() || '—';
      if (pvEdges) {
        pvEdges.innerHTML = '';
        // Display panel: show only the selected person's own (outgoing) connections
        const conns = Array.isArray(d.connections)
          ? d.connections.filter(c => c && c.direction === 'out')
          : [];
        if (!conns.length) {
          const e = document.createElement('div'); e.className='text-xs text-stone-600'; e.textContent='No connections yet'; pvEdges.appendChild(e);
        } else {
          conns.forEach(c => {
            const row = document.createElement('div'); row.className='text-sm my-0.5';
            const roleLabel = relLabel(c.rel_type);

            // Always subject=displayed person, object=neighbor
            const subjName = (pvName.textContent || ('#'+d.id));
            const objName  = (c.name || ('#'+c.person_id));
            // Fallback label first using edge's rel_type
            row.textContent = `${subjName} is ${roleLabel} of ${objName}`;

            // Upgrade to kinship label asynchronously using ego=object, alter=subject
            (async () => {
              try {
                const subjectId = d.id;        // displayed person
                const objectId  = c.person_id; // neighbor
                const lbl = await fetchKinshipLabel(objectId, subjectId);
                if (lbl && String(lbl).toLowerCase() !== 'related') {
                  row.textContent = `${subjName} is ${lbl} of ${objName}`;
                } else if (gen && (rt==='aunt-of' || rt==='niece-of')) {
                  row.textContent = `${subjName} is ${roleLabel} of ${objName}`;
                }
              } catch {}
            })();

            pvEdges.appendChild(row);
          });
        }
      }
    } catch {}
    panelView.style.display = 'block';
  }

  // Panel close buttons
  // Off-click handling for both panels: edit (save+close) and display (close)
  async function persistPanelIfOpen(ev){
    const target = ev.target;
    // Edit panel: if click is inside the panel DOM, do nothing
    if (panel && panel.style.display !== 'none') {
      if (panel.contains(target)) return;
      await savePanel?.();
      panel.style.display = 'none';
      state.selectedId = null;
      // if edit panel was open, consume this off-click
      return;
    }
    // Display panel: if click is inside the display panel DOM, do nothing
    if (panelView && panelView.style.display !== 'none') {
      if (panelView.contains(target)) return;
      panelView.style.display = 'none';
      state.selectedId = null;
    }
  }
  document.addEventListener('mousedown', persistPanelIfOpen);
  pvClose?.addEventListener('click', () => { panelView.style.display='none'; state.selectedId=null; });
  // Prevent off-click handler from firing when pressing Edit
  pvEdit?.addEventListener('mousedown', (ev) => { ev.stopPropagation(); });
  pvEdit?.addEventListener('click', async (ev) => {
    ev.stopPropagation();
    if (!state.selectedId) return;
    panelView.style.display='none';
    await openPanel(state.selectedId);
  });

  // (Set as Me button removed)

  // ---- People search ----
  function showSearch(results) {
    if (!pgDD) return;
    if (!results || !results.length) { pgDD.classList.add('hidden'); pgDD.innerHTML=''; return; }
    pgDD.innerHTML = results.map(r => `
      <button type="button" data-id="${r.id}" class="w-full text-left px-2 py-1 hover:bg-stone-100 border-b last:border-b-0 border-stone-200">
        ${escapeHtml(r.label || ('#'+r.id))}
      </button>
    `).join('');
    pgDD.classList.remove('hidden');
  }
  function filterPeople(q) {
    const s = (q||'').trim().toLowerCase(); if (!s) return [];
    const items = state.nodes.filter(n => n.kind==='person');
    const scored = [];
    for (const n of items) {
      const name = (n.label||'').toLowerCase();
      const idx = name.indexOf(s);
      if (idx >= 0) scored.push({ id:n.id, label:n.label, score: (idx===0?0:(idx<=2?1:2)), idx });
    }
    scored.sort((a,b) => a.score - b.score || a.idx - b.idx || String(a.label).localeCompare(String(b.label)));
    return scored.slice(0, 12);
  }
  function applyFilterFromQuery(q){
    const s = (q||'').trim().toLowerCase();
    state.filterText = s;
    state.filterIds = new Set();
    if (!s) return;
    for (const n of state.nodes){
      if (n.kind==='person' && String(n.label||'').toLowerCase().includes(s)){
        state.filterIds.add(String(n.id));
      }
    }
  }
  function gotoPerson(id){
    const node = state.nodes.find(n => String(n.id)===String(id));
    if (!node) return;
    state.selectedId = node.id;
    openDisplay(node.id);
    // close dropdown
    if (pgDD) { pgDD.classList.add('hidden'); pgDD.innerHTML=''; }
  }
  pgSearch?.addEventListener('input', (e)=>{
    const q = e.target.value;
    applyFilterFromQuery(q);
    const res = filterPeople(q);
    showSearch(res);
  });
  pgSearch?.addEventListener('keydown', (e)=>{
    if (e.key==='Enter'){
      const q = e.target.value;
      const res = filterPeople(q);
      if (res.length){ gotoPerson(res[0].id); }
    } else if (e.key==='Escape'){ showSearch([]); }
  });
  pgDD?.addEventListener('click', (e)=>{
    const btn = e.target.closest('button[data-id]');
    if (!btn) return;
    gotoPerson(btn.getAttribute('data-id'));
  });
  document.addEventListener('click', (e)=>{
    if (pgDD && !pgDD.contains(e.target) && e.target !== pgSearch){ showSearch([]); }
  });

  // Infer connections for selected person (preview + confirm)
  const pInfer = document.getElementById('p_infer');
  pInfer?.addEventListener('click', async () => {
    if (!state.selectedId) return;
    try {
      const r = await fetch(`/api/people/${state.selectedId}/infer/preview`, { credentials:'include', cache:'no-store' });
      const j = await r.json();
      const edges = Array.isArray(j.edges) ? j.edges : [];
      if (!edges.length) { alert('No inferred connections found to add.'); return; }
      // Build a tiny preview string (limit to 8 lines)
      const lines = [];
      const maxShow = Math.min(edges.length, 8);
      for (let i=0;i<maxShow;i++) {
        const e = edges[i];
        const a = getNode(e.src_id), b = getNode(e.dst_id);
        const an = a ? (a.label || ('#'+e.src_id)) : ('#'+e.src_id);
        const bn = b ? (b.label || ('#'+e.dst_id)) : ('#'+e.dst_id);
        lines.push(`${an} — ${String(e.rel_type||'').replace(/-of$/,'').replace(/-/g,' ')} — ${bn}`);
      }
      const extra = edges.length > maxShow ? `\n(+${edges.length - maxShow} more…)` : '';
      const ok = confirm(`Add ${edges.length} inferred connection(s)?\n\n` + lines.join('\n') + extra);
      if (!ok) return;
      const c = await fetch(`/api/people/${state.selectedId}/infer/commit`, { method:'POST', credentials:'include' });
      const jj = await c.json().catch(()=>({}));
      await refreshGraph();
      await openPanel(state.selectedId);
    } catch (err) {
      console.warn('infer failed:', err);
      alert('Could not infer connections right now.');
    }
  });

  // Graph data load/refresh
  async function refreshGraph(){
    try {
      const res = await fetch('/api/people/graph', { credentials: 'include' });
      const data = await res.json();
      build(data.nodes||[], data.edges||[]);
    } catch {
      build([], []);
    }
    // Clear paths when graph changes (avoid dangling ids)
    clearStickyPath();
    clearHoverPath();
    state.anchorId = null;
  }

  async function load() {
    await refreshGraph();
    loop();
  }

  // Boot
  (async () => {
    try { const r = await fetch('/api/roles', { credentials:'include' }); const j = await r.json(); window.__ROLES__ = j.roles || []; } catch {}
    await renderToolbarRoles();

    await load();
    await loadInferred();
  })();
})();



