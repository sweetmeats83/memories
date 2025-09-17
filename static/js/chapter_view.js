(() => {
  const $ = (sel, el=document) => el.querySelector(sel);
  const $$ = (sel, el=document) => Array.from(el.querySelectorAll(sel));

  async function getJSON(url, opts={}) {
    const r = await fetch(url, { credentials: 'include', ...opts });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  function setLoading(on) {
    const sp = $("#spinner");
    if (!sp) return;
    sp.classList.toggle("hidden", !on);
  }

  async function refreshStatus() {
    try {
      const data = await getJSON(`/api/chapter/${encodeURIComponent(window.CHAPTER_KEY)}/status`);
      const badge = $("#chapterStatusBadge");
      if (badge) badge.textContent = data.ready ? "Ready" : "Locked";
      // populate right rail gaps if present
      if (data.latest_compilation && data.latest_compilation.gap_questions?.length) {
        const box = $("#gapChips");
        if (box) {
          box.innerHTML = "";
          for (const g of data.latest_compilation.gap_questions) {
            const b = document.createElement("button");
            b.className = "pill";
            b.textContent = g.question;
            if (g.why) b.title = g.why;
            box.appendChild(b);
          }
        }
        $("#btnPublish")?.classList.remove("hidden");
        $("#btnRecompile")?.classList.remove("hidden");
      }
    } catch (e) {
      console.warn("status err", e);
    }
  }

  async function compileNow() {
    setLoading(true);
    try {
      const dto = await getJSON(`/api/chapter/${encodeURIComponent(window.CHAPTER_KEY)}/compile`, {
        method: 'POST',
      });
      // render draft
      const art = $("#draftArea");
      if (art) {
        art.innerHTML = dto.compiled_markdown;
      }
      // refresh right rail
      const box = $("#gapChips");
      if (box) {
        box.innerHTML = "";
        (dto.gap_questions || []).forEach(g => {
          const b = document.createElement("button");
          b.className = "pill";
          b.textContent = g.question;
          if (g.why) b.title = g.why;
          box.appendChild(b);
        });
      }
      $("#btnPublish")?.classList.remove("hidden");
      $("#btnRecompile")?.classList.remove("hidden");
    } catch (e) {
      alert("Compile failed: " + (e.message || e));
    } finally {
      setLoading(false);
    }
  }

  async function publishNow() {
    try {
      const r = await getJSON(`/api/chapter/${encodeURIComponent(window.CHAPTER_KEY)}/publish`, { method: "POST" });
      if (r?.ok) {
        const badge = $("#chapterStatusBadge");
        if (badge) badge.textContent = "Published";
      }
    } catch (e) {
      alert("Publish failed: " + (e.message || e));
    }
  }

  $("#btnCompile")?.addEventListener("click", compileNow);
  $("#btnRecompile")?.addEventListener("click", compileNow);
  $("#btnPublish")?.addEventListener("click", publishNow);

  // initial
  refreshStatus();
})();
