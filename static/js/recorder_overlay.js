// static/js/recorder_overlay.js
(function(){
  // Public API: window.initRecorderOverlay(config)
  // config = {
  //   triggerSelector: '#btn-record' or '#btn-record-segment',
  //   modeDefault: 'audio' | 'video',
  //   context: 'user_record' | 'response_edit',
  //   endpoints: {
  //     upload: '/api/upload/segment' OR '/api/upload/primary-or-segment',
  //     jobStatus: (id)=> `/api/jobs/${id}`,
  //     // optional hooks to refresh UI
  //     fetchSegments: ( ) => Promise<void>,        // response_edit
  //     insertTranscript: (text, segId) => void,    // response_edit
  //   }
  // }

  const S = {
    overlay:   ()=>document.getElementById('captureOverlay'),
    vEl:       ()=>document.getElementById('liveVideo'),
    aView:     ()=>document.getElementById('audioView'),
    aMeter:    ()=>document.getElementById('audioMeter'),
    btnPause:  ()=>document.getElementById('btnPauseResume'),
    btnFinish: ()=>document.getElementById('btnFinish'),
    btnCancel: ()=>document.getElementById('btnCancel'),
    tEl:       ()=>document.getElementById('overlayTimer'),
    transCover:()=>document.getElementById('transcribeCover'),
    transErr:  ()=>document.getElementById('transcribeErr'),
    btnAudio:  ()=>document.getElementById('mode-audio'),
    btnVideo:  ()=>document.getElementById('mode-video'),
  };

  function openOverlay(){ S.overlay()?.classList.remove('hidden'); }
  function closeOverlay(){ S.overlay()?.classList.add('hidden'); }
  function showTranscribing(msg){ if(S.transErr()) S.transErr().textContent = msg || ''; S.transCover()?.classList.remove('hidden'); }
  function hideTranscribing(){ S.transCover()?.classList.add('hidden'); if(S.transErr()) S.transErr().textContent=''; }

  let mode='audio', stream=null, rec=null, chunks=[], paused=false, t0=0, raf=0, audioCtx=null, analyser=null, meterRAF=0;

  function setMode(m){
    mode=m;
    S.btnAudio()?.classList.toggle('active-pill', m==='audio');
    S.btnVideo()?.classList.toggle('active-pill', m==='video');
  }

  function tick(){
    const el = S.tEl(); if (!el) return;
    const s=(performance.now()-t0)/1000;
    el.textContent=s.toFixed(1)+'s';
    raf=requestAnimationFrame(tick);
  }
  function stopTick(){ cancelAnimationFrame(raf); }

  function drawMeter(){
    const aM = S.aMeter();
    if (!analyser || !aM) return;
    const ctx=aM.getContext('2d');
    const w=aM.width=aM.clientWidth, h=aM.height=aM.clientHeight;
    const data=new Uint8Array(analyser.fftSize); analyser.getByteTimeDomainData(data);
    let peak=0; for (let i=0;i<data.length;i++){ const v=Math.abs(data[i]-128)/128; if (v>peak) peak=v; }
    ctx.clearRect(0,0,w,h); const barW=Math.max(4, w*peak); ctx.fillStyle='rgba(255,255,255,0.85)'; ctx.fillRect(0,0,barW,h);
    meterRAF=requestAnimationFrame(drawMeter);
  }
  function stopMeter(){ cancelAnimationFrame(meterRAF); }

  function cleanupPreview(){
    try { stream?.getTracks().forEach(t=>t.stop()); } catch {}
    if (audioCtx){ try{ audioCtx.close(); } catch{} }
    stream=null; rec=null; analyser=null; audioCtx=null; chunks=[];
    stopTick(); stopMeter(); const vEl=S.vEl(), aV=S.aView();
    if (S.tEl()) S.tEl().textContent='0.0s';
    if (vEl) vEl.style.display='none';
    if (aV) aV.style.display='none';
  }

  async function getStream(){
    if (mode==='video') return await navigator.mediaDevices.getUserMedia({ audio:true, video:{ facingMode:'user' }});
    return await navigator.mediaDevices.getUserMedia({ audio:true });
  }

  async function startCapture(){
    try{
      stream = await getStream(); chunks=[]; paused=false;
      const vEl=S.vEl(), aV=S.aView();

      if (mode==='video'){
        if (vEl){ vEl.srcObject=stream; vEl.style.display=''; }
        if (aV){ aV.style.display='none'; }
      } else {
        if (vEl) vEl.style.display='none';
        if (aV) aV.style.display='';
        audioCtx=new (window.AudioContext||window.webkitAudioContext)();
        const src=audioCtx.createMediaStreamSource(stream);
        analyser=audioCtx.createAnalyser(); analyser.fftSize=2048; src.connect(analyser); drawMeter();
      }

      const mimeV='video/webm;codecs=vp9,opus', mimeA='audio/webm;codecs=opus';
      const want=(mode==='video')?mimeV:mimeA;
      const opts = MediaRecorder.isTypeSupported?.(want) ? { mimeType: want } : {};
      rec = new MediaRecorder(stream, opts);
      rec.ondataavailable = e=>{ if(e.data && e.data.size) chunks.push(e.data); };
      rec.onstop = handleStop;

      rec.start();
      t0=performance.now(); tick(); openOverlay();
      if (S.btnPause()) S.btnPause().textContent='Pause';
    }catch(err){
      console.warn('capture error',err);
      alert('Could not start recording. Please allow microphone/camera access.');
    }
  }

  async function postBlob(url, extraFields, blob){
    const ext = 'webm';
    const mime= (mode==='video') ? 'video/webm' : 'audio/webm';
    const fd = new FormData();
    fd.append('file', new File([blob], `capture-${Date.now()}.${ext}`, { type: mime }));
    Object.entries(extraFields || {}).forEach(([k,v])=>fd.append(k, String(v)));
    const r = await fetch(url, { method:'POST', body:fd, credentials:'include' });
    if (!r.ok) {
      let m=''; try{ m=await r.text(); }catch{}
      throw new Error(m || `Upload failed (${r.status})`);
    }
    return r.json(); // { job_id: "..." }
  }

  async function pollJob(jobUrl, onProgress, maxMs=10*60*1000){
    const deadline = Date.now() + maxMs;
    while (Date.now() < deadline){
      const r = await fetch(jobUrl, { credentials:'include', cache:'no-store' });
      if (!r.ok) throw new Error(`Job poll ${r.status}`);
      const j = await r.json();
      onProgress?.(j);
      if (j.status === 'done' || j.status === 'error') return j;
      await new Promise(r=>setTimeout(r, 1500));
    }
    throw new Error('Timed out waiting for job');
  }

  async function handleStop(){
    const blob = new Blob(chunks, { type: (mode==='video')?'video/webm':'audio/webm' });
    try {
      showTranscribing('Uploading & queuing…');
      const job = await postBlob(CONFIG.endpoints.upload, CONFIG._extraFields||{}, blob);
      const jobUrl = CONFIG.endpoints.jobStatus(job.job_id);
      showTranscribing('Processing on server…');
      const result = await pollJob(jobUrl, (j)=>{
        // Optional: if you want to reflect numeric progress later
      });

      if (result.status === 'done'){
        // response_edit enhancements
        if (CONFIG.context === 'response_edit'){
          try { await CONFIG.endpoints.fetchSegments?.(); } catch {}
          if (result.transcript && CONFIG.endpoints.insertTranscript){
            CONFIG.endpoints.insertTranscript(result.transcript, result.segment_id);
          }
        }
      } else {
        if (S.transErr()) S.transErr().textContent = 'Processing failed.';
      }
    } catch (e){
      console.error(e);
      if (S.transErr()) S.transErr().textContent = e.message || 'Upload failed.';
    } finally {
      hideTranscribing();
    }
  }

  // wiring
  let CONFIG = {};
  window.initRecorderOverlay = function(cfg){
    CONFIG = cfg || {};
    // default mode: audio on desktop, video on touch
    if (!CONFIG.modeDefault){
      if (matchMedia && matchMedia('(pointer: coarse)').matches) setMode('video'); else setMode('audio');
    } else {
      setMode(CONFIG.modeDefault);
    }
    S.btnAudio()?.addEventListener('click', ()=>setMode('audio'));
    S.btnVideo()?.addEventListener('click', ()=>setMode('video'));

    const trigger = document.querySelector(CONFIG.triggerSelector);
    trigger?.addEventListener('click', startCapture);

    S.btnPause()?.addEventListener('click', ()=>{
      if (!rec) return;
      if (!paused){ rec.pause(); paused=true; S.btnPause().textContent='Resume'; stopTick(); }
      else { rec.resume(); paused=false; S.btnPause().textContent='Pause';
             const now=parseFloat(S.tEl()?.textContent||'0'); t0 = performance.now() - now*1000; tick(); }
    });
    S.btnFinish()?.addEventListener('click', ()=>{
      if (!rec) return;
      showTranscribing('Finishing…');
      try{ rec.stop(); } finally { closeOverlay(); cleanupPreview(); }
    });
    S.btnCancel()?.addEventListener('click', ()=>{
      try{ rec?.stop(); }catch{}
      closeOverlay(); cleanupPreview(); hideTranscribing();
    });
    document.addEventListener('keydown', (e)=>{ if(e.key==='Escape' && !S.overlay()?.classList.contains('hidden')) S.btnCancel()?.click(); });
  };
})();
