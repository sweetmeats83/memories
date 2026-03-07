/**
 * ChunkedUploader — splits a File/Blob into chunks, uploads via
 * POST /api/upload/init → /chunk → /complete, and resolves with the
 * upload_id token that the server can use as a "staged file".
 *
 * Automatically uses smaller chunks on cellular/slow connections.
 * Failed chunks are retried up to 3 times with exponential backoff.
 *
 * Usage:
 *   const uploader = new ChunkedUploader(file, { onProgress });
 *   const uploadId = await uploader.upload();
 */
class ChunkedUploader {
  /**
   * @param {File|Blob} file
   * @param {object}   [opts]
   * @param {function} [opts.onProgress]  called with (bytesUploaded, totalBytes, chunkIndex, totalChunks)
   * @param {number}   [opts.chunkSize]   override chunk size (bytes)
   * @param {number}   [opts.maxRetries]  max retries per chunk (default 3)
   */
  constructor(file, opts = {}) {
    this.file        = file;
    this.onProgress  = opts.onProgress  || null;
    this._chunkSize  = opts.chunkSize   || null;
    this._maxRetries = opts.maxRetries  ?? 3;
  }

  /** Detect effective connection type and return an appropriate chunk size. */
  static _adaptiveChunkSize() {
    try {
      const conn = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
      if (conn) {
        const type = conn.effectiveType || conn.type || '';
        // 2g / slow-2g: very small chunks to avoid timeouts
        if (type === '2g' || type === 'slow-2g') return 512 * 1024;       // 512 KB
        // 3g / cellular: keep chunks manageable
        if (type === '3g' || conn.type === 'cellular') return 2 * 1024 * 1024; // 2 MB
        // 4g / wifi: comfortable default
        if (type === '4g') return 8 * 1024 * 1024;                         // 8 MB
      }
    } catch {}
    // Safe default for unknown — works on most mobile connections
    return 5 * 1024 * 1024; // 5 MB
  }

  /** True if the device appears to be on cellular. */
  static onCellular() {
    try {
      const conn = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
      if (!conn) return false;
      return conn.type === 'cellular' ||
             ['slow-2g','2g','3g'].includes(conn.effectiveType || '');
    } catch { return false; }
  }

  async upload() {
    const file        = this.file;
    const filename    = file.name || 'upload';
    const contentType = file.type || 'application/octet-stream';

    // Prefer caller override, then adaptive detection, then server hint
    const adaptiveSize = this._chunkSize || ChunkedUploader._adaptiveChunkSize();

    // ── 1. Init ──────────────────────────────────────────────────────────────
    const guessChunks = Math.max(1, Math.ceil(file.size / adaptiveSize));
    const initForm = new FormData();
    initForm.append('filename',     filename);
    initForm.append('content_type', contentType);
    initForm.append('total_chunks', String(guessChunks));

    const initResp = await fetch('/api/upload/init', { method: 'POST', body: initForm });
    if (!initResp.ok) {
      throw new Error(`Upload init failed: ${initResp.status} ${await initResp.text()}`);
    }
    const { upload_id, chunk_size_hint } = await initResp.json();

    // Use our adaptive size; fall back to server hint only if we have no preference
    const chunkSize      = this._chunkSize || adaptiveSize || chunk_size_hint || (5 * 1024 * 1024);
    const totalChunks    = Math.max(1, Math.ceil(file.size / chunkSize));

    // ── 2. Upload chunks with retry ──────────────────────────────────────────
    let uploaded = 0;
    for (let i = 0; i < totalChunks; i++) {
      const start = i * chunkSize;
      const end   = Math.min(start + chunkSize, file.size);
      const blob  = file.slice(start, end);

      let lastErr;
      for (let attempt = 0; attempt <= this._maxRetries; attempt++) {
        if (attempt > 0) {
          // Exponential backoff: 1s, 2s, 4s
          await new Promise(r => setTimeout(r, 1000 * Math.pow(2, attempt - 1)));
        }
        try {
          const chunkForm = new FormData();
          chunkForm.append('chunk_index', String(i));
          chunkForm.append('file', blob, filename);

          const chunkResp = await fetch(`/api/upload/${upload_id}/chunk`, {
            method: 'POST',
            body:   chunkForm,
          });
          if (!chunkResp.ok) {
            throw new Error(`HTTP ${chunkResp.status}`);
          }
          lastErr = null;
          break; // success
        } catch (err) {
          lastErr = err;
        }
      }
      if (lastErr) {
        throw new Error(`Chunk ${i + 1}/${totalChunks} failed after ${this._maxRetries} retries: ${lastErr.message}`);
      }

      uploaded += (end - start);
      if (this.onProgress) {
        this.onProgress(uploaded, file.size, i + 1, totalChunks);
      }
    }

    // ── 3. Complete ───────────────────────────────────────────────────────────
    const completeResp = await fetch(`/api/upload/${upload_id}/complete`, { method: 'POST' });
    if (!completeResp.ok) {
      throw new Error(`Upload complete failed: ${completeResp.status} ${await completeResp.text()}`);
    }

    return upload_id;
  }
}

/** Threshold above which we use chunked upload (bytes). */
ChunkedUploader.LARGE_FILE_THRESHOLD = 5 * 1024 * 1024; // 5 MB — lower threshold so mobile always uses chunked
