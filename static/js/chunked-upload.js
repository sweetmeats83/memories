/**
 * ChunkedUploader — splits a File/Blob into 50 MB chunks, uploads via
 * POST /api/upload/init → /chunk → /complete, and resolves with the
 * upload_id token that the server can use as a "staged file".
 *
 * Usage:
 *   const uploader = new ChunkedUploader(file, { onProgress });
 *   const uploadId = await uploader.upload();
 */
class ChunkedUploader {
  /**
   * @param {File|Blob} file
   * @param {object}   [opts]
   * @param {function} [opts.onProgress]  called with (bytesUploaded, totalBytes)
   * @param {number}   [opts.chunkSize]   override chunk size (bytes); default from server hint
   */
  constructor(file, opts = {}) {
    this.file       = file;
    this.onProgress = opts.onProgress || null;
    this._chunkSize = opts.chunkSize  || null; // null → use server hint
  }

  async upload() {
    const file = this.file;
    const filename     = file.name || 'upload';
    const contentType  = file.type || 'application/octet-stream';

    // ── 1. Init ──────────────────────────────────────────────────────────────
    // We need total_chunks, but we don't know chunk_size until after init.
    // Use the hint we'll request, or fall back to 50 MB.
    const guessChunk = this._chunkSize || (50 * 1024 * 1024);
    const totalChunks = Math.max(1, Math.ceil(file.size / guessChunk));

    const initForm = new FormData();
    initForm.append('filename',     filename);
    initForm.append('content_type', contentType);
    initForm.append('total_chunks', String(totalChunks));

    const initResp = await fetch('/api/upload/init', {
      method: 'POST',
      body:   initForm,
    });
    if (!initResp.ok) {
      throw new Error(`Upload init failed: ${initResp.status} ${await initResp.text()}`);
    }
    const { upload_id, chunk_size_hint } = await initResp.json();

    // Use the server-suggested chunk size if not overridden
    const chunkSize = this._chunkSize || chunk_size_hint || guessChunk;

    // Recalculate in case server hint differs from our guess
    const realTotalChunks = Math.max(1, Math.ceil(file.size / chunkSize));

    // ── 2. Upload chunks ─────────────────────────────────────────────────────
    let uploaded = 0;
    for (let i = 0; i < realTotalChunks; i++) {
      const start = i * chunkSize;
      const end   = Math.min(start + chunkSize, file.size);
      const blob  = file.slice(start, end);

      const chunkForm = new FormData();
      chunkForm.append('chunk_index', String(i));
      chunkForm.append('file', blob, filename);

      const chunkResp = await fetch(`/api/upload/${upload_id}/chunk`, {
        method: 'POST',
        body:   chunkForm,
      });
      if (!chunkResp.ok) {
        throw new Error(`Chunk ${i} upload failed: ${chunkResp.status} ${await chunkResp.text()}`);
      }

      uploaded += (end - start);
      if (this.onProgress) {
        this.onProgress(uploaded, file.size);
      }
    }

    // ── 3. Complete ───────────────────────────────────────────────────────────
    const completeResp = await fetch(`/api/upload/${upload_id}/complete`, {
      method: 'POST',
    });
    if (!completeResp.ok) {
      throw new Error(`Upload complete failed: ${completeResp.status} ${await completeResp.text()}`);
    }

    return upload_id;
  }
}

/** Threshold above which we use chunked upload (bytes). */
ChunkedUploader.LARGE_FILE_THRESHOLD = 50 * 1024 * 1024; // 50 MB
