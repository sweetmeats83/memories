package com.memories.app.net

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.util.concurrent.TimeUnit

/**
 * Native client for the Memories upload + create-response API. Mirrors the web
 * app's flow exactly:
 *
 *   1. POST /api/upload/init            -> { upload_id }
 *   2. POST /api/upload/{id}/chunk      (repeat per chunk, with retry)
 *   3. POST /api/upload/{id}/complete
 *   4. POST /responses/                 (primary_staged_id + base fields)
 *        with Accept: application/json  -> { id, processing_state }
 *
 * Auth is carried by the WebView session cookie via [CookieBridge].
 */
class MemoriesApi(private val baseUrl: String) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    /** Progress callback: (bytesUploaded, totalBytes). */
    fun interface ProgressListener {
        fun onProgress(uploaded: Long, total: Long)
    }

    /**
     * Full pipeline: stage [file] via chunked upload, then create a Response.
     * @param baseFields form fields for POST /responses/ — either
     *   {prompt_id, response_text} or {title, chapter, response_text}.
     * @return the new response id.
     */
    suspend fun uploadPrimary(
        file: File,
        contentType: String,
        baseFields: Map<String, String>,
        progress: ProgressListener? = null,
    ): Int = withContext(Dispatchers.IO) {
        val stagedId = stageFile(file, contentType, progress)
        createResponse(stagedId, baseFields)
    }

    /**
     * Stage a segment recording onto an existing response.
     * POST /response/{id}/segments with staged_id.
     */
    suspend fun uploadSegment(
        responseId: Int,
        file: File,
        contentType: String,
        progress: ProgressListener? = null,
    ): Unit = withContext(Dispatchers.IO) {
        val stagedId = stageFile(file, contentType, progress)
        val body = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("staged_id", stagedId)
            .build()
        execExpectOk(post("$baseUrl/response/$responseId/segments", body))
    }

    // ---- Steps -------------------------------------------------------------

    private fun stageFile(
        file: File,
        contentType: String,
        progress: ProgressListener?,
    ): String {
        val total = file.length()
        val totalChunks = maxOf(1, ((total + CHUNK_SIZE - 1) / CHUNK_SIZE).toInt())

        // 1. init
        val initBody = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("filename", file.name)
            .addFormDataPart("content_type", contentType)
            .addFormDataPart("total_chunks", totalChunks.toString())
            .build()
        val uploadId = execJson(post("$baseUrl/api/upload/init", initBody))
            .optString("upload_id")
            .ifBlank { throw ApiException("upload/init returned no upload_id") }

        // 2. chunks (with retry + exponential backoff)
        var uploaded = 0L
        RandomAccessFile(file, "r").use { raf ->
            for (index in 0 until totalChunks) {
                val start = index.toLong() * CHUNK_SIZE
                val size = minOf(CHUNK_SIZE.toLong(), total - start).toInt()
                val buffer = ByteArray(size)
                raf.seek(start)
                raf.readFully(buffer)

                val chunkBody = MultipartBody.Builder().setType(MultipartBody.FORM)
                    .addFormDataPart("chunk_index", index.toString())
                    .addFormDataPart(
                        "file", file.name,
                        buffer.toRequestBody(OCTET_STREAM, 0, size)
                    )
                    .build()

                uploadChunkWithRetry("$baseUrl/api/upload/$uploadId/chunk", chunkBody)
                uploaded += size
                progress?.onProgress(uploaded, total)
            }
        }

        // 3. complete
        execExpectOk(post("$baseUrl/api/upload/$uploadId/complete", EMPTY_BODY))
        return uploadId
    }

    private fun createResponse(stagedId: String, baseFields: Map<String, String>): Int {
        val builder = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("primary_staged_id", stagedId)
        for ((k, v) in baseFields) builder.addFormDataPart(k, v)

        val req = Request.Builder()
            .url("$baseUrl/responses/")
            .header("Accept", "application/json")
            .apply { CookieBridge.cookieHeaderFor(baseUrl)?.let { header("Cookie", it) } }
            .post(builder.build())
            .build()

        val json = execJson(req)
        val id = json.optInt("id", -1)
        if (id <= 0) throw ApiException("create response returned no id: $json")
        return id
    }

    // ---- HTTP helpers ------------------------------------------------------

    private fun post(url: String, body: okhttp3.RequestBody): Request =
        Request.Builder()
            .url(url)
            .apply { CookieBridge.cookieHeaderFor(baseUrl)?.let { header("Cookie", it) } }
            .post(body)
            .build()

    private fun uploadChunkWithRetry(url: String, bodyFactory: MultipartBody) {
        var lastErr: Exception? = null
        for (attempt in 0..MAX_RETRIES) {
            if (attempt > 0) Thread.sleep(1000L * (1 shl (attempt - 1))) // 1s,2s,4s
            try {
                execExpectOk(
                    Request.Builder().url(url)
                        .apply { CookieBridge.cookieHeaderFor(baseUrl)?.let { header("Cookie", it) } }
                        .post(bodyFactory)
                        .build()
                )
                return
            } catch (e: Exception) {
                lastErr = e
            }
        }
        throw ApiException("chunk upload failed after $MAX_RETRIES retries", lastErr)
    }

    private fun execExpectOk(req: Request) {
        client.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) {
                throw ApiException("HTTP ${resp.code}: ${resp.body?.string()?.take(300)}")
            }
        }
    }

    private fun execJson(req: Request): JSONObject {
        client.newCall(req).execute().use { resp ->
            val text = resp.body?.string().orEmpty()
            if (!resp.isSuccessful) {
                throw ApiException("HTTP ${resp.code}: ${text.take(300)}")
            }
            return try {
                JSONObject(text)
            } catch (e: Exception) {
                throw ApiException("Expected JSON, got: ${text.take(300)}", e)
            }
        }
    }

    class ApiException(message: String, cause: Throwable? = null) : Exception(message, cause)

    companion object {
        private const val CHUNK_SIZE = 5 * 1024 * 1024   // 5 MB — matches web mobile default
        private const val MAX_RETRIES = 3
        private val OCTET_STREAM = "application/octet-stream".toMediaTypeOrNull()
        private val EMPTY_BODY = ByteArray(0).toRequestBody(null, 0, 0)
    }
}
