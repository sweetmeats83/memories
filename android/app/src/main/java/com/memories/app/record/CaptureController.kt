package com.memories.app.record

import java.io.File

/**
 * Common contract for the audio and video recorders so [RecordActivity] can
 * treat them uniformly. Implementations write to a cache file and report it
 * (with its MIME type) when [stop] finalises.
 */
interface CaptureController {

    /** Acquire devices / start preview. Safe to call before [start]. */
    fun prepare()

    fun start()
    fun pause()
    fun resume()

    /** Finalise the recording. [onResult] receives the file + content type. */
    fun stop(onResult: (File, String) -> Unit, onError: (Throwable) -> Unit)

    /** Release all resources (camera, encoder). */
    fun release()

    /** Instantaneous input level 0..100 for the on-screen meter (0 if N/A). */
    fun level(): Int = 0
}

/** Where in-progress captures are written before upload. */
fun captureFile(dir: File, ext: String): File {
    val captures = File(dir, "captures").apply { mkdirs() }
    return File(captures, "capture-${System.currentTimeMillis()}.$ext")
}
