package com.memories.app.record

import android.content.Context
import android.media.MediaRecorder
import android.os.Build
import java.io.File
import kotlin.math.log10

/**
 * Audio capture via [MediaRecorder]: AAC in an MP4 container (.m4a). The server
 * transcodes uploads with ffmpeg, so an AAC/MP4 file is accepted the same as the
 * web app's webm/opus.
 */
class AudioRecorderController(private val context: Context) : CaptureController {

    private var recorder: MediaRecorder? = null
    private var outputFile: File? = null
    private var started = false

    override fun prepare() { /* MediaRecorder needs no live preview. */ }

    override fun start() {
        val file = captureFile(context.cacheDir, "m4a")
        outputFile = file
        val rec = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S)
            MediaRecorder(context) else @Suppress("DEPRECATION") MediaRecorder()

        rec.apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            setAudioEncodingBitRate(128_000)
            setAudioSamplingRate(44_100)
            setOutputFile(file.absolutePath)
            prepare()
            start()
        }
        recorder = rec
        started = true
    }

    override fun pause() {
        if (started) runCatching { recorder?.pause() }
    }

    override fun resume() {
        if (started) runCatching { recorder?.resume() }
    }

    override fun stop(onResult: (File, String) -> Unit, onError: (Throwable) -> Unit) {
        val rec = recorder
        val file = outputFile
        if (rec == null || file == null) {
            onError(IllegalStateException("Recorder not started")); return
        }
        try {
            rec.stop()
            rec.release()
            recorder = null
            started = false
            onResult(file, "audio/mp4")
        } catch (e: Exception) {
            // stop() throws if it was stopped too quickly (no valid data captured).
            runCatching { rec.release() }
            recorder = null
            started = false
            onError(e)
        }
    }

    override fun release() {
        runCatching { recorder?.release() }
        recorder = null
        started = false
    }

    /** Map max amplitude (~0..32767) to a 0..100 log scale for the meter. */
    override fun level(): Int {
        if (!started) return 0
        val amp = runCatching { recorder?.maxAmplitude ?: 0 }.getOrDefault(0)
        if (amp <= 0) return 0
        val db = 20.0 * log10(amp.toDouble())          // ~0..90 dB
        return (db / 90.0 * 100).coerceIn(0.0, 100.0).toInt()
    }
}
