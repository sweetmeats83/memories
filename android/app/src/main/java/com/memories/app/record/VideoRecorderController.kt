package com.memories.app.record

import android.annotation.SuppressLint
import android.content.Context
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.File

/**
 * Video capture via CameraX: H.264 + AAC in an MP4 container with a live preview.
 * Supports front/back switching (when not recording) and pause/resume.
 */
class VideoRecorderController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
) : CaptureController {

    private var cameraProvider: ProcessCameraProvider? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var activeRecording: Recording? = null
    private var outputFile: File? = null
    private var lensFacing = CameraSelector.LENS_FACING_FRONT

    private var pendingResult: ((File, String) -> Unit)? = null
    private var pendingError: ((Throwable) -> Unit)? = null

    /** Invoked once the camera preview is bound and ready to record. */
    var onReady: (() -> Unit)? = null

    override fun prepare() {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            cameraProvider = future.get()
            bind()
            onReady?.invoke()
        }, ContextCompat.getMainExecutor(context))
    }

    private fun bind() {
        val provider = cameraProvider ?: return
        provider.unbindAll()

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }
        val recorder = Recorder.Builder()
            .setQualitySelector(
                QualitySelector.fromOrderedList(listOf(Quality.HD, Quality.SD))
            )
            .build()
        val capture = VideoCapture.withOutput(recorder)
        videoCapture = capture

        val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        provider.bindToLifecycle(lifecycleOwner, selector, preview, capture)
    }

    /** Toggle front/back. No-op while a recording is active. */
    fun switchCamera() {
        if (activeRecording != null) return
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
            CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT
        bind()
    }

    @SuppressLint("MissingPermission") // RECORD_AUDIO is enforced by RecordActivity.
    override fun start() {
        val capture = videoCapture ?: run {
            pendingError?.invoke(IllegalStateException("Camera not ready")); return
        }
        val file = captureFile(context.cacheDir, "mp4")
        outputFile = file
        val options = FileOutputOptions.Builder(file).build()

        activeRecording = capture.output
            .prepareRecording(context, options)
            .withAudioEnabled()
            .start(ContextCompat.getMainExecutor(context)) { event ->
                if (event is VideoRecordEvent.Finalize) {
                    val out = outputFile
                    if (event.hasError() || out == null) {
                        pendingError?.invoke(
                            IllegalStateException("Recording error: ${event.error}")
                        )
                    } else {
                        pendingResult?.invoke(out, "video/mp4")
                    }
                    pendingResult = null
                    pendingError = null
                }
            }
    }

    override fun pause() { runCatching { activeRecording?.pause() } }
    override fun resume() { runCatching { activeRecording?.resume() } }

    override fun stop(onResult: (File, String) -> Unit, onError: (Throwable) -> Unit) {
        val rec = activeRecording
        if (rec == null) { onError(IllegalStateException("Not recording")); return }
        pendingResult = onResult
        pendingError = onError
        rec.stop()            // Finalize event delivers the file to the consumer above.
        activeRecording = null
    }

    override fun release() {
        runCatching { activeRecording?.stop() }
        activeRecording = null
        runCatching { cameraProvider?.unbindAll() }
    }
}
