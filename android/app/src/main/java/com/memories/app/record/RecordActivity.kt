package com.memories.app.record

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.view.View
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.memories.app.databinding.ActivityRecordBinding
import com.memories.app.net.MemoriesApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

/**
 * Full-screen native recorder. Launched from the WebView's record button via the
 * JS bridge. Captures audio or video, uploads it through [MemoriesApi], and
 * returns the new response id so the host can open its edit page.
 */
class RecordActivity : AppCompatActivity() {

    private lateinit var binding: ActivityRecordBinding
    private lateinit var api: MemoriesApi

    private var mode = "audio"
    private val baseFields = mutableMapOf<String, String>()

    private var controller: CaptureController? = null
    private var recording = false
    private var paused = false

    // Elapsed-time bookkeeping (pause-aware).
    private var accumulatedMs = 0L
    private var segStartMs = 0L
    private var tickJob: Job? = null

    // Last successful capture, kept so "retry" can re-upload without re-recording.
    private var lastFile: File? = null
    private var lastContentType: String? = null

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
            if (grants[Manifest.permission.RECORD_AUDIO] == false) {
                Toast.makeText(this, com.memories.app.R.string.rec_perm_needed, Toast.LENGTH_LONG).show()
                finish()
            } else {
                initAfterPermissions()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityRecordBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val baseUrl = intent.getStringExtra(EXTRA_BASE_URL).orEmpty()
        api = MemoriesApi(baseUrl)
        parsePayload(intent.getStringExtra(EXTRA_PAYLOAD))

        binding.promptText.apply {
            val p = baseFields["_promptText"] ?: ""
            if (p.isNotBlank()) { text = p; visibility = View.VISIBLE }
        }
        baseFields.remove("_promptText")

        wireControls()
        requestPermissions()
    }

    private fun parsePayload(json: String?) {
        if (json.isNullOrBlank()) return
        runCatching {
            val obj = JSONObject(json)
            mode = obj.optString("mode").ifBlank { "audio" }
            obj.optString("promptText").takeIf { it.isNotBlank() }
                ?.let { baseFields["_promptText"] = it }
            obj.optJSONObject("baseFields")?.let { bf ->
                bf.keys().forEach { k -> baseFields[k] = bf.get(k).toString() }
            }
        }
    }

    private fun requestPermissions() {
        permissionLauncher.launch(
            arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA)
        )
    }

    private fun initAfterPermissions() {
        // Fall back to audio if the requested video mode lacks camera permission.
        if (mode == "video" && !hasCamera()) mode = "audio"
        applyMode(mode)
        binding.modeToggle.check(
            if (mode == "video") binding.btnModeVideo.id else binding.btnModeAudio.id
        )
    }

    private fun wireControls() {
        binding.btnModeAudio.setOnClickListener { if (!recording) applyMode("audio") }
        binding.btnModeVideo.setOnClickListener {
            if (recording) return@setOnClickListener
            if (hasCamera()) applyMode("video")
            else permissionLauncher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO))
        }
        binding.btnRecord.setOnClickListener { onRecordToggle() }
        binding.btnFinish.setOnClickListener { onFinish() }
        binding.btnClose.setOnClickListener { cancelAndExit() }
        binding.btnFlip.setOnClickListener {
            (controller as? VideoRecorderController)?.switchCamera()
        }
        binding.veilRetry.setOnClickListener {
            val f = lastFile; val ct = lastContentType
            if (f != null && ct != null) uploadFile(f, ct)
        }
        onBackPressedDispatcher.addCallback(this) { cancelAndExit() }
    }

    /** Switch capture mode; rebuilds the controller and preview. */
    private fun applyMode(newMode: String) {
        controller?.release()
        mode = newMode
        val video = newMode == "video"
        binding.previewView.visibility = if (video) View.VISIBLE else View.GONE
        binding.audioView.visibility = if (video) View.GONE else View.VISIBLE
        binding.btnFlip.visibility = if (video) View.VISIBLE else View.GONE

        controller = if (video) {
            binding.btnRecord.isEnabled = false // enable once camera is ready
            VideoRecorderController(this, this, binding.previewView).apply {
                onReady = { runOnUiThread { binding.btnRecord.isEnabled = true } }
                prepare()
            }
        } else {
            binding.btnRecord.isEnabled = true
            AudioRecorderController(this).apply { prepare() }
        }
    }

    private fun onRecordToggle() {
        val c = controller ?: return
        if (!recording) {
            runCatching { c.start() }.onFailure {
                Toast.makeText(this, "Could not start recording.", Toast.LENGTH_SHORT).show()
                return
            }
            recording = true; paused = false
            accumulatedMs = 0L; segStartMs = SystemClock.elapsedRealtime()
            lockModeToggle()
            binding.btnRecord.text = getString(com.memories.app.R.string.rec_pause)
            startTicker()
        } else if (!paused) {
            c.pause(); paused = true
            accumulatedMs += SystemClock.elapsedRealtime() - segStartMs
            binding.btnRecord.text = getString(com.memories.app.R.string.rec_resume)
        } else {
            c.resume(); paused = false
            segStartMs = SystemClock.elapsedRealtime()
            binding.btnRecord.text = getString(com.memories.app.R.string.rec_pause)
        }
    }

    private fun onFinish() {
        val c = controller
        if (!recording || c == null) { cancelAndExit(); return }
        stopTicker()
        showVeil(getString(com.memories.app.R.string.rec_saving))
        c.stop(
            onResult = { file, contentType ->
                recording = false
                lastFile = file; lastContentType = contentType
                uploadFile(file, contentType)
            },
            onError = { err -> showRetry(err.message ?: "Recording failed.") }
        )
    }

    private fun uploadFile(file: File, contentType: String) {
        showVeil(getString(com.memories.app.R.string.rec_uploading))
        binding.veilProgress.visibility = View.VISIBLE
        binding.veilRetry.visibility = View.GONE
        lifecycleScope.launch {
            try {
                val id = withContext(Dispatchers.IO) {
                    api.uploadPrimary(
                        file = file,
                        contentType = contentType,
                        baseFields = baseFields,
                        progress = { uploaded, total ->
                            val pct = if (total > 0) (uploaded * 100 / total).toInt() else 0
                            runOnUiThread { binding.veilProgress.progress = pct }
                        },
                    )
                }
                file.delete()
                setResult(RESULT_OK, android.content.Intent().putExtra(RESULT_RESPONSE_ID, id))
                finish()
            } catch (e: Exception) {
                showRetry(e.message ?: getString(com.memories.app.R.string.rec_failed))
            }
        }
    }

    // ---- Ticker (timer + level meter) --------------------------------------

    private fun startTicker() {
        tickJob?.cancel()
        tickJob = lifecycleScope.launch {
            while (true) {
                val ms = accumulatedMs +
                    if (recording && !paused) SystemClock.elapsedRealtime() - segStartMs else 0L
                binding.timerText.text = String.format("%.1fs", ms / 1000.0)
                if (mode == "audio") binding.audioMeter.progress = controller?.level() ?: 0
                delay(100)
            }
        }
    }

    private fun stopTicker() { tickJob?.cancel(); tickJob = null }

    // ---- Veil / UI state ---------------------------------------------------

    private fun showVeil(title: String) {
        binding.veil.visibility = View.VISIBLE
        binding.veilSpinner.visibility = View.VISIBLE
        binding.veilTitle.text = title
        binding.veilRetry.visibility = View.GONE
    }

    private fun showRetry(message: String) {
        binding.veil.visibility = View.VISIBLE
        binding.veilSpinner.visibility = View.GONE
        binding.veilProgress.visibility = View.GONE
        binding.veilTitle.text = message
        binding.veilRetry.visibility = View.VISIBLE
    }

    private fun lockModeToggle() {
        binding.btnModeAudio.isEnabled = false
        binding.btnModeVideo.isEnabled = false
    }

    private fun hasCamera(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun cancelAndExit() {
        stopTicker()
        controller?.release()
        controller = null
        setResult(RESULT_CANCELED)
        finish()
    }

    override fun onDestroy() {
        stopTicker()
        controller?.release()
        super.onDestroy()
    }

    companion object {
        const val EXTRA_PAYLOAD = "payload"
        const val EXTRA_BASE_URL = "base_url"
        const val RESULT_RESPONSE_ID = "response_id"
    }
}
