package com.memories.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.webkit.PermissionRequest
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.addCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.memories.app.databinding.ActivityMainBinding
import com.memories.app.record.RecordActivity
import com.memories.app.web.NativeBridge

/**
 * Host screen. Renders the entire Memories web app in a WebView so every feature
 * is available, while the "record a memory" button is hijacked to launch the
 * native recorder (see [NativeBridge]).
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var settings: SettingsStore
    private val baseUrl get() = settings.baseUrl.orEmpty()

    // Pending WebView <input type=file> callback.
    private var fileChooserCallback: ValueCallback<Array<Uri>>? = null

    // A web-originated getUserMedia permission awaiting OS permission grant.
    private var pendingWebPermission: PermissionRequest? = null

    private val fileChooserLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            val callback = fileChooserCallback ?: return@registerForActivityResult
            fileChooserCallback = null
            callback.onReceiveValue(parseChooserResult(result.resultCode, result.data))
        }

    private val recordLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            val id = result.data?.getIntExtra(RecordActivity.RESULT_RESPONSE_ID, -1) ?: -1
            if (id > 0) {
                binding.webView.loadUrl("$baseUrl/response/$id/edit")
            }
        }

    private val webPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
            val req = pendingWebPermission
            pendingWebPermission = null
            if (req != null) {
                val allGranted = grants.values.all { it }
                if (allGranted) req.grant(req.resources) else req.deny()
            }
        }

    private val notificationPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { /* no-op either way */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        settings = SettingsStore(this)
        if (!settings.hasBaseUrl()) {
            startActivity(Intent(this, SetupActivity::class.java))
            finish()
            return
        }

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        configureWebView()
        requestNotificationPermissionIfNeeded()
        binding.retryButton.setOnClickListener {
            binding.errorView.visibility = android.view.View.GONE
            binding.webView.loadUrl(baseUrl + "/")
        }
        binding.changeServerButton.setOnClickListener { openServerSetup() }

        if (savedInstanceState == null) {
            binding.webView.loadUrl(baseUrl + "/")
        } else {
            binding.webView.restoreState(savedInstanceState)
        }

        onBackPressedDispatcher.addCallback(this) {
            if (binding.webView.canGoBack()) binding.webView.goBack() else finish()
        }
    }

    @Suppress("SetJavaScriptEnabled")
    private fun configureWebView() {
        val web = binding.webView
        web.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            mediaPlaybackRequiresUserGesture = false
            allowFileAccess = true
            loadWithOverviewMode = true
            useWideViewPort = true
            userAgentString = "$userAgentString MemoriesAndroid/1.0"
        }
        android.webkit.CookieManager.getInstance().setAcceptThirdPartyCookies(web, true)

        web.addJavascriptInterface(
            NativeBridge(onStartPrimary = ::launchNativeRecorder),
            NativeBridge.NAME,
        )

        web.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView, url: String?) {
                CookieBridgeFlush()
                view.evaluateJavascript(NativeBridge.HOOK_SCRIPT, null)
            }

            override fun shouldOverrideUrlLoading(
                view: WebView,
                request: WebResourceRequest,
            ): Boolean {
                val url = request.url
                // Keep our own host inside the WebView; send everything else out.
                return if (isSameHost(url)) {
                    false
                } else {
                    runCatching { startActivity(Intent(Intent.ACTION_VIEW, url)) }
                    true
                }
            }

            override fun onReceivedError(
                view: WebView,
                request: WebResourceRequest,
                error: WebResourceError,
            ) {
                if (request.isForMainFrame) showError()
            }
        }

        web.webChromeClient = object : WebChromeClient() {
            override fun onProgressChanged(view: WebView, newProgress: Int) {
                binding.topProgress.apply {
                    progress = newProgress
                    visibility = if (newProgress in 1..99) android.view.View.VISIBLE
                    else android.view.View.GONE
                }
            }

            // Grant mic/camera to the web recorder (segment path) — needs OS perms.
            override fun onPermissionRequest(request: PermissionRequest) {
                val needed = mutableListOf<String>()
                request.resources.forEach { res ->
                    when (res) {
                        PermissionRequest.RESOURCE_AUDIO_CAPTURE ->
                            needed += Manifest.permission.RECORD_AUDIO
                        PermissionRequest.RESOURCE_VIDEO_CAPTURE ->
                            needed += Manifest.permission.CAMERA
                    }
                }
                val missing = needed.filter {
                    ContextCompat.checkSelfPermission(this@MainActivity, it) !=
                        PackageManager.PERMISSION_GRANTED
                }
                if (missing.isEmpty()) {
                    request.grant(request.resources)
                } else {
                    pendingWebPermission = request
                    webPermissionLauncher.launch(missing.toTypedArray())
                }
            }

            override fun onShowFileChooser(
                view: WebView,
                filePathCallback: ValueCallback<Array<Uri>>,
                params: FileChooserParams,
            ): Boolean {
                fileChooserCallback?.onReceiveValue(null)
                fileChooserCallback = filePathCallback
                val intent = params.createIntent()
                return try {
                    fileChooserLauncher.launch(intent)
                    true
                } catch (e: Exception) {
                    fileChooserCallback = null
                    false
                }
            }
        }
    }

    /** Parse the JSON payload from the web hook and start [RecordActivity]. */
    private fun launchNativeRecorder(payloadJson: String) {
        runOnUiThread {
            val intent = Intent(this, RecordActivity::class.java).apply {
                putExtra(RecordActivity.EXTRA_PAYLOAD, payloadJson)
                putExtra(RecordActivity.EXTRA_BASE_URL, baseUrl)
            }
            recordLauncher.launch(intent)
        }
    }

    private fun isSameHost(url: Uri): Boolean {
        val base = Uri.parse(baseUrl)
        return url.host != null && url.host.equals(base.host, ignoreCase = true)
    }

    private fun parseChooserResult(resultCode: Int, data: Intent?): Array<Uri>? {
        if (resultCode != RESULT_OK || data == null) return null
        data.clipData?.let { clip ->
            return Array(clip.itemCount) { clip.getItemAt(it).uri }
        }
        return data.data?.let { arrayOf(it) }
    }

    private fun showError() {
        binding.errorView.visibility = android.view.View.VISIBLE
    }

    private fun openServerSetup() {
        startActivity(Intent(this, SetupActivity::class.java))
    }

    /** Weekly-prompt reminder notifications need this on Android 13+. */
    private fun requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) ==
            PackageManager.PERMISSION_GRANTED
        if (!granted) notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
    }

    private fun CookieBridgeFlush() {
        // Persist the session cookie so native uploads (and relaunches) stay authed.
        com.memories.app.net.CookieBridge.flush()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        // May be destroyed before setContentView on the "no server URL yet" path.
        if (::binding.isInitialized) binding.webView.saveState(outState)
    }

    override fun onDestroy() {
        if (::binding.isInitialized) binding.webView.destroy()
        super.onDestroy()
    }
}
