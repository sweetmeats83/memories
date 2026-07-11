package com.memories.app.web

import android.webkit.JavascriptInterface

/**
 * JavaScript -> native bridge, exposed to the WebView as `AndroidNative`.
 *
 * The injected hook (see MainActivity) rewrites the web app's
 * `window.RecordOverlay.start(cfg)` so that a *primary* memory recording calls
 * [startPrimaryRecord] with a JSON payload instead of using the browser recorder.
 * Segment recording and everything else stay on the web path.
 */
class NativeBridge(private val onStartPrimary: (String) -> Unit) {

    /** Called from JS with a JSON string: {mode, promptText, baseFields:{...}}. */
    @JavascriptInterface
    fun startPrimaryRecord(payloadJson: String) {
        onStartPrimary(payloadJson)
    }

    /** Lets us detect the native app from web code if ever needed. */
    @JavascriptInterface
    fun isNativeApp(): Boolean = true

    companion object {
        const val NAME = "AndroidNative"

        /**
         * JS injected on every page load. Idempotent. Wraps RecordOverlay.start so
         * primary captures route to the native recorder; retries briefly because
         * RecordOverlay binds shortly after DOMContentLoaded.
         */
        val HOOK_SCRIPT = """
            (function(){
              // Defensive: the recorder overlay's open/closed check keys off aria-modal,
              // but its setMode() guard checks a 'hidden' class that isn't present until
              // the overlay is closed once. On the edit page, select('audio') runs on load
              // and would fire getUserMedia prematurely. Marking a not-open overlay as
              // 'hidden' makes that guard evaluate correctly even on un-patched servers.
              try {
                var ov = document.getElementById('captureOverlay');
                if (ov && ov.getAttribute('aria-modal') !== 'true') ov.classList.add('hidden');
              } catch (e) {}
              function install(){
                if (!window.RecordOverlay || !window.RecordOverlay.start) return false;
                if (window.RecordOverlay.__nativePatched) return true;
                var orig = window.RecordOverlay.start.bind(window.RecordOverlay);
                window.RecordOverlay.start = function(cfg){
                  try {
                    cfg = cfg || {};
                    var kind = cfg.kind || 'segment';
                    if (kind === 'primary' && window.${NAME} && window.${NAME}.startPrimaryRecord) {
                      var base = (cfg.primary && cfg.primary.baseFields) || {};
                      window.${NAME}.startPrimaryRecord(JSON.stringify({
                        mode: cfg.mode || '',
                        promptText: cfg.promptText || '',
                        baseFields: base
                      }));
                      return;
                    }
                  } catch (e) { console.warn('native record hook error', e); }
                  return orig(cfg);
                };
                window.RecordOverlay.__nativePatched = true;
                return true;
              }
              if (!install()){
                var n = 0;
                var iv = setInterval(function(){ if (install() || ++n > 40) clearInterval(iv); }, 100);
              }
            })();
        """.trimIndent()
    }
}
