package com.memories.app.net

import android.webkit.CookieManager

/**
 * Bridges the WebView's session cookie to the native OkHttp uploader.
 *
 * The Memories backend authenticates with an httpOnly `session` cookie set by
 * the HTML login flow. JavaScript can't read httpOnly cookies, but Android's
 * native [CookieManager] can — so after the user logs in inside the WebView we
 * simply read the cookie here and replay it on native upload requests.
 */
object CookieBridge {

    /** Full `Cookie:` header value for [url], or null if none is stored. */
    fun cookieHeaderFor(url: String): String? {
        val raw = CookieManager.getInstance().getCookie(url)
        return raw?.takeIf { it.isNotBlank() }
    }

    /** True if a `session` cookie is present for [url]. */
    fun hasSession(url: String): Boolean =
        cookieHeaderFor(url)?.contains("session=") == true

    /** Persist cookies to disk so a killed-and-relaunched app stays logged in. */
    fun flush() {
        CookieManager.getInstance().flush()
    }
}
