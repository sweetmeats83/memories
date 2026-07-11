package com.memories.app

import android.content.Context

/**
 * Tiny persistence wrapper around SharedPreferences. Holds the one piece of
 * config the app truly needs before it can do anything: the server base URL.
 */
class SettingsStore(context: Context) {

    private val prefs = context.applicationContext
        .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    /** Normalised base URL (scheme + host [+ port]), no trailing slash, or null. */
    var baseUrl: String?
        get() = prefs.getString(KEY_BASE_URL, null)
        set(value) {
            prefs.edit().putString(KEY_BASE_URL, value?.let { normalize(it) }).apply()
        }

    fun hasBaseUrl(): Boolean = !baseUrl.isNullOrBlank()

    /** Epoch day (see java.time.LocalDate.toEpochDay) the reminder notification last fired, or -1. */
    var lastReminderEpochDay: Long
        get() = prefs.getLong(KEY_LAST_REMINDER_DAY, -1L)
        set(value) = prefs.edit().putLong(KEY_LAST_REMINDER_DAY, value).apply()

    /** The weekly prompt id the last reminder notification was about, or -1. */
    var lastReminderPromptId: Int
        get() = prefs.getInt(KEY_LAST_REMINDER_PROMPT_ID, -1)
        set(value) = prefs.edit().putInt(KEY_LAST_REMINDER_PROMPT_ID, value).apply()

    companion object {
        const val PREFS_NAME = "memories_prefs"
        private const val KEY_BASE_URL = "server_url"
        private const val KEY_LAST_REMINDER_DAY = "last_reminder_epoch_day"
        private const val KEY_LAST_REMINDER_PROMPT_ID = "last_reminder_prompt_id"

        /**
         * Coerce user input into a usable base URL:
         *  - trims whitespace
         *  - adds https:// if no scheme was typed
         *  - drops any trailing slash and path
         * Returns null if it doesn't look like a valid http(s) URL.
         */
        fun normalize(raw: String): String? {
            var s = raw.trim()
            if (s.isEmpty()) return null
            if (!s.startsWith("http://", true) && !s.startsWith("https://", true)) {
                s = "https://$s"
            }
            return try {
                val uri = android.net.Uri.parse(s)
                val scheme = uri.scheme?.lowercase() ?: return null
                if (scheme != "http" && scheme != "https") return null
                val host = uri.host ?: return null
                val port = if (uri.port > 0) ":${uri.port}" else ""
                "$scheme://$host$port"
            } catch (_: Exception) {
                null
            }
        }
    }
}
