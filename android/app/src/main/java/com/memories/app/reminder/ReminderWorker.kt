package com.memories.app.reminder

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.memories.app.SettingsStore
import com.memories.app.net.CookieBridge
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.time.LocalDate
import java.util.concurrent.TimeUnit

/**
 * Periodic background check: is the signed-in user's weekly prompt still
 * unanswered? If so, show a local notification — a distinct "ready" message
 * the first day a given prompt is pending, then a gentler daily nudge for
 * as long as it stays unanswered. Runs entirely against [SettingsStore]'s
 * saved server URL and the WebView's session cookie (via [CookieBridge]);
 * no server-side push infra required.
 */
class ReminderWorker(appContext: Context, params: WorkerParameters) :
    CoroutineWorker(appContext, params) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val settings = SettingsStore(applicationContext)
        val baseUrl = settings.baseUrl ?: return@withContext Result.success()
        if (!CookieBridge.hasSession(baseUrl)) return@withContext Result.success()

        val status = try {
            fetchStatus(baseUrl)
        } catch (e: Exception) {
            // Transient network failure — the next periodic run will try again.
            return@withContext Result.success()
        } ?: return@withContext Result.success()

        if (!status.pending || status.promptId == null) return@withContext Result.success()

        val today = LocalDate.now().toEpochDay()
        val alreadyNotifiedToday =
            settings.lastReminderEpochDay == today && settings.lastReminderPromptId == status.promptId
        if (alreadyNotifiedToday) return@withContext Result.success()

        val isNewPrompt = settings.lastReminderPromptId != status.promptId
        val title = applicationContext.getString(
            if (isNewPrompt) com.memories.app.R.string.reminder_title_new_prompt
            else com.memories.app.R.string.reminder_title_gentle
        )
        val body = status.message
            ?: status.promptText
            ?: applicationContext.getString(com.memories.app.R.string.reminder_channel_desc)

        NotificationHelper.ensureChannel(applicationContext)
        NotificationHelper.show(applicationContext, title, body)

        settings.lastReminderEpochDay = today
        settings.lastReminderPromptId = status.promptId

        Result.success()
    }

    private data class WeeklyStatus(
        val pending: Boolean,
        val promptId: Int?,
        val promptText: String?,
        val message: String?,
    )

    private fun fetchStatus(baseUrl: String): WeeklyStatus? {
        val req = Request.Builder()
            .url("$baseUrl/api/weekly/status")
            .apply { CookieBridge.cookieHeaderFor(baseUrl)?.let { header("Cookie", it) } }
            .get()
            .build()

        client.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) return null
            val json = JSONObject(resp.body?.string().orEmpty())
            return WeeklyStatus(
                pending = json.optBoolean("pending", false),
                promptId = if (json.isNull("prompt_id")) null else json.optInt("prompt_id"),
                promptText = json.optString("prompt_text").takeIf { it.isNotBlank() && !json.isNull("prompt_text") },
                message = json.optString("message").takeIf { it.isNotBlank() && !json.isNull("message") },
            )
        }
    }
}
