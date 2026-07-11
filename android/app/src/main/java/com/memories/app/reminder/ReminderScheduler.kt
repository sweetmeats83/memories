package com.memories.app.reminder

import android.content.Context
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import java.util.concurrent.TimeUnit

/** Schedules the periodic weekly-prompt reminder check. */
object ReminderScheduler {

    private const val UNIQUE_WORK_NAME = "weekly-prompt-reminder"

    // WorkManager's minimum periodic interval is 15 minutes; twice a day is plenty
    // for a "gentle nudge" without being naggy, and gives Doze/battery-optimized
    // devices a wide window to actually run it.
    fun schedule(context: Context) {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .build()

        val request = PeriodicWorkRequestBuilder<ReminderWorker>(12, TimeUnit.HOURS)
            .setConstraints(constraints)
            .setBackoffCriteria(BackoffPolicy.LINEAR, 15, TimeUnit.MINUTES)
            .build()

        WorkManager.getInstance(context).enqueueUniquePeriodicWork(
            UNIQUE_WORK_NAME,
            ExistingPeriodicWorkPolicy.KEEP,
            request,
        )
    }
}
