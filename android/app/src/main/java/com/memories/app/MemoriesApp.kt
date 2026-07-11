package com.memories.app

import android.app.Application
import com.memories.app.reminder.NotificationHelper
import com.memories.app.reminder.ReminderScheduler

/**
 * Application entry point. Kept minimal; exists so we have a stable place to
 * initialise shared singletons (settings, notification channels) as the app grows.
 */
class MemoriesApp : Application() {
    override fun onCreate() {
        super.onCreate()
        NotificationHelper.ensureChannel(this)
        ReminderScheduler.schedule(this)
    }
}
