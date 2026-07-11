package com.memories.app.reminder

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import com.memories.app.MainActivity
import com.memories.app.R

/** Channel + builder for the local "your weekly prompt is waiting" reminder. */
object NotificationHelper {

    const val CHANNEL_ID = "memory_reminders"
    private const val NOTIFICATION_ID = 1001

    fun ensureChannel(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = context.getSystemService(NotificationManager::class.java) ?: return
        val channel = NotificationChannel(
            CHANNEL_ID,
            context.getString(R.string.reminder_channel_name),
            NotificationManager.IMPORTANCE_DEFAULT,
        ).apply {
            description = context.getString(R.string.reminder_channel_desc)
        }
        manager.createNotificationChannel(channel)
    }

    /** Show (or replace) the reminder notification. Tapping opens the app. */
    fun show(context: Context, title: String, body: String) {
        val tapIntent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val flags = PendingIntent.FLAG_UPDATE_CURRENT or
            (if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) PendingIntent.FLAG_IMMUTABLE else 0)
        val pendingIntent = PendingIntent.getActivity(context, 0, tapIntent, flags)

        val notification = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .build()

        // Runtime POST_NOTIFICATIONS permission may not be granted yet on Android 13+;
        // NotificationManagerCompat silently no-ops rather than throwing in that case,
        // but we still guard explicitly to avoid a SecurityException on stricter OEMs.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
            androidx.core.content.ContextCompat.checkSelfPermission(
                context, android.Manifest.permission.POST_NOTIFICATIONS
            ) == android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            NotificationManagerCompat.from(context).notify(NOTIFICATION_ID, notification)
        }
    }
}
