package com.memories.app.reminder

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.PowerManager
import android.provider.Settings
import androidx.appcompat.app.AlertDialog
import com.memories.app.R
import com.memories.app.SettingsStore

/**
 * Weekly-prompt reminders run on a WorkManager periodic job (see [ReminderScheduler]),
 * not push — many OEM Android skins (MIUI, ColorOS, FuntouchOS, EMUI, ...) kill or
 * defer that kind of background work unless the app is explicitly exempted from
 * battery optimization, and some additionally gate it behind an OEM-specific
 * "autostart" toggle with no portable API. This offers the standard exemption
 * dialog plus a best-effort deep link into the manufacturer's autostart screen,
 * falling back to the app's system settings page if nothing resolves.
 */
object BackgroundReliabilityHelper {

    fun isIgnoringBatteryOptimizations(context: Context): Boolean {
        val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager ?: return true
        return pm.isIgnoringBatteryOptimizations(context.packageName)
    }

    /** Auto-prompt path: shows once per install, and only if reminders are actually at risk. */
    fun maybeShow(activity: Activity) {
        val settings = SettingsStore(activity)
        if (settings.hasPromptedBackgroundReliability) return
        if (isIgnoringBatteryOptimizations(activity)) return
        settings.hasPromptedBackgroundReliability = true
        show(activity)
    }

    /** Manual path (e.g. a "Background & notifications" button in Setup) — always shows. */
    fun show(activity: Activity) {
        val autostartIntent = autostartIntentFor(activity)
        val builder = AlertDialog.Builder(activity)
            .setTitle(R.string.bg_dialog_title)
            .setMessage(R.string.bg_dialog_message)
            .setNegativeButton(R.string.bg_dialog_not_now, null)
            .setPositiveButton(R.string.bg_dialog_allow) { _, _ -> requestIgnoreBatteryOptimizations(activity) }

        if (autostartIntent != null) {
            builder.setNeutralButton(R.string.bg_dialog_autostart) { _, _ -> launchAutostart(activity, autostartIntent) }
        }
        builder.show()
    }

    private fun requestIgnoreBatteryOptimizations(context: Context) {
        val intent = Intent(
            Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
            Uri.parse("package:${context.packageName}"),
        )
        try {
            context.startActivity(intent)
        } catch (_: ActivityNotFoundException) {
            openAppSettings(context)
        }
    }

    private fun launchAutostart(context: Context, intent: Intent) {
        try {
            context.startActivity(intent)
        } catch (_: Exception) {
            // Undocumented OEM screen didn't resolve on this ROM version — fall back.
            openAppSettings(context)
        }
    }

    private fun openAppSettings(context: Context) {
        val intent = Intent(
            Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
            Uri.parse("package:${context.packageName}"),
        )
        runCatching { context.startActivity(intent) }
    }

    /**
     * Best-effort deep link into the manufacturer's autostart/background-permission
     * screen. These component names are undocumented, unofficial, and vary by ROM
     * and OS version — every caller must be ready for them to fail
     * (ActivityNotFoundException / SecurityException) and fall back gracefully,
     * which [launchAutostart] does.
     */
    private fun autostartIntentFor(context: Context): Intent? {
        val candidates: List<Pair<String, String>> = when (Build.MANUFACTURER.lowercase()) {
            "xiaomi" -> listOf(
                "com.miui.securitycenter" to "com.miui.permcenter.autostart.AutoStartManagementActivity",
            )
            "oppo" -> listOf(
                "com.coloros.safecenter" to "com.coloros.safecenter.permission.startup.StartupAppListActivity",
                "com.coloros.safecenter" to "com.coloros.safecenter.startupapp.StartupAppListActivity",
                "com.oppo.safe" to "com.oppo.safe.permission.startup.StartupAppListActivity",
            )
            "vivo" -> listOf(
                "com.vivo.permissionmanager" to "com.vivo.permissionmanager.activity.BgStartUpManagerActivity",
                "com.iqoo.secure" to "com.iqoo.secure.ui.phoneoptimize.AddWhiteListActivity",
            )
            "huawei", "honor" -> listOf(
                "com.huawei.systemmanager" to "com.huawei.systemmanager.startupmgr.ui.StartupNormalAppListActivity",
                "com.huawei.systemmanager" to "com.huawei.systemmanager.optimize.process.ProtectActivity",
            )
            "oneplus" -> listOf(
                "com.oneplus.security" to "com.oneplus.security.chainlaunch.view.ChainLaunchAppListActivity",
            )
            "letv" -> listOf(
                "com.letv.android.letvsafe" to "com.letv.android.letvsafe.AutobootManageActivity",
            )
            "asus" -> listOf(
                "com.asus.mobilemanager" to "com.asus.mobilemanager.autostart.AutoStartActivity",
            )
            else -> emptyList()
        }
        val pm = context.packageManager
        for ((pkg, cls) in candidates) {
            val intent = Intent().apply {
                component = ComponentName(pkg, cls)
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            if (intent.resolveActivity(pm) != null) return intent
        }
        return null
    }
}
