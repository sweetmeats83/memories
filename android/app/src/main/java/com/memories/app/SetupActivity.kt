package com.memories.app

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.textfield.TextInputEditText

/**
 * One-time (and re-openable) screen to capture the Memories server URL.
 * On success it stores the URL and launches [MainActivity].
 */
class SetupActivity : AppCompatActivity() {

    private lateinit var settings: SettingsStore

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_setup)
        settings = SettingsStore(this)

        val input = findViewById<TextInputEditText>(R.id.serverUrlInput)
        input.setText(settings.baseUrl ?: "")

        findViewById<Button>(R.id.saveButton).setOnClickListener {
            val normalized = SettingsStore.normalize(input.text?.toString() ?: "")
            if (normalized == null) {
                Toast.makeText(this, R.string.setup_invalid, Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            settings.baseUrl = normalized
            startActivity(Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            })
            setResult(Activity.RESULT_OK)
            finish()
        }
    }
}
