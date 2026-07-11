# Memories — Android app

A thin native shell around the existing Memories web app. The whole feature set
runs in a WebView (so nothing has to be re-implemented), while **recording a new
memory launches a native audio/video recorder** with reliable chunked, resumable
uploads.

No backend changes are required.

---

## How it works

```
MainActivity (WebView)                 RecordActivity (native)
  ├─ loads your Memories site            ├─ CameraX (video) / MediaRecorder (audio)
  ├─ user logs in on /login              ├─ pause / resume / flip camera
  │    → session cookie stored in        ├─ chunked upload  /api/upload/init|chunk|complete
  │      Android CookieManager           │     (auth = session cookie, bridged from WebView)
  ├─ injects a JS hook that rewrites      └─ POST /responses/ (primary_staged_id + fields)
  │  window.RecordOverlay.start(...)            → { id }  ⇒ WebView opens /response/{id}/edit
  │  so a *primary* recording calls
  │  AndroidNative.startPrimaryRecord()
  └─ everything else stays in the web UI
     (segment recording still uses the in-page recorder)
```

- **Auth:** the backend uses an httpOnly `session` cookie. JS can't read it, but
  Android's native `CookieManager` can — `CookieBridge` copies it onto the native
  OkHttp upload requests. Log in once in the WebView and native uploads are authed.
- **Upload contract** (identical to the web `ChunkedUploader`): `MemoriesApi`
  posts `filename` / `content_type` / `total_chunks` to `/api/upload/init`, streams
  5 MB chunks (retry w/ backoff) to `/api/upload/{id}/chunk`, calls
  `/api/upload/{id}/complete`, then `POST /responses/` with `primary_staged_id`
  plus the base fields (`prompt_id` or `title`+`chapter`, and `response_text`) and
  `Accept: application/json`.

## Project layout

```
android/
├── app/src/main/java/com/memories/app/
│   ├── MainActivity.kt            WebView host, JS bridge, file chooser, permissions
│   ├── SetupActivity.kt           first-run server URL entry
│   ├── SettingsStore.kt           persists + normalises the server URL
│   ├── net/CookieBridge.kt        WebView session cookie → OkHttp
│   ├── net/MemoriesApi.kt         chunked upload + create-response client
│   ├── web/NativeBridge.kt        JS interface + the RecordOverlay hook script
│   └── record/
│       ├── RecordActivity.kt          native recorder screen + upload
│       ├── CaptureController.kt       shared interface
│       ├── AudioRecorderController.kt MediaRecorder (AAC/MP4)
│       └── VideoRecorderController.kt CameraX (H.264/MP4, front/back)
└── app/src/main/res/…             layouts, theme, adaptive icon, xml config
```

## Build & install (sideload)

Prereqs: **Android Studio** (Koala/Ladybug or newer) or a standalone JDK 17 +
Android SDK. Min Android 8.0 (API 26).

### Option A — Android Studio (easiest)
1. `File ▸ Open…` and select the `android/` folder.
2. Let Gradle sync (it fetches the Gradle wrapper and dependencies automatically).
3. Plug in a phone with USB debugging on, then **Run ▸ Run 'app'**.

### Option B — command line
The Gradle **wrapper jar** is not committed (it's a binary). Generate it once,
then build:

```bash
cd android
# one-time: create gradle/wrapper/gradle-wrapper.jar + gradlew scripts
gradle wrapper --gradle-version 8.7      # needs a system Gradle, OR skip and use Android Studio

# debug APK
./gradlew assembleDebug
# → app/build/outputs/apk/debug/app-debug.apk

# install to a connected device
./gradlew installDebug
```

Copy `app-debug.apk` to the phone and tap it to install (enable
“Install unknown apps” for your file manager). Debug APKs are self-signed and fine
for family sideloading.

### First launch
On first open you'll be asked for your **server URL** (e.g.
`https://memories.yourfamily.com`). It's stored locally; change it later from the
error screen's **Change server** button. Then log in on the web login page as usual.

## Notes & knobs

- **Cleartext http** is permitted (see `res/xml/network_security_config.xml`) so a
  LAN/dev server (`http://192.168.x.x`, emulator `http://10.0.2.2:8000`) works.
  For an https-only production install, flip `cleartextTrafficPermitted` to `false`.
- **Video** defaults to the front camera at HD; **audio** records AAC. Both are
  transcoded server-side by the existing ffmpeg pipeline, so format parity with the
  web app's webm/opus is not required.
- **Segment recording** (adding clips to an existing story) intentionally stays on
  the in-page web recorder; the app grants it mic/camera via `onPermissionRequest`.
- **Package id** is `com.memories.app` — change `applicationId`/`namespace` in
  `app/build.gradle.kts` if you want your own.

## Release build (distributing to family phones)

The app is release-signed so the same APK installs on every phone and updates in
place. Signing is driven by `keystore.properties` (gitignored) which points at the
keystore. **Both are secrets — keep them safe and backed up:**

- Keystore: `C:\Users\blank\keystores\memories-release.jks`
- Credentials: `android/keystore.properties`  (alias `memories`)

> ⚠️ If you lose the keystore or its password you can't ship an *update* your family
> can install over the top — they'd have to uninstall and reinstall. Back up both.

### Important: build release from a LOCAL copy, not the `\\wildspace` share
AAPT2 (icon PNG crunching) and Gradle file-watching time out over the SMB share.
The reliable flow mirrors the project to local disk first:

```powershell
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
$gradle = "C:\Users\blank\gradle-dl\gradle-8.7\bin\gradle.bat"

# 1. mirror source -> local (excludes build/.gradle/.idea)
robocopy "\\wildspace\compose\memories3\android" "C:\Users\blank\memories-android" `
    /MIR /XD build .gradle .idea /XF *.iml

# 2. build the signed release APK
& $gradle -p "C:\Users\blank\memories-android" assembleRelease
# → C:\Users\blank\memories-android\app\build\outputs\apk\release\app-release.apk
```

A copy is also staged at `android/dist/Memories-1.0.apk` for easy sharing.

### Install on a family phone
Send them the APK (email/Drive/messaging) and have them tap it, allowing
"install unknown apps" for that source. Or via USB:

```powershell
$adb = "C:\Users\blank\AppData\Local\Android\Sdk\platform-tools\adb.exe"
& $adb install "C:\Users\blank\memories-android\app\build\outputs\apk\release\app-release.apk"
```

On first launch each phone asks for the server URL (`https://memories.blankenship.casa`),
then they log in on the web page.

### Shipping an update
Bump `versionCode` (and `versionName`) in `app/build.gradle.kts`, rebuild with the
**same** keystore, and redistribute. Because the signing key matches, phones install
it over the existing app and keep their data. (Switching debug→release, or changing
the key, requires an uninstall/reinstall once.)

## Possible next steps
- Native web-push forwarding (the backend already has a push router + VAPID) via
  Firebase or a foreground service.
- Background upload with `WorkManager` so a recording finishes uploading even if the
  app is backgrounded.
- Route segment recording through the native recorder too (currently uses the
  in-WebView recorder).
