# Keep the JavaScript bridge interface reachable from the WebView.
-keepclassmembers class com.memories.app.web.NativeBridge {
   public *;
}
-keepattributes JavascriptInterface

# OkHttp / Okio ship their own consumer rules; nothing extra needed here.
