mod auth;
mod sidecar;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_oauth::init())
        .manage(auth::OAuthState::default())
        .invoke_handler(tauri::generate_handler![
            auth::secret_set,
            auth::secret_get,
            auth::secret_delete,
            auth::get_sidecar_token,
            auth::oauth_start,
            auth::oauth_await_code,
            auth::oauth_cancel,
        ])
        .setup(|app| {
            auth::init(app)?;
            let handle = app.handle().clone();
            // Spawn the Python server in the background. The sidecar
            // module generates the per-launch Bearer token before
            // launching the child process, then publishes it via
            // `auth::SIDECAR_TOKEN` for the renderer to consume.
            tauri::async_runtime::spawn(async move {
                if let Err(e) = sidecar::spawn_server(&handle).await {
                    eprintln!("Failed to start server: {}", e);
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running SPARC desktop app");
}
