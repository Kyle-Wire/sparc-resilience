mod sidecar;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let handle = app.handle().clone();
            // Spawn the Python server in the background
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
