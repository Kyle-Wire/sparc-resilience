mod sidecar;

use tauri::RunEvent;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .manage(sidecar::SidecarHandle::new())
        .invoke_handler(tauri::generate_handler![sidecar::stop_sidecar, sidecar::get_sidecar_token])
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
        .build(tauri::generate_context!())
        .expect("error while building SPARC desktop app")
        .run(|app, event| {
            if let RunEvent::ExitRequested { .. } | RunEvent::Exit = event {
                sidecar::kill_server(app);
            }
        });
}
