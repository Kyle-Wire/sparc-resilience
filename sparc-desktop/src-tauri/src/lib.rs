mod sidecar;
mod setup;

use tauri::{Manager, RunEvent};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .manage(sidecar::SidecarHandle::new())
        .invoke_handler(tauri::generate_handler![
            sidecar::stop_sidecar,
            sidecar::get_sidecar_token,
            setup::setup_create_venv,
            setup::setup_install_engine,
            setup::setup_upgrade_engine,
            setup::setup_mark_complete,
            setup::setup_cleanup_env,
            setup::setup_finish,
        ])
        .setup(|app| {
            let handle = app.handle().clone();

            if setup::engine_ready() {
                // Normal launch — engine installed and version matches.
                tauri::async_runtime::spawn(async move {
                    if let Err(e) = sidecar::spawn_server(&handle).await {
                        eprintln!("Failed to start server: {}", e);
                    }
                });
            } else {
                // First launch or version mismatch — show setup window,
                // keep main window hidden until setup_finish() is called.
                if let Some(main) = app.get_webview_window("main") {
                    main.hide().ok();
                }
                if let Some(setup_win) = app.get_webview_window("setup") {
                    setup_win.show().ok();
                    setup_win.set_focus().ok();
                }
            }

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
