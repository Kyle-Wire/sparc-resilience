use std::path::PathBuf;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandEvent;

/// Wheel URL baked in at build time.
const WHEEL_URL: &str = {
    match option_env!("SPARC_WHEEL_URL") {
        Some(url) => url,
        None => concat!(
            "https://github.com/Kyle-Wire/sparc-status/releases/download/v",
            env!("CARGO_PKG_VERSION"),
            "/sparc-",
            env!("CARGO_PKG_VERSION"),
            "-py3-none-any.whl[server]"
        ),
    }
};

pub fn env_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".sparc")
        .join("env")
}

pub fn version_sentinel() -> PathBuf {
    env_dir().join(".sparc-version")
}

pub fn engine_ready() -> bool {
    let sentinel = version_sentinel();
    if !sentinel.exists() {
        return false;
    }
    let installed = std::fs::read_to_string(&sentinel)
        .unwrap_or_default()
        .trim()
        .to_string();
    installed == env!("CARGO_PKG_VERSION")
}

#[tauri::command]
pub async fn setup_create_venv(app: AppHandle) -> Result<(), String> {
    let venv = env_dir();
    if venv.exists() {
        std::fs::remove_dir_all(&venv)
            .map_err(|e| format!("Failed to clean up existing env: {e}"))?;
    }
    run_uv(
        &app,
        &["venv", venv.to_str().unwrap_or(".sparc/env"), "--python", "3.11"],
    )
    .await
}

#[tauri::command]
pub async fn setup_install_engine(app: AppHandle) -> Result<(), String> {
    let python = env_dir().join(if cfg!(windows) { "Scripts/python.exe" } else { "bin/python" });
    run_uv(
        &app,
        &["pip", "install", "--python", python.to_str().unwrap_or("python"), WHEEL_URL],
    )
    .await
}

#[tauri::command]
pub async fn setup_upgrade_engine(app: AppHandle) -> Result<(), String> {
    let python = env_dir().join(if cfg!(windows) { "Scripts/python.exe" } else { "bin/python" });
    run_uv(
        &app,
        &["pip", "install", "--upgrade", "--python", python.to_str().unwrap_or("python"), WHEEL_URL],
    )
    .await
}

#[tauri::command]
pub fn setup_mark_complete() -> Result<(), String> {
    let sentinel = version_sentinel();
    std::fs::create_dir_all(sentinel.parent().unwrap_or(&sentinel))
        .map_err(|e| format!("mkdir failed: {e}"))?;
    std::fs::write(&sentinel, env!("CARGO_PKG_VERSION"))
        .map_err(|e| format!("write sentinel failed: {e}"))
}

#[tauri::command]
pub fn setup_cleanup_env() -> Result<(), String> {
    let venv = env_dir();
    if venv.exists() {
        std::fs::remove_dir_all(&venv)
            .map_err(|e| format!("cleanup failed: {e}"))?;
    }
    Ok(())
}

#[tauri::command]
pub fn setup_finish(app: AppHandle) -> Result<(), String> {
    if let Some(main) = app.get_webview_window("main") {
        main.show().ok();
        main.set_focus().ok();
    }
    if let Some(setup_win) = app.get_webview_window("setup") {
        setup_win.close().ok();
    }
    Ok(())
}

// ── Internal helper ──────────────────────────────────────────────────────────

/// Spawn the `uv` sidecar via tauri_plugin_shell (correct platform binary
/// resolution) and stream stdout/stderr as `setup://progress` events.
async fn run_uv(app: &AppHandle, args: &[&str]) -> Result<(), String> {
    let (mut rx, _child) = app
        .shell()
        .sidecar("uv")
        .map_err(|e| format!("uv sidecar init: {e}"))?
        .args(args)
        .spawn()
        .map_err(|e| format!("Failed to spawn uv: {e}"))?;

    loop {
        match rx.recv().await {
            Some(CommandEvent::Stdout(bytes)) => {
                let line = String::from_utf8_lossy(&bytes);
                let t = line.trim();
                if !t.is_empty() { app.emit("setup://progress", t).ok(); }
            }
            Some(CommandEvent::Stderr(bytes)) => {
                let line = String::from_utf8_lossy(&bytes);
                let t = line.trim();
                if !t.is_empty() { app.emit("setup://progress", t).ok(); }
            }
            Some(CommandEvent::Terminated(payload)) => {
                return match payload.code {
                    Some(0) => Ok(()),
                    Some(code) => Err(format!("uv exited with code {code}")),
                    None => Err("uv was terminated by a signal".to_string()),
                };
            }
            Some(CommandEvent::Error(e)) => return Err(format!("uv error: {e}")),
            None => return Ok(()), // channel closed
            Some(_) => {}
        }
    }
}
