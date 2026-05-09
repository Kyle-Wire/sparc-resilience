use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tauri::{AppHandle, Emitter, Manager};

/// Wheel URL baked in at build time.
/// Set SPARC_WHEEL_URL in the build environment (CI injects it from the version tag).
/// Falls back to the GitHub releases convention using CARGO_PKG_VERSION.
const WHEEL_URL: &str = {
    // SPARC_WHEEL_URL is set by build.rs; fallback is constructed at compile time.
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

/// Path where the SPARC Python engine venv lives.
pub fn env_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".sparc")
        .join("env")
}

/// Sentinel file that records the installed engine version.
pub fn version_sentinel() -> PathBuf {
    env_dir().join(".sparc-version")
}

/// Returns `true` when a valid, version-matched engine is installed.
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

/// Resolve the path to the bundled `uv` sidecar binary.
fn uv_path(app: &AppHandle) -> Option<PathBuf> {
    let resource_dir = app.path().resource_dir().ok()?;
    let bin_name = if cfg!(windows) {
        format!("uv-{}.exe", std::env::consts::ARCH)
    } else {
        format!("uv-{}-apple-darwin", std::env::consts::ARCH)
    };

    // Tauri bundles externalBin as <resource_dir>/binaries/<name>-<target-triple>[.exe]
    // We try both the explicit target-triple name and the simplified arch name.
    let candidates = [
        resource_dir.join("binaries").join(&bin_name),
        // Tauri 2 places external bins directly in resource dir on some platforms
        resource_dir.join(&bin_name),
    ];
    candidates.into_iter().find(|p| p.exists())
}

/// Run `uv venv ~/.sparc/env --python 3.11` to create the isolated venv.
/// Streams output lines back to the frontend as `setup://progress` events.
#[tauri::command]
pub async fn setup_create_venv(app: AppHandle) -> Result<(), String> {
    let uv = uv_path(&app).ok_or("uv binary not found in bundle")?;
    let venv = env_dir();

    // Remove partial venv if it exists (e.g. after a failed previous attempt)
    if venv.exists() {
        std::fs::remove_dir_all(&venv)
            .map_err(|e| format!("Failed to clean up existing env: {e}"))?;
    }

    run_uv_streaming(
        &app,
        &uv,
        &[
            "venv",
            venv.to_str().unwrap_or(".sparc/env"),
            "--python",
            "3.11",
        ],
    )
    .await
}

/// Run `uv pip install --python ~/.sparc/env <WHEEL_URL>`.
/// Streams progress lines back to the frontend as `setup://progress` events.
#[tauri::command]
pub async fn setup_install_engine(app: AppHandle) -> Result<(), String> {
    let uv = uv_path(&app).ok_or("uv binary not found in bundle")?;
    let python = env_dir().join(if cfg!(windows) {
        "Scripts/python.exe"
    } else {
        "bin/python"
    });

    run_uv_streaming(
        &app,
        &uv,
        &[
            "pip",
            "install",
            "--python",
            python.to_str().unwrap_or("python"),
            WHEEL_URL,
        ],
    )
    .await
}

/// Run `uv pip install --upgrade --python ~/.sparc/env <WHEEL_URL>` for silent upgrades.
#[tauri::command]
pub async fn setup_upgrade_engine(app: AppHandle) -> Result<(), String> {
    let uv = uv_path(&app).ok_or("uv binary not found in bundle")?;
    let python = env_dir().join(if cfg!(windows) {
        "Scripts/python.exe"
    } else {
        "bin/python"
    });

    run_uv_streaming(
        &app,
        &uv,
        &[
            "pip",
            "install",
            "--upgrade",
            "--python",
            python.to_str().unwrap_or("python"),
            WHEEL_URL,
        ],
    )
    .await
}

/// Write the version sentinel so future launches skip setup.
#[tauri::command]
pub fn setup_mark_complete() -> Result<(), String> {
    let sentinel = version_sentinel();
    std::fs::create_dir_all(sentinel.parent().unwrap_or(&sentinel))
        .map_err(|e| format!("mkdir failed: {e}"))?;
    std::fs::write(&sentinel, env!("CARGO_PKG_VERSION"))
        .map_err(|e| format!("write sentinel failed: {e}"))
}

/// Delete the venv and sentinel — called before a retry.
#[tauri::command]
pub fn setup_cleanup_env() -> Result<(), String> {
    let venv = env_dir();
    if venv.exists() {
        std::fs::remove_dir_all(&venv)
            .map_err(|e| format!("cleanup failed: {e}"))?;
    }
    Ok(())
}

/// Open the main SPARC window and close the setup window.
#[tauri::command]
pub fn setup_finish(app: AppHandle) -> Result<(), String> {
    // Show main window (created hidden initially when setup is shown)
    if let Some(main) = app.get_webview_window("main") {
        main.show().ok();
        main.set_focus().ok();
    }
    // Close setup window
    if let Some(setup) = app.get_webview_window("setup") {
        setup.close().ok();
    }
    Ok(())
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Spawn a `uv` subprocess, stream stdout+stderr lines to the frontend as
/// `setup://progress` Tauri events, and return Ok/Err on completion.
async fn run_uv_streaming(
    app: &AppHandle,
    uv: &PathBuf,
    args: &[&str],
) -> Result<(), String> {
    let app = app.clone();
    let uv = uv.clone();
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();

    tauri::async_runtime::spawn_blocking(move || {
        let mut child = Command::new(&uv)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn uv: {e}"))?;

        // Stream stdout
        if let Some(stdout) = child.stdout.take() {
            let app2 = app.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    app2.emit("setup://progress", &line).ok();
                }
            });
        }

        // Stream stderr (uv writes progress to stderr)
        if let Some(stderr) = child.stderr.take() {
            let app2 = app.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    app2.emit("setup://progress", &line).ok();
                }
            });
        }

        let status = child.wait().map_err(|e| format!("uv wait failed: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!(
                "uv exited with code {}",
                status.code().unwrap_or(-1)
            ))
        }
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}
