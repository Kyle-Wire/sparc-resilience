use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};

/// Wheel URL baked in at build time.
///
/// CI sets `SPARC_WHEEL_URL` to the exact URL of the freshly-built wheel.
/// Local dev builds fall back to a URL built from `SPARC_PY_VERSION`, which
/// build.rs reads from `pyproject.toml` — independent of the desktop version.
const WHEEL_URL: &str = {
    match option_env!("SPARC_WHEEL_URL") {
        Some(url) => url,
        None => concat!(
            "https://github.com/Kyle-Wire/sparc-status/releases/download/v",
            env!("SPARC_PY_VERSION"),
            "/sparc-",
            env!("SPARC_PY_VERSION"),
            "-py3-none-any.whl[server]"
        ),
    }
};

/// SHA-256 hash of the wheel, baked in by CI at build time.
/// None in local dev builds — hash verification is skipped.
const WHEEL_HASH: Option<&str> = option_env!("SPARC_WHEEL_HASH");

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

// ── Binary resolution ────────────────────────────────────────────────────────

/// Find the bundled uv binary regardless of the exact arch suffix in the name.
///
/// Tauri places external binaries in `<resource_dir>/binaries/`. We scan that
/// directory for any file whose name starts with "uv-" (and isn't sparc-related)
/// rather than hard-coding the full target-triple name. This is resilient to:
///   - arm64 / x86_64 mismatches in local dev
///   - Windows MSVC vs GNU variants
///   - Any future triple changes
fn find_uv(app: &AppHandle) -> Result<PathBuf, String> {
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("resource_dir: {e}"))?;

    // Where Tauri 2 actually places externalBin sidecars varies by platform:
    //   - macOS:   <bundle>.app/Contents/MacOS/  (next to the main exe)
    //   - Windows: next to the main .exe
    //   - Linux:   varies (AppImage uses resource_dir)
    // We also check resource_dir + resource_dir/binaries for local dev
    // (`pnpm tauri dev`) where the layout is flatter.
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));

    let mut search_dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = exe_dir.clone() {
        search_dirs.push(d);
    }
    search_dirs.push(resource_dir.clone());
    search_dirs.push(resource_dir.join("binaries"));

    let arch = std::env::consts::ARCH; // "aarch64" | "x86_64"

    let exact_name = if cfg!(windows) {
        format!("uv-{}-pc-windows-msvc.exe", arch)
    } else if cfg!(target_os = "macos") {
        format!("uv-{}-apple-darwin", arch)
    } else {
        format!("uv-{}-unknown-linux-musl", arch)
    };

    // 1. Try exact arch name in each search dir
    for dir in &search_dirs {
        let p = dir.join(&exact_name);
        if p.exists() {
            return Ok(p);
        }
    }

    // 2. Scan each dir for any uv-* (handles arch mismatch, future triples)
    for dir in &search_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            let found = entries
                .flatten()
                .find(|e| {
                    let n = e.file_name();
                    let s = n.to_string_lossy();
                    (s.starts_with("uv-") || s == "uv" || s == "uv.exe")
                        && !s.contains("sparc")
                })
                .map(|e| e.path());
            if let Some(p) = found {
                return Ok(p);
            }
        }
    }

    Err(format!(
        "uv binary not found (expected {}; searched: {})",
        exact_name,
        search_dirs
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

// ── Tauri commands ───────────────────────────────────────────────────────────

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
    let python = env_dir()
        .join(if cfg!(windows) { "Scripts/python.exe" } else { "bin/python" });
    let python_str = python.to_str().unwrap_or("python").to_string();
    let mut args = vec!["pip".to_string(), "install".to_string(),
                        "--python".to_string(), python_str, WHEEL_URL.to_string()];
    if let Some(hash) = WHEEL_HASH {
        args.push(format!("--hash=sha256:{}", hash));
    }
    let args_ref: Vec<&str> = args.iter().map(String::as_str).collect();
    run_uv(&app, &args_ref).await
}

#[tauri::command]
pub async fn setup_upgrade_engine(app: AppHandle) -> Result<(), String> {
    let python = env_dir()
        .join(if cfg!(windows) { "Scripts/python.exe" } else { "bin/python" });
    let python_str = python.to_str().unwrap_or("python").to_string();
    let mut args = vec!["pip".to_string(), "install".to_string(), "--upgrade".to_string(),
                        "--python".to_string(), python_str, WHEEL_URL.to_string()];
    if let Some(hash) = WHEEL_HASH {
        args.push(format!("--hash=sha256:{}", hash));
    }
    let args_ref: Vec<&str> = args.iter().map(String::as_str).collect();
    run_uv(&app, &args_ref).await
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

/// Spawn uv directly (std::process::Command) and stream stdout+stderr as
/// `setup://progress` Tauri events.
///
/// We use two concurrent reader threads so neither pipe buffer can fill and
/// deadlock the process, regardless of which stream uv writes to.
async fn run_uv(app: &AppHandle, args: &[&str]) -> Result<(), String> {
    let uv = find_uv(app)?;
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let app = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let mut child = Command::new(&uv)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn uv: {e}"))?;

        // Shared buffer — collects all output so we can include it in errors
        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        // Drain stdout in a separate thread
        let stdout_handle = child.stdout.take().map(|stdout| {
            let app2 = app.clone();
            let log2 = Arc::clone(&log);
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    let t = line.trim().to_string();
                    if !t.is_empty() {
                        log2.lock().unwrap().push(t.clone());
                        app2.emit("setup://progress", &t).ok();
                    }
                }
            })
        });

        // Drain stderr in a separate thread (uv writes progress here)
        let stderr_handle = child.stderr.take().map(|stderr| {
            let app2 = app.clone();
            let log2 = Arc::clone(&log);
            std::thread::spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    let t = line.trim().to_string();
                    if !t.is_empty() {
                        log2.lock().unwrap().push(t.clone());
                        app2.emit("setup://progress", &t).ok();
                    }
                }
            })
        });

        // Wait for process to finish (pipes are being drained concurrently)
        let status = child.wait().map_err(|e| format!("uv wait: {e}"))?;

        // Join reader threads (they exit when their pipe closes)
        if let Some(h) = stdout_handle { h.join().ok(); }
        if let Some(h) = stderr_handle { h.join().ok(); }

        if status.success() {
            Ok(())
        } else {
            // Include the last 20 lines of output so the UI can display the real error
            let lines = log.lock().unwrap();
            let tail: Vec<&str> = lines.iter().rev().take(20).rev().map(String::as_str).collect();
            Err(format!(
                "uv exited with code {}\n{}",
                status.code().unwrap_or(-1),
                tail.join("\n")
            ))
        }
    })
    .await
    .map_err(|e| format!("task join: {e}"))?
}
