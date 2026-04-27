//! Authentication / licensing helpers exposed to the renderer.
//!
//! Three responsibilities:
//!   1. Store secrets in the OS keyring (Windows Credential Manager,
//!      macOS Keychain, Linux Secret Service) — used for the Supabase
//!      JWT, refresh token, license key, and migrated Anthropic key.
//!   2. Hand the renderer the per-launch sidecar Bearer token that the
//!      FastAPI middleware requires (see `sidecar.rs`).
//!   3. Run a one-shot loopback HTTP server for OAuth callbacks
//!      (`tauri-plugin-oauth`) so providers can redirect back without
//!      registering a `sparc://` URL scheme.
//!
//! All commands return JSON-friendly `Result<T, String>` so errors
//! surface in the renderer as plain strings.

use std::sync::Mutex;

use keyring::Entry;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, State};

const KEYRING_SERVICE: &str = "com.sparclabs.desktop";

/// Process-wide store for the per-launch sidecar token.
///
/// Set once by `sidecar::spawn_server` before the renderer can read it
/// via [`get_sidecar_token`].
pub static SIDECAR_TOKEN: OnceCell<String> = OnceCell::new();

/// Active OAuth callback servers keyed by `state`.
///
/// `tauri-plugin-oauth::start_with_config` returns a port; we keep the
/// returned receiver of the auth code in a slot so the renderer can
/// `await` it via [`oauth_await_code`].
#[derive(Default)]
pub struct OAuthState {
    pending: Mutex<Vec<PendingOAuth>>,
}

struct PendingOAuth {
    state: String,
    port: u16,
    /// Receiver resolved when the loopback server gets the redirect.
    rx: std::sync::mpsc::Receiver<Result<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OAuthStartResult {
    /// Localhost port the loopback server is bound to. Caller embeds
    /// `http://127.0.0.1:{port}/` as the OAuth `redirect_to`.
    pub port: u16,
    /// CSRF token that must be echoed back in the OAuth `state` param.
    pub state: String,
}

// ─── Keyring commands ───────────────────────────────────────────────

#[tauri::command]
pub fn secret_set(key: String, value: String) -> Result<(), String> {
    let entry = Entry::new(KEYRING_SERVICE, &key).map_err(|e| e.to_string())?;
    entry.set_password(&value).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn secret_get(key: String) -> Result<Option<String>, String> {
    let entry = Entry::new(KEYRING_SERVICE, &key).map_err(|e| e.to_string())?;
    match entry.get_password() {
        Ok(v) => Ok(Some(v)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

#[tauri::command]
pub fn secret_delete(key: String) -> Result<(), String> {
    let entry = Entry::new(KEYRING_SERVICE, &key).map_err(|e| e.to_string())?;
    match entry.delete_credential() {
        Ok(()) => Ok(()),
        Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

// ─── Sidecar token ──────────────────────────────────────────────────

/// Renderer-callable accessor for the per-launch sidecar Bearer token.
///
/// Returns `None` if the sidecar hasn't started yet (the renderer
/// should retry alongside `useServer`'s health poll).
#[tauri::command]
pub fn get_sidecar_token() -> Option<String> {
    SIDECAR_TOKEN.get().cloned()
}

// ─── OAuth loopback ─────────────────────────────────────────────────

/// Start a one-shot loopback HTTP server that will capture the OAuth
/// `code` (and echoed `state`) from the provider's redirect.
///
/// Returns the bound port and a CSRF `state` value that the caller
/// must include when constructing the provider authorization URL.
#[tauri::command]
pub fn oauth_start(oauth: State<'_, OAuthState>) -> Result<OAuthStartResult, String> {
    let csrf = uuid::Uuid::new_v4().to_string();
    let (tx, rx) = std::sync::mpsc::channel::<Result<String, String>>();
    let csrf_for_cb = csrf.clone();

    let config = tauri_plugin_oauth::OauthConfig {
        ports: None,
        // Minimal HTML closer; provider redirects here, we extract the code.
        response: Some(
            "<html><body style=\"font-family:system-ui;padding:32px;\">\
             <h2>Sign-in complete</h2>\
             <p>You can close this window and return to SPARC Labs.</p>\
             <script>setTimeout(()=>window.close(),300)</script>\
             </body></html>"
                .into(),
        ),
    };

    let port = tauri_plugin_oauth::start_with_config(config, move |url| {
        // `url` is the full request line, e.g. "/?code=…&state=…"
        let parsed = url::Url::parse(&format!("http://localhost{}", url))
            .or_else(|_| url::Url::parse(&url));
        let result = match parsed {
            Ok(u) => {
                let mut code: Option<String> = None;
                let mut state: Option<String> = None;
                let mut error: Option<String> = None;
                for (k, v) in u.query_pairs() {
                    match k.as_ref() {
                        "code" => code = Some(v.into_owned()),
                        "state" => state = Some(v.into_owned()),
                        "error" | "error_description" => {
                            error = Some(v.into_owned());
                        }
                        _ => {}
                    }
                }
                if let Some(e) = error {
                    Err(format!("oauth provider error: {e}"))
                } else if state.as_deref() != Some(&csrf_for_cb) {
                    Err("oauth state mismatch (possible CSRF)".into())
                } else if let Some(c) = code {
                    Ok(c)
                } else {
                    Err("oauth callback missing code".into())
                }
            }
            Err(e) => Err(format!("oauth callback parse failed: {e}")),
        };
        let _ = tx.send(result);
    })
    .map_err(|e| e.to_string())?;

    let mut pending = oauth.pending.lock().map_err(|e| e.to_string())?;
    pending.push(PendingOAuth {
        state: csrf.clone(),
        port,
        rx,
    });

    Ok(OAuthStartResult { port, state: csrf })
}

/// Await the loopback server for the matching `state` and return the
/// authorization code. Blocks the async runtime via `spawn_blocking`
/// until the provider redirects (or the user closes the browser).
#[tauri::command]
pub async fn oauth_await_code(
    state_token: String,
    oauth: State<'_, OAuthState>,
) -> Result<String, String> {
    let rx = {
        let mut pending = oauth.pending.lock().map_err(|e| e.to_string())?;
        let idx = pending
            .iter()
            .position(|p| p.state == state_token)
            .ok_or_else(|| "no pending oauth flow for this state".to_string())?;
        pending.swap_remove(idx).rx
    };

    tauri::async_runtime::spawn_blocking(move || {
        rx.recv().unwrap_or_else(|e| Err(e.to_string()))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Cancel a pending OAuth flow (e.g. user clicked Cancel).
#[tauri::command]
pub fn oauth_cancel(state_token: String, oauth: State<'_, OAuthState>) -> Result<(), String> {
    let mut pending = oauth.pending.lock().map_err(|e| e.to_string())?;
    if let Some(idx) = pending.iter().position(|p| p.state == state_token) {
        let entry = pending.swap_remove(idx);
        // Best-effort: stopping the loopback drops the listener.
        let _ = tauri_plugin_oauth::cancel(entry.port);
    }
    Ok(())
}

// ─── Setup hook ─────────────────────────────────────────────────────

/// Register OAuth state into the Tauri app. Call from `lib.rs::run`.
pub fn init(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let _ = app; // reserved for future setup
    Ok(())
}

// Bring `AppHandle` into scope when needed by future commands.
#[allow(dead_code)]
fn _unused(_app: &AppHandle) {}
