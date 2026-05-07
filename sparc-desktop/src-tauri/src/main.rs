// Suppress the Windows console window in release builds. In debug builds we
// keep the console so `println!` from the host (e.g. sidecar startup logs)
// remains visible during development. No-op on macOS/Linux.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    sparc_desktop_lib::run();
}
