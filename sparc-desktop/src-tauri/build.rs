fn main() {
    // Forward SUPABASE_URL from the build environment into the compiled binary
    // so the sidecar spawn path can inject it without touching user code.
    // Set this in your shell or .env.local before running `cargo tauri build`.
    let supabase_url = std::env::var("SUPABASE_URL").unwrap_or_default();
    println!("cargo:rustc-env=SPARC_SUPABASE_URL={supabase_url}");
    // Re-run if the env var changes.
    println!("cargo:rerun-if-env-changed=SUPABASE_URL");

    tauri_build::build();
}
