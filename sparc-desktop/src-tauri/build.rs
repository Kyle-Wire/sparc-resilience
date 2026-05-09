fn main() {
    // Forward SUPABASE_URL from the build environment into the compiled binary
    // so the sidecar spawn path can inject it without touching user code.
    // Set this in your shell or .env.local before running `cargo tauri build`.
    let supabase_url = std::env::var("SUPABASE_URL").unwrap_or_default();
    println!("cargo:rustc-env=SPARC_SUPABASE_URL={supabase_url}");
    println!("cargo:rerun-if-env-changed=SUPABASE_URL");

    // Bake the Python wheel URL into the binary so the bootstrapper knows
    // exactly what to download without a runtime API call.
    // CI sets this to the tagged release URL; dev builds fall back to the
    // default constructed from CARGO_PKG_VERSION in setup.rs.
    if let Ok(url) = std::env::var("SPARC_WHEEL_URL") {
        println!("cargo:rustc-env=SPARC_WHEEL_URL={url}");
    }
    println!("cargo:rerun-if-env-changed=SPARC_WHEEL_URL");

    tauri_build::build();
}
