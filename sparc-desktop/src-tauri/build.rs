fn main() {
    // Forward SUPABASE_URL from the build environment into the compiled binary
    // so the sidecar spawn path can inject it without touching user code.
    // Set this in your shell or .env.local before running `cargo tauri build`.
    let supabase_url = std::env::var("SUPABASE_URL").unwrap_or_default();
    println!("cargo:rustc-env=SPARC_SUPABASE_URL={supabase_url}");
    println!("cargo:rerun-if-env-changed=SUPABASE_URL");

    // Bake the Python wheel URL into the binary so the bootstrapper knows
    // exactly what to download without a runtime API call.
    // CI sets SPARC_WHEEL_URL to the exact URL of the freshly-built wheel;
    // local dev builds fall back to a URL derived from the Python package
    // version read directly from pyproject.toml (independent of the Tauri
    // desktop version, which has its own cadence).
    //
    // SPARC_PY_VERSION is always emitted so env!("SPARC_PY_VERSION") in
    // setup.rs compiles even when SPARC_WHEEL_URL is also set (both sides
    // of option_env! are evaluated by the compiler at macro expansion time).
    let py_ver = read_pyproject_version().unwrap_or_else(|| "0.0.0".to_string());
    println!("cargo:rustc-env=SPARC_PY_VERSION={py_ver}");
    if let Ok(url) = std::env::var("SPARC_WHEEL_URL") {
        println!("cargo:rustc-env=SPARC_WHEEL_URL={url}");
    }
    println!("cargo:rerun-if-env-changed=SPARC_WHEEL_URL");
    if let Ok(hash) = std::env::var("SPARC_WHEEL_HASH") {
        println!("cargo:rustc-env=SPARC_WHEEL_HASH={hash}");
    }
    println!("cargo:rerun-if-env-changed=SPARC_WHEEL_HASH");
    println!("cargo:rerun-if-changed=../../pyproject.toml");

    tauri_build::build();
}

/// Parse `version = "x.y.z"` from the workspace pyproject.toml.
fn read_pyproject_version() -> Option<String> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    // src-tauri is two levels below the repo root
    let toml_path = std::path::Path::new(&manifest)
        .join("../../pyproject.toml");
    let text = std::fs::read_to_string(toml_path).ok()?;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("version") {
            if let Some(val) = line.splitn(2, '=').nth(1) {
                return Some(val.trim().trim_matches('"').to_string());
            }
        }
    }
    None
}
