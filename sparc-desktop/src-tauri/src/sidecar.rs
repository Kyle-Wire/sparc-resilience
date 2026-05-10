use std::fs::OpenOptions;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};

/// Generate a cryptographically-random 32-byte (256-bit) token using the OS
/// CSPRNG via `getrandom`. The result is hex-encoded to a 64-character string.
fn generate_token() -> String {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).expect("OS CSPRNG unavailable");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Port the FastAPI sidecar listens on. Kept as a constant so the shutdown
/// path agrees with the spawn path.
const SIDECAR_PORT: &str = "8008";

/// Holds the child handle of the spawned Python sidecar so it can be
/// killed when the desktop app exits. On Windows we additionally hold a
/// Job Object handle that the OS uses to reap the entire process tree if
/// the host dies abruptly (crash, force-kill, console-X) and our normal
/// `RunEvent::Exit` cleanup never runs.
///
/// `token` is a 64-char hex string generated once at startup and injected
/// into the sidecar as `SPARC_SERVER_TOKEN`. The webview retrieves it via
/// the `get_sidecar_token` Tauri command and sends it in every HTTP request
/// as `X-SPARC-Token`, so the sidecar can reject calls from any other
/// process on the machine.
pub struct SidecarHandle {
    pub child: Mutex<Option<Child>>,
    pub token: String,
    #[cfg(windows)]
    pub job: Mutex<Option<job::JobHandle>>,
}

impl SidecarHandle {
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
            token: generate_token(),
            #[cfg(windows)]
            job: Mutex::new(None),
        }
    }
}

// ─── Windows Job Object ─────────────────────────────────────────────────────

#[cfg(windows)]
pub mod job {
    //! Wraps a Windows Job Object configured with
    //! `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. When the last handle to the job
    //! is closed (which happens automatically when our process exits for ANY
    //! reason), the kernel kills every process assigned to the job. This is
    //! the only mechanism Windows offers that survives host crashes /
    //! force-quits and reliably reaps PyInstaller's `_MEI` child Python.

    use std::io;
    use windows::core::PCWSTR;
    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
        JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows::Win32::System::Threading::{OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE};

    /// RAII wrapper around a Job Object HANDLE.
    pub struct JobHandle(HANDLE);

    // SAFETY: A Win32 HANDLE is just an opaque pointer-sized integer; the
    // kernel object it refers to is safe to use from any thread. We only
    // close it on Drop.
    unsafe impl Send for JobHandle {}
    unsafe impl Sync for JobHandle {}

    impl JobHandle {
        /// Create a new Job Object with KILL_ON_JOB_CLOSE set.
        pub fn create() -> io::Result<Self> {
            unsafe {
                let h = CreateJobObjectW(None, PCWSTR::null())
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                if h.is_invalid() {
                    return Err(io::Error::last_os_error());
                }

                let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

                let info_ptr = &info as *const _ as *const _;
                let info_size = std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32;
                SetInformationJobObject(
                    h,
                    JobObjectExtendedLimitInformation,
                    info_ptr,
                    info_size,
                )
                .map_err(|e| {
                    let _ = CloseHandle(h);
                    io::Error::new(io::ErrorKind::Other, e.to_string())
                })?;

                Ok(JobHandle(h))
            }
        }

        /// Assign a process (by PID) to this job. Newly spawned children of
        /// that process are automatically inherited into the job by Windows,
        /// which is exactly what we need for the PyInstaller bootloader.
        pub fn assign_pid(&self, pid: u32) -> io::Result<()> {
            unsafe {
                let proc_handle = OpenProcess(
                    PROCESS_TERMINATE | PROCESS_SET_QUOTA,
                    false,
                    pid,
                )
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                let result = AssignProcessToJobObject(self.0, proc_handle)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()));
                let _ = CloseHandle(proc_handle);
                result
            }
        }
    }

    impl Drop for JobHandle {
        fn drop(&mut self) {
            // Closing the last handle to the job triggers KILL_ON_JOB_CLOSE
            // and the kernel terminates every process in it.
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }
}

// ─── Platform-specific helpers ───────────────────────────────────────────────

/// Return the user's home directory.
fn home_dir() -> String {
    #[cfg(target_os = "windows")]
    { std::env::var("USERPROFILE").unwrap_or_else(|_| "C:\\".to_string()) }
    #[cfg(not(target_os = "windows"))]
    { std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()) }
}

/// Return the platform log directory: ~/Library/Logs/SPARC (mac) or %APPDATA%\SPARC\Logs (win).
fn log_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        let appdata = std::env::var("APPDATA")
            .unwrap_or_else(|_| format!("{}\\AppData\\Roaming", home_dir()));
        PathBuf::from(appdata).join("SPARC").join("Logs")
    }
    #[cfg(not(target_os = "windows"))]
    {
        PathBuf::from(home_dir()).join("Library/Logs/SPARC")
    }
}

// ─── Python resolution ──────────────────────────────────────────────────────

/// Resolve the best Python interpreter that has the `sparc` package.
///
/// macOS / Linux checks:
///   1. pyenv  — `~/.pyenv/bin/pyenv which python3`
///   2. Interactive shell — `$SHELL -ic 'which python3'`
///   3. Well-known paths — Homebrew, system
///   4. Bare `python3`
///
/// Windows checks:
///   1. pyenv-win — `%USERPROFILE%\.pyenv\pyenv-win\bin\pyenv.bat which python`
///   2. `py -3` launcher (standard Python.org installs)
///   3. Well-known paths — AppData, Program Files
///   4. Bare `python`
fn resolve_python() -> String {
    let home = home_dir();

    // ── macOS / Linux ────────────────────────────────────────────────────
    #[cfg(not(target_os = "windows"))]
    {
        // 1. pyenv
        let pyenv_root = std::env::var("PYENV_ROOT")
            .unwrap_or_else(|_| format!("{home}/.pyenv"));
        let pyenv_bin = format!("{pyenv_root}/bin/pyenv");

        if std::path::Path::new(&pyenv_bin).exists() {
            if let Ok(output) = Command::new(&pyenv_bin)
                .args(["which", "python3"])
                .env("PYENV_ROOT", &pyenv_root)
                .env("PATH", format!(
                    "{pyenv_root}/bin:{pyenv_root}/shims:/usr/local/bin:/usr/bin:/bin"
                ))
                .output()
            {
                let p = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !p.is_empty() && std::path::Path::new(&p).exists() {
                    println!("Resolved python via pyenv: {p}");
                    return p;
                }
            }
        }

        // 2. Interactive shell (picks up .zshrc / .bashrc)
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".to_string());
        if let Ok(output) = Command::new(&shell)
            .args(["-ic", "which python3"])
            .output()
        {
            let p = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !p.is_empty() && !p.contains("not found") && std::path::Path::new(&p).exists() {
                println!("Resolved python via interactive shell: {p}");
                return p;
            }
        }

        // 3. Well-known paths
        let candidates = [
            format!("{home}/.pyenv/shims/python3"),
            "/usr/local/bin/python3".to_string(),
            "/opt/homebrew/bin/python3".to_string(),
        ];
        for c in &candidates {
            if std::path::Path::new(c).exists() {
                println!("Resolved python at well-known path: {c}");
                return c.clone();
            }
        }

        println!("Falling back to bare python3");
        "python3".to_string()
    }

    // ── Windows ──────────────────────────────────────────────────────────
    #[cfg(target_os = "windows")]
    {
        // 1. pyenv-win
        let pyenv_root = std::env::var("PYENV_ROOT").or_else(|_| std::env::var("PYENV"))
            .unwrap_or_else(|_| format!("{home}\\.pyenv\\pyenv-win"));
        let pyenv_bat = format!("{pyenv_root}\\bin\\pyenv.bat");

        if std::path::Path::new(&pyenv_bat).exists() {
            if let Ok(output) = Command::new("cmd")
                .args(["/C", &pyenv_bat, "which", "python"])
                .output()
            {
                let p = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !p.is_empty() && std::path::Path::new(&p).exists() {
                    println!("Resolved python via pyenv-win: {p}");
                    return p;
                }
            }
        }

        // 2. Python Launcher (`py -3`) — ships with python.org installs
        if let Ok(output) = Command::new("py")
            .args(["-3", "-c", "import sys; print(sys.executable)"])
            .output()
        {
            if output.status.success() {
                let p = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !p.is_empty() && std::path::Path::new(&p).exists() {
                    println!("Resolved python via py launcher: {p}");
                    return p;
                }
            }
        }

        // 3. Well-known Windows paths
        let localappdata = std::env::var("LOCALAPPDATA").unwrap_or_default();
        let candidates = [
            format!("{localappdata}\\Programs\\Python\\Python311\\python.exe"),
            format!("{localappdata}\\Programs\\Python\\Python312\\python.exe"),
            format!("{localappdata}\\Programs\\Python\\Python310\\python.exe"),
            format!("{home}\\.pyenv\\pyenv-win\\shims\\python.bat"),
            "C:\\Python311\\python.exe".to_string(),
            "C:\\Python312\\python.exe".to_string(),
            "C:\\Python310\\python.exe".to_string(),
        ];
        for c in &candidates {
            if std::path::Path::new(c).exists() {
                println!("Resolved python at well-known path: {c}");
                return c.clone();
            }
        }

        println!("Falling back to bare python");
        "python".to_string()
    }
}

// ─── Server spawn ────────────────────────────────────────────────────────────

/// Spawn the SPARC FastAPI server.
///
/// Strategy (in order):
///   1. Bundled PyInstaller binary next to the app binary.
///   2. Resolve the user's Python and run `python -m sparc server`.
///
/// Server stdout/stderr go to a platform-appropriate log file. On Windows
/// the spawned process is assigned to a Job Object so it (and any child
/// processes it spawns) is killed automatically if the host dies abruptly.
pub async fn spawn_server(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let port = SIDECAR_PORT;

    // Evict any stale process squatting on our port (e.g. a leftover
    // `python -m sparc server` from a previous dev session). If we don't,
    // our spawn will succeed but we'll silently talk to the zombie, which
    // is almost certainly running an older code version with missing routes.
    evict_port_squatter(port);

    // Retrieve the token from managed state so both spawn paths inject it.
    let token = app
        .try_state::<SidecarHandle>()
        .map(|s| s.token.clone())
        .unwrap_or_default();

    // On Windows, create the Job Object up front so we can assign whichever
    // spawn path actually succeeds.
    #[cfg(windows)]
    {
        match job::JobHandle::create() {
            Ok(j) => {
                if let Some(state) = app.try_state::<SidecarHandle>() {
                    *state.job.lock().unwrap() = Some(j);
                }
            }
            Err(e) => {
                eprintln!("Failed to create sidecar Job Object: {e}. \
                          Falling back to best-effort cleanup on exit.");
            }
        }
    }

    // Prepare log file (truncated each launch).
    // Prepare log file (truncated each launch).
    let ld = log_dir();
    std::fs::create_dir_all(&ld).ok();
    let log_path = ld.join("server.log");
    let log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
        .ok();
    println!("Server log: {}", log_path.display());

    let stdio = |f: &Option<std::fs::File>| -> Stdio {
        f.as_ref()
            .and_then(|f| f.try_clone().ok())
            .map(Stdio::from)
            .unwrap_or_else(Stdio::inherit)
    };

    // Primary: launch the engine from the managed venv (~/.sparc/env).
    // The setup wizard ensures this venv exists before spawn_server is called.
    let venv_python: PathBuf = crate::setup::env_dir().join(if cfg!(windows) {
        "Scripts/python.exe"
    } else {
        "bin/python"
    });

    let python = if venv_python.exists() {
        venv_python.to_string_lossy().to_string()
    } else {
        // Fallback: resolve system Python (dev installs, or venv not ready).
        eprintln!(
            "WARNING: venv not found at {}, falling back to system Python",
            venv_python.display()
        );
        resolve_python()
    };

    println!("Spawning: {python} -m sparc server --port {port}");

    // On Windows, hide the console window so no cmd.exe flash appears.
    #[cfg(target_os = "windows")]
    let mut cmd = {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        let mut c = Command::new(&python);
        c.args(["-m", "sparc", "server", "--port", port])
            .env("SPARC_SERVER_TOKEN", &token)
            .env("SUPABASE_URL", env!("SPARC_SUPABASE_URL"))
            .creation_flags(CREATE_NO_WINDOW)
            .stdout(stdio(&log_file))
            .stderr(stdio(&log_file));
        c
    };
    #[cfg(not(target_os = "windows"))]
    let mut cmd = {
        let mut c = Command::new(&python);
        c.args(["-m", "sparc", "server", "--port", port])
            .env("SPARC_SERVER_TOKEN", &token)
            .env("SUPABASE_URL", env!("SPARC_SUPABASE_URL"))
            .stdout(stdio(&log_file))
            .stderr(stdio(&log_file));
        c
    };

    match cmd.spawn() {
        Ok(child) => {
            println!("Server started on port {port} via {python}");
            register_child(app, child);
            return Ok(());
        }
        Err(e) => {
            eprintln!("Failed to start server with {python}: {e}");
        }
    }

    eprintln!(
        "Could not start the SPARC server. \
         Please run 'python -m sparc server --port {port}' manually."
    );
    Ok(())
}

/// Store the child handle and (on Windows) assign it to the Job Object so
/// the kernel will reap the whole tree if we die unexpectedly.
fn register_child(app: &AppHandle, child: Child) {
    #[cfg(windows)]
    {
        let pid = child.id();
        if let Some(state) = app.try_state::<SidecarHandle>() {
            if let Some(j) = state.job.lock().unwrap().as_ref() {
                if let Err(e) = j.assign_pid(pid) {
                    eprintln!("Failed to assign sidecar pid {pid} to Job Object: {e}");
                }
            }
        }
    }
    if let Some(state) = app.try_state::<SidecarHandle>() {
        *state.child.lock().unwrap() = Some(child);
    }
}

/// Detect and evict any process already listening on the sidecar port
/// before we spawn our own. Without this, a stale `python -m sparc server`
/// from a previous dev session (or a crashed prior run) keeps the port
/// bound; our new spawn fails silently and the webview ends up talking to
/// the zombie, which serves an older code version with missing routes —
/// surfacing as mysterious 404s on /data/select, /project/create, etc.
///
/// Strategy:
///   1. Fast TCP probe — if the port is free, return immediately.
///   2. Try `POST /shutdown` (localhost-only, no auth) so a real SPARC
///      sidecar can flush state and exit cleanly.
///   3. Poll for up to ~5 s for the port to free.
///   4. If still bound (unknown squatter or shutdown ignored), fall back
///      to OS-level kill of whichever PID owns the port.
fn evict_port_squatter(port: &str) {
    let parsed: u16 = match port.parse() {
        Ok(p) => p,
        Err(_) => return,
    };
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], parsed));

    if !port_in_use(&addr) {
        return;
    }

    eprintln!("Port {port} is in use; attempting to evict existing listener");

    // Phase 1: graceful /shutdown. Works for any prior SPARC sidecar.
    let url = format!("http://127.0.0.1:{port}/shutdown");
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(2))
        .build();
    let _ = agent.post(&url).call();

    // Phase 2: wait briefly for the port to free.
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(150));
        if !port_in_use(&addr) {
            eprintln!("Port {port} freed after /shutdown");
            return;
        }
    }

    // Phase 3: not a SPARC server (or it ignored /shutdown). Hard-kill.
    if let Some(pid) = pid_listening_on(parsed) {
        eprintln!("Force-killing PID {pid} holding port {port}");
        force_kill_pid(pid);
        std::thread::sleep(Duration::from_millis(300));
    }

    if port_in_use(&addr) {
        eprintln!(
            "WARNING: port {port} is still in use after eviction attempt; \
             sidecar spawn will likely fail"
        );
    }
}

fn port_in_use(addr: &std::net::SocketAddr) -> bool {
    std::net::TcpStream::connect_timeout(addr, Duration::from_millis(250)).is_ok()
}

#[cfg(unix)]
fn pid_listening_on(port: u16) -> Option<u32> {
    let out = Command::new("lsof")
        .args(["-i", &format!(":{port}"), "-sTCP:LISTEN", "-t", "-n", "-P"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next().and_then(|l| l.trim().parse().ok())
}

#[cfg(windows)]
fn pid_listening_on(port: u16) -> Option<u32> {
    let out = Command::new("netstat").args(["-ano", "-p", "TCP"]).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let needle = format!(":{port}");
    for line in s.lines() {
        if !line.contains("LISTENING") || !line.contains(&needle) {
            continue;
        }
        if let Some(pid_str) = line.split_whitespace().last() {
            if let Ok(pid) = pid_str.parse::<u32>() {
                return Some(pid);
            }
        }
    }
    None
}

#[cfg(unix)]
fn force_kill_pid(pid: u32) {
    let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
}

#[cfg(windows)]
fn force_kill_pid(pid: u32) {
    let _ = Command::new("taskkill")
        .args(["/F", "/PID", &pid.to_string()])
        .status();
}

/// Ask the Python server to shut itself down cleanly via its `/shutdown`
/// endpoint. Returns quickly; callers should still wait on the child
/// afterwards. Best-effort: any error (server already gone, port closed,
/// timeout) is logged and ignored.
fn request_graceful_shutdown() {
    let url = format!("http://127.0.0.1:{SIDECAR_PORT}/shutdown");
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(3))
        .build();
    match agent.post(&url).call() {
        Ok(_) => println!("Sent /shutdown to sidecar"),
        Err(e) => println!("Sidecar /shutdown request failed (continuing to hard-kill): {e}"),
    }
}

/// Two-phase termination of the sidecar:
///   1. POST /shutdown so FastAPI/uvicorn can flush DB writes and close
///      WebSocket subscribers cleanly.
///   2. Poll up to ~5 s for the child to exit on its own.
///   3. If still alive, hard-kill. On Windows the Job Object is the
///      ultimate safety net for any descendants.
pub fn kill_server(app: &AppHandle) {
    let Some(state) = app.try_state::<SidecarHandle>() else { return };
    let Some(mut child) = state.child.lock().unwrap().take() else { return };

    request_graceful_shutdown();

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                println!("Sidecar exited gracefully");
                drop_job(&state);
                return;
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                eprintln!("try_wait on sidecar failed: {e}");
                break;
            }
        }
    }

    let _ = child.kill();
    let _ = child.wait();
    println!("Sidecar server terminated (forced)");
    drop_job(&state);
}

/// Drop the Job Object handle, which on Windows triggers KILL_ON_JOB_CLOSE
/// and reaps any straggler processes (e.g. PyInstaller's `_MEI` child).
fn drop_job(_state: &SidecarHandle) {
    #[cfg(windows)]
    {
        // Taking the Option drops the JobHandle, which closes the kernel
        // handle, which triggers the kill.
        let _ = _state.job.lock().unwrap().take();
    }
}

/// Tauri command that gives the webview its copy of the sidecar token.
/// Called once on app load; stored in React memory (never localStorage).
#[tauri::command]
pub fn get_sidecar_token(app: AppHandle) -> String {
    app.try_state::<SidecarHandle>()
        .map(|s| s.token.clone())
        .unwrap_or_default()
}

/// Tauri command exposed to the frontend so the updater UI can stop the
/// sidecar BEFORE invoking `downloadAndInstall`. Without this the running
/// `sparc-sidecar.exe` is locked and the installer fails.
#[tauri::command]
pub fn stop_sidecar(app: AppHandle) -> Result<(), String> {
    kill_server(&app);
    Ok(())
}

/// Tauri command: gracefully kill any existing sidecar, then re-spawn it.
///
/// Called from the webview when:
///   - The startup splash times out (no /health response within ~15 s)
///   - The server-lost overlay wants an automatic recovery
///
/// Returns immediately after spawning (does not wait for /health).
/// The webview's useServer hook continues polling and will detect when
/// the new sidecar is ready.
#[tauri::command]
pub async fn restart_sidecar(app: AppHandle) -> Result<(), String> {
    // Phase 1: shut down any existing child.
    kill_server(&app);
    // Phase 2: brief pause so the OS can release the port (blocking is fine
    // here — this runs on Tauri's async runtime thread pool, not the UI thread).
    std::thread::sleep(std::time::Duration::from_millis(500));
    // Phase 3: spawn a fresh sidecar (includes the port-eviction guard).
    spawn_server(&app).await.map_err(|e| e.to_string())
}

