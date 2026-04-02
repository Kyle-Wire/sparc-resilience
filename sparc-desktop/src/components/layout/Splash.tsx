/**
 * Splash screen shown while waiting for the FastAPI server to start.
 * Brand rule: logo is white on black background.
 */
export default function Splash() {
  return (
    <div className="flex h-screen w-screen flex-col items-center justify-center bg-black text-white">
      <img
        src="/logo.svg"
        alt="SPARC Labs"
        className="mb-8 h-40 w-40 invert"
      />
      <div className="mb-2 text-3xl font-bold tracking-tight">SPARC</div>
      <div className="mb-8 text-xs tracking-[0.25em] text-sparc-gray-300 uppercase">Labs</div>
      <div className="flex gap-1.5">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="h-1.5 w-1.5 animate-pulse rounded-full bg-sparc-pink"
            style={{ animationDelay: `${i * 200}ms` }}
          />
        ))}
      </div>
      <p className="mt-8 text-[11px] text-sparc-gray-600">Starting server…</p>
    </div>
  );
}
