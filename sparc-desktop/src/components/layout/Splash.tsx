/**
 * Splash screen shown while waiting for the FastAPI server to start.
 * Uses the animated CubeLogo with a loading bar.
 */
import CubeLogo from "../brand/CubeLogo";

export default function Splash() {
  return (
    <div
      className="flex h-screen w-screen flex-col items-center justify-center gap-6"
      style={{ background: 'var(--color-sparc-paper, #f7f4ee)' }}
    >
      <CubeLogo size={180} animate hue="ink" density={1.4} intensity={1} />
      <div className="text-center">
        <div
          className="text-[22px] font-extrabold"
          style={{ letterSpacing: '-0.02em', color: 'var(--color-sparc-ink)' }}
        >
          SPARC LABS
        </div>
        <div
          className="mono mt-1 text-[10px] font-medium uppercase"
          style={{ letterSpacing: '0.2em', color: 'var(--color-sparc-muted)' }}
        >
          Spatial Analysis &amp; Research Core · v0.4.2
        </div>
      </div>
      {/* Loading bar */}
      <div
        style={{
          width: 220,
          height: 3,
          background: 'rgba(0,0,0,0.08)',
          borderRadius: 2,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: '100%',
            height: '100%',
            background: 'var(--color-sparc-crimson)',
            animation: 'loadBar 1.3s ease-out',
          }}
        />
      </div>
      <p
        className="mono text-[10px]"
        style={{ color: 'var(--color-sparc-muted)', letterSpacing: '0.06em' }}
      >
        Starting sidecar…
      </p>
    </div>
  );
}
