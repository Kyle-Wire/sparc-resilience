import Link from "next/link";
import { Nav, Footer } from "@/components/shell/Shell";
import { PageHero, Section } from "@/components/marketing/page-shell";
import { VERSION } from "@/lib/brand";

export const metadata = { title: "Download SPARC" };

const BUILDS = [
  { os: "macOS · Apple Silicon", file: `SPARC-${VERSION}-arm64.dmg`, requires: "macOS 13+" },
  { os: "macOS · Intel", file: `SPARC-${VERSION}-x64.dmg`, requires: "macOS 13+" },
  { os: "Windows · x64", file: `SPARC-${VERSION}-setup.exe`, requires: "Windows 10/11" },
  { os: "Linux · deb", file: `sparc_${VERSION.replace("v", "")}_amd64.deb`, requires: "Ubuntu 22.04+, Debian 12+" },
  { os: "Linux · AppImage", file: `SPARC-${VERSION}.AppImage`, requires: "glibc 2.31+" },
];

export default function DownloadPage() {
  return (
    <>
      <Nav active="download" />
      <PageHero
        kicker={`download · ${VERSION} · private beta`}
        title="Get the SPARC desktop client."
        lead="One installer per OS. Sign in with the email you used here and your license activates automatically. Builds will appear here when the beta opens."
        accent="var(--gold)"
      />

      <Section>
        <div style={{ display: "grid", gap: 10 }}>
          {BUILDS.map((b) => (
            <div key={b.file} className="row" style={{ justifyContent: "space-between", alignItems: "center", padding: "16px 20px", border: "1px solid var(--line)", borderRadius: 8 }}>
              <div>
                <div style={{ fontSize: 16, fontWeight: 700 }}>{b.os}</div>
                <div className="mono" style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>{b.file} · {b.requires}</div>
              </div>
              <button className="btn" disabled>Coming soon</button>
            </div>
          ))}
        </div>
      </Section>

      <Section sunken kicker="system requirements" title="What you'll need.">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
          {[
            { k: "minimum", v: ["8 GB RAM", "4-core CPU", "20 GB disk", "OpenGL 3.3+"] },
            { k: "recommended", v: ["32 GB RAM", "NVIDIA GPU + CUDA 11.8", "100 GB SSD", "Discrete GPU (Apple Silicon = native MPS)"] },
            { k: "for big runs", v: ["64 GB+ RAM", "Multi-GPU node", "1 TB NVMe", "Wired network for sidecar telemetry"] },
          ].map((r, i) => (
            <div key={i} className="card" style={{ padding: 22 }}>
              <div className="kicker">{r.k}</div>
              <ul style={{ listStyle: "none", padding: 0, margin: "12px 0 0", display: "grid", gap: 6 }}>
                {r.v.map((x, j) => <li key={j} style={{ fontSize: 13, color: "var(--ink-2)" }}>· {x}</li>)}
              </ul>
            </div>
          ))}
        </div>
      </Section>

      <Section kicker="not ready yet?" title="Join the waitlist.">
        <Link href="/signup" className="btn btn-accent">Request access →</Link>
      </Section>

      <Footer />
    </>
  );
}
