import Link from "next/link";
import { redirect } from "next/navigation";
import { Nav, Footer } from "@/components/shell/Shell";
import { createSupabaseServerClient } from "@/lib/supabase/server";

const NAV = [
  { href: "/account", label: "Overview" },
  { href: "/account/subscription", label: "Subscription" },
  { href: "/account/seats", label: "Seats" },
  { href: "/account/downloads", label: "Downloads & licenses" },
  { href: "/account/api-keys", label: "API keys (BYOK)" },
  { href: "/account/usage", label: "Usage" },
  { href: "/account/devices", label: "Devices" },
  { href: "/account/settings", label: "Settings" },
] as const;

export default async function AccountLayout({ children }: { children: React.ReactNode }) {
  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) redirect("/login?next=/account");

  return (
    <>
      <Nav />
      <section className="grid-bg" style={{ borderBottom: "1px solid var(--line)", padding: "32px 0 18px" }}>
        <div className="wrap">
          <div className="kicker">account</div>
          <div className="row" style={{ alignItems: "baseline", justifyContent: "space-between", gap: 14, flexWrap: "wrap" }}>
            <h1 style={{ fontSize: 30, fontWeight: 800, letterSpacing: "-0.025em", margin: "6px 0 0" }}>{user.email}</h1>
            <form action="/auth/signout" method="post"><button className="btn btn-sm" type="submit">Sign out</button></form>
          </div>
        </div>
      </section>
      <section style={{ padding: "32px 0 80px" }}>
        <div className="wrap" style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: 32, alignItems: "start" }}>
          <nav style={{ position: "sticky", top: 80, display: "flex", flexDirection: "column", gap: 2 }}>
            {NAV.map((n) => (
              <Link key={n.href} href={n.href} style={{
                padding: "9px 12px", borderRadius: 6, fontSize: 14, color: "var(--ink-2)",
                borderLeft: "2px solid transparent",
              }}>{n.label}</Link>
            ))}
          </nav>
          <div style={{ minWidth: 0 }}>{children}</div>
        </div>
      </section>
      <Footer />
    </>
  );
}
