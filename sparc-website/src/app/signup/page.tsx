import Link from "next/link";
import { Suspense } from "react";
import { Nav, Footer } from "@/components/shell/Shell";
import { PageHero, Section } from "@/components/marketing/page-shell";
import { LoginForm } from "../login/login-parts";

export const metadata = { title: "Request access" };

export default function SignupPage() {
  return (
    <>
      <Nav />
      <PageHero
        kicker="create account"
        title="Request access to SPARC."
        lead="One magic link, no passwords. We'll create your account on first sign-in. Add Pursue, Converge, or Transcend after you're in."
        accent="var(--crimson)"
      />
      <Section>
        <div style={{ maxWidth: 460, margin: "0 auto" }}>
          <Suspense fallback={<div className="card" style={{ padding: 24 }}>Loading…</div>}>
            <LoginForm mode="signup" />
          </Suspense>
          <p style={{ textAlign: "center", marginTop: 18, fontSize: 13, color: "var(--muted)" }}>
            Already have an account? <Link href="/login" style={{ color: "var(--ink)" }}>Sign in →</Link>
          </p>
        </div>
      </Section>
      <Footer />
    </>
  );
}
