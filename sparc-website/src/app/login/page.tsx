import Link from "next/link";
import { Suspense } from "react";
import { Nav, Footer } from "@/components/shell/Shell";
import { PageHero, Section } from "@/components/marketing/page-shell";
import { LoginForm } from "./login-parts";

export const metadata = { title: "Sign in" };

export default function LoginPage() {
  return (
    <>
      <Nav />
      <PageHero
        kicker="sign in"
        title="Welcome back."
        lead="Sign in to manage your subscription, seats, license keys, and downloads."
        accent="var(--purple)"
      />
      <Section>
        <div style={{ maxWidth: 460, margin: "0 auto" }}>
          <Suspense fallback={<div className="card" style={{ padding: 24 }}>Loading…</div>}>
            <LoginForm mode="signin" />
          </Suspense>
          <p style={{ textAlign: "center", marginTop: 18, fontSize: 13, color: "var(--muted)" }}>
            Don&apos;t have an account? <Link href="/signup" style={{ color: "var(--ink)" }}>Request access →</Link>
          </p>
        </div>
      </Section>
      <Footer />
    </>
  );
}
