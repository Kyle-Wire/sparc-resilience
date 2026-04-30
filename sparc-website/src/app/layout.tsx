import type { Metadata } from "next";
import "../styles/globals.css";
import { COMPANY, BRAND_LONG, TAGLINE_LONG, SITE_URL } from "@/lib/brand";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: `${COMPANY} · ${BRAND_LONG}`,
    template: `%s · ${COMPANY}`,
  },
  description: TAGLINE_LONG,
  openGraph: {
    title: `${COMPANY} · ${BRAND_LONG}`,
    description: TAGLINE_LONG,
    url: SITE_URL,
    siteName: COMPANY,
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: `${COMPANY} · ${BRAND_LONG}`,
    description: TAGLINE_LONG,
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
