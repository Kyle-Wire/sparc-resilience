import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  experimental: {
    typedRoutes: true,
  },
  async redirects() {
    return [
      // Old static-prototype URLs → App Router routes
      { source: "/index.html", destination: "/", permanent: true },
      { source: "/product.html", destination: "/product", permanent: true },
      { source: "/templates.html", destination: "/templates", permanent: true },
      { source: "/pricing.html", destination: "/pricing", permanent: true },
      { source: "/about.html", destination: "/about", permanent: true },
      { source: "/contact.html", destination: "/contact", permanent: true },
    ];
  },
};

export default config;
