// SPARC Labs · brand constants. Single source of truth — every page reads from here.

export const BRAND_LONG = "Spatial Research";
export const BRAND_SHORT = "SPARC";
export const COMPANY = "SPARC Labs";
export const COMPANY_LONG = "Spatial Research Labs";
export const COMPANY_LEGAL = "SPARC Labs LLC";

export const DOMAIN = "sparclabs.co";
export const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? `https://${DOMAIN}`;

export const EMAIL_PRIMARY = "SparcUrbanLabs@gmail.com";

export const OFFICE_STREET = "800 South Street";
export const OFFICE_CITY = "Philadelphia";
export const OFFICE_STATE = "PA";
export const OFFICE_ZIP = "19147";
export const OFFICE_FULL = `${OFFICE_STREET} · ${OFFICE_CITY} · ${OFFICE_STATE} ${OFFICE_ZIP}`;

export const VERSION = "v0.4.2";
export const FOUNDED_YEAR = 2024;
export const COPYRIGHT_YEAR = 2026;

export const TAGLINE_SHORT = `${BRAND_LONG} · ${VERSION}`;
export const TAGLINE_LONG =
  `${BRAND_LONG}. A neural model with built-in causal inference — for spatial scientists ` +
  `who need a pipeline that can simulate adaptation, not just describe the world.`;
