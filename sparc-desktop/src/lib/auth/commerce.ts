/**
 * SPARC commerce URLs. Update here when the Lemon Squeezy store is
 * approved and final variant URLs are available.
 *
 * Storefront base:  https://sparclabs.lemonsqueezy.com
 * Customer portal:  Lemon Squeezy hosts a magic-link customer area
 *                   at https://app.lemonsqueezy.com/my-orders that any
 *                   customer can sign into with their purchase email.
 *
 * To wire up a *direct* checkout for a specific variant (skipping the
 * storefront browse page), use:
 *   https://sparclabs.lemonsqueezy.com/buy/<VARIANT_UUID>
 * Note: LS provides a UUID for each variant separate from the
 * numeric variant ID — find it under Variants → Share Buy Link.
 */

export const LEMON_STOREFRONT_URL = "https://sparclabs.lemonsqueezy.com";

/** Where the "Upgrade" button sends free-tier users. */
export const LEMON_UPGRADE_URL = LEMON_STOREFRONT_URL;

/** Where "Manage subscription" sends paid users. LS hosts a generic
 *  customer self-service area; customers sign in with the email they
 *  used at checkout. Replace with a per-user portal URL once the
 *  app starts minting them via the LS API. */
export const LEMON_CUSTOMER_PORTAL_URL = "https://app.lemonsqueezy.com/my-orders";
