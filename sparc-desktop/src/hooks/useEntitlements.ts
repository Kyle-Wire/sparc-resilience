/**
 * Plan-aware feature checks. Wraps `featuresFor()` against the
 * AuthProvider's current cached license.
 */
import { useMemo } from "react";
import { useAuth } from "@/lib/auth/AuthProvider";
import { featuresFor } from "@/lib/auth/entitlements";
import type { EntitlementSet } from "@/lib/auth/types";

export function useEntitlements(): EntitlementSet {
  const { license } = useAuth();
  return useMemo(() => featuresFor(license?.plan ?? "free"), [license?.plan]);
}
