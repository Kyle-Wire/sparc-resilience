/**
 * Plan → entitlement set mapping.
 *
 * Pure data: no React, no I/O. Consumed by `useEntitlements()` and
 * by tests.
 */
import type { EntitlementSet, Plan } from "./types";

export function featuresFor(plan: Plan): EntitlementSet {
  switch (plan) {
    case "team":
      return {
        plan,
        maxProjects: -1,
        maxRunsPerMonth: -1,
        templates: "all",
        scenarioRunner: true,
        causalDiscoveryLearn: true,
        aiChat: true,
        reportExport: true,
        multiSeat: true,
      };
    case "pro":
      return {
        plan,
        maxProjects: -1,
        maxRunsPerMonth: -1,
        templates: "all",
        scenarioRunner: true,
        causalDiscoveryLearn: true,
        aiChat: true,
        reportExport: true,
        multiSeat: false,
      };
    case "free":
    default:
      return {
        plan: "free",
        maxProjects: 1,
        maxRunsPerMonth: 5,
        templates: "blank_only",
        scenarioRunner: false,
        causalDiscoveryLearn: false,
        aiChat: false,
        reportExport: false,
        multiSeat: false,
      };
  }
}
