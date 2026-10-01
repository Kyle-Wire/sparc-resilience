// Foundation routes: the catch-all "not found" page (SPEC §3.2, owner foundation).
import { lazy } from "react";
import type { RouteDef } from "../../router";

export const routes: RouteDef[] = [{ path: "*", component: lazy(() => import("../NotFound")), title: "Not found" }];
