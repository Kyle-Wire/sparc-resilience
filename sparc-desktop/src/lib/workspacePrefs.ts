/**
 * Default workspace folder preference.
 *
 * Stored via localStore so it travels with the install but not across
 * machines. When set, new projects (template click + wizard) default to
 * this directory instead of prompting the user every time.
 */

import { localStore } from "./localStore";

export function getWorkspaceDir(): string | null {
  try {
    return localStore.get("workspace-dir");
  } catch {
    return null;
  }
}

export function setWorkspaceDir(path: string | null): void {
  try {
    if (path && path.trim().length > 0) {
      localStore.set("workspace-dir", path);
    } else {
      localStore.remove("workspace-dir");
    }
  } catch {
    /* ignore */
  }
}

/** Join a parent dir + child folder name with the platform separator. */
export function joinPath(parent: string, child: string): string {
  const sep = parent.includes("\\") && !parent.includes("/") ? "\\" : "/";
  const trimmed = parent.endsWith(sep) ? parent.slice(0, -sep.length) : parent;
  return `${trimmed}${sep}${child}`;
}
