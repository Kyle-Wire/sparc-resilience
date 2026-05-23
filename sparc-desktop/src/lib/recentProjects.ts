import { localStore } from "./localStore";

const MAX_RECENTS = 10;

export interface RecentProject {
  path: string;
  name: string;
  template: string;
  lastOpened: string; // ISO date string
}

export function getRecents(): RecentProject[] {
  try {
    const raw = localStore.get("recent-projects");
    if (!raw) return [];
    const parsed: RecentProject[] = JSON.parse(raw);
    return parsed.sort(
      (a, b) => new Date(b.lastOpened).getTime() - new Date(a.lastOpened).getTime(),
    );
  } catch {
    return [];
  }
}

export function addRecent(entry: Omit<RecentProject, "lastOpened">): void {
  const recents = getRecents().filter((r) => r.path !== entry.path);
  recents.unshift({ ...entry, lastOpened: new Date().toISOString() });
  localStore.set(
    "recent-projects",
    JSON.stringify(recents.slice(0, MAX_RECENTS)),
  );
}

export function removeRecent(path: string): void {
  const recents = getRecents().filter((r) => r.path !== path);
  localStore.set("recent-projects", JSON.stringify(recents));
}

export function clearRecents(): void {
  localStore.remove("recent-projects");
}
