// Inline SVG icons (24×24 viewBox, stroke = currentColor). Icons are decorative by default;
// pass `label` when an icon is the only content of a control.

const PATHS = {
  check: "M5 12.5l4.5 4.5L19 7.5",
  x: "M6 6l12 12M18 6L6 18",
  alert: "M12 4l9 16H3zM12 10v4.5M12 17.6v.1",
  info: "M12 21a9 9 0 110-18 9 9 0 010 18zM12 11v5.5M12 7.6v.1",
  clock: "M12 21a9 9 0 110-18 9 9 0 010 18zM12 7v5l3.5 2",
  play: "M8 5.5v13l10.5-6.5z",
  pause: "M8 5.5v13M16 5.5v13",
  stop: "M6.5 6.5h11v11h-11z",
  cached: "M4 7c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 7v10c0 1.7 3.6 3 8 3s8-1.3 8-3V7M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  skip: "M5 12h14M13 6l6 6-6 6",
  minus: "M5 12h14",
  plus: "M12 5v14M5 12h14",
  ban: "M12 21a9 9 0 110-18 9 9 0 010 18zM5.6 5.6l12.8 12.8",
  pulse: "M3 12h4l2.5-6 4 12 2.5-6H21",
  search: "M10.5 17a6.5 6.5 0 110-13 6.5 6.5 0 010 13zM15.5 15.5L20 20",
  sun: "M12 16.5a4.5 4.5 0 110-9 4.5 4.5 0 010 9zM12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4",
  moon: "M20 14.5A8 8 0 019.5 4a8 8 0 1010.5 10.5z",
  monitor: "M3.5 5h17v11h-17zM8.5 20h7M12 16v4",
  menu: "M4 7h16M4 12h16M4 17h16",
  chevronDown: "M6 9l6 6 6-6",
  chevronUp: "M6 15l6-6 6 6",
  chevronLeft: "M15 6l-6 6 6 6",
  chevronRight: "M9 6l6 6-6 6",
  external: "M14 4h6v6M20 4l-9 9M18 14v5H5V6h5",
  download: "M12 4v11M7 10.5l5 5 5-5M5 20h14",
  copy: "M8.5 8.5h11v11h-11zM15.5 8.5v-4h-11v11h4",
  pin: "M9 4h6l-1 6 3.5 3.5H6.5L10 10zM12 13.5V20",
  table: "M4 5h16v14H4zM4 10h16M4 14.5h16M10 5v14",
  image: "M4 5h16v14H4zM4 16l5-5 4 4 2.5-2.5L20 17M15.5 9.5v.1",
  refresh: "M20 11a8 8 0 10-2.4 5.7M20 4.5V11h-6.5",
  dot: "M12 15a3 3 0 110-6 3 3 0 010 6z",
  bolt: "M13 3L5 13.5h6.5L10.5 21 19 10.5h-6.5z",
  layers: "M12 4l9 5-9 5-9-5zM3 14l9 5 9-5",
  filter: "M4 5h16l-6 7.5V19l-4-2v-4.5z",
  eye: "M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12zM12 15a3 3 0 110-6 3 3 0 010 6z",
  lock: "M6.5 11h11v9h-11zM8.5 11V8a3.5 3.5 0 017 0v3",
  unlock: "M6.5 11h11v9h-11zM8.5 11V8a3.5 3.5 0 016.8-1.2",
  hourglass: "M7 3.5h10M7 20.5h10M8 3.5c0 5 8 5 8 8.5s-8 3.5-8 8.5M16 3.5c0 5-8 5-8 8.5s8 3.5 8 8.5",
  wifi: "M3 9.5a13 13 0 0118 0M6 13a8.5 8.5 0 0112 0M9 16.5a4 4 0 016 0M12 20v.1",
  wifiOff: "M3 3l18 18M9 16.5a4 4 0 016 0M6 13a8.5 8.5 0 015.2-2.5M16 11.2a8.5 8.5 0 012 1.8M3 9.5a13 13 0 014.3-2.9M12.2 6a13 13 0 018.8 3.5M12 20v.1",
  folder: "M3.5 6.5h6l2 2h9v10h-17z",
  file: "M6 3.5h8l4 4v13H6zM14 3.5v4h4",
  grip: "M9 6v.1M15 6v.1M9 12v.1M15 12v.1M9 18v.1M15 18v.1",
  undo: "M9 7L4.5 11.5 9 16M5 11.5h9.5a5 5 0 010 10H12",
  redo: "M15 7l4.5 4.5L15 16M19 11.5H9.5a5 5 0 000 10H12",
  hand: "M8 12V6.5a1.5 1.5 0 013 0V11M11 10.5V5a1.5 1.5 0 013 0v5.5M14 10.5V6.5a1.5 1.5 0 013 0V14a6 6 0 01-6 6h-.5a6 6 0 01-5-2.7L3.5 14a1.5 1.5 0 012.5-1.7L8 14.5",
  brush: "M14.5 4.5l5 5-8 8-5-5zM6.5 12.5L4 20l7.5-2.5",
  square: "M5 5h14v14H5z",
  circle: "M12 20a8 8 0 110-16 8 8 0 010 16z",
  polygon: "M12 3.5l8.5 6-3.2 10H6.7l-3.2-10z",
  target: "M12 20a8 8 0 110-16 8 8 0 010 16zM12 15a3 3 0 110-6 3 3 0 010 6zM12 2v3M12 19v3M2 12h3M19 12h3",
  swap: "M7 4.5L3.5 8 7 11.5M3.5 8h13M17 12.5l3.5 3.5-3.5 3.5M20.5 16h-13",
  command: "M9 6.5a2.5 2.5 0 10-2.5 2.5H9zM9 9h6v6H9zM15 6.5A2.5 2.5 0 1117.5 9H15zM9 17.5A2.5 2.5 0 116.5 15H9zM15 17.5a2.5 2.5 0 102.5-2.5H15z",
  home: "M3.5 11L12 4l8.5 7M6 9.5V20h12V9.5",
  settings: "M12 15a3 3 0 110-6 3 3 0 010 6zM19.4 13.5l1.6 1.2-2 3.4-1.9-.7a7 7 0 01-2.1 1.2L14.7 21h-4l-.3-2.4a7 7 0 01-2.1-1.2l-1.9.7-2-3.4 1.6-1.2a7 7 0 010-3L4.4 9.3l2-3.4 1.9.7a7 7 0 012.1-1.2L10.7 3h4l.3 2.4a7 7 0 012.1 1.2l1.9-.7 2 3.4-1.6 1.2a7 7 0 010 3z",
  activity: "M3 12h4l3-8 4 16 3-8h4",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, label, size = 16, className }: { name: IconName; label?: string; size?: number; className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      focusable="false"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
