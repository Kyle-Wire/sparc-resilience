import { useState, type ReactNode, type UIEvent } from "react";

export type VirtualListProps = {
  count: number;
  rowHeight: number;
  /** Viewport height in px. */
  height: number;
  overscan?: number;
  renderRow: (index: number) => ReactNode;
  getKey?: (index: number) => string | number;
  label?: string;
  /** Called with the scroll position (e.g. follow-tail logic). */
  onScroll?: (scrollTop: number, atEnd: boolean) => void;
  scrollRef?: (el: HTMLDivElement | null) => void;
  className?: string;
};

/** Visible rows of a window over `count` fixed-height rows. */
export function visibleRange(scrollTop: number, height: number, rowHeight: number, count: number, overscan: number): [number, number] {
  const first = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const last = Math.min(count, Math.ceil((scrollTop + height) / rowHeight) + overscan);
  return [first, Math.max(first, last)];
}

/** Fixed-row-height windowed list: only rows in (or near) the viewport are rendered. */
export function VirtualList({ count, rowHeight, height, overscan = 8, renderRow, getKey, label, onScroll, scrollRef, className }: VirtualListProps) {
  const [top, setTop] = useState(0);
  const [first, last] = visibleRange(top, height, rowHeight, count, overscan);
  const rows: ReactNode[] = [];
  for (let i = first; i < last; i++) {
    rows.push(
      <div key={getKey ? getKey(i) : i} className="vlist-row" role="listitem" style={{ top: i * rowHeight, height: rowHeight }}>
        {renderRow(i)}
      </div>,
    );
  }
  const handle = (e: UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    setTop(el.scrollTop);
    onScroll?.(el.scrollTop, el.scrollTop + el.clientHeight >= el.scrollHeight - rowHeight);
  };
  return (
    <div ref={scrollRef} className={["vlist", className ?? ""].filter(Boolean).join(" ")} style={{ height }} onScroll={handle} role="list" aria-label={label} tabIndex={0}>
      <div className="vlist-inner" style={{ height: count * rowHeight }}>
        {rows}
      </div>
    </div>
  );
}
