import { useState, useCallback, type ReactNode, type DragEvent } from "react";

interface DropZoneProps {
  onFileDrop: (files: File[]) => void;
  /** File extensions to accept, e.g. [".csv", ".parquet"]. If empty, accepts all. */
  accept?: string[];
  children: ReactNode;
}

export default function DropZone({ onFileDrop, accept = [], children }: DropZoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const [rejected, setRejected] = useState(false);

  const checkAccepted = useCallback(
    (files: FileList) => {
      if (accept.length === 0) return true;
      return Array.from(files).every((f) =>
        accept.some((ext) => f.name.toLowerCase().endsWith(ext)),
      );
    },
    [accept],
  );

  const handleDragOver = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.dataTransfer.items.length > 0) {
        setDragOver(true);
        setRejected(false);
      }
    },
    [],
  );

  const handleDragEnter = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Only leave if we're actually leaving the drop zone (not entering a child)
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setDragOver(false);
    setRejected(false);
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOver(false);

      const { files } = e.dataTransfer;
      if (!files || files.length === 0) return;

      if (!checkAccepted(files)) {
        setRejected(true);
        setTimeout(() => setRejected(false), 1500);
        return;
      }

      onFileDrop(Array.from(files));
    },
    [onFileDrop, checkAccepted],
  );

  return (
    <div
      className="relative h-full"
      onDragOver={handleDragOver}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {children}

      {/* Drag overlay */}
      {(dragOver || rejected) && (
        <div
          className={`pointer-events-none absolute inset-0 z-50 flex items-center justify-center rounded-lg border-2 border-dashed transition-colors ${
            rejected
              ? "border-red-400 bg-red-50/80"
              : "border-sparc-purple bg-sparc-purple/5"
          }`}
        >
          <div className="rounded-lg bg-white px-6 py-4 shadow-lg">
            {rejected ? (
              <p className="text-sm font-medium text-red-600">
                File type not accepted
                {accept.length > 0 && (
                  <span className="block text-xs font-normal text-red-500">
                    Expected: {accept.join(", ")}
                  </span>
                )}
              </p>
            ) : (
              <p className="text-sm font-medium text-sparc-purple">
                Drop file here
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
