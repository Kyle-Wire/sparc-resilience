import { useRef, useState, type DragEvent, type ReactNode } from "react";
import { errorMessage, putRaw } from "../../api/client";
import { fmtBytes, fmtBytesOf } from "../../theme/format";
import { ProgressBar } from "./ProgressBar";

export type UploadSpec<R> = {
  /** Upload URL for a file, e.g. `/api/projects/p_1/files/data/${name}`. */
  url: (file: File) => string;
  contentType?: (file: File) => string;
  onDone: (result: R, file: File) => void;
  onError?: (error: unknown, file: File) => void;
};

export type FileDropProps<R> = {
  label: string;
  hint?: ReactNode;
  accept?: string;
  multiple?: boolean;
  disabled?: boolean;
  /** Receive the files and upload them yourself… */
  onFiles?: (files: File[]) => void;
  /** …or let FileDrop stream them with a raw PUT and show progress. */
  upload?: UploadSpec<R>;
};

type Progress = { name: string; sent: number; total: number; error?: string; done?: boolean };

/** Drop zone + file picker (click, Enter or Space). Uploads are streamed raw PUT bodies. */
export function FileDrop<R = unknown>({ label, hint, accept, multiple, disabled, onFiles, upload }: FileDropProps<R>) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [progress, setProgress] = useState<Progress[]>([]);

  const handle = async (list: File[]) => {
    if (!list.length || disabled) return;
    const files = multiple ? list : [list[0]];
    onFiles?.(files);
    if (!upload) return;
    setProgress(files.map((f) => ({ name: f.name, sent: 0, total: f.size })));
    await Promise.all(
      files.map(async (f, idx) => {
        try {
          const r = await putRaw<R>(upload.url(f), f, {
            contentType: upload.contentType?.(f) ?? "application/octet-stream",
            onProgress: (sent, total) => setProgress((p) => p.map((x, k) => (k === idx ? { ...x, sent, total: total || x.total } : x))),
          });
          setProgress((p) => p.map((x, k) => (k === idx ? { ...x, sent: x.total, done: true } : x)));
          upload.onDone(r, f);
        } catch (e) {
          setProgress((p) => p.map((x, k) => (k === idx ? { ...x, error: errorMessage(e) } : x)));
          upload.onError?.(e, f);
        }
      }),
    );
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    void handle([...(e.dataTransfer.files ?? [])]);
  };

  return (
    <div className="stack" style={{ gap: 8 }}>
      <div
        className="filedrop"
        data-over={over || undefined}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled || undefined}
        aria-label={label}
        onClick={() => !disabled && input.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && !disabled) {
            e.preventDefault();
            input.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
      >
        <strong>{label}</strong>
        <span className="cap">{hint ?? "Drop a file here, or click to choose one."}</span>
        <input
          ref={input}
          type="file"
          hidden
          accept={accept}
          multiple={multiple}
          onChange={(e) => {
            const files = [...(e.target.files ?? [])];
            // Clear the picker so choosing the same file again (after fixing it) fires again.
            e.target.value = "";
            void handle(files);
          }}
        />
      </div>
      {progress.map((p) => (
        <div key={p.name} className="stack" style={{ gap: 4 }}>
          <div className="row cap">
            <span className="mono">{p.name}</span>
            <span className="spacer" />
            <span>{p.error ? `Failed: ${p.error}` : p.done ? `Uploaded ${fmtBytes(p.total)}` : fmtBytesOf(p.sent, p.total)}</span>
          </div>
          <ProgressBar value={p.total ? p.sent / p.total : null} label={`Uploading ${p.name}`} tone={p.error ? "crit" : p.done ? "good" : undefined} />
        </div>
      ))}
    </div>
  );
}
