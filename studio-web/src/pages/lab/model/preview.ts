// Live preview requests (SPEC §7.5; api.md §7.3). Every request carries a monotonically
// increasing `request_seq`; inputs are debounced at 120 ms; a response is applied only when
// its sequence number is newer than the last one applied, so a superseded (older) response
// arriving late never overwrites a newer map. The server is single-flight latest-wins and
// answers queued older requests with `409 superseded`, which is ignored silently.
import { ApiError } from "../../../api/client";
import type { PreviewRequest, PreviewResult } from "../../../api/lab";
import { debounce, type Limited } from "./timing";

export const PREVIEW_DEBOUNCE_MS = 120;

export type PreviewInput = Omit<PreviewRequest, "request_seq">;

export type PreviewControllerOptions = {
  send: (body: PreviewRequest) => Promise<PreviewResult>;
  onResult: (r: PreviewResult) => void;
  onError?: (e: unknown) => void;
  /** In flight or waiting for the debounce. */
  onBusy?: (busy: boolean) => void;
  delayMs?: number;
};

export class PreviewController {
  private seq = 0;
  private applied = 0;
  private inflight = 0;
  private disposed = false;
  private readonly opts: PreviewControllerOptions;
  private readonly debounced: Limited<[PreviewInput]>;

  constructor(opts: PreviewControllerOptions) {
    this.opts = opts;
    this.debounced = debounce((input: PreviewInput) => void this.send(input), opts.delayMs ?? PREVIEW_DEBOUNCE_MS);
  }

  /** The last request_seq sent. */
  get lastSent(): number {
    return this.seq;
  }

  /** The request_seq of the result on screen. */
  get lastApplied(): number {
    return this.applied;
  }

  /** Ask for a preview of these inputs (debounced). */
  request(input: PreviewInput): void {
    if (this.disposed) return;
    this.debounced(input);
    this.opts.onBusy?.(true);
  }

  /** Send any pending request now. */
  flush(): void {
    this.debounced.flush();
  }

  /** Forget pending work and drop every response still in flight (e.g. edits were cleared). */
  reset(): void {
    this.debounced.cancel();
    this.applied = this.seq; // anything in flight is now superseded
    this.opts.onBusy?.(this.inflight > 0);
  }

  dispose(): void {
    this.disposed = true;
    this.debounced.cancel();
  }

  /** Send one request immediately (exported for tests through `request` + timers). */
  async send(input: PreviewInput): Promise<void> {
    const seq = ++this.seq;
    this.inflight++;
    try {
      const r = await this.opts.send({ ...input, request_seq: seq });
      const got = typeof r.summary.request_seq === "number" ? r.summary.request_seq : seq;
      if (this.disposed || got <= this.applied || got > this.seq) return; // superseded or foreign
      this.applied = got;
      this.opts.onResult(r);
    } catch (e) {
      if (this.disposed) return;
      if (e instanceof ApiError && (e.code === "superseded" || e.code === "aborted")) return;
      if (seq < this.seq) return; // an error of an older request no longer matters
      this.opts.onError?.(e);
    } finally {
      this.inflight--;
      if (!this.disposed) this.opts.onBusy?.(this.inflight > 0 || this.debounced.pending());
    }
  }
}
