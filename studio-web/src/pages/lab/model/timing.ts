// Rate limiting for live controls: the plan budget slider (≤ 1 preview per 150 ms while it
// moves, SPEC §7.10) and the selection live count. Timers come from the global
// setTimeout/clearTimeout so fake timers drive them in tests.

export type Limited<A extends unknown[]> = {
  (...args: A): void;
  /** Run a pending call now. */
  flush: () => void;
  cancel: () => void;
  pending: () => boolean;
};

/**
 * Trailing-edge rate limiter: the first call in a quiet period waits `ms`, then fires with the
 * latest arguments; calls arriving meanwhile only replace the arguments. Under continuous
 * input this fires at most once per `ms` (live updates while dragging) and always fires once
 * more with the final value.
 */
export function rateLimit<A extends unknown[]>(fn: (...args: A) => void, ms: number): Limited<A> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let args: A | null = null;
  const fire = () => {
    timer = null;
    if (!args) return;
    const a = args;
    args = null;
    fn(...a);
  };
  const call = ((...a: A) => {
    args = a;
    if (timer === null) timer = setTimeout(fire, ms);
  }) as Limited<A>;
  call.flush = () => {
    if (timer !== null) clearTimeout(timer);
    fire();
  };
  call.cancel = () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    args = null;
  };
  call.pending = () => timer !== null;
  return call;
}

/** Classic trailing debounce: fires `ms` after the last call. */
export function debounce<A extends unknown[]>(fn: (...args: A) => void, ms: number): Limited<A> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let args: A | null = null;
  const fire = () => {
    timer = null;
    if (!args) return;
    const a = args;
    args = null;
    fn(...a);
  };
  const call = ((...a: A) => {
    args = a;
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(fire, ms);
  }) as Limited<A>;
  call.flush = () => {
    if (timer !== null) clearTimeout(timer);
    fire();
  };
  call.cancel = () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    args = null;
  };
  call.pending = () => timer !== null;
  return call;
}
