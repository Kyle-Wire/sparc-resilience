const MAX_BUDGET = 1_000_000_000_000; // 1 trillion

export interface BudgetParseResult {
  value: number | null;
  error: string | null;
}

/**
 * Parses a raw user-supplied budget string into a validated number.
 * Returns { value: null, error: null } for intentionally empty input.
 * Returns { value: null, error: string } for invalid input.
 * Returns { value: number, error: null } for valid input.
 */
export function parseBudget(raw: string): BudgetParseResult {
  const trimmed = raw.trim();
  if (trimmed === "") return { value: null, error: null };

  const n = Number(trimmed);

  if (!Number.isFinite(n)) {
    return { value: null, error: "Budget must be a valid number." };
  }

  if (n <= 0) {
    return { value: null, error: "Budget must be a positive number." };
  }

  if (n > MAX_BUDGET) {
    return { value: null, error: "Budget value is too large." };
  }

  return { value: n, error: null };
}
