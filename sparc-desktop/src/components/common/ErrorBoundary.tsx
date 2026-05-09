import React from "react";

interface Props {
  page: string;
  children: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  reportState: "idle" | "sending" | "sent" | "failed";
}

/**
 * Page-level error boundary. Catches render errors, displays a recovery card,
 * and lets the user send a crash report to the team via Supabase Edge Function.
 */
export default class ErrorBoundary extends React.Component<Props, State> {
  state: State = { hasError: false, error: null, reportState: "idle" };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    console.error("[ErrorBoundary]", this.props.page, error, info.componentStack);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, reportState: "idle" });
  };

  handleReport = async () => {
    if (this.state.reportState !== "idle") return;
    this.setState({ reportState: "sending" });
    try {
      const { reportCrash } = await import("@/lib/crashReporter");
      await reportCrash({ page: this.props.page, error: this.state.error! });
      this.setState({ reportState: "sent" });
    } catch {
      this.setState({ reportState: "failed" });
    }
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    const { error, reportState } = this.state;
    const reportLabel =
      reportState === "sending" ? "Sending…"
      : reportState === "sent" ? "Report sent ✓"
      : reportState === "failed" ? "Send failed — try again"
      : "Send crash report";

    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          minHeight: 300,
          padding: 32,
          gap: 16,
        }}
      >
        <div
          style={{
            background: "var(--surface, #fff)",
            border: "1px solid var(--line, #e0e0e0)",
            borderRadius: 8,
            padding: "24px 32px",
            maxWidth: 480,
            width: "100%",
          }}
        >
          <div
            style={{
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 11,
              color: "var(--crimson, #e94461)",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              marginBottom: 8,
            }}
          >
            Render error · {this.props.page}
          </div>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>
            Something went wrong on this page.
          </div>
          {error && (
            <pre
              style={{
                fontSize: 11,
                color: "var(--muted, #666)",
                whiteSpace: "pre-wrap",
                wordBreak: "break-all",
                marginBottom: 16,
                maxHeight: 140,
                overflow: "auto",
              }}
            >
              {error.message}
            </pre>
          )}
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button
              onClick={this.handleReset}
              style={{
                padding: "7px 14px",
                fontSize: 12,
                fontWeight: 600,
                border: "none",
                borderRadius: 4,
                background: "var(--ink, #111)",
                color: "#fff",
                cursor: "pointer",
              }}
            >
              Try again
            </button>
            <button
              onClick={this.handleReport}
              disabled={reportState === "sending" || reportState === "sent"}
              style={{
                padding: "7px 14px",
                fontSize: 12,
                fontWeight: 600,
                border: "1px solid var(--line, #e0e0e0)",
                borderRadius: 4,
                background: "transparent",
                color:
                  reportState === "sent"
                    ? "var(--green, #2e7d32)"
                    : "var(--ink, #111)",
                cursor:
                  reportState === "sending" || reportState === "sent"
                    ? "default"
                    : "pointer",
              }}
            >
              {reportLabel}
            </button>
          </div>
        </div>
      </div>
    );
  }
}
