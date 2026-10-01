// Keeps a crashing page from taking the shell down; shows the error with a reload button.
// `resetKey` clears the error when it changes (e.g. on navigation) without remounting the
// children, so a layout around the page keeps its state and focus.
import { Component, type ErrorInfo, type ReactNode } from "react";

type State = { error: Error | null };

type Props = { children: ReactNode; resetKey?: unknown };

export class ErrorBoundary extends Component<Props, State> {
  override state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  override componentDidUpdate(prev: Props): void {
    if (this.state.error && prev.resetKey !== this.props.resetKey) this.setState({ error: null });
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[studio] page crashed", error, info.componentStack);
  }

  override render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="empty" role="alert">
        <div className="empty-title">This view failed to render</div>
        <div className="empty-body mono">{this.state.error.message}</div>
        <button type="button" className="btn" onClick={() => this.setState({ error: null })}>
          Try again
        </button>
      </div>
    );
  }
}
