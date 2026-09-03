import { Component } from 'react';
import { Shield, RefreshCw } from 'lucide-react';

/**
 * Error Boundary (L3) — catches unhandled React component errors and displays
 * a graceful fallback instead of a blank screen.
 *
 * Usage:
 *   <PageErrorBoundary name="Upload Page">
 *     <UploadPage />
 *   </PageErrorBoundary>
 */
export default class PageErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error('[ErrorBoundary]', this.props.name, error, info.componentStack);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <div className="flex flex-col items-center justify-center h-full min-h-96 p-12 gap-6">
        <div className="w-16 h-16 rounded-2xl bg-danger/10 border border-danger/30 flex items-center justify-center">
          <Shield className="w-8 h-8 text-danger" />
        </div>
        <div className="text-center max-w-md">
          <h2 className="font-mono text-lg text-danger mb-2">
            {this.props.name || 'Page'} — Render Error
          </h2>
          <p className="text-text3 font-mono text-xs mb-4">
            An unexpected error occurred in this component. Your session and
            analysis data are safe — only this view failed to render.
          </p>
          {this.state.error && (
            <pre className="text-left bg-surface border border-border rounded p-3 text-xs text-text2 font-mono overflow-auto max-h-32 mb-4">
              {this.state.error.message}
            </pre>
          )}
        </div>
        <button
          onClick={this.handleReset}
          className="flex items-center gap-2 px-4 py-2 bg-accent/10 border border-accent/30 text-accent font-mono text-xs rounded hover:bg-accent/20 transition-all"
        >
          <RefreshCw className="w-3 h-3" />
          Retry
        </button>
      </div>
    );
  }
}
