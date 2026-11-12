import { Component } from 'react';
import { Link } from 'react-router-dom';
import { RotateCw, TriangleAlert } from 'lucide-react';
import { Button, Page } from './ui';

/**
 * Catches a render error inside one page so the shell (navigation, session) keeps working.
 *
 *   <PageErrorBoundary name="History"><HistoryPage /></PageErrorBoundary>
 */
export default class PageErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error('[PageErrorBoundary]', this.props.name, error, info.componentStack);
  }

  reset = () => this.setState({ error: null });

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <Page>
        <div role="alert" className="edge-card quiet max-w-2xl">
          <div className="edge-inner p-8 md:p-10">
            <span className="icon-chip h-11 w-11"><TriangleAlert className="h-4 w-4 text-statusWarn" strokeWidth={1.5} aria-hidden /></span>
            <h1 className="font-display mt-6 text-[clamp(2rem,4vw,3rem)] font-medium leading-none">
              This page hit a <span className="serif-accent">problem</span>
            </h1>
            <p className="mt-4 max-w-[56ch] text-textSecondary">
              {this.props.name ? <>The {this.props.name} view</> : 'This view'} failed to draw. Your session and your results are safe; only this page is affected.
            </p>
            {error.message && <p className="mt-5 rounded-2xl bg-white/[0.04] px-4 py-3 font-mono text-xs text-textMuted break-words">{error.message}</p>}
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <Button onClick={this.reset}><RotateCw className="h-4 w-4" aria-hidden /> Try again</Button>
              <Link to="/" onClick={this.reset} className="inline-flex min-h-[44px] items-center rounded-2xl px-4 text-sm text-violet hover:underline">Back to the Overview</Link>
            </div>
          </div>
        </div>
      </Page>
    );
  }
}
