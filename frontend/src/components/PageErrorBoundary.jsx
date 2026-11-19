import { Component } from 'react';

/**
 * Catches an error while a page renders and shows what happened instead of a blank screen. The navigation and the
 * rest of the app keep working.
 *
 *   <PageErrorBoundary name="Check data"><UploadPage /></PageErrorBoundary>
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

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    // A page's code could not be downloaded: usually a new version was deployed while this tab was open, or the
    // connection dropped. Trying again in place cannot fix that (the failed download is remembered); a reload can.
    const stale = /dynamically imported module|Importing a module script failed|ChunkLoadError|Loading chunk/i.test(error.message || '');
    return (
      <main className="in-main"><div className="in-page" role="alert">
        <h1 className="in-h1">{this.props.name ? `${this.props.name} could not be shown` : 'This page could not be shown'}</h1>
        <p className="in-lede">
          {stale ? 'This page could not be downloaded. Veritas may have been updated since this tab was opened, or the connection dropped. Reloading fixes both.'
            : 'Something in this page failed while it was drawn. Your session and stored results are not affected.'}
        </p>
        {!stale && error.message && <details className="in-hint"><summary>Technical detail</summary><pre className="in-code">{error.message}</pre></details>}
        <div className="in-actions" style={{ marginTop: 20 }}>
          {stale ? <button type="button" className="in-btn" onClick={() => window.location.reload()}>Reload Veritas</button>
            : <button type="button" className="in-btn" onClick={() => this.setState({ error: null })}>Try again</button>}
        </div>
      </div></main>
    );
  }
}
