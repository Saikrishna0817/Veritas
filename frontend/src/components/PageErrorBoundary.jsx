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
    if (!this.state.error) return this.props.children;
    return (
      <main className="in-main"><div className="in-page" role="alert">
        <h1 className="in-h1">{this.props.name ? `${this.props.name} could not be shown` : 'This page could not be shown'}</h1>
        <p className="in-lede">Something in this page failed while it was drawn. Your session and stored results are not affected.</p>
        {this.state.error.message && <pre className="in-code">{this.state.error.message}</pre>}
        <div className="in-actions" style={{ marginTop: 20 }}>
          <button type="button" className="in-btn" onClick={() => this.setState({ error: null })}>Try again</button>
        </div>
      </div></main>
    );
  }
}
