/**
 * A page section that failed to load. "Try again" is offered only when trying again can help: no answer from the
 * server (network), a server error or rate limiting. Signed out, not allowed or not found will not change on a retry.
 */
export function LoadError({ error, onRetry }) {
  const status = error?.status;
  const passing = !(status >= 400 && status < 500 && status !== 429);
  const message = status ? error.message : 'Could not reach Veritas. Check your connection.';
  return (
    <div className="in-load-error" role="alert">
      <p className="in-error">{message}</p>
      {passing && onRetry && <button type="button" className="in-btn in-btn-quiet" onClick={onRetry}>Try again</button>}
    </div>
  );
}
