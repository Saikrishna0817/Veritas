/** "Load more" under a list. `more` is the result of useMore; renders nothing on the last page. */
export function LoadMoreButton({ more, what }) {
  if (!more.next) return null;
  return (
    <div className="in-actions">
      <button type="button" className="in-btn in-btn-quiet" onClick={more.load} disabled={more.busy}>{more.busy ? 'Loading…' : `Load more ${what}`}</button>
      {more.error && <span className="in-error" role="alert">More {what} could not be loaded: {more.error.message}</span>}
    </div>
  );
}
