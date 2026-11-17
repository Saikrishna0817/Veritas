/**
 * Loading placeholders shaped like what is about to appear (a title and lines, or table rows), so the page does not
 * jump when the data arrives. Screen readers get one short status line instead of the shapes.
 */
export function Skeleton({ label = 'Loading…', title = false, lines = 2, rows = 0 }) {
  return (
    <div role="status" aria-live="polite">
      <span className="in-sr">{label}</span>
      <div aria-hidden="true">
        {title && <span className="in-skel in-skel-title" />}
        {Array.from({ length: lines }, (_, i) => <span key={i} className={`in-skel in-skel-line${i === lines - 1 ? ' short' : ''}`} />)}
        {rows > 0 && (
          <div className="in-skel-rows">
            {Array.from({ length: rows }, (_, i) => <span key={i} className="in-skel in-skel-row" />)}
          </div>
        )}
      </div>
    </div>
  );
}
