import { Component, lazy, Suspense } from 'react';

// three.js loads with this chunk only, when a result with row scores is shown.
const ScoreCloud = lazy(() => import('./ScoreCloud'));

class Quiet extends Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? null : this.props.children; }
}

const n = (x) => x.toLocaleString('en');

/** The 3D view of a table result's row scores, with its text alternative. Nothing without real scores or WebGL. */
export default function ScoreCloudSlot({ scores, flagged }) {
  if (!scores?.length || !window.WebGL2RenderingContext) return null;
  const top = scores.reduce((a, b) => Math.max(a, b), 0);
  const label = `Every row of the batch as a point: ${n(scores.length)} rows scored, ${n(flagged.length)} flagged, highest row score ${top.toFixed(1)}. The list of rows to review has the flagged rows in text.`;
  return (
    <div className="in-cloud" role="img" aria-label={label}>
      <Quiet><Suspense fallback={null}><ScoreCloud scores={scores} flagged={flagged} /></Suspense></Quiet>
    </div>
  );
}
