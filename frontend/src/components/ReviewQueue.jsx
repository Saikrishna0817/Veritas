import { useCallback, useState } from 'react';
import { api } from '../services/api';
import { useLoad } from '../hooks/useLoad';
import { LoadError } from './LoadError';
import { ConfirmAction, Notice, shortId, when } from '../pages/admin/shared';

const ev = (x) => (typeof x === 'number' ? x.toFixed(2) : '—');

/**
 * The signed-in person's own review cases (GET /defense/hitl/pending): table results between the 1-in-100 and
 * 1-in-10,000 false-alarm levels wait here for a person. Each is decided with POST /defense/hitl/decide; the server
 * records who decided. Shows nothing while there are no cases, so the Overview stays quiet.
 */
export default function ReviewQueue() {
  const { data: cases, error, retry } = useLoad(useCallback(() => api.getPendingReviews().then((d) => d.cases || []), []));
  const [notice, setNotice] = useState(null);

  const decide = async (c, decision) => {
    try {
      await api.submitReviewDecision(c.case_id, decision);
      setNotice({ text: decision === 'approve_quarantine'
        ? `Case ${shortId(c.case_id)}: its flagged rows are recorded as held back. No data was changed.`
        : `Case ${shortId(c.case_id)} closed with no action.` });
      retry();
    } catch (err) {
      setNotice({ error: true, text: `Case ${shortId(c.case_id)} was not updated: ${err.message || 'the server refused it.'}` });
    }
  };

  if (!error && !cases?.length && !notice) return null;
  return (
    <section className="in-section" aria-labelledby="review-title">
      <h2 id="review-title" className="in-h2">Waiting for your review</h2>
      <p className="in-sub">
        Table results with evidence between the 1-in-100 and 1-in-10,000 false-alarm levels wait here for a person.
        Holding back a case's rows only records the decision: Veritas does not change your data.
      </p>
      <Notice notice={notice} onClose={() => setNotice(null)} />
      {error && <LoadError error={error} onRetry={retry} />}
      {cases && !cases.length && <p className="in-empty">No cases are waiting for review.</p>}
      {cases?.length > 0 && (
        <div className="in-table-wrap">
          <table className="in-table in-table-stack">
            <caption className="in-sr">Your cases waiting for review, newest first</caption>
            <thead>
              <tr>
                <th scope="col">Case</th><th scope="col">Opened</th><th scope="col" className="in-right">Rows in batch</th>
                <th scope="col" className="in-right">Label evidence</th><th scope="col" className="in-right">Shift evidence</th>
                <th scope="col"><span className="in-sr">Decision</span></th>
              </tr>
            </thead>
            <tbody>
              {cases.map((c) => (
                <tr key={c.case_id}>
                  <td data-label="Case" className="in-path" title={c.case_id}>{shortId(c.case_id)}</td>
                  <td data-label="Opened" className="in-tag">{when(c.created_at)}</td>
                  <td data-label="Rows in batch" className="in-num">{c.n_samples ?? '—'}</td>
                  <td data-label="Label evidence" className="in-num">{ev(c.evidence_summary?.label_evidence)}</td>
                  <td data-label="Shift evidence" className="in-num">{ev(c.evidence_summary?.shift_evidence)}</td>
                  <td data-label="" className="in-cell-actions">
                    <ConfirmAction
                      label="Hold back rows"
                      confirmLabel="Record them"
                      message={`Record the ${c.sample_ids?.length ?? ''} flagged rows of case ${shortId(c.case_id)} as held back? No data is changed.`}
                      onConfirm={() => decide(c, 'approve_quarantine')}
                    />
                    <ConfirmAction
                      label="Close, no action"
                      confirmLabel="Close case"
                      message={`Close case ${shortId(c.case_id)} without holding anything back? It leaves the queue.`}
                      onConfirm={() => decide(c, 'mark_safe')}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {cases?.length > 0 && <p className="in-hint">Evidence is −log₁₀ of the check's p-value: 2 is the 1-in-100 level, 4 the 1-in-10,000 level.</p>}
    </section>
  );
}
