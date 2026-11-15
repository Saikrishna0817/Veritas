import { api } from '../../services/api';
import { ConfirmAction, LoadState, shortId, when } from './shared';

const ACTION_WORDS = { quarantine: 'rows recorded as held back', soft_quarantine: 'rows recorded for down-weighting', monitor: 'no action', observe: 'observed only', cooldown: 'skipped (repeated too soon)' };
const ev = (x) => (typeof x === 'number' ? x.toFixed(2) : '—');

/**
 * Recorded defence actions and the review queue. GET /defense/status, POST /defense/quarantine,
 * GET /defense/hitl/pending, POST /defense/hitl/decide. Both only record decisions: nothing in Veritas reads
 * these records to change data or block uploads yet, and the copy says so.
 */
export default function DefenseTab({ status, cases, self, notify, onChanged }) {
  const s = status.data;
  const last = s?.last_action;

  const holdBack = async () => {
    try {
      const res = await api.triggerQuarantine();
      const n = res.samples_affected ?? 0;
      notify({ text: n
        ? `Recorded ${n} flagged row${n === 1 ? '' : 's'} of your latest table result as held back.`
        : 'Nothing new was recorded: your latest table result has no flagged rows that are not already held back.' });
      onChanged();
    } catch (err) {
      notify({ error: true, text: `Nothing was recorded: ${err.message || 'the server refused it.'}` });
    }
  };

  const decide = async (c, decision) => {
    try {
      await api.submitReviewDecision(c.case_id, decision, self?.name || 'admin');
      notify({ text: decision === 'approve_quarantine'
        ? `Case ${shortId(c.case_id)}: its flagged rows are recorded as held back.`
        : `Case ${shortId(c.case_id)} closed with no action.` });
      onChanged();
    } catch (err) {
      notify({ error: true, text: `Case ${shortId(c.case_id)} was not updated: ${err.message || 'the server refused it.'}` });
    }
  };

  return (
    <>
      <section className="in-panel" aria-labelledby="hold-title">
        <h2 id="hold-title" className="in-h2">Hold back flagged rows</h2>
        <p className="in-sub">
          Records the rows your latest table result flagged (at most 5% of that batch) as held back from training. It is a
          record only: Veritas does not change the data, stop uploads or retrain anything, and nothing reads this record yet.
          It needs a table result with flagged rows.
        </p>
        <LoadState resource={status} what="recorded actions" isEmpty={() => false}>
          <dl className="in-kv in-kv-wide">
            <dt>Rows recorded as held back</dt><dd className="in-num">{s?.total_quarantined ?? '—'}</dd>
            <dt>Actions recorded</dt><dd className="in-num">{s?.n_defense_actions ?? '—'}</dd>
            <dt>Last action</dt>
            <dd>{last ? <>{last.samples_affected ?? '—'} {ACTION_WORDS[last.action] || last.action}, {when(last.timestamp)}</> : 'none yet'}</dd>
          </dl>
        </LoadState>
        <div className="in-actions" style={{ marginTop: 16 }}>
          <ConfirmAction
            quiet={false}
            label="Record flagged rows as held back"
            confirmLabel="Record them"
            message="Record the flagged rows of your latest table result as held back? This adds a record that cannot be removed here. No data is changed."
            onConfirm={holdBack}
          />
        </div>
      </section>

      <section className="in-section" aria-labelledby="queue-title">
        <h2 id="queue-title" className="in-h2">Review queue</h2>
        <p className="in-sub">
          Table results between the 1-in-100 and 1-in-10,000 false-alarm levels wait here for a person. Only cases from your own
          analyses are listed. Holding back a case's rows adds the same record as above; closing it records that you looked
          and took no action.
        </p>
        <LoadState resource={cases} what="the review queue" isEmpty={(d) => !d?.length} empty="No cases are waiting for review.">
          <div className="in-table-wrap">
            <table className="in-table in-table-stack">
              <caption className="in-sr">Cases waiting for review, newest first</caption>
              <thead>
                <tr>
                  <th scope="col">Case</th><th scope="col">Opened</th><th scope="col" className="in-right">Rows in batch</th>
                  <th scope="col" className="in-right">Label evidence</th><th scope="col" className="in-right">Shift evidence</th>
                  <th scope="col"><span className="in-sr">Decision</span></th>
                </tr>
              </thead>
              <tbody>
                {(cases.data || []).map((c) => (
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
          <p className="in-hint">Evidence is −log₁₀ of the check's p-value: 2 is the 1-in-100 level, 4 the 1-in-10,000 level.</p>
        </LoadState>
      </section>
    </>
  );
}
