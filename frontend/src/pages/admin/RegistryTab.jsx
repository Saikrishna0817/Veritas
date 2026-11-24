import { api } from '../../services/api';
import { KINDS, KIND_OF_SOURCE, MODEL_VERDICTS, verdictWords, fmt } from '../../components/results';
import { ConfirmAction, LoadMore, LoadState, when } from './shared';

const typeOf = (r) => (r.source === 'model_scan' ? 'Model file' : KINDS[KIND_OF_SOURCE[r.source]]?.label || r.source || '—');
const verdictOf = (r) => (r.source === 'model_scan'
  ? (MODEL_VERDICTS[r.verdict] || [r.verdict ? String(r.verdict).replace(/_/g, ' ').toLowerCase() : 'no verdict'])[0]
  : verdictWords(r.verdict).title);

/** Every stored analysis and model scan, all accounts. GET /history?limit=100, DELETE /history/{id}. */
export default function RegistryTab({ records, userNames, notify, onChanged }) {
  const list = records.data || [];
  const remove = async (r) => {
    try {
      await api.deleteAnalysisHistory(r.id);
      notify({ text: `Deleted the stored result for “${r.filename || r.model_filename || r.id}”.` });
      onChanged();
    } catch (err) {
      notify({ error: true, text: `The result was not deleted: ${err.message || 'the server refused it.'}` });
    }
  };
  return (
    <>
      <h2 className="in-h2">Stored results</h2>
      <p className="in-sub">Every dataset check and model scan kept on the server, from all accounts. Deleting one removes it from that person's History and reports.</p>
      <LoadState resource={records} what="stored results" isEmpty={(d) => !d?.length} empty="No results are stored.">
        <div className="in-table-wrap">
          <table className="in-table in-table-stack">
            <caption className="in-sr">Stored results, newest first</caption>
            <thead>
              <tr>
                <th scope="col">File</th><th scope="col">Type</th><th scope="col">Verdict</th><th scope="col" className="in-right">Items</th>
                <th scope="col">Account</th><th scope="col">When</th><th scope="col"><span className="in-sr">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              {list.map((r) => {
                const name = r.filename || r.model_filename || r.id;
                return (
                  <tr key={`${r.source}-${r.id}`}>
                    <td data-label="File" className="in-path">{name}</td>
                    <td data-label="Type" className="in-tag">{typeOf(r)}</td>
                    <td data-label="Verdict"><span className="in-chip" data-v={r.verdict}>{verdictOf(r)}</span></td>
                    <td data-label="Items" className="in-num">{r.n_samples != null ? fmt(r.n_samples) : '—'}</td>
                    <td data-label="Account">{userNames[r.owner_id] || <span className="in-tag">{r.owner_id ? 'deleted or unknown' : '—'}</span>}</td>
                    <td data-label="When" className="in-tag">{when(r.created_at)}</td>
                    <td data-label="" className="in-cell-actions">
                      <ConfirmAction
                        label="Delete"
                        confirmLabel="Delete result"
                        message={`Delete the stored result for “${name}”? It disappears from its owner's History. This cannot be undone.`}
                        danger
                        onConfirm={() => remove(r)}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <LoadMore resource={records} what="results" />
      </LoadState>
    </>
  );
}
