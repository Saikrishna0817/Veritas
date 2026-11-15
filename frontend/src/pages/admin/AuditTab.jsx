import { Fragment, useState } from 'react';
import { LoadState, shortId, when } from './shared';

/** The audit log, newest first. GET /audit/events?limit=100 (everyone's events for an admin). */
export default function AuditTab({ events, userNames }) {
  const [open, setOpen] = useState(null);
  const list = events.data || [];
  return (
    <>
      <h2 className="in-h2">Audit log</h2>
      <p className="in-sub">What each account did: sign-ins, analyses, scans, account changes and recorded actions. Open an entry for everything stored with it.</p>
      <LoadState resource={events} what="the audit log" isEmpty={(d) => !d?.length} empty="No events recorded yet.">
        <div className="in-table-wrap">
          <table className="in-table in-table-stack">
            <caption className="in-sr">Audit events, newest first</caption>
            <thead><tr><th scope="col">When</th><th scope="col">Account</th><th scope="col">Action</th><th scope="col">On</th><th scope="col"><span className="in-sr">Details</span></th></tr></thead>
            <tbody>
              {list.map((e) => {
                const expanded = open === e.id;
                return (
                  <Fragment key={e.id}>
                    <tr>
                      <td data-label="When" className="in-tag">{when(e.created_at)}</td>
                      <td data-label="Account">{userNames[e.actor_id] || <span className="in-path" title={e.actor_id}>{shortId(e.actor_id)}</span>}</td>
                      <td data-label="Action" className="in-path">{e.action}</td>
                      <td data-label="On" className="in-tag">{e.resource_type || '—'}{e.resource_id ? <> · <span className="in-path" title={e.resource_id}>{shortId(e.resource_id)}</span></> : null}</td>
                      <td data-label="" className="in-cell-actions">
                        <button type="button" className="in-btn in-btn-quiet in-btn-sm" aria-expanded={expanded} aria-controls={`audit-${e.id}`} onClick={() => setOpen(expanded ? null : e.id)}>
                          {expanded ? 'Hide details' : 'Details'}
                        </button>
                      </td>
                    </tr>
                    {expanded && (
                      <tr id={`audit-${e.id}`} className="in-row-detail">
                        <td colSpan={5}><pre className="in-code">{JSON.stringify(e, null, 2)}</pre></td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
        {list.length >= 100 && <p className="in-hint">Showing the newest 100 events.</p>}
      </LoadState>
    </>
  );
}
