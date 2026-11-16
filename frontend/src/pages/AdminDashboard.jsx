import { useCallback, useMemo, useState } from 'react';
import { api } from '../services/api';
import { useAuth } from '../hooks/useAuth';
import { Notice, TabPanel, Tabs, useResource } from './admin/shared';
import UsersTab from './admin/UsersTab';
import DefenseTab from './admin/DefenseTab';
import AuditTab from './admin/AuditTab';
import RegistryTab from './admin/RegistryTab';

// Stable fetchers, so each resource loads once on mount and again on reload().
const fetchUsers = () => api.getUsers();
const fetchStatus = () => api.getDefenseStatus();
const fetchCases = () => api.getPendingReviews().then((d) => d.cases || []);
const fetchEvents = () => api.getAuditEvents(100).then((d) => d.events || []);
const fetchRecords = () => api.getAnalysisHistory(100).then((d) => d.results || []);

/** Admin console: accounts, recorded defence actions and the review queue, the audit log, stored results. */
export default function AdminDashboard() {
  const { user: self } = useAuth();
  const [tab, setTab] = useState('users');
  const [notice, setNotice] = useState(null);

  const users = useResource(fetchUsers);
  const status = useResource(fetchStatus);
  const cases = useResource(fetchCases);
  const events = useResource(fetchEvents);
  const records = useResource(fetchRecords);

  const userNames = useMemo(() => Object.fromEntries((users.data || []).map((u) => [u.id, u.username])), [users.data]);
  const count = (r) => (r.data ? r.data.length : null);
  const tabs = [
    { id: 'users', label: 'Accounts', count: count(users) },
    { id: 'defense', label: 'Review queue', count: count(cases) },
    { id: 'audit', label: 'Audit log', count: count(events) },
    { id: 'registry', label: 'Stored results', count: count(records) },
  ];

  const { reload: reloadUsers } = users;
  const { reload: reloadStatus } = status;
  const { reload: reloadCases } = cases;
  const { reload: reloadEvents } = events;
  const { reload: reloadRecords } = records;
  const reloadAll = useCallback(() => {
    reloadUsers(); reloadStatus(); reloadCases(); reloadEvents(); reloadRecords();
  }, [reloadUsers, reloadStatus, reloadCases, reloadEvents, reloadRecords]);
  const anyLoading = [users, status, cases, events, records].some((r) => r.loading);

  return (
    <main className="in-main"><div className="in-page">
      <div className="in-tab-head">
        <div>
          <h1 className="in-h1">Admin console</h1>
          <p className="in-lede" style={{ marginBottom: 0 }}>Accounts, the review queue and recorded actions, the audit log, and every stored result. Changes here apply to all accounts.</p>
        </div>
        <button type="button" className="in-btn in-btn-quiet" onClick={reloadAll} disabled={anyLoading}>{anyLoading ? 'Loading…' : 'Reload all'}</button>
      </div>

      <Notice notice={notice} onClose={() => setNotice(null)} />

      <Tabs label="Admin sections" tabs={tabs} value={tab} onChange={setTab} />
      <TabPanel id={tab}>
        {tab === 'users' && <UsersTab users={users} self={self} notify={setNotice} onChanged={() => { reloadUsers(); reloadEvents(); }} />}
        {tab === 'defense' && <DefenseTab status={status} cases={cases} self={self} notify={setNotice} onChanged={() => { reloadStatus(); reloadCases(); reloadEvents(); }} />}
        {tab === 'audit' && <AuditTab events={events} userNames={userNames} />}
        {tab === 'registry' && <RegistryTab records={records} userNames={userNames} notify={setNotice} onChanged={() => { reloadRecords(); reloadEvents(); }} />}
      </TabPanel>
    </div></main>
  );
}
