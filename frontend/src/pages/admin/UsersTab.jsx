import { useId, useState } from 'react';
import { api } from '../../services/api';
import { ConfirmAction, LoadState, when } from './shared';

const ROLE_WORDS = { admin: 'Administrator', analyst: 'Analyst', user: 'User' };

function CreateUserForm({ onCreated, onCancel, notify }) {
  const id = useId();
  const [form, setForm] = useState({ username: '', password: '', role: 'user' });
  const [busy, setBusy] = useState(false);
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });
  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    try {
      await api.createUser(form);
      notify({ text: `Account “${form.username}” created as ${ROLE_WORDS[form.role].toLowerCase()}.` });
      onCreated();
    } catch (err) {
      notify({ error: true, text: `The account was not created: ${err.message || 'the server refused it.'}` });
    } finally {
      setBusy(false);
    }
  };
  return (
    <form className="in-panel in-admin-form" onSubmit={submit} aria-labelledby={`${id}-title`}>
      <h3 id={`${id}-title`} className="in-h2">Create an account</h3>
      <p className="in-hint">The person signs in with this username and password. Accounts created here have no email address, so give them the password yourself; they cannot reset it by email.</p>
      <label htmlFor={`${id}-u`}>Username</label>
      <input id={`${id}-u`} className="in-input" value={form.username} onChange={set('username')} required minLength={3} maxLength={64} autoComplete="off" aria-describedby={`${id}-u-hint`} />
      <small id={`${id}-u-hint`} className="in-hint">3 to 64 characters.</small>
      <label htmlFor={`${id}-p`}>Password</label>
      <input id={`${id}-p`} className="in-input" type="password" value={form.password} onChange={set('password')} required minLength={10} maxLength={128} autoComplete="new-password" aria-describedby={`${id}-p-hint`} />
      <small id={`${id}-p-hint`} className="in-hint">At least 10 characters.</small>
      <label htmlFor={`${id}-r`}>Role</label>
      <select id={`${id}-r`} className="in-input" value={form.role} onChange={set('role')}>
        <option value="user">User: checks data and scans model files</option>
        <option value="admin">Administrator: also manages accounts and stored results</option>
      </select>
      <div className="in-actions">
        <button type="submit" className="in-btn" disabled={busy}>{busy ? 'Creating…' : 'Create account'}</button>
        <button type="button" className="in-btn in-btn-quiet" onClick={onCancel} disabled={busy}>Cancel</button>
      </div>
    </form>
  );
}

/** Accounts: list, create, delete. GET /users, POST /users, DELETE /users/{id}. */
export default function UsersTab({ users, self, notify, onChanged }) {
  const [creating, setCreating] = useState(false);
  const remove = async (u) => {
    try {
      await api.deleteUser(u.id);
      notify({ text: `Account “${u.username}” deleted.` });
      onChanged();
    } catch (err) {
      notify({ error: true, text: `“${u.username}” was not deleted: ${err.message || 'the server refused it.'}` });
    }
  };
  return (
    <>
      <div className="in-tab-head">
        <div>
          <h2 className="in-h2">Accounts</h2>
          <p className="in-sub">Everyone who can sign in. Deleting an account removes its sign-in; their stored results stay until you delete them under Stored results.</p>
        </div>
        {!creating && <button type="button" className="in-btn" onClick={() => setCreating(true)}>Create an account</button>}
      </div>
      {creating && <CreateUserForm notify={notify} onCancel={() => setCreating(false)} onCreated={() => { setCreating(false); onChanged(); }} />}
      <LoadState resource={users} what="accounts" isEmpty={(d) => !d?.length} empty="No accounts yet.">
        <div className="in-table-wrap">
          <table className="in-table in-table-stack">
            <caption className="in-sr">Accounts, newest first</caption>
            <thead><tr><th scope="col">Username</th><th scope="col">Role</th><th scope="col">Created</th><th scope="col">ID</th><th scope="col"><span className="in-sr">Actions</span></th></tr></thead>
            <tbody>
              {(users.data || []).map((u) => {
                const isSelf = u.id === self?.id;
                return (
                  <tr key={u.id}>
                    <td data-label="Username"><b>{u.username}</b>{isSelf && <span className="in-tag"> · you</span>}</td>
                    <td data-label="Role">{ROLE_WORDS[u.role] || u.role}</td>
                    <td data-label="Created" className="in-tag">{when(u.created_at)}</td>
                    <td data-label="ID" className="in-path in-tag">{u.id}</td>
                    <td data-label="" className="in-cell-actions">
                      <ConfirmAction
                        label="Delete"
                        confirmLabel={`Delete ${u.username}`}
                        message={`Delete the account “${u.username}”? They can no longer sign in. This cannot be undone.`}
                        danger
                        disabled={isSelf}
                        disabledReason="You cannot delete the account you are signed in with."
                        onConfirm={() => remove(u)}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {users.data?.length >= 200 && <p className="in-hint">Showing the newest 200 accounts, the most the server returns.</p>}
      </LoadState>
    </>
  );
}
