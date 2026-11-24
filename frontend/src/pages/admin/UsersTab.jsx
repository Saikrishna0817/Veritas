import { useId, useState } from 'react';
import { api } from '../../services/api';
import { ConfirmAction, LoadMore, LoadState, when } from './shared';

const ROLE_WORDS = { admin: 'Administrator', analyst: 'Analyst', user: 'User' };

function CreateUserForm({ onCreated, onCancel, notify }) {
  const id = useId();
  const [form, setForm] = useState({ username: '', password: '', role: 'user' });
  const [busy, setBusy] = useState(false);
  const [fields, setFields] = useState({});       // the server's per-field validation messages
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });
  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setFields({});
    try {
      await api.createUser(form);
      notify({ text: `Account “${form.username}” created as ${ROLE_WORDS[form.role].toLowerCase()}.` });
      onCreated();
    } catch (err) {
      setFields(err.fields || {});
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
      <input id={`${id}-u`} className="in-input" value={form.username} onChange={set('username')} required minLength={3} maxLength={64} autoComplete="off" aria-describedby={`${id}-u-hint${fields.username ? ` ${id}-u-err` : ''}`} aria-invalid={fields.username ? true : undefined} />
      <small id={`${id}-u-hint`} className="in-hint">3 to 64 characters.</small>
      {fields.username && <small id={`${id}-u-err`} className="in-error in-field-error">{fields.username}</small>}
      <label htmlFor={`${id}-p`}>Password</label>
      <input id={`${id}-p`} className="in-input" type="password" value={form.password} onChange={set('password')} required minLength={10} maxLength={128} autoComplete="new-password" aria-describedby={`${id}-p-hint${fields.password ? ` ${id}-p-err` : ''}`} aria-invalid={fields.password ? true : undefined} />
      <small id={`${id}-p-hint`} className="in-hint">At least 10 characters.</small>
      {fields.password && <small id={`${id}-p-err`} className="in-error in-field-error">{fields.password}</small>}
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

/** One account's role: pick another, then save. The server refuses a change to your own role, so the select is off there. */
function RoleCell({ u, isSelf, notify, onChanged }) {
  const id = useId();
  const [pick, setPick] = useState(u.role);
  const [busy, setBusy] = useState(false);
  const save = async () => {
    setBusy(true);
    try {
      await api.setUserRole(u.id, pick);
      notify({ text: `“${u.username}” is now ${(ROLE_WORDS[pick] || pick).toLowerCase()}. It applies from their next request.` });
      onChanged();
    } catch (err) {
      setPick(u.role);
      notify({ error: true, text: `The role of “${u.username}” was not changed: ${err.message || 'the server refused it.'}` });
    } finally {
      setBusy(false);
    }
  };
  if (isSelf) return ROLE_WORDS[u.role] || u.role;
  return (
    <div className="in-actions in-role-cell">
      <label htmlFor={id} className="in-sr">Role of {u.username}</label>
      <select id={id} className="in-input in-input-sm" value={pick} onChange={(e) => setPick(e.target.value)} disabled={busy}>
        {Object.entries(ROLE_WORDS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
      </select>
      {pick !== u.role && (
        <>
          <button type="button" className="in-btn in-btn-sm" onClick={save} disabled={busy}>{busy ? 'Saving…' : 'Save role'}</button>
          <button type="button" className="in-btn in-btn-quiet in-btn-sm" onClick={() => setPick(u.role)} disabled={busy}>Cancel</button>
        </>
      )}
    </div>
  );
}

/** Accounts: list, create, change role, delete. GET /users, POST /users, PATCH /users/{id}, DELETE /users/{id}. */
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
  const setDisabled = async (u, disable) => {
    try {
      await (disable ? api.disableUser(u.id) : api.enableUser(u.id));
      notify({ text: disable ? `“${u.username}” is disabled. Their sessions and API keys stopped working; their data is kept.` : `“${u.username}” can sign in again.` });
      onChanged();
    } catch (err) {
      notify({ error: true, text: `“${u.username}” was not ${disable ? 'disabled' : 'enabled'}: ${err.message || 'the server refused it.'}` });
    }
  };
  return (
    <>
      <div className="in-tab-head">
        <div>
          <h2 className="in-h2">Accounts</h2>
          <p className="in-sub">Everyone who can sign in. A role change applies from the person's next request. Disabling an account blocks sign-in and keeps its data; you can enable it again. Deleting an account removes its sign-in; their stored results stay until you delete them under Stored results.</p>
        </div>
        {!creating && <button type="button" className="in-btn" onClick={() => setCreating(true)}>Create an account</button>}
      </div>
      {creating && <CreateUserForm notify={notify} onCancel={() => setCreating(false)} onCreated={() => { setCreating(false); onChanged(); }} />}
      <LoadState resource={users} what="accounts" isEmpty={(d) => !d?.length} empty="No accounts yet.">
        <div className="in-table-wrap">
          <table className="in-table in-table-stack">
            <caption className="in-sr">Accounts, newest first</caption>
            <thead><tr><th scope="col">Username</th><th scope="col">Role</th><th scope="col">Status</th><th scope="col">Created</th><th scope="col">ID</th><th scope="col"><span className="in-sr">Actions</span></th></tr></thead>
            <tbody>
              {(users.data || []).map((u) => {
                const isSelf = u.id === self?.id;
                return (
                  <tr key={u.id}>
                    <td data-label="Username"><b>{u.username}</b>{isSelf && <span className="in-tag"> · you</span>}</td>
                    <td data-label="Role"><RoleCell key={u.role} u={u} isSelf={isSelf} notify={notify} onChanged={onChanged} /></td>
                    <td data-label="Status" className="in-tag">{u.disabled_at ? `Disabled ${when(u.disabled_at)}` : 'Active'}</td>
                    <td data-label="Created" className="in-tag">{when(u.created_at)}</td>
                    <td data-label="ID" className="in-path in-tag">{u.id}</td>
                    <td data-label="" className="in-cell-actions">
                      {!isSelf && (u.disabled_at ? (
                        <ConfirmAction
                          label="Enable"
                          confirmLabel={`Enable ${u.username}`}
                          message={`Let “${u.username}” sign in again? Their old sessions and API keys stay ended.`}
                          onConfirm={() => setDisabled(u, false)}
                        />
                      ) : (
                        <ConfirmAction
                          label="Disable"
                          confirmLabel={`Disable ${u.username}`}
                          message={`Disable “${u.username}”? They are signed out everywhere and their API keys stop working. Their data is kept.`}
                          onConfirm={() => setDisabled(u, true)}
                        />
                      ))}
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
        <LoadMore resource={users} what="accounts" />
      </LoadState>
    </>
  );
}
