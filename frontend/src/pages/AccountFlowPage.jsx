import { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { api } from '../services/api';
import Brand from '../components/Brand';

const COPY = {
  signup: { title: 'Create your account', cta: 'Create account' },
  verify: { title: 'Confirm your email', cta: '' },
  forgot: { title: 'Reset your password', cta: 'Send reset link' },
  reset: { title: 'Choose a new password', cta: 'Change password' },
};

/** Public account pages: sign-up, email confirmation, forgot password, reset password. */
export default function AccountFlowPage({ mode }) {
  const [params] = useSearchParams();
  const token = params.get('token') || '';
  const [form, setForm] = useState({ username: '', email: '', password: '' });
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  useEffect(() => {
    if (mode !== 'verify') return;
    if (!token) { setError('This link is missing its token.'); return; }
    api.verifyEmail(token).then((r) => setMessage(r.message)).catch((e) => setError(e.message));
  }, [mode, token]);

  async function submit(event) {
    event.preventDefault();
    setError(''); setMessage(''); setBusy(true);
    try {
      let r;
      if (mode === 'signup') r = await api.signup(form.username, form.email, form.password);
      else if (mode === 'forgot') r = await api.requestPasswordReset(form.email);
      else r = await api.confirmPasswordReset(token, form.password);
      setMessage(r.message);
    } catch (err) {
      setError(err.message || 'Something went wrong.');
    } finally {
      setBusy(false);
    }
  }

  const copy = COPY[mode];
  return (
    <main className="instrument in-auth">
      <div className="in-auth-box">
        <Brand />
        <h1 className="in-h2">{copy.title}</h1>
        <form className="in-auth-form" onSubmit={submit}>
          {mode === 'signup' && <label>Username<input className="in-input" placeholder="Username" value={form.username} onChange={set('username')} required minLength={3} maxLength={64} pattern="[A-Za-z0-9._\-]+" title="Letters, digits, '.', '_' and '-' only" autoComplete="username" /></label>}
          {(mode === 'signup' || mode === 'forgot') && <label>Email<input className="in-input" type="email" placeholder="Email" value={form.email} onChange={set('email')} required autoComplete="email" /></label>}
          {(mode === 'signup' || mode === 'reset') && (
            <label>Password<input className="in-input" type="password" placeholder="Password (10+ characters)" value={form.password} onChange={set('password')} required minLength={10} maxLength={72} autoComplete="new-password" /></label>
          )}
          {error && <p className="in-error" role="alert" style={{ margin: 0 }}>{error}</p>}
          {message && <p className="in-ok" role="status" style={{ margin: 0 }}>{message}</p>}
          {mode !== 'verify' && !message && <button className="in-btn" type="submit" disabled={busy}>{busy ? 'Please wait…' : copy.cta}</button>}
        </form>
        <div className="in-auth-alt">
          <Link to="/login" className="in-link-text">Sign in</Link>
          {mode !== 'signup' ? <Link to="/signup" className="in-link-text">Create an account</Link> : <Link to="/forgot-password" className="in-link-text">Forgot your password?</Link>}
        </div>
      </div>
    </main>
  );
}
