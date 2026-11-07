import { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { api } from '../services/api';

const COPY = {
  signup: { title: 'Create your account', cta: 'Create account' },
  verify: { title: 'Confirm your email', cta: '' },
  forgot: { title: 'Reset your password', cta: 'Send reset link' },
  reset: { title: 'Choose a new password', cta: 'Change password' },
};

const input = 'w-full rounded-xl p-3 bg-black/40 border border-white/10 text-white';

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
    <main className="min-h-screen flex items-center justify-center p-6" style={{ backgroundColor: 'var(--bg-void)' }}>
      <form onSubmit={submit} className="w-full max-w-md space-y-4 rounded-[24px] p-10" style={{ border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(19,19,22,0.85)' }}>
        <h1 className="text-2xl font-bold text-white m-0">{copy.title}</h1>
        {mode === 'signup' && <input className={input} placeholder="Username" value={form.username} onChange={set('username')} required minLength={3} autoComplete="username" />}
        {(mode === 'signup' || mode === 'forgot') && <input className={input} type="email" placeholder="Email" value={form.email} onChange={set('email')} required autoComplete="email" />}
        {(mode === 'signup' || mode === 'reset') && (
          <input className={input} type="password" placeholder="Password (10+ characters)" value={form.password} onChange={set('password')} required minLength={10} maxLength={72} autoComplete="new-password" />
        )}
        {error && <div className="text-red-400 text-sm font-bold" role="alert">{error}</div>}
        {message && <div className="text-green-400 text-sm font-bold" role="status">{message}</div>}
        {mode !== 'verify' && !message && (
          <button type="submit" disabled={busy} className="w-full py-3 rounded-xl bg-redPrimary text-white font-bold disabled:opacity-50">{busy ? 'Please wait…' : copy.cta}</button>
        )}
        <div className="text-sm text-gray-400 flex justify-between">
          <Link to="/login" className="text-gray-300">Sign in</Link>
          {mode !== 'signup' ? <Link to="/signup" className="text-gray-300">Create account</Link> : <Link to="/forgot-password" className="text-gray-300">Forgot password?</Link>}
        </div>
      </form>
    </main>
  );
}
