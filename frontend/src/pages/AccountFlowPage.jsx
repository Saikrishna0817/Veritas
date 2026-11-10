import { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { api } from '../services/api';
import AuthLayout, { Field } from '../components/AuthLayout';
import { Button, ErrorNote } from '../components/ui';
import HumanCheck from '../components/HumanCheck';
import { TURNSTILE_SITE_KEY } from '../services/config';

const COPY = {
  signup: { title: 'Create an account', lede: 'We email you a link to confirm the address.', cta: 'Create account' },
  verify: { title: 'Confirm your email', lede: 'Checking your link…', cta: '' },
  forgot: { title: 'Reset your password', lede: 'If the address has an account, a reset link is on its way.', cta: 'Send reset link' },
  reset: { title: 'Choose a new password', lede: 'At least 10 characters.', cta: 'Change password' },
  resend: { title: 'Confirm your email', lede: 'Lost the confirmation email or it expired? We send a new link, valid for 24 hours.', cta: 'Send a new link' },
};

/** Public account pages: sign-up, email confirmation, forgot password, reset password. */
export default function AccountFlowPage({ mode }) {
  const [params] = useSearchParams();
  const token = params.get('token') || '';
  const [form, setForm] = useState({ username: '', email: '', password: '' });
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [human, setHuman] = useState('');
  const [checkRound, setCheckRound] = useState(0);
  const needsCheck = Boolean(TURNSTILE_SITE_KEY) && ['signup', 'forgot', 'resend'].includes(mode);
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  useEffect(() => {
    if (mode !== 'verify') return;
    if (!token) { setError('This link is missing its token. Open the link from the email again.'); return; }
    api.verifyEmail(token).then((r) => setMessage(r.message)).catch((e) => setError(e.message));
  }, [mode, token]);

  async function submit(event) {
    event.preventDefault();
    setError(''); setMessage(''); setBusy(true);
    try {
      let r;
      if (mode === 'signup') r = await api.signup(form.username, form.email, form.password, human);
      else if (mode === 'forgot') r = await api.requestPasswordReset(form.email, human);
      else if (mode === 'resend') r = await api.resendVerification(form.email, human);
      else r = await api.confirmPasswordReset(token, form.password);
      setMessage(r.message);
    } catch (err) {
      setError(err.message || 'Something went wrong. Try again in a moment.');
    } finally {
      setBusy(false);
      if (needsCheck) { setHuman(''); setCheckRound((n) => n + 1); }   // a check token works once
    }
  }

  const copy = COPY[mode];
  return (
    <AuthLayout
      title={copy.title}
      lede={copy.lede}
      footer={(
        <>
          <Link to="/login" className="text-violet hover:text-[var(--red-bright)]">Sign in</Link>
          {mode !== 'signup'
            ? <Link to="/signup" className="text-textSecondary hover:text-textPrimary">Create an account</Link>
            : <Link to="/forgot-password" className="text-textSecondary hover:text-textPrimary">Forgot password?</Link>}
        </>
      )}
    >
      <form onSubmit={submit} className="space-y-5">
        {mode === 'signup' && <Field label="Username" placeholder="Username" value={form.username} onChange={set('username')} required minLength={3} autoComplete="username" />}
        {(mode === 'signup' || mode === 'forgot' || mode === 'resend') && <Field label="Email" type="email" placeholder="Email" value={form.email} onChange={set('email')} required autoComplete="email" />}
        {(mode === 'signup' || mode === 'reset') && (
          <Field label="Password" type="password" placeholder="Password (10+ characters)" value={form.password} onChange={set('password')} required minLength={10} maxLength={72} autoComplete="new-password" />
        )}
        {needsCheck && !message && <HumanCheck onToken={setHuman} onError={setError} resetKey={checkRound} />}
        <ErrorNote>{error}</ErrorNote>
        {message && <div role="status" className="rounded-2xl border border-statusSafe/40 bg-statusSafe/[0.07] px-4 py-3 text-sm text-statusSafe">{message}</div>}
        {mode !== 'verify' && !message && (
          <Button type="submit" variant="uv" loading={busy} disabled={needsCheck && !human} className="w-full py-3.5 text-[15px]">{copy.cta}</Button>
        )}
      </form>
    </AuthLayout>
  );
}
