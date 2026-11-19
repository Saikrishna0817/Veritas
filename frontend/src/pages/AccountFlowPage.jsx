import { useEffect, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { api } from '../services/api';
import AuthFrame from '../components/AuthFrame';
import Turnstile, { TURNSTILE_SITE_KEY } from '../components/Turnstile';

const COPY = {
  signup: { title: 'Create your account', intro: 'A username, your email, and a password of 10 characters or more.', cta: 'Create account' },
  verify: { title: 'Confirm your email', intro: 'Confirming the address from your link.', cta: '' },
  forgot: { title: 'Reset your password', intro: 'Enter the email on your account and we will send a link to choose a new password.', cta: 'Send reset link' },
  reset: { title: 'Choose a new password', intro: 'Use 10 characters or more.', cta: 'Change password' },
};

/** Public account pages: sign-up, email confirmation, forgot password, reset password. */
export default function AccountFlowPage({ mode }) {
  const [params] = useSearchParams();
  const token = params.get('token') || '';
  const [form, setForm] = useState({ username: '', email: '', password: '' });
  const [message, setMessage] = useState('');
  const [error, setError] = useState(mode === 'reset' && !token ? 'This link is missing its token. Ask for a new reset link.' : '');
  const [fields, setFields] = useState({});      // the server's per-field validation messages
  const [busy, setBusy] = useState(false);
  const [checking, setChecking] = useState(mode === 'verify' && !!token);
  const [resend, setResend] = useState({ email: '', message: '', error: '', busy: false });
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });
  // The human check on public forms (only when the build has a Turnstile site key). A token works once: after each
  // attempt the widget is reset and issues a new one.
  const [human, setHuman] = useState('');
  const [humanReset, setHumanReset] = useState(0);
  const needsHuman = Boolean(TURNSTILE_SITE_KEY) && (mode === 'signup' || mode === 'forgot');
  const [resendHuman, setResendHuman] = useState('');
  const [resendReset, setResendReset] = useState(0);

  // A link's token works once, so it is sent once per token, even when React runs this effect twice in development.
  const sent = useRef('');
  useEffect(() => {
    if (mode !== 'verify' || sent.current === token) return;
    sent.current = token;
    if (!token) { setError('This link is missing its token.'); return; }
    api.verifyEmail(token).then((r) => setMessage(r.message)).catch((e) => setError(e.message)).finally(() => setChecking(false));
  }, [mode, token]);

  async function submit(event) {
    event.preventDefault();
    setError(''); setMessage(''); setFields({}); setBusy(true);
    try {
      let r;
      if (mode === 'signup') r = await api.signup(form.username, form.email, form.password, human || undefined);
      else if (mode === 'forgot') r = await api.requestPasswordReset(form.email, human || undefined);
      else r = await api.confirmPasswordReset(token, form.password);
      setMessage(r.message);
    } catch (err) {
      const byField = err.fields || {};
      setFields(byField);
      // messages that belong to a field are shown beside it; anything else (a token, a taken name) goes here
      const shown = new Set(['username', 'email', 'password', 'new_password']);
      setError(Object.keys(byField).length && Object.keys(byField).every((k) => shown.has(k))
        ? 'Check the fields marked below.' : err.message || 'Something went wrong.');
    } finally {
      setBusy(false);
      if (needsHuman) setHumanReset((n) => n + 1);
    }
  }
  const fieldNote = (...names) => {
    const text = names.map((n) => fields[n]).find(Boolean);
    return text ? <small className="in-error in-field-error" id={`f-${names[0]}`}>{text}</small> : null;
  };
  const described = (name) => (fields[name] || (name === 'password' && fields.new_password) ? { 'aria-invalid': true, 'aria-describedby': `f-${name}` } : {});

  // the link was bad or expired: ask for a new one (the answer is the same whether or not the address has an account)
  async function resendLink(event) {
    event.preventDefault();
    setResend((r) => ({ ...r, message: '', error: '', busy: true }));
    try {
      const r = await api.resendVerification(resend.email, resendHuman || undefined);
      setResend((s) => ({ ...s, message: r.message, busy: false }));
    } catch (err) {
      setResend((s) => ({ ...s, error: err.message || 'Something went wrong.', busy: false }));
      if (TURNSTILE_SITE_KEY) setResendReset((n) => n + 1);
    }
  }

  const copy = COPY[mode];
  return (
    <AuthFrame title={copy.title} intro={mode === 'verify' && error ? 'That link did not work. Enter your email and we will send a new one.' : mode === 'verify' && message ? '' : copy.intro} busy={busy || checking || resend.busy}>
        <form className="in-auth-form" onSubmit={submit}>
          {mode === 'signup' && <label>Username<input className="in-input" placeholder="Username" value={form.username} onChange={set('username')} required minLength={3} maxLength={64} pattern="[A-Za-z0-9._\-]+" title="Letters, digits, '.', '_' and '-' only" autoComplete="username" {...described('username')} />{fieldNote('username')}</label>}
          {(mode === 'signup' || mode === 'forgot') && <label>Email<input className="in-input" type="email" placeholder="Email" value={form.email} onChange={set('email')} required autoComplete="email" {...described('email')} />{fieldNote('email')}</label>}
          {(mode === 'signup' || mode === 'reset') && (
            <label>Password<input className="in-input" type="password" placeholder="Password (10+ characters)" value={form.password} onChange={set('password')} required minLength={10} maxLength={72} autoComplete="new-password" {...described('password')} />{fieldNote('password', 'new_password')}</label>
          )}
          {needsHuman && !message && <Turnstile onToken={setHuman} onError={setError} reset={humanReset} />}
          {error && <p className="in-error" role="alert" style={{ margin: 0 }}>{error}</p>}
          {message && <p className="in-ok" role="status" style={{ margin: 0 }}>{message}</p>}
          {mode !== 'verify' && !message && !(mode === 'reset' && !token) && <button className="in-btn" type="submit" disabled={busy || (needsHuman && !human)}>{busy ? 'Please wait…' : copy.cta}</button>}
        </form>
        {mode === 'verify' && error && (
          <form className="in-auth-form" onSubmit={resendLink} aria-label="Send a new link">
            <label>Email<input className="in-input" type="email" placeholder="Email" value={resend.email} onChange={(e) => setResend({ ...resend, email: e.target.value })} required autoComplete="email" /></label>
            {!resend.message && <Turnstile onToken={setResendHuman} onError={(e) => setResend((s) => ({ ...s, error: e }))} reset={resendReset} />}
            {resend.error && <p className="in-error" role="alert" style={{ margin: 0 }}>{resend.error}</p>}
            {resend.message ? <p className="in-ok" role="status" style={{ margin: 0 }}>{resend.message}</p>
              : <button className="in-btn" type="submit" disabled={resend.busy || (Boolean(TURNSTILE_SITE_KEY) && !resendHuman)}>{resend.busy ? 'Please wait…' : 'Send a new link'}</button>}
          </form>
        )}
        <div className="in-auth-alt">
          <Link to="/login" className="in-link-text">Sign in</Link>
          {mode !== 'signup' ? <Link to="/signup" className="in-link-text">Create an account</Link> : <Link to="/forgot-password" className="in-link-text">Forgot your password?</Link>}
        </div>
    </AuthFrame>
  );
}
