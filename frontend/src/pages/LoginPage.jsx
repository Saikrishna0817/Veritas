import { useEffect, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import AuthFrame from '../components/AuthFrame';

// After signing in: the page the person was sent here from (?next=, only an address inside the app), else their home.
function destination(next, role) {
  const admin = role === 'admin';
  const safe = typeof next === 'string' && next.startsWith('/') && !next.startsWith('//') && !next.includes('\\');
  if (!safe || /^\/(admin\/)?login(\/|\?|$)/.test(next)) return admin ? '/admin' : '/';
  if (!admin && next.startsWith('/admin')) return '/';
  return next;
}

/** Sign-in for users and, with ``admin``, for the admin console (same form, different role check). */
export default function LoginPage({ admin = false }) {
  const { login, user } = useAuth();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const next = params.get('next');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {                                   // already signed in: go straight to the right home
    if (user) navigate(destination(next, user.role), { replace: true });
  }, [user, navigate, next]);

  const submit = async (e) => {
    e.preventDefault();
    setError(''); setBusy(true);
    try {
      const who = await login(username, password, admin ? 'admin' : 'user');
      if (admin && who?.role !== 'admin') setError('This account is not an administrator.');
      else navigate(destination(next, who?.role), { replace: true });
    } catch (err) {
      setError(err.message || 'Sign-in failed.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthFrame
      title={admin ? 'Administrator sign-in' : 'Sign in'}
      intro={admin ? 'For managing accounts and the service.' : 'Check training data for poisoning before you train on it.'}
      busy={busy}
    >
        <form className="in-auth-form" onSubmit={submit}>
          <label>Username<input className="in-input" placeholder="Your username" value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" required /></label>
          <label>Password<input className="in-input" type="password" placeholder="Your password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required /></label>
          {error && <p className="in-error" role="alert" style={{ margin: 0 }}>{error}</p>}
          <button className="in-btn" type="submit" disabled={busy}>{busy ? 'Signing in…' : 'Sign in'}</button>
        </form>
        <div className="in-auth-alt">
          {admin ? <Link to="/login" className="in-link-text">User sign-in</Link> : (
            <>
              <span>No account? <Link to="/signup" className="in-link-text">Create one</Link></span>
              <Link to="/forgot-password" className="in-link-text">Forgot your password?</Link>
            </>
          )}
        </div>
    </AuthFrame>
  );
}
