import { useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import AuthLayout, { Field } from '../components/AuthLayout';
import { Button, ErrorNote } from '../components/ui';

export default function LoginPage() {
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [unverified, setUnverified] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (user) navigate(user.role === 'admin' ? '/admin' : '/', { replace: true });
  }, [user, navigate]);

  async function submit(event) {
    event.preventDefault();
    setError('');
    setUnverified(false);
    setLoading(true);
    try {
      const loggedIn = await login(username, password, 'user');
      navigate(loggedIn?.role === 'admin' ? '/admin' : '/', { replace: true });
    } catch (err) {
      setError(err.message || 'Unable to sign in.');
      // The backend answers 403 when the password is right but the email address is not confirmed yet.
      setUnverified(err.status === 403 && /verify your email/i.test(err.message || ''));
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthLayout
      title="Sign in"
      lede="Pick up where your last examination left off."
      footer={(
        <>
          <Link to="/signup" className="text-violet hover:text-[var(--red-bright)]">Create an account</Link>
          <Link to="/forgot-password" className="text-textSecondary hover:text-textPrimary">Forgot password?</Link>
        </>
      )}
    >
      <form onSubmit={submit} className="space-y-5">
        <Field label="Username" placeholder="Your username" value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" required autoFocus />
        <Field label="Password" type="password" placeholder="Your password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required />
        <ErrorNote>{error}</ErrorNote>
        {unverified && (
          <Link to="/resend-verification" className="block rounded-2xl bg-white/[0.06] px-4 py-3 text-sm hover:bg-white/[0.1]">
            Didn't get the email? <span className="text-uv">Send a new confirmation link →</span>
          </Link>
        )}
        <Button type="submit" variant="uv" loading={loading} className="w-full py-3.5 text-[15px]">
          Sign in <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
        </Button>
      </form>
      <p className="mt-6 text-xs text-textMuted">
        Administrators sign in at <Link to="/admin/login" className="underline hover:text-textPrimary">the admin console</Link>.
      </p>
    </AuthLayout>
  );
}
