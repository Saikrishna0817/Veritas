import { useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { ShieldCheck } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import AuthLayout, { Field } from '../components/AuthLayout';
import { Button, ErrorNote } from '../components/ui';

export default function AdminLoginPage() {
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (user) navigate(user.role === 'admin' ? '/admin' : '/', { replace: true });
  }, [user, navigate]);

  async function submit(event) {
    event.preventDefault();
    setError('');
    setLoading(true);
    try {
      const loggedIn = await login(username, password, 'admin');
      if (loggedIn?.role === 'admin') navigate('/admin', { replace: true });
      else setError('This account does not have administrator access.');
    } catch (err) {
      setError(err.message || 'Unable to sign in as administrator.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthLayout
      tone="admin"
      title="Admin console"
      lede="Manage analyst accounts. Analysis pages are for analyst accounts."
      footer={<Link to="/login" className="text-textSecondary hover:text-textPrimary">← Analyst sign-in</Link>}
    >
      <form onSubmit={submit} className="space-y-5">
        <Field id="admin-username" label="Admin username" placeholder="Enter admin username" value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" required autoFocus />
        <Field id="admin-password" label="Password" type="password" placeholder="Enter admin password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required />
        <ErrorNote>{error}</ErrorNote>
        <Button type="submit" variant="primary" loading={loading} className="w-full py-3.5 text-[15px]">
          <ShieldCheck className="h-4 w-4" /> Sign in as administrator
        </Button>
      </form>
    </AuthLayout>
  );
}
