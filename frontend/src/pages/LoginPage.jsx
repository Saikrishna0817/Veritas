import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { Shield } from 'lucide-react';

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(username, password);
      navigate('/', { replace: true });
    } catch (err) {
      setError(err.message || 'Unable to sign in');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen flex items-center justify-center bg-bg p-6">
      <form
        onSubmit={submit}
        className="w-full max-w-sm space-y-4 rounded-xl border border-border bg-bg2 p-8 shadow-accent"
      >
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-lg bg-accent/10 border border-accent/30 flex items-center justify-center">
            <Shield className="w-5 h-5 text-accent" />
          </div>
          <div>
            <h1 className="font-mono text-accent text-sm font-bold tracking-widest">AI TRUST FORENSICS</h1>
            <p className="font-mono text-text3 text-xs">Analyst Access</p>
          </div>
        </div>

        <div className="space-y-3">
          <input
            id="login-username"
            aria-label="Username"
            className="w-full p-3 bg-surface border border-border rounded-lg font-mono text-sm text-text1 placeholder-text3 focus:outline-none focus:border-accent transition-colors"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Username"
            autoComplete="username"
            required
          />
          <input
            id="login-password"
            aria-label="Password"
            className="w-full p-3 bg-surface border border-border rounded-lg font-mono text-sm text-text1 placeholder-text3 focus:outline-none focus:border-accent transition-colors"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            type="password"
            autoComplete="current-password"
            required
          />
        </div>

        {error && (
          <p className="text-danger text-xs font-mono bg-danger/5 border border-danger/20 rounded p-2">{error}</p>
        )}

        <button
          className="w-full p-3 bg-accent text-bg font-mono text-sm font-bold rounded-lg hover:bg-accent/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          type="submit"
          disabled={loading}
        >
          {loading ? 'Authenticating...' : 'Sign in'}
        </button>
      </form>
    </main>
  );
}
