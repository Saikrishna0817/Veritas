import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { Lock } from 'lucide-react';
import SoftProtection3D from '../components/SoftProtection3D';

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
    <main className="min-h-screen flex items-center justify-center bg-bg p-6 relative overflow-hidden">
      {/* 3D Background Visual Element */}
      <div className="absolute inset-0 pointer-events-none flex items-center justify-center opacity-40">
        <SoftProtection3D height="550px" className="w-full max-w-2xl" interactive={false} />
      </div>

      <form
        onSubmit={submit}
        className="w-full max-w-md space-y-5 rounded-3xl border border-white/[0.08] bg-surface/90 backdrop-blur-2xl p-8 lg:p-10 shadow-[0_20px_60px_rgba(0,0,0,0.5)] relative z-10 animate-fadeInUp"
      >
        <div className="flex items-center gap-3.5 mb-2">
          <div className="w-11 h-11 rounded-2xl bg-accent/10 border border-accent/30 flex items-center justify-center shadow-[0_0_20px_rgba(61,127,255,0.3)]">
            <Lock className="w-5 h-5 text-accent" />
          </div>
          <div>
            <h1 className="font-semibold text-text1 text-base tracking-tight">SPECTRA / VERITAS</h1>
            <p className="eyebrow-label text-[10px] text-accentCyan">Analyst Access Authentication</p>
          </div>
        </div>

        <div className="space-y-4 pt-2">
          <div>
            <label className="eyebrow-label mb-1.5 block">Username</label>
            <input
              id="login-username"
              aria-label="Username"
              className="w-full p-3.5 bg-surface2/60 border border-white/[0.06] rounded-xl text-sm text-text1 placeholder-text3 focus:outline-none focus:border-accent focus:bg-surface2 focus:ring-1 focus:ring-accent/40 transition-all font-mono"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username (admin)"
              autoComplete="username"
              required
            />
          </div>

          <div>
            <label className="eyebrow-label mb-1.5 block">Password</label>
            <input
              id="login-password"
              aria-label="Password"
              className="w-full p-3.5 bg-surface2/60 border border-white/[0.06] rounded-xl text-sm text-text1 placeholder-text3 focus:outline-none focus:border-accent focus:bg-surface2 focus:ring-1 focus:ring-accent/40 transition-all font-mono"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password (admin)"
              type="password"
              autoComplete="current-password"
              required
            />
          </div>
        </div>

        {error && (
          <p className="text-danger text-xs font-mono bg-danger/10 border border-danger/30 rounded-xl p-3">
            ⚠️ {error}
          </p>
        )}

        <button
          className="btn-soft-primary w-full p-3.5 text-sm font-semibold rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed mt-2"
          type="submit"
          disabled={loading}
        >
          {loading ? 'Authenticating Analyst...' : 'Sign in to Platform'}
        </button>
      </form>
    </main>
  );
}
