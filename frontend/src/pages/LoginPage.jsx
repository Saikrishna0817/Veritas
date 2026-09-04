import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { Lock, AlertTriangle } from 'lucide-react';
import Tactile3DHero from '../components/Tactile3DHero';

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
    <main className="min-h-screen flex items-center justify-center p-6 relative overflow-hidden"
      style={{ backgroundColor: 'var(--bg-void)' }}>

      {/* Full-screen 3D animated background */}
      <div className="fixed inset-0 z-0 pointer-events-none" style={{ opacity: 0.35 }}>
        <Tactile3DHero intensity={0.6} />
      </div>

      {/* Gradient vignette overlay */}
      <div className="fixed inset-0 z-[1] pointer-events-none"
        style={{
          background: `
            radial-gradient(ellipse at center, transparent 30%, rgba(0,0,0,0.7) 100%),
            linear-gradient(180deg, rgba(0,0,0,0.3) 0%, transparent 40%, transparent 60%, rgba(0,0,0,0.5) 100%)
          `
        }}
      />

      {/* Login card */}
      <form
        onSubmit={submit}
        className="w-full max-w-md space-y-6 rounded-[24px] relative z-10"
        style={{
          border: '1px solid rgba(255,255,255,0.08)',
          background: 'rgba(19,19,22,0.75)',
          backdropFilter: 'blur(40px)',
          WebkitBackdropFilter: 'blur(40px)',
          padding: '40px',
          boxShadow: `
            0 0 0 1px rgba(255,255,255,0.04),
            0 20px 60px rgba(0,0,0,0.6),
            0 0 120px rgba(228,36,43,0.06)
          `,
          animation: 'loginFadeIn 0.8s ease-out',
        }}
      >
        {/* Brand header */}
        <div className="flex items-center gap-4 mb-8">
          <div style={{
            width: 48, height: 48,
            borderRadius: 16,
            background: 'rgba(228,36,43,0.1)',
            border: '1px solid rgba(228,36,43,0.3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Lock style={{ width: 24, height: 24, color: 'var(--red-primary)' }} />
          </div>
          <div>
            <h1 style={{
              fontFamily: "'Space Grotesk', sans-serif",
              fontWeight: 900,
              color: 'var(--text-primary)',
              fontSize: 24,
              letterSpacing: '-0.02em',
              lineHeight: 1,
            }}>
              SPECTRA / VERITAS
            </h1>
            <p style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 10,
              fontWeight: 700,
              color: 'var(--red-primary)',
              textTransform: 'uppercase',
              letterSpacing: '0.15em',
              marginTop: 8,
            }}>
              Analyst Access Authentication
            </p>
          </div>
        </div>

        {/* Form fields */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div>
            <label style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 10,
              fontWeight: 700,
              color: 'var(--text-secondary)',
              textTransform: 'uppercase',
              letterSpacing: '0.15em',
              display: 'block',
              marginBottom: 8,
            }}>
              Username
            </label>
            <input
              id="login-username"
              aria-label="Username"
              style={{
                width: '100%',
                padding: 16,
                background: 'var(--bg-void)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 12,
                fontSize: 14,
                color: 'var(--text-primary)',
                fontFamily: "'JetBrains Mono', monospace",
                fontWeight: 500,
                outline: 'none',
                transition: 'border-color 0.2s, box-shadow 0.2s',
              }}
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username (admin)"
              autoComplete="username"
              required
              onFocus={(e) => {
                e.target.style.borderColor = 'var(--red-primary)';
                e.target.style.boxShadow = '0 0 0 3px rgba(228,36,43,0.15)';
              }}
              onBlur={(e) => {
                e.target.style.borderColor = 'rgba(255,255,255,0.08)';
                e.target.style.boxShadow = 'none';
              }}
            />
          </div>

          <div>
            <label style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 10,
              fontWeight: 700,
              color: 'var(--text-secondary)',
              textTransform: 'uppercase',
              letterSpacing: '0.15em',
              display: 'block',
              marginBottom: 8,
            }}>
              Password
            </label>
            <input
              id="login-password"
              aria-label="Password"
              style={{
                width: '100%',
                padding: 16,
                background: 'var(--bg-void)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 12,
                fontSize: 14,
                color: 'var(--text-primary)',
                fontFamily: "'JetBrains Mono', monospace",
                fontWeight: 500,
                outline: 'none',
                transition: 'border-color 0.2s, box-shadow 0.2s',
              }}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password (admin)"
              type="password"
              autoComplete="current-password"
              required
              onFocus={(e) => {
                e.target.style.borderColor = 'var(--red-primary)';
                e.target.style.boxShadow = '0 0 0 3px rgba(228,36,43,0.15)';
              }}
              onBlur={(e) => {
                e.target.style.borderColor = 'rgba(255,255,255,0.08)';
                e.target.style.boxShadow = 'none';
              }}
            />
          </div>
        </div>

        {/* Error */}
        {error && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 12,
            background: 'rgba(242,184,75,0.08)',
            border: '1px solid rgba(242,184,75,0.25)',
            borderRadius: 12, padding: 16,
          }}>
            <AlertTriangle style={{ width: 16, height: 16, color: 'var(--status-warn)', flexShrink: 0 }} />
            <p style={{ color: 'var(--status-warn)', fontSize: 12, fontFamily: "'JetBrains Mono', monospace", fontWeight: 700 }}>
              {error}
            </p>
          </div>
        )}

        {/* Submit button */}
        <button
          type="submit"
          disabled={loading}
          style={{
            width: '100%',
            padding: 16,
            marginTop: 8,
            fontSize: 13,
            fontFamily: "'JetBrains Mono', monospace",
            fontWeight: 700,
            color: '#ffffff',
            background: loading ? 'rgba(228,36,43,0.6)' : 'var(--red-primary)',
            border: '1px solid rgba(228,36,43,0.5)',
            borderRadius: 12,
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.7 : 1,
            textTransform: 'uppercase',
            letterSpacing: '0.15em',
            transition: 'all 0.2s',
            boxShadow: '0 8px 32px rgba(228,36,43,0.25)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
          }}
          onMouseEnter={(e) => {
            if (!loading) {
              e.target.style.background = 'var(--red-bright)';
              e.target.style.boxShadow = '0 12px 40px rgba(228,36,43,0.4)';
            }
          }}
          onMouseLeave={(e) => {
            if (!loading) {
              e.target.style.background = 'var(--red-primary)';
              e.target.style.boxShadow = '0 8px 32px rgba(228,36,43,0.25)';
            }
          }}
        >
          {loading ? 'Authenticating Analyst...' : 'Sign in to Platform'}
        </button>

        {/* Bottom decorative line */}
        <div style={{
          height: 2,
          borderRadius: 1,
          background: 'linear-gradient(90deg, transparent, var(--red-primary), transparent)',
          opacity: 0.3,
          marginTop: 8,
        }} />
      </form>

      <style>{`
        @keyframes loginFadeIn {
          0% { opacity: 0; transform: translateY(30px) scale(0.97); }
          100% { opacity: 1; transform: translateY(0) scale(1); }
        }
        input::placeholder {
          color: var(--text-muted) !important;
        }
      `}</style>
    </main>
  );
}
