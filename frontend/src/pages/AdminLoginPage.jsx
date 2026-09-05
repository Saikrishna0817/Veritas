import { useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { ShieldAlert, AlertTriangle, ShieldCheck, ArrowLeft } from 'lucide-react';
import Tactile3DHero from '../components/Tactile3DHero';

export default function AdminLoginPage() {
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (user) {
      if (user.role === 'admin') {
        navigate('/admin', { replace: true });
      } else {
        navigate('/', { replace: true });
      }
    }
  }, [user, navigate]);

  async function submit(event) {
    event.preventDefault();
    setError('');
    setLoading(true);
    try {
      const loggedInUser = await login(username, password, 'admin');
      if (loggedInUser?.role === 'admin') {
        navigate('/admin', { replace: true });
      } else {
        setError('Access denied: Account does not have administrator privileges.');
      }
    } catch (err) {
      setError(err.message || 'Unable to sign in as Administrator');
    } finally {
      setLoading(false);
    }
  }

  const fillAdminCredentials = () => {
    setUsername('admin');
    setPassword('admin');
    setError('');
  };

  return (
    <main
      className="min-h-screen flex items-center justify-center p-6 relative overflow-hidden"
      style={{ backgroundColor: 'var(--bg-void)' }}
    >
      {/* Full-screen 3D animated background */}
      <div className="fixed inset-0 z-0 pointer-events-none" style={{ opacity: 0.3 }}>
        <Tactile3DHero intensity={0.8} />
      </div>

      {/* Crimson security vignette overlay */}
      <div
        className="fixed inset-0 z-[1] pointer-events-none"
        style={{
          background: `
            radial-gradient(ellipse at center, transparent 20%, rgba(0,0,0,0.85) 100%),
            linear-gradient(180deg, rgba(228,36,43,0.08) 0%, transparent 50%, rgba(0,0,0,0.7) 100%)
          `
        }}
      />

      {/* Admin Login Card */}
      <form
        onSubmit={submit}
        className="w-full max-w-md space-y-6 rounded-[24px] relative z-10"
        style={{
          border: '1px solid rgba(228,36,43,0.3)',
          background: 'rgba(15,10,12,0.85)',
          backdropFilter: 'blur(40px)',
          WebkitBackdropFilter: 'blur(40px)',
          padding: '40px',
          boxShadow: `
            0 0 0 1px rgba(228,36,43,0.15),
            0 20px 60px rgba(0,0,0,0.8),
            0 0 140px rgba(228,36,43,0.15)
          `,
          animation: 'adminLoginFadeIn 0.8s ease-out',
        }}
      >
        {/* Portal Header */}
        <div className="flex items-center gap-4 mb-6">
          <div
            style={{
              width: 52,
              height: 52,
              borderRadius: 16,
              background: 'rgba(228,36,43,0.15)',
              border: '1px solid rgba(228,36,43,0.4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 20px rgba(228,36,43,0.2)',
            }}
          >
            <ShieldAlert style={{ width: 26, height: 26, color: 'var(--red-primary)' }} />
          </div>
          <div>
            <h1
              style={{
                fontFamily: "'Space Grotesk', sans-serif",
                fontWeight: 900,
                color: '#ffffff',
                fontSize: 22,
                letterSpacing: '-0.02em',
                lineHeight: 1.1,
              }}
            >
              ADMINISTRATOR CONTROL
            </h1>
            <p
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 10,
                fontWeight: 700,
                color: 'var(--red-primary)',
                textTransform: 'uppercase',
                letterSpacing: '0.15em',
                marginTop: 6,
              }}
            >
              Restricted Portal Access
            </p>
          </div>
        </div>

        {/* Security Warning Notice */}
        <div
          style={{
            background: 'rgba(228,36,43,0.08)',
            border: '1px solid rgba(228,36,43,0.2)',
            borderRadius: 12,
            padding: '12px 16px',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
          }}
        >
          <ShieldCheck style={{ width: 16, height: 16, color: 'var(--red-primary)', flexShrink: 0 }} />
          <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: 'var(--text-secondary)' }}>
            Authorized administrator credentials required. All authentication attempts are logged.
          </span>
        </div>

        {/* Quick Fill Button */}
        <div>
          <button
            type="button"
            onClick={fillAdminCredentials}
            style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
              padding: '10px 14px',
              borderRadius: 10,
              border: username === 'admin' ? '1px solid var(--red-primary)' : '1px solid rgba(255,255,255,0.08)',
              background: username === 'admin' ? 'rgba(228,36,43,0.2)' : 'rgba(255,255,255,0.03)',
              color: username === 'admin' ? 'var(--red-bright)' : 'var(--text-muted)',
              fontSize: 11,
              fontFamily: "'JetBrains Mono', monospace",
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
          >
            <ShieldAlert style={{ width: 14, height: 14 }} />
            Fill Administrator Credentials (admin / admin)
          </button>
        </div>

        {/* Form fields */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div>
            <label
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 10,
                fontWeight: 700,
                color: 'var(--text-secondary)',
                textTransform: 'uppercase',
                letterSpacing: '0.15em',
                display: 'block',
                marginBottom: 8,
              }}
            >
              Administrator Username
            </label>
            <input
              id="admin-username"
              aria-label="Administrator Username"
              style={{
                width: '100%',
                padding: 16,
                background: 'var(--bg-void)',
                border: '1px solid rgba(228,36,43,0.3)',
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
              placeholder="Enter admin username"
              autoComplete="username"
              required
              onFocus={(e) => {
                e.target.style.borderColor = 'var(--red-primary)';
                e.target.style.boxShadow = '0 0 0 3px rgba(228,36,43,0.2)';
              }}
              onBlur={(e) => {
                e.target.style.borderColor = 'rgba(228,36,43,0.3)';
                e.target.style.boxShadow = 'none';
              }}
            />
          </div>

          <div>
            <label
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 10,
                fontWeight: 700,
                color: 'var(--text-secondary)',
                textTransform: 'uppercase',
                letterSpacing: '0.15em',
                display: 'block',
                marginBottom: 8,
              }}
            >
              Administrator Password
            </label>
            <input
              id="admin-password"
              aria-label="Administrator Password"
              style={{
                width: '100%',
                padding: 16,
                background: 'var(--bg-void)',
                border: '1px solid rgba(228,36,43,0.3)',
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
              placeholder="Enter admin password"
              type="password"
              autoComplete="current-password"
              required
              onFocus={(e) => {
                e.target.style.borderColor = 'var(--red-primary)';
                e.target.style.boxShadow = '0 0 0 3px rgba(228,36,43,0.2)';
              }}
              onBlur={(e) => {
                e.target.style.borderColor = 'rgba(228,36,43,0.3)';
                e.target.style.boxShadow = 'none';
              }}
            />
          </div>
        </div>

        {/* Error */}
        {error && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              background: 'rgba(228,36,43,0.12)',
              border: '1px solid rgba(228,36,43,0.4)',
              borderRadius: 12,
              padding: 16,
            }}
          >
            <AlertTriangle style={{ width: 18, height: 18, color: 'var(--red-bright)', flexShrink: 0 }} />
            <p style={{ color: '#ff6b6b', fontSize: 12, fontFamily: "'JetBrains Mono', monospace", fontWeight: 700 }}>
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
            border: '1px solid rgba(228,36,43,0.6)',
            borderRadius: 12,
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.7 : 1,
            textTransform: 'uppercase',
            letterSpacing: '0.15em',
            transition: 'all 0.2s',
            boxShadow: '0 8px 32px rgba(228,36,43,0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
          }}
        >
          {loading ? 'Authenticating Administrator...' : 'Authenticate Admin Session'}
        </button>

        {/* Link back to User Login */}
        <div style={{ textAlign: 'center', marginTop: 16 }}>
          <Link
            to="/login"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              fontSize: 12,
              fontFamily: "'JetBrains Mono', monospace",
              color: 'var(--text-muted)',
              textDecoration: 'none',
              transition: 'color 0.2s',
            }}
            onMouseEnter={(e) => (e.target.style.color = '#ffffff')}
            onMouseLeave={(e) => (e.target.style.color = 'var(--text-muted)')}
          >
            <ArrowLeft style={{ width: 14, height: 14 }} />
            Return to User Access Portal
          </Link>
        </div>

        {/* Bottom decorative line */}
        <div
          style={{
            height: 2,
            borderRadius: 1,
            background: 'linear-gradient(90deg, transparent, var(--red-primary), transparent)',
            opacity: 0.4,
            marginTop: 12,
          }}
        />
      </form>

      <style>{`
        @keyframes adminLoginFadeIn {
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
