import { useAuth } from '../hooks/useAuth';
import { ShieldAlert } from 'lucide-react';
import { Link, Navigate } from 'react-router-dom';

const ROLE_NAMES = { admin: 'administrator', analyst: 'analyst', user: 'user' };

export default function ProtectedRoute({ children, fallback, requiredRole }) {
  const { user, loading } = useAuth();

  if (loading) {
    return <div className="p-8 text-sm text-textMuted" role="status">Checking your session…</div>;
  }

  if (!user) {
    if (requiredRole === 'admin' || (typeof window !== 'undefined' && window.location.pathname.startsWith('/admin'))) {
      return <Navigate to="/admin/login" replace />;
    }
    return fallback || <Navigate to="/login" replace />;
  }

  if (requiredRole && user.role !== requiredRole) {
    const needsAdmin = requiredRole === 'admin';
    return (
      <div className="flex min-h-[70vh] items-center justify-center p-6 text-textPrimary">
        <div className="edge-card w-full max-w-md">
          <div className="edge-inner space-y-6 p-8 text-center">
            <span className="icon-chip mx-auto"><ShieldAlert className="h-5 w-5 text-uv" aria-hidden /></span>
            <div>
              <h1 className="font-display text-[28px] font-medium leading-tight tracking-tight">This page is for {ROLE_NAMES[requiredRole] || requiredRole}s</h1>
              <p className="mt-2 text-sm text-textSecondary">
                You're signed in as {user.name || user.username || 'this account'}, with the {ROLE_NAMES[user.role] || user.role} role.
              </p>
            </div>
            <Link
              to={needsAdmin ? '/' : '/admin'}
              className="inline-flex w-full items-center justify-center rounded-2xl bg-white px-6 py-3.5 text-sm font-medium text-ink-900 transition-colors hover:bg-[#EDC5FC]"
            >
              {needsAdmin ? 'Back to the overview' : 'Back to the admin console'}
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return children;
}
