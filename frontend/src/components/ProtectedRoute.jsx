import { useAuth } from '../hooks/useAuth';
import { ShieldAlert } from 'lucide-react';
import { Link, Navigate } from 'react-router-dom';

export default function ProtectedRoute({ children, fallback, requiredRole }) {
  const { user, loading } = useAuth();

  if (loading) {
    return <div className="p-8 text-textMuted font-mono text-sm">Checking session…</div>;
  }

  if (!user) {
    if (requiredRole === 'admin' || (typeof window !== 'undefined' && window.location.pathname.startsWith('/admin'))) {
      return <Navigate to="/admin/login" replace />;
    }
    return fallback || <Navigate to="/login" replace />;
  }

  if (requiredRole && user.role !== requiredRole) {
    const isTargetingAdmin = requiredRole === 'admin';
    return (
      <div className="min-h-screen flex items-center justify-center p-6 bg-bgVoid text-textPrimary">
        <div className="max-w-md w-full p-8 rounded-2xl bg-bgPanel border border-redPrimary/30 shadow-red-glow text-center space-y-6">
          <div className="w-14 h-14 mx-auto rounded-2xl bg-redPrimary/10 border border-redPrimary/30 flex items-center justify-center">
            <ShieldAlert className="w-7 h-7 text-redPrimary" />
          </div>
          <div>
            <h2 className="font-display text-2xl font-bold tracking-tight text-white mb-2">403 Access Denied</h2>
            <p className="text-sm font-mono text-textMuted leading-relaxed">
              This module requires <span className="text-redPrimary font-bold uppercase">{requiredRole}</span> privilege. Your active role is <span className="text-textPrimary font-bold uppercase">{user.role}</span>.
            </p>
          </div>
          <Link
            to={isTargetingAdmin ? '/' : '/admin'}
            className="inline-block w-full py-3 px-6 rounded-xl bg-redPrimary text-white font-mono text-xs font-bold uppercase tracking-wider hover:bg-redBright transition-all shadow-red-glow"
          >
            {isTargetingAdmin ? 'Return to User Dashboard' : 'Return to Admin Console'}
          </Link>
        </div>
      </div>
    );
  }

  return children;
}

