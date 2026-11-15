import { Link, Navigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

const ROLE_WORDS = { admin: 'administrator', analyst: 'analyst', user: 'user' };

export default function ProtectedRoute({ children, fallback, requiredRole }) {
  const { user, loading } = useAuth();

  if (loading) {
    return <div className="instrument in-main in-hint" role="status">Checking your session…</div>;
  }

  if (!user) {
    if (requiredRole === 'admin' || (typeof window !== 'undefined' && window.location.pathname.startsWith('/admin'))) {
      return <Navigate to="/admin/login" replace />;
    }
    return fallback || <Navigate to="/login" replace />;
  }

  if (requiredRole && user.role !== requiredRole) {
    const forAdmins = requiredRole === 'admin';
    return (
      <main className="in-main"><div className="in-page">
        <h1 className="in-h1">{forAdmins ? 'This page is for administrators' : 'This page is not available to your account'}</h1>
        <p className="in-lede">
          You are signed in as <b>{user.name || user.username}</b> ({ROLE_WORDS[user.role] || user.role}).
          {forAdmins ? ' An administrator can change your role.' : ''}
        </p>
        <Link to={forAdmins ? '/' : '/admin'} className="in-link-text">{forAdmins ? 'Go to the overview' : 'Go to the admin console'}</Link>
      </div></main>
    );
  }

  return children;
}
