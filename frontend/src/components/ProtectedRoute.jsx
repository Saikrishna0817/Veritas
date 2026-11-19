import { useEffect } from 'react';
import { Link, Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { LoadError } from './LoadError';

const ROLE_WORDS = { admin: 'administrator', analyst: 'analyst', user: 'user' };

// Where to sign in, carrying the address the person asked for so they land there afterwards.
function signInPath(location, admin) {
  const here = location.pathname + location.search;
  const base = admin ? '/admin/login' : '/login';
  return here === '/' || here === base ? base : `${base}?next=${encodeURIComponent(here)}`;
}

// The landing page is its own build at /welcome/ (nginx in production, the dev proxy locally), so it is a page load.
export const LANDING = '/welcome/';
function ToLanding() {
  useEffect(() => { window.location.replace(LANDING); }, []);
  return <div className="instrument in-shell"><div className="in-main in-hint" role="status">Opening Veritas…</div></div>;
}

export default function ProtectedRoute({ children, requiredRole }) {
  const { user, loading, sessionError, recheck } = useAuth();
  const location = useLocation();

  if (loading) {
    return <div className="instrument in-shell"><div className="in-main in-hint" role="status">Checking your session…</div></div>;
  }

  // The server did not answer the session check: say so instead of treating it as signed out.
  if (!user && sessionError) {
    return (
      // in-shell lifts it above the 3D scene, as every other page is
      <div className="instrument in-shell"><main className="in-main"><div className="in-page">
        <h1 className="in-h1">Veritas did not answer</h1>
        <p className="in-lede">Your session could not be checked, so nothing is shown yet. Your results are not affected.</p>
        <LoadError error={sessionError} onRetry={recheck} />
      </div></main></div>
    );
  }

  // A signed-out visitor at the site's front door sees the landing page first; its Sign in leads back here.
  if (!user && location.pathname === '/' && !location.search && requiredRole !== 'admin') return <ToLanding />;

  if (!user) {
    return <Navigate to={signInPath(location, requiredRole === 'admin' || location.pathname.startsWith('/admin'))} replace />;
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
