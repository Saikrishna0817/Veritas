import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate, Link, useNavigate } from 'react-router-dom';
import Brand from './components/Brand';
import GlassNav from './components/glass-nav/GlassNav';
import ProtectedRoute, { LANDING } from './components/ProtectedRoute';
import PageErrorBoundary from './components/PageErrorBoundary';
import SceneLayer from './scene/SceneLayer';
import AppCursor from './ui/AppCursor';
import { useAuth } from './hooks/useAuth';
import './styles/argilla.css';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const AdminDashboard = lazy(() => import('./pages/AdminDashboard'));
const ReportsPage = lazy(() => import('./pages/ReportsPage'));
const UploadPage = lazy(() => import('./pages/UploadPage'));
const ModelScanPage = lazy(() => import('./pages/ModelScanPage'));
const HistoryPage = lazy(() => import('./pages/HistoryPage'));
const BenchmarksPage = lazy(() => import('./pages/BenchmarksPage'));
const LoginPage = lazy(() => import('./pages/LoginPage'));
const AccountFlowPage = lazy(() => import('./pages/AccountFlowPage'));
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'));
const AccountPage = lazy(() => import('./pages/AccountPage'));
const LegalPage = lazy(() => import('./pages/LegalPage'));

const PageFallback = () => <div className="in-main in-hint">Loading…</div>;

const NAV = [
  { to: '/upload', label: 'Check data', icon: 'upload' },
  { to: '/history', label: 'History', icon: 'history' },
  { to: '/model-scan', label: 'Scanner', icon: 'scanner' },
  { to: '/reports', label: 'Reports', icon: 'reports' },
  { to: '/benchmarks', label: 'Benchmarks', icon: 'benchmarks' },
  { to: '/account', label: 'Account', icon: 'account' },
];
const ADMIN_NAV = [{ to: '/admin', label: 'Admin console', icon: 'admin' }, NAV[1], NAV[5]];

function AppNav() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const admin = user?.role === 'admin';
  const links = admin ? ADMIN_NAV : NAV;
  // a user goes back to the landing page (a separate page, so a full load); an admin to the admin sign-in
  const signOut = async () => { await logout(); if (admin) navigate('/admin/login', { replace: true }); else window.location.assign(LANDING); };
  return (
    <header className="in-nav">
      <div className="in-nav-inner">
        <Brand />
        <GlassNav links={links} />
        {user && (
          <div className="in-user">
            <span>{user.name}{admin ? ' · admin' : ''}</span>
            <button className="in-btn in-btn-quiet" style={{ padding: '7px 14px' }} onClick={signOut}>Sign out</button>
          </div>
        )}
      </div>
    </header>
  );
}

function AppFooter() {
  return (
    <footer className="in-footer">
      <div className="in-footer-inner">
        <span>Veritas checks training data for poisoning. A verdict is evidence at a stated false-alarm level, not proof.</span>
        <span className="in-footer-links"><Link to="/benchmarks">How it was measured</Link><Link to="/terms">Terms (draft)</Link><Link to="/privacy">Privacy (draft)</Link></span>
      </div>
    </footer>
  );
}

// keyed by page, so an error on one page does not stay on screen after moving to another
const page = (name, element) => <PageErrorBoundary key={name} name={name}>{element}</PageErrorBoundary>;

function AppShell() {
  return (
    <div className="instrument in-shell">
      <AppNav />
      <div className="in-content">
        <Suspense fallback={<PageFallback />}>
          <Routes>
            <Route path="/" element={page('Overview', <Dashboard />)} />
            <Route path="/upload" element={page('Check data', <UploadPage />)} />
            <Route path="/history" element={page('History', <HistoryPage />)} />
            <Route path="/history/:id" element={page('History', <HistoryPage />)} />
            <Route path="/model-scan" element={page('Model scanner', <ModelScanPage />)} />
            <Route path="/model-scan/:id" element={page('Model scanner', <ModelScanPage />)} />
            <Route path="/reports" element={page('Reports', <ReportsPage />)} />
            <Route path="/benchmarks" element={page('Benchmarks', <BenchmarksPage />)} />
            <Route path="/account" element={page('Account', <AccountPage />)} />
            <Route path="/admin" element={<ProtectedRoute requiredRole="admin">{page('Admin console', <AdminDashboard />)}</ProtectedRoute>} />
            {/* Pages folded into others; old links still land somewhere sensible. */}
            <Route path="/forensics" element={<Navigate to="/history" replace />} />
            <Route path="/blue-team" element={<Navigate to="/reports" replace />} />
            <Route path="/real-datasets" element={<Navigate to="/upload" replace />} />
            <Route path="*" element={page('Not found', <NotFoundPage />)} />
          </Routes>
        </Suspense>
      </div>
      <AppFooter />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      {/* one 3D world behind every page, sign-in included; it persists across route changes */}
      <SceneLayer />
      <AppCursor />
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route path="/login" element={page('Sign in', <LoginPage />)} />
          <Route path="/admin/login" element={page('Sign in', <LoginPage admin />)} />
          <Route path="/signup" element={page('Sign up', <AccountFlowPage mode="signup" />)} />
          <Route path="/verify-email" element={page('Email confirmation', <AccountFlowPage mode="verify" />)} />
          <Route path="/forgot-password" element={page('Password reset', <AccountFlowPage mode="forgot" />)} />
          <Route path="/reset-password" element={page('Password reset', <AccountFlowPage mode="reset" />)} />
          <Route path="/terms" element={page('Terms', <LegalPage doc="terms" />)} />
          <Route path="/privacy" element={page('Privacy', <LegalPage doc="privacy" />)} />
          <Route path="/*" element={<ProtectedRoute><AppShell /></ProtectedRoute>} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
