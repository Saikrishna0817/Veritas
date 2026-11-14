import { lazy, Suspense, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate, NavLink, Link, useNavigate } from 'react-router-dom';
import Brand from './components/Brand';
import ProtectedRoute from './components/ProtectedRoute';
import PageErrorBoundary from './components/PageErrorBoundary';
import { useAuth } from './hooks/useAuth';
import './styles/instrument.css';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const AdminDashboard = lazy(() => import('./pages/AdminDashboard'));
const ReportsPage = lazy(() => import('./pages/ReportsPage'));
const UploadPage = lazy(() => import('./pages/UploadPage'));
const ModelScanPage = lazy(() => import('./pages/ModelScanPage'));
const HistoryPage = lazy(() => import('./pages/HistoryPage'));
const BenchmarksPage = lazy(() => import('./pages/BenchmarksPage'));
const LoginPage = lazy(() => import('./pages/LoginPage'));
const AccountFlowPage = lazy(() => import('./pages/AccountFlowPage'));

const PageFallback = () => <div className="in-main in-hint">Loading…</div>;

const NAV = [
  ['/upload', 'Check data'], ['/history', 'History'], ['/model-scan', 'Scanner'], ['/reports', 'Reports'], ['/benchmarks', 'Benchmarks'],
];

function AppNav() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const admin = user?.role === 'admin';
  const links = admin ? [['/admin', 'Admin console'], ...NAV.slice(1, 2)] : NAV;
  const signOut = async () => { await logout(); navigate(admin ? '/admin/login' : '/login', { replace: true }); };
  return (
    <nav className="in-nav" aria-label="Main">
      <div className="in-nav-inner">
        <Brand />
        <div className="in-links">
          {links.map(([to, label]) => <NavLink key={to} to={to} className={({ isActive }) => `in-link${isActive ? ' active' : ''}`}>{label}</NavLink>)}
        </div>
        {user && (
          <div className="in-user">
            <span>{user.name}{admin ? ' · admin' : ''}</span>
            <button className="in-btn in-btn-quiet" style={{ padding: '7px 14px' }} onClick={signOut}>Sign out</button>
          </div>
        )}
      </div>
    </nav>
  );
}

function AppFooter() {
  return (
    <footer className="in-footer">
      <div className="in-footer-inner">
        <span>Veritas checks training data for poisoning. A verdict is evidence at a stated false-alarm level, not proof.</span>
        <span><Link to="/benchmarks">How it was measured</Link></span>
      </div>
    </footer>
  );
}

const page = (name, element) => <PageErrorBoundary name={name}>{element}</PageErrorBoundary>;

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
            <Route path="/reports" element={page('Reports', <ReportsPage />)} />
            <Route path="/benchmarks" element={page('Benchmarks', <BenchmarksPage />)} />
            <Route path="/admin" element={<ProtectedRoute requiredRole="admin" fallback={<Navigate to="/admin/login" replace />}>{page('Admin console', <AdminDashboard />)}</ProtectedRoute>} />
            {/* Pages folded into others; old links still land somewhere sensible. */}
            <Route path="/forensics" element={<Navigate to="/history" replace />} />
            <Route path="/blue-team" element={<Navigate to="/reports" replace />} />
            <Route path="/real-datasets" element={<Navigate to="/upload" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </div>
      <AppFooter />
    </div>
  );
}

function LightBody() {
  useEffect(() => {                   // the old dark frame stays only behind pages that have not been redesigned
    document.body.classList.add('light');
    return () => document.body.classList.remove('light');
  }, []);
  return null;
}

export default function App() {
  return (
    <BrowserRouter>
      <LightBody />
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/admin/login" element={<LoginPage admin />} />
          <Route path="/signup" element={<AccountFlowPage mode="signup" />} />
          <Route path="/verify-email" element={<AccountFlowPage mode="verify" />} />
          <Route path="/forgot-password" element={<AccountFlowPage mode="forgot" />} />
          <Route path="/reset-password" element={<AccountFlowPage mode="reset" />} />
          <Route path="/*" element={<ProtectedRoute fallback={<Navigate to="/login" replace />}><AppShell /></ProtectedRoute>} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
