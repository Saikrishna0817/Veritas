import { lazy, Suspense, useEffect, useState } from 'react';
import {
  BrowserRouter,
  Routes,
  Route,
  NavLink,
  Navigate,
  useNavigate,
  useLocation,
} from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LogOut, Menu, X, LayoutGrid, ScanLine, FileCode2, Database, History, Microscope, Siren, FileText, ShieldCheck, UserRound,
} from 'lucide-react';
import ProtectedRoute from './components/ProtectedRoute';
import PageErrorBoundary from './components/PageErrorBoundary';
import Footer from './components/Footer';
import Wordmark from './components/Wordmark';
import { useWebSocket } from './hooks/useWebSocket';
import { useAuth } from './hooks/useAuth';

// Pages load on demand so the sign-in screen doesn't ship three.js.
const Dashboard = lazy(() => import('./pages/Dashboard'));
const AdminDashboard = lazy(() => import('./pages/AdminDashboard'));
const ForensicsPage = lazy(() => import('./pages/ForensicsPage'));
const ReportsPage = lazy(() => import('./pages/ReportsPage'));
const UploadPage = lazy(() => import('./pages/UploadPage'));
const ModelScanPage = lazy(() => import('./pages/ModelScanPage'));
const RealDatasetsPage = lazy(() => import('./pages/RealDatasetsPage'));
const HistoryPage = lazy(() => import('./pages/HistoryPage'));
const BlueTeamPage = lazy(() => import('./pages/BlueTeamPage'));
const LoginPage = lazy(() => import('./pages/LoginPage'));
const AdminLoginPage = lazy(() => import('./pages/AdminLoginPage'));
const AccountFlowPage = lazy(() => import('./pages/AccountFlowPage'));
const AccountPage = lazy(() => import('./pages/AccountPage'));
const LandingPage = lazy(() => import('./pages/LandingPage'));

const PageFallback = () => (
  <div className="p-10 text-sm text-textMuted" role="status">Loading…</div>
);

const BASE_NAV_ITEMS = [
  { path: '/', label: 'Overview', icon: LayoutGrid },
  { path: '/upload', label: 'Examine a batch', icon: ScanLine },
  { path: '/model-scan', label: 'Scan a model file', icon: FileCode2 },
  { path: '/real-datasets', label: 'Reference datasets', icon: Database },
  { path: '/history', label: 'History', icon: History },
  { path: '/reports', label: 'Evidence reports', icon: FileText },
  { path: '/forensics', label: 'Forensics', icon: Microscope, preview: true },
  { path: '/blue-team', label: 'Response desk', icon: Siren, preview: true },
  { path: '/account', label: 'Account', icon: UserRound },
];
const ADMIN_NAV_ITEMS = [{ path: '/admin', label: 'Admin console', icon: ShieldCheck }];
const ROLE_NAMES = { admin: 'Administrator', analyst: 'Analyst', user: 'User' };

function NavItems({ items, onNavigate }) {
  return (
    <ul className="flex flex-col gap-0.5">
      {items.map((item) => {
        const { path, label, preview } = item;
        return (
        <li key={path}>
          <NavLink
            to={path}
            end={path === '/'}
            onClick={onNavigate}
            className={({ isActive }) =>
              `relative flex items-center gap-3 rounded-2xl px-3 py-2.5 text-[14px] font-medium transition-colors ${
                isActive ? 'bg-white/[0.08] text-textPrimary' : 'text-textSecondary hover:bg-white/[0.03] hover:text-textPrimary'
              }`
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.span layoutId="rail-mark" className="absolute bottom-2 left-0 top-2 w-[2px] rounded bg-uv" transition={{ type: 'spring', stiffness: 500, damping: 40 }} />
                )}
                <item.icon className={`h-4 w-4 shrink-0 ${isActive ? 'text-uv' : ''}`} aria-hidden />
                <span className="flex-1">{label}</span>
                {preview && <span className="rounded-full border border-statusWarn/40 px-2 py-0.5 text-[11px] text-statusWarn/90" title="Research preview: heuristic output">Preview</span>}
              </>
            )}
          </NavLink>
        </li>
        );
      })}
    </ul>
  );
}

function Rail({ connected }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  // The phone menu covers the page, so Escape closes it.
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);
  const items = user?.role === 'admin' ? ADMIN_NAV_ITEMS : BASE_NAV_ITEMS;

  const handleLogout = async () => {
    const wasAdmin = user?.role === 'admin';
    await logout();
    navigate(wasAdmin ? '/admin/login' : '/login', { replace: true });
  };

  const body = (
    <>
      <nav aria-label="Main" className="flex-1 overflow-y-auto px-3 py-4">
        <NavItems items={items} onNavigate={() => setOpen(false)} />
      </nav>
      <div className="space-y-4 border-t border-borderHairline p-4">
        <div className="flex items-center gap-2 font-mono text-[11px] text-textMuted" aria-live="polite">
          <span className={`h-2 w-2 rounded-full ${connected ? 'animate-pulse bg-statusSafe' : 'bg-textMuted'}`} />
          {connected ? 'Live event stream' : 'Event stream reconnecting'}
        </div>
        {user && (
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-2xl border border-borderStrong font-display text-sm font-bold uppercase text-uv">
              {(user.name || user.username || '?').charAt(0)}
            </div>
            <div className="min-w-0 flex-1 leading-tight">
              <div className="truncate text-sm font-semibold">{user.name || user.username}</div>
              <div className="text-[12px] text-textMuted">{ROLE_NAMES[user.role] || user.role}</div>
            </div>
            <button onClick={handleLogout} className="rounded-2xl p-2 text-textMuted hover:bg-white/[0.04] hover:text-textPrimary" aria-label="Sign out" title="Sign out">
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        )}
      </div>
    </>
  );

  return (
    <>
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-[248px] flex-col border-r border-borderHairline bg-ink-950/70 backdrop-blur-xl lg:flex">
        <button type="button" onClick={() => navigate('/')} className="px-6 pb-4 pt-6 text-left" aria-label="Veritas overview">
          <Wordmark />
        </button>
        {body}
      </aside>
      <header className="sticky top-0 z-40 flex items-center justify-between border-b border-borderHairline bg-ink-950/90 px-4 py-3 backdrop-blur-md lg:hidden">
        <button type="button" onClick={() => navigate('/')} aria-label="Veritas overview"><Wordmark /></button>
        <button onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-label={open ? 'Close menu' : 'Open menu'} className="rounded-2xl border border-borderStrong p-2">
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </header>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="fixed inset-x-0 bottom-0 top-[61px] z-40 flex flex-col bg-ink-950 lg:hidden"
          >
            {body}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

function AnimatedRoutes({ events }) {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -6 }}
        transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
      >
        <Suspense fallback={<PageFallback />}>
          <Routes location={location}>
            <Route path="/" element={<PageErrorBoundary name="Overview"><Dashboard wsEvents={events} /></PageErrorBoundary>} />
            <Route path="/admin" element={<ProtectedRoute requiredRole="admin" fallback={<Navigate to="/admin/login" replace />}><PageErrorBoundary name="Admin console"><AdminDashboard /></PageErrorBoundary></ProtectedRoute>} />
            <Route path="/upload" element={<PageErrorBoundary name="Examine a batch"><UploadPage /></PageErrorBoundary>} />
            <Route path="/model-scan" element={<PageErrorBoundary name="Model scanner"><ModelScanPage /></PageErrorBoundary>} />
            <Route path="/real-datasets" element={<PageErrorBoundary name="Reference datasets"><RealDatasetsPage /></PageErrorBoundary>} />
            <Route path="/forensics" element={<PageErrorBoundary name="Forensics"><ForensicsPage /></PageErrorBoundary>} />
            <Route path="/blue-team" element={<PageErrorBoundary name="Response desk"><BlueTeamPage /></PageErrorBoundary>} />
            <Route path="/reports" element={<PageErrorBoundary name="Evidence reports"><ReportsPage /></PageErrorBoundary>} />
            <Route path="/history" element={<PageErrorBoundary name="History"><HistoryPage /></PageErrorBoundary>} />
            <Route path="/account" element={<PageErrorBoundary name="Account"><AccountPage /></PageErrorBoundary>} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </motion.div>
    </AnimatePresence>
  );
}

function AppShell() {
  const { events, connected } = useWebSocket();
  return (
    <div className="uv-ground min-h-screen text-textPrimary">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[70] focus:rounded focus:bg-uv focus:px-3 focus:py-2 focus:text-ink-950">
        Skip to content
      </a>
      <Rail connected={connected} />
      <div className="flex min-h-screen flex-col lg:pl-[248px]">
        <main id="main" className="w-full flex-1">
          <AnimatedRoutes events={events} />
        </main>
        <Footer />
      </div>
    </div>
  );
}

/** Signed-out visitors see the public landing page at "/"; every other path asks them to sign in. */
function Gate() {
  const { user, loading } = useAuth();
  const { pathname } = useLocation();
  if (!loading && !user && pathname === '/') return <LandingPage />;
  return (
    <ProtectedRoute fallback={<Navigate to="/login" replace />}>
      <AppShell />
    </ProtectedRoute>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/admin/login" element={<AdminLoginPage />} />
          <Route path="/signup" element={<AccountFlowPage mode="signup" />} />
          <Route path="/verify-email" element={<AccountFlowPage mode="verify" />} />
          <Route path="/forgot-password" element={<AccountFlowPage mode="forgot" />} />
          <Route path="/reset-password" element={<AccountFlowPage mode="reset" />} />
          <Route path="/resend-verification" element={<AccountFlowPage mode="resend" />} />
          <Route path="/welcome" element={<LandingPage />} />
          <Route path="/*" element={<Gate />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
