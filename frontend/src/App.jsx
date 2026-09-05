import {
  BrowserRouter,
  Routes,
  Route,
  NavLink,
  Navigate,
  useNavigate,
} from 'react-router-dom';
import { LogOut, Lock, Globe, ChevronRight } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import AdminDashboard from './pages/AdminDashboard';
import ForensicsPage from './pages/ForensicsPage';
import FederatedPage from './pages/FederatedPage';
import ReportsPage from './pages/ReportsPage';
import UploadPage from './pages/UploadPage';
import ModelScanPage from './pages/ModelScanPage';
import RealDatasetsPage from './pages/RealDatasetsPage';
import HistoryPage from './pages/HistoryPage';
import BlueTeamPage from './pages/BlueTeamPage';
import LoginPage from './pages/LoginPage';
import ProtectedRoute from './components/ProtectedRoute';
import PageErrorBoundary from './components/PageErrorBoundary';
import { useWebSocket } from './hooks/useWebSocket';
import { useAuth } from './hooks/useAuth';

const BASE_NAV_ITEMS = [
  { path: '/', label: 'Dashboard' },
  { path: '/upload', label: 'Upload' },
  { path: '/model-scan', label: 'Scanner' },
  { path: '/forensics', label: 'Forensics' },
  { path: '/blue-team', label: 'SOC' },
  { path: '/reports', label: 'Reports' },
];

// ── Top Navigation ────────────────────────────────────────────────────────────

function TopNav() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const navItems = user?.role === 'admin'
    ? [{ path: '/admin', label: 'Admin Console' }]
    : BASE_NAV_ITEMS;

  const handleLogout = () => {
    const wasAdmin = user?.role === 'admin';
    logout();
    if (wasAdmin) {
      navigate('/admin/login', { replace: true });
    } else {
      navigate('/login', { replace: true });
    }
  };


  return (
    <nav className="sticky top-0 z-50 w-full bg-bgVoid/80 backdrop-blur-md border-b border-borderHairline px-6 py-4 flex items-center justify-between">
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-redPrimary/10 border border-redPrimary/20 flex items-center justify-center">
          <Lock className="w-5 h-5 text-redPrimary" />
        </div>
        <div className="font-display font-bold text-lg tracking-tight text-textPrimary">SPECTRA</div>
      </div>

      {/* Pill Links */}
      <div className="hidden md:flex items-center gap-1 bg-bgPanel p-1 rounded-full border border-borderHairline">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            className={({ isActive }) =>
              `relative px-4 py-2 rounded-full text-sm font-medium transition-colors z-10 ${
                isActive ? 'text-redBright tracking-wide' : 'text-textMuted hover:text-textPrimary'
              }`
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.div
                    layoutId="nav-indicator"
                    className="absolute inset-0 bg-redDim/30 border border-redPrimary/20 rounded-full shadow-red-glow z-[-1]"
                    transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                  />
                )}
                {item.label}
              </>
            )}
          </NavLink>
        ))}
      </div>

      {/* Right Actions */}
      <div className="flex items-center gap-4">
        <button className="hidden sm:flex items-center gap-2 text-sm font-medium text-textMuted hover:text-textPrimary transition-colors">
          <Globe className="w-4 h-4" /> EN
        </button>

        {user ? (
          <>
            {/* User badge */}
            <div className={`hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-bgPanel border ${
              user.role === 'admin'
                ? 'border-redPrimary/40 shadow-red-glow'
                : 'border-cyan-500/40'
            }`}>
              <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold uppercase ${
                user.role === 'admin'
                  ? 'bg-redPrimary/20 border border-redPrimary/40 text-redBright'
                  : 'bg-cyan-500/20 border border-cyan-500/40 text-cyan-400'
              }`}>
                {user.name?.charAt(0) || '?'}
              </div>
              <div className="flex flex-col leading-none">
                <span className="text-xs font-semibold text-textPrimary">{user.name}</span>
                <span className={`text-[9px] font-mono font-extrabold uppercase tracking-wider ${
                  user.role === 'admin' ? 'text-redBright' : 'text-cyan-400'
                }`}>
                  {user.role === 'admin' ? 'ADMINISTRATOR' : 'REGULAR USER'}
                </span>
              </div>
            </div>


            {/* Log Out button */}
            <button
              onClick={handleLogout}
              className="group flex items-center gap-2 bg-bgPanelRaised hover:bg-redPrimary text-textMuted hover:text-white px-4 py-2.5 rounded-full text-sm font-semibold border border-borderHairline hover:border-redPrimary transition-all"
            >
              <LogOut className="w-4 h-4" />
              Log Out
            </button>
          </>
        ) : (
          <button
            onClick={() => navigate('/login')}
            className="group flex items-center gap-2 bg-redPrimary text-white px-5 py-2.5 rounded-full text-sm font-semibold hover:bg-redBright hover:shadow-red-glow transition-all"
          >
            Sign In
            <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </button>
        )}
      </div>
    </nav>
  );
}

import AdminLoginPage from './pages/AdminLoginPage';

// ── Protected app shell ────────────────────────────────────────────────────────

function AppShell() {
  const { events } = useWebSocket();

  return (
    <div className="flex flex-col min-h-screen bg-bgSurface text-textPrimary relative">

      <TopNav />
      <main className="flex-1 w-full relative z-10">
        <Routes>
          <Route path="/" element={<PageErrorBoundary name="Trust Dashboard"><Dashboard wsEvents={events} /></PageErrorBoundary>} />
          <Route path="/admin" element={<ProtectedRoute requiredRole="admin" fallback={<Navigate to="/admin/login" replace />}><PageErrorBoundary name="Admin Console"><AdminDashboard /></PageErrorBoundary></ProtectedRoute>} />
          <Route path="/upload" element={<PageErrorBoundary name="Upload Dataset"><UploadPage /></PageErrorBoundary>} />
          <Route path="/model-scan" element={<PageErrorBoundary name="Model Scanner"><ModelScanPage /></PageErrorBoundary>} />
          <Route path="/real-datasets" element={<PageErrorBoundary name="Real Datasets"><RealDatasetsPage /></PageErrorBoundary>} />
          <Route path="/forensics" element={<PageErrorBoundary name="Poison Forensics"><ForensicsPage /></PageErrorBoundary>} />
          <Route path="/federated" element={<PageErrorBoundary name="Federated Trust"><FederatedPage /></PageErrorBoundary>} />
          <Route path="/blue-team" element={<PageErrorBoundary name="Blue Team SOC"><BlueTeamPage /></PageErrorBoundary>} />
          <Route path="/reports" element={<PageErrorBoundary name="Evidence Reports"><ReportsPage /></PageErrorBoundary>} />
          <Route path="/history" element={<PageErrorBoundary name="Analysis History"><HistoryPage /></PageErrorBoundary>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

// ── Root App ──────────────────────────────────────────────────────────────────

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/admin/login" element={<AdminLoginPage />} />
        <Route
          path="/*"
          element={
            <ProtectedRoute fallback={<Navigate to="/login" replace />}>
              <AppShell />
            </ProtectedRoute>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}

