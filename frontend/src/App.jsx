import { useState } from 'react';
import {
  BrowserRouter,
  Routes,
  Route,
  NavLink,
  Navigate,
  useNavigate,
} from 'react-router-dom';
import {
  Shield, Search, Users, FileText, Upload, Cpu, BookOpen, History, LogOut,
} from 'lucide-react';

import Dashboard from './pages/Dashboard';
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

const NAV_ITEMS = [
  { path: '/',               label: 'Trust Dashboard',          icon: Shield },
  { path: '/upload',         label: 'Upload Dataset',           icon: Upload },
  { path: '/model-scan',     label: 'Model Scanner',            icon: Cpu },
  { path: '/real-datasets',  label: 'Real World Dataset Library', icon: BookOpen },
  { path: '/forensics',      label: 'Poison Forensics',         icon: Search },
  { path: '/federated',      label: 'Federated Trust',          icon: Users },
  { path: '/blue-team',      label: 'Blue Team SOC',            icon: Shield },
  { path: '/reports',        label: 'Evidence Reports',         icon: FileText },
  { path: '/history',        label: 'Analysis History',         icon: History },
];

// ── Sidebar ───────────────────────────────────────────────────────────────────

function Sidebar({ events, connected, clearEvents }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <aside className="w-64 flex-shrink-0 bg-bg2 border-r border-border flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-accent/10 border border-accent/30 flex items-center justify-center">
            <Shield className="w-4 h-4 text-accent" />
          </div>
          <div>
            <div className="font-mono text-xs text-accent font-bold tracking-widest">AI TRUST</div>
            <div className="font-mono text-xs text-text3 tracking-wider">FORENSICS v2.2</div>
          </div>
        </div>
      </div>

      {/* WS Status */}
      <div className="px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${connected ? 'bg-accent3' : 'bg-danger'}`}
            style={{ boxShadow: connected ? '0 0 6px #00ffc8' : '0 0 6px #ff4d6a' }}
          />
          <span className="font-mono text-xs text-text3">
            {connected ? 'LIVE STREAM ACTIVE' : 'CONNECTING...'}
          </span>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 p-4 space-y-1">
        {NAV_ITEMS.map((item) => {
          const ItemIcon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) =>
                `w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-left transition-all duration-200 font-mono text-xs tracking-wide ${
                  isActive
                    ? 'bg-accent/10 text-accent border border-accent/30'
                    : 'text-text3 hover:text-text2 hover:bg-surface'
                }`
              }
            >
              <ItemIcon className="w-4 h-4 flex-shrink-0" />
              {item.label}
            </NavLink>
          );
        })}
      </nav>

      {/* Live Events Feed */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center justify-between mb-2">
          <span className="font-mono text-xs text-text3 tracking-widest uppercase">Live Events</span>
          {events.length > 0 && (
            <button onClick={clearEvents} className="font-mono text-xs text-text3 hover:text-accent">
              clear
            </button>
          )}
        </div>
        <div className="space-y-1 max-h-32 overflow-y-auto">
          {events.length === 0 ? (
            <div className="font-mono text-xs text-text3 italic">No events yet...</div>
          ) : (
            events.slice(0, 5).map((evt, i) => (
              <div
                key={i}
                className={`font-mono text-xs px-2 py-1 rounded border-l-2 ${
                  evt.event === 'attack_confirmed'
                    ? 'border-danger text-danger bg-danger/5'
                    : evt.event === 'defense_triggered'
                    ? 'border-purple text-purple bg-purple/5'
                    : evt.event === 'human_review_required'
                    ? 'border-yellow text-yellow bg-yellow/5'
                    : 'border-accent text-accent bg-accent/5'
                }`}
              >
                {evt.event?.replace(/_/g, ' ')}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center justify-between gap-2 font-mono text-xs text-text3">
          <span className="truncate">{user?.name || 'Authenticated analyst'}</span>
          <button
            onClick={handleLogout}
            className="flex items-center gap-1 hover:text-accent"
            aria-label="Sign out"
          >
            <LogOut className="w-3 h-3" /> Sign out
          </button>
        </div>
      </div>
    </aside>
  );
}

// ── Protected app shell ────────────────────────────────────────────────────────

function AppShell() {
  const { events, connected, clearEvents } = useWebSocket();

  return (
    <div className="flex h-screen bg-bg overflow-hidden relative z-10">
      <Sidebar events={events} connected={connected} clearEvents={clearEvents} />
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<PageErrorBoundary name="Trust Dashboard"><Dashboard wsEvents={events} /></PageErrorBoundary>} />
          <Route path="/upload" element={<PageErrorBoundary name="Upload Dataset"><UploadPage /></PageErrorBoundary>} />
          <Route path="/model-scan" element={<PageErrorBoundary name="Model Scanner"><ModelScanPage /></PageErrorBoundary>} />
          <Route path="/real-datasets" element={<PageErrorBoundary name="Real Datasets"><RealDatasetsPage /></PageErrorBoundary>} />
          <Route path="/forensics" element={<PageErrorBoundary name="Poison Forensics"><ForensicsPage /></PageErrorBoundary>} />
          <Route path="/federated" element={<PageErrorBoundary name="Federated Trust"><FederatedPage /></PageErrorBoundary>} />
          <Route path="/blue-team" element={<PageErrorBoundary name="Blue Team SOC"><BlueTeamPage /></PageErrorBoundary>} />
          <Route path="/reports" element={<PageErrorBoundary name="Evidence Reports"><ReportsPage /></PageErrorBoundary>} />
          <Route path="/history" element={<PageErrorBoundary name="Analysis History"><HistoryPage /></PageErrorBoundary>} />
          {/* Catch-all: redirect unknown paths to dashboard */}
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
