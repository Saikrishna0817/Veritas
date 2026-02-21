import { useState } from 'react';
import { Shield, Search, Users, FileText, Upload, Cpu, BookOpen, History } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import ForensicsPage from './pages/ForensicsPage';

import FederatedPage from './pages/FederatedPage';
import ReportsPage from './pages/ReportsPage';
import UploadPage from './pages/UploadPage';
import ModelScanPage from './pages/ModelScanPage';
import RealDatasetsPage from './pages/RealDatasetsPage';
import HistoryPage from './pages/HistoryPage';
import { useWebSocket } from './hooks/useWebSocket';

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Trust Dashboard', icon: Shield },
  { id: 'upload', label: 'Upload Dataset', icon: Upload },
  { id: 'model_scan', label: 'Model Scanner', icon: Cpu },
  { id: 'real_datasets', label: 'Real World Dataset Library', icon: BookOpen },
  { id: 'forensics', label: 'Poison Forensics', icon: Search },

  { id: 'federated', label: 'Federated Trust', icon: Users },
  { id: 'reports', label: 'Evidence Reports', icon: FileText },
  { id: 'history', label: 'Analysis History', icon: History },
];

export default function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const { events, connected, clearEvents } = useWebSocket();

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard': return <Dashboard wsEvents={events} />;
      case 'upload': return <UploadPage />;
      case 'model_scan': return <ModelScanPage />;
      case 'real_datasets': return <RealDatasetsPage />;
      case 'forensics': return <ForensicsPage />;

      case 'federated': return <FederatedPage />;
      case 'reports': return <ReportsPage />;
      case 'history': return <HistoryPage />;
      default: return <Dashboard wsEvents={events} />;
    }
  };

  return (
    <div className="flex h-screen bg-bg overflow-hidden relative z-10">
      {/* Sidebar */}
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
            <div className={`w-2 h-2 rounded-full ${connected ? 'bg-accent3' : 'bg-danger'}`}
              style={{ boxShadow: connected ? '0 0 6px #00ffc8' : '0 0 6px #ff4d6a' }} />
            <span className="font-mono text-xs text-text3">
              {connected ? 'LIVE STREAM ACTIVE' : 'CONNECTING...'}
            </span>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-4 space-y-1">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActivePage(id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-left transition-all duration-200 font-mono text-xs tracking-wide
                ${activePage === id
                  ? 'bg-accent/10 text-accent border border-accent/30'
                  : 'text-text3 hover:text-text2 hover:bg-surface'
                }`}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {label}
            </button>
          ))}
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
                <div key={i} className={`font-mono text-xs px-2 py-1 rounded border-l-2 ${evt.event === 'attack_confirmed' ? 'border-danger text-danger bg-danger/5' :
                  evt.event === 'defense_triggered' ? 'border-purple text-purple bg-purple/5' :
                    evt.event === 'human_review_required' ? 'border-yellow text-yellow bg-yellow/5' :
                      'border-accent text-accent bg-accent/5'
                  }`}>
                  {evt.event?.replace(/_/g, ' ')}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border">
          <div className="font-mono text-xs text-text3">
            🛡️ Secure AI for Everyone
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        {renderPage()}
      </main>
    </div>
  );
}
