import { useState, useEffect, useCallback } from 'react';
import {
  Shield,
  Users,
  UserPlus,
  Trash2,
  AlertTriangle,
  CheckCircle,
  Activity,
  Lock,
  RefreshCw,
  Radio,
  Eye,
  X,
  Play,
  Layers,
} from 'lucide-react';
import { api } from '../services/api';
import { useAuth } from '../hooks/useAuth';

export default function AdminDashboard() {
  const { user: activeUser } = useAuth();
  const [activeTab, setActiveTab] = useState('users');

  // Governance state
  const [users, setUsers] = useState([]);
  const [loadingUsers, setLoadingUsers] = useState(true);
  const [userError, setUserError] = useState('');
  const [userSuccess, setUserSuccess] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newRole, setNewRole] = useState('user');
  const [submittingUser, setSubmittingUser] = useState(false);

  // Security & Quarantine state
  const [quarantineStatus, setQuarantineStatus] = useState(null);
  const [quarantining, setQuarantining] = useState(false);
  const [hitlCases, setHitlCases] = useState([]);
  const [loadingHitl, setLoadingHitl] = useState(false);
  const [selectedAttack, setSelectedAttack] = useState('label_flip');
  const [simResult, setSimResult] = useState(null);
  const [runningSim, setRunningSim] = useState(false);

  // Audit feed state
  const [auditEvents, setAuditEvents] = useState([]);
  const [loadingAudit, setLoadingAudit] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState(null);

  // Scan registry state
  const [scanHistory, setScanHistory] = useState([]);
  const [loadingRegistry, setLoadingRegistry] = useState(false);

  const fetchUsers = useCallback(async () => {
    setLoadingUsers(true);
    try {
      const data = await api.getUsers();
      setUsers(data);
    } catch (err) {
      setUserError(err.message || 'Failed to load user directory.');
    } finally {
      setLoadingUsers(false);
    }
  }, []);

  const fetchSecurityData = useCallback(async () => {
    setLoadingHitl(true);
    try {
      const [qs, hitl] = await Promise.all([
        api.getDefenseStatus().catch(() => null),
        api.getPendingReviews().catch(() => ({ cases: [] })),
      ]);
      setQuarantineStatus(qs);
      setHitlCases(hitl.cases || []);
    } catch {
      // ignore
    } finally {
      setLoadingHitl(false);
    }
  }, []);

  const fetchAuditEvents = useCallback(async () => {
    setLoadingAudit(true);
    try {
      const res = await api.getAuditEvents(100);
      setAuditEvents(res.events || []);
    } catch {
      // ignore
    } finally {
      setLoadingAudit(false);
    }
  }, []);

  const fetchScanRegistry = useCallback(async () => {
    setLoadingRegistry(true);
    try {
      const res = await api.getAnalysisHistory(100);
      setScanHistory(res.results || []);
    } catch {
      // ignore
    } finally {
      setLoadingRegistry(false);
    }
  }, []);

  const refreshAll = useCallback(() => {
    fetchUsers();
    fetchSecurityData();
    fetchAuditEvents();
    fetchScanRegistry();
  }, [fetchUsers, fetchSecurityData, fetchAuditEvents, fetchScanRegistry]);

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  // ── Actions ──────────────────────────────────────────────────────────────────

  const handleCreateUser = async (e) => {
    e.preventDefault();
    setUserError('');
    setUserSuccess('');
    setSubmittingUser(true);
    try {
      await api.createUser({ username: newUsername, password: newPassword, role: newRole });
      setUserSuccess(`User '${newUsername}' successfully created with role [${newRole.toUpperCase()}].`);
      setNewUsername('');
      setNewPassword('');
      setNewRole('user');
      setShowCreateModal(false);
      fetchUsers();
      fetchAuditEvents();
    } catch (err) {
      setUserError(err.message || 'Failed to create user account.');
    } finally {
      setSubmittingUser(false);
    }
  };

  const handleDeleteUser = async (userId, username) => {
    if (userId === activeUser?.id) {
      setUserError('Cannot revoke your own active administrator account.');
      return;
    }
    if (!window.confirm(`Are you sure you want to revoke access for user '${username}'?`)) return;
    setUserError('');
    setUserSuccess('');
    try {
      await api.deleteUser(userId);
      setUserSuccess(`Revoked account access for user '${username}'.`);
      fetchUsers();
      fetchAuditEvents();
    } catch (err) {
      setUserError(err.message || 'Failed to revoke user access.');
    }
  };

  const handleTriggerQuarantine = async () => {
    if (!window.confirm('WARNING: Triggering emergency quarantine will isolate all pending model pipelines and suspend unverified ingestion. Proceed?')) return;
    setQuarantining(true);
    setUserError('');
    try {
      const res = await api.triggerQuarantine();
      setQuarantineStatus(res);
      setUserSuccess('Emergency System Quarantine executed successfully.');
      fetchAuditEvents();
    } catch (err) {
      setUserError(err.message || 'Quarantine trigger failed.');
    } finally {
      setQuarantining(false);
    }
  };

  const handleResolveHitl = async (caseId, decision) => {
    try {
      await api.submitReviewDecision(caseId, decision, activeUser?.name || 'admin');
      setUserSuccess(`HITL case '${caseId.slice(0, 8)}' resolved with action [${decision.toUpperCase()}].`);
      fetchSecurityData();
      fetchAuditEvents();
    } catch (err) {
      setUserError(err.message || 'Failed to submit HITL review decision.');
    }
  };

  const handleRunRedTeam = async () => {
    setRunningSim(true);
    setSimResult(null);
    try {
      const res = await api.runRedTeamSimulation(selectedAttack);
      setSimResult(res);
      setUserSuccess(`Red team simulation '${selectedAttack}' executed successfully.`);
      fetchAuditEvents();
    } catch (err) {
      setUserError(err.message || 'Red team simulation failed.');
    } finally {
      setRunningSim(false);
    }
  };

  const handleDeleteScanRecord = async (runId) => {
    if (!window.confirm(`Purge analysis record '${runId.slice(0, 8)}'? This action is permanent.`)) return;
    try {
      await api.deleteAnalysisHistory(runId);
      setUserSuccess(`Purged analysis record '${runId.slice(0, 8)}'.`);
      fetchScanRegistry();
      fetchAuditEvents();
    } catch (err) {
      setUserError(err.message || 'Failed to purge scan record.');
    }
  };

  return (
    <div className="p-6 md:p-10 max-w-7xl mx-auto space-y-8">

      {/* ── Console Header ───────────────────────────────────────────────────────── */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 border-b border-borderHairline pb-6">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <span className="px-3 py-1 rounded-md bg-redPrimary/20 border border-redPrimary/40 font-mono text-[11px] font-bold text-redBright uppercase tracking-widest flex items-center gap-1.5 shadow-red-glow">
              <Lock className="w-3.5 h-3.5" /> Admin Privilege Level 0
            </span>
            <span className="font-mono text-xs text-textMuted">• Dedicated Console</span>
          </div>
          <h1 className="font-display font-black text-3xl md:text-4xl text-textPrimary tracking-tight">
            Administrator Governance & Security Control
          </h1>
          <p className="text-sm font-sans text-textMuted mt-1">
            Centralized administration for user credentials, AI defense overrides, HITL review cases, security audit feeds, and model registries.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={refreshAll}
            className="p-3 rounded-xl bg-bgPanel border border-borderHairline text-textMuted hover:text-white hover:border-redPrimary/50 transition-all flex items-center gap-2 font-mono text-xs font-semibold"
            title="Refresh All System Data"
          >
            <RefreshCw className={`w-4 h-4 ${loadingUsers || loadingAudit ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="flex items-center gap-2 bg-redPrimary text-white font-mono text-xs font-bold uppercase tracking-wider px-5 py-3 rounded-xl hover:bg-redBright shadow-red-glow transition-all"
          >
            <UserPlus className="w-4 h-4" />
            Provision Account
          </button>
        </div>
      </div>

      {/* ── Notifications ────────────────────────────────────────────────────────── */}
      {userSuccess && (
        <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-500/30 text-emerald-400 font-mono text-xs flex items-center justify-between shadow-lg">
          <div className="flex items-center gap-3">
            <CheckCircle className="w-4 h-4 shrink-0" />
            <span>{userSuccess}</span>
          </div>
          <button onClick={() => setUserSuccess('')} className="text-emerald-400 hover:text-white font-bold">✕</button>
        </div>
      )}

      {userError && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-500/40 text-red-400 font-mono text-xs flex items-center justify-between shadow-lg">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-4 h-4 shrink-0 text-redPrimary" />
            <span>{userError}</span>
          </div>
          <button onClick={() => setUserError('')} className="text-red-400 hover:text-white font-bold">✕</button>
        </div>
      )}

      {/* ── Executive Metric Cards ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="p-5 rounded-2xl bg-bgPanel border border-borderHairline space-y-2 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-1 h-full bg-cyan-500" />
          <div className="flex items-center justify-between">
            <span className="font-mono text-[11px] font-bold uppercase text-textMuted tracking-wider">Total Accounts</span>
            <Users className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="font-display text-3xl font-bold text-white">{users.length}</div>
          <div className="font-mono text-[11px] text-textMuted flex items-center gap-2">
            <span className="text-amber-400 font-bold">{users.filter((u) => u.role === 'admin').length} Admins</span>
            <span>•</span>
            <span className="text-cyan-400 font-bold">{users.filter((u) => u.role !== 'admin').length} Regular</span>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-bgPanel border border-borderHairline space-y-2 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-1 h-full bg-redPrimary" />
          <div className="flex items-center justify-between">
            <span className="font-mono text-[11px] font-bold uppercase text-textMuted tracking-wider">Quarantine Mode</span>
            <Shield className={`w-4 h-4 ${quarantineStatus?.active ? 'text-redBright' : 'text-emerald-400'}`} />
          </div>
          <div className={`font-display text-2xl font-bold ${quarantineStatus?.active ? 'text-redBright' : 'text-emerald-400'}`}>
            {quarantineStatus?.active ? 'LOCKDOWN ACTIVE' : 'NOMINAL'}
          </div>
          <div className="font-mono text-[11px] text-textMuted">
            {quarantineStatus?.total_quarantined || 0} Isolated Samples
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-bgPanel border border-borderHairline space-y-2 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-1 h-full bg-amber-500" />
          <div className="flex items-center justify-between">
            <span className="font-mono text-[11px] font-bold uppercase text-textMuted tracking-wider">HITL Pending Queue</span>
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="font-display text-3xl font-bold text-amber-400">{hitlCases.length}</div>
          <div className="font-mono text-[11px] text-textMuted">Cases awaiting manual review</div>
        </div>

        <div className="p-5 rounded-2xl bg-bgPanel border border-borderHairline space-y-2 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-1 h-full bg-purple-500" />
          <div className="flex items-center justify-between">
            <span className="font-mono text-[11px] font-bold uppercase text-textMuted tracking-wider">Audit Trail Records</span>
            <Activity className="w-4 h-4 text-purple-400" />
          </div>
          <div className="font-display text-3xl font-bold text-purple-400">{auditEvents.length}</div>
          <div className="font-mono text-[11px] text-textMuted">Persisted security audit logs</div>
        </div>
      </div>

      {/* ── Navigation Tabs ──────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-2 border-b border-borderHairline pb-1 overflow-x-auto">
        <button
          onClick={() => setActiveTab('users')}
          className={`flex items-center gap-2 px-5 py-3 rounded-xl font-mono text-xs font-bold uppercase tracking-wider transition-all border ${
            activeTab === 'users'
              ? 'bg-redPrimary/20 border-redPrimary text-redBright shadow-red-glow'
              : 'bg-bgPanel border-borderHairline text-textMuted hover:text-white'
          }`}
        >
          <Users className="w-4 h-4" />
          User Governance ({users.length})
        </button>

        <button
          onClick={() => setActiveTab('security')}
          className={`flex items-center gap-2 px-5 py-3 rounded-xl font-mono text-xs font-bold uppercase tracking-wider transition-all border ${
            activeTab === 'security'
              ? 'bg-redPrimary/20 border-redPrimary text-redBright shadow-red-glow'
              : 'bg-bgPanel border-borderHairline text-textMuted hover:text-white'
          }`}
        >
          <Shield className="w-4 h-4" />
          AI Defense & HITL Controls ({hitlCases.length})
        </button>

        <button
          onClick={() => setActiveTab('audit')}
          className={`flex items-center gap-2 px-5 py-3 rounded-xl font-mono text-xs font-bold uppercase tracking-wider transition-all border ${
            activeTab === 'audit'
              ? 'bg-redPrimary/20 border-redPrimary text-redBright shadow-red-glow'
              : 'bg-bgPanel border-borderHairline text-textMuted hover:text-white'
          }`}
        >
          <Activity className="w-4 h-4" />
          Security Audit Feed ({auditEvents.length})
        </button>

        <button
          onClick={() => setActiveTab('registry')}
          className={`flex items-center gap-2 px-5 py-3 rounded-xl font-mono text-xs font-bold uppercase tracking-wider transition-all border ${
            activeTab === 'registry'
              ? 'bg-redPrimary/20 border-redPrimary text-redBright shadow-red-glow'
              : 'bg-bgPanel border-borderHairline text-textMuted hover:text-white'
          }`}
        >
          <Layers className="w-4 h-4" />
          Asset Scan Registry ({scanHistory.length})
        </button>
      </div>

      {/* ── Tab Content 1: User Governance ────────────────────────────────────────── */}
      {activeTab === 'users' && (
        <div className="space-y-6">
          <div className="p-6 rounded-2xl bg-bgPanel border border-borderHairline space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-borderHairline pb-4">
              <div>
                <h2 className="font-display text-xl font-bold text-white">Platform User Directory</h2>
                <p className="font-mono text-xs text-textMuted mt-1">
                  Manage platform authentication accounts, assign roles, and revoke privileges.
                </p>
              </div>
              <button
                onClick={() => setShowCreateModal(true)}
                className="flex items-center gap-2 bg-redPrimary text-white font-mono text-xs font-bold uppercase tracking-wider px-4 py-2.5 rounded-xl hover:bg-redBright transition-all shadow-red-glow self-start sm:self-auto"
              >
                <UserPlus className="w-4 h-4" />
                Provision Account
              </button>
            </div>

            {loadingUsers ? (
              <div className="py-16 text-center font-mono text-xs text-textMuted">Loading platform user registry…</div>
            ) : users.length === 0 ? (
              <div className="py-16 text-center font-mono text-xs text-textMuted">No user accounts found.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-borderHairline text-textMuted uppercase text-[10px] tracking-wider bg-bgVoid/50">
                      <th className="py-3.5 px-4">User Account</th>
                      <th className="py-3.5 px-4">Role Privilege</th>
                      <th className="py-3.5 px-4">User ID</th>
                      <th className="py-3.5 px-4">Provisioned At</th>
                      <th className="py-3.5 px-4 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-borderHairline">
                    {users.map((u) => {
                      const isSelf = u.id === activeUser?.id;
                      return (
                        <tr key={u.id} className="hover:bg-bgSurface/50 transition-colors">
                          <td className="py-4 px-4 font-semibold text-textPrimary">
                            <div className="flex items-center gap-3">
                              <div
                                className={`w-9 h-9 rounded-full flex items-center justify-center font-bold uppercase ${
                                  u.role === 'admin'
                                    ? 'bg-redPrimary/20 border border-redPrimary/40 text-redBright'
                                    : 'bg-cyan-500/20 border border-cyan-500/40 text-cyan-400'
                                }`}
                              >
                                {u.username.charAt(0)}
                              </div>
                              <div>
                                <div className="text-white font-bold text-sm flex items-center gap-2">
                                  {u.username}
                                  {isSelf && (
                                    <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-500/30">
                                      CURRENT SESSION
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                          </td>
                          <td className="py-4 px-4">
                            <span
                              className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                                u.role === 'admin'
                                  ? 'bg-redPrimary/20 text-redBright border border-redPrimary/40 shadow-red-glow'
                                  : 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                              }`}
                            >
                              {u.role === 'admin' ? 'ADMINISTRATOR' : 'REGULAR USER'}
                            </span>
                          </td>
                          <td className="py-4 px-4 text-textMuted font-mono text-[11px]">{u.id}</td>
                          <td className="py-4 px-4 text-textMuted text-[11px]">
                            {u.created_at ? new Date(u.created_at).toLocaleString() : 'N/A'}
                          </td>
                          <td className="py-4 px-4 text-right">
                            <button
                              disabled={isSelf}
                              onClick={() => handleDeleteUser(u.id, u.username)}
                              className={`p-2 rounded-lg border transition-all ${
                                isSelf
                                  ? 'opacity-30 cursor-not-allowed bg-bgPanelRaised border-borderHairline text-textMuted'
                                  : 'bg-bgPanelRaised hover:bg-redPrimary/20 text-textMuted hover:text-redBright border-borderHairline hover:border-redPrimary/40'
                              }`}
                              title={isSelf ? 'Cannot delete your own active session' : 'Revoke Account Access'}
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Tab Content 2: AI Security & Defense Controls ────────────────────────── */}
      {activeTab === 'security' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Emergency Lockdown Control */}
          <div className="p-6 rounded-2xl bg-bgPanel border border-redPrimary/40 shadow-red-glow space-y-6">
            <div className="flex items-center gap-3 text-redBright font-mono text-xs font-bold uppercase tracking-wider">
              <Lock className="w-5 h-5" />
              Emergency Administrative Override
            </div>
            <div>
              <h3 className="font-display font-bold text-white text-xl">Emergency System Quarantine</h3>
              <p className="text-xs font-sans text-textMuted mt-1 leading-relaxed">
                Instantly trigger a platform-wide security lockdown. Suspends unverified ingestion pipelines and quarantines suspect training data.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-bgVoid border border-borderHairline space-y-2">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-textMuted">Current Platform Status:</span>
                <span className={`font-bold ${quarantineStatus?.active ? 'text-redBright' : 'text-emerald-400'}`}>
                  {quarantineStatus?.active ? 'QUARANTINE ACTIVE' : 'NOMINAL OPERATION'}
                </span>
              </div>
              <div className="flex justify-between text-xs font-mono">
                <span className="text-textMuted">Quarantined Samples:</span>
                <span className="text-white font-bold">{quarantineStatus?.total_quarantined || 0}</span>
              </div>
            </div>

            <button
              onClick={handleTriggerQuarantine}
              disabled={quarantining}
              className="w-full py-4 px-6 rounded-xl bg-redPrimary hover:bg-redBright text-white font-mono text-xs font-bold uppercase tracking-wider transition-all shadow-red-glow disabled:opacity-50 flex items-center justify-center gap-2"
            >
              <AlertTriangle className="w-5 h-5" />
              {quarantining ? 'Executing Lockdown...' : 'Trigger System Quarantine Lockdown'}
            </button>
          </div>

          {/* Red Team Attack Simulator */}
          <div className="p-6 rounded-2xl bg-bgPanel border border-borderHairline space-y-6">
            <div className="flex items-center gap-3 text-purple-400 font-mono text-xs font-bold uppercase tracking-wider">
              <Radio className="w-5 h-5" />
              Adversarial Resilience Tester
            </div>
            <div>
              <h3 className="font-display font-bold text-white text-xl">Red Team Attack Simulation</h3>
              <p className="text-xs font-sans text-textMuted mt-1 leading-relaxed">
                Inject synthetic poisoning vectors into test workloads to evaluate 5-layer detection coverage and resilience scores.
              </p>
            </div>

            <div className="space-y-3">
              <label className="block font-mono text-[10px] font-bold text-textMuted uppercase tracking-wider">
                Select Attack Vector:
              </label>
              <select
                value={selectedAttack}
                onChange={(e) => setSelectedAttack(e.target.value)}
                className="w-full p-3 rounded-xl bg-bgVoid border border-borderHairline text-white font-mono text-xs focus:border-redPrimary outline-none"
              >
                <option value="label_flip">Label Flip (Class Boundary Corruption)</option>
                <option value="backdoor">Backdoor Trojan (Inference Trigger)</option>
                <option value="clean_label">Clean Label (Feature Collision Outliers)</option>
                <option value="gradient_poisoning">Gradient Poisoning (Federated Inversion)</option>
                <option value="boiling_frog">Boiling Frog (Slow Cumulative Drift)</option>
              </select>
            </div>

            <button
              onClick={handleRunRedTeam}
              disabled={runningSim}
              className="w-full py-3.5 px-6 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-mono text-xs font-bold uppercase tracking-wider transition-all shadow-lg disabled:opacity-50 flex items-center justify-center gap-2"
            >
              <Play className="w-4 h-4" />
              {runningSim ? 'Running Simulation...' : 'Execute Red Team Simulation'}
            </button>

            {simResult && (
              <div className="p-4 rounded-xl bg-bgVoid border border-purple-500/30 font-mono text-xs space-y-2">
                <div className="flex justify-between text-textMuted">
                  <span>Detected Status:</span>
                  <span className={simResult.detected ? 'text-emerald-400 font-bold' : 'text-redBright font-bold'}>
                    {simResult.detected ? 'CAUGHT BY SPECTRA' : 'MISSED'}
                  </span>
                </div>
                <div className="flex justify-between text-textMuted">
                  <span>Resilience Score:</span>
                  <span className="text-purple-400 font-bold">{simResult.resilience_score} / 10.0</span>
                </div>
                <div className="flex justify-between text-textMuted">
                  <span>Detection Speed:</span>
                  <span className="text-white">{simResult.detection_speed_ms} ms</span>
                </div>
              </div>
            )}
          </div>

          {/* Pending HITL Cases Table (Full Width) */}
          <div className="lg:col-span-2 p-6 rounded-2xl bg-bgPanel border border-borderHairline space-y-6">
            <div className="flex items-center justify-between border-b border-borderHairline pb-4">
              <div>
                <h3 className="font-display font-bold text-white text-lg">HITL Review Queue Escalations</h3>
                <p className="font-mono text-xs text-textMuted">Cases flagged by proxy disagreement requiring human review</p>
              </div>
              <span className="font-mono text-xs text-amber-400 bg-amber-950/40 px-3 py-1 rounded-full border border-amber-500/30 font-bold">
                {hitlCases.length} Pending
              </span>
            </div>

            {loadingHitl ? (
              <div className="py-12 text-center font-mono text-xs text-textMuted">Loading pending review cases…</div>
            ) : hitlCases.length === 0 ? (
              <div className="py-12 text-center font-mono text-xs text-textMuted">No pending HITL cases awaiting review.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-borderHairline text-textMuted uppercase text-[10px] tracking-wider">
                      <th className="py-3 px-4">Case ID</th>
                      <th className="py-3 px-4">Suspicion Score</th>
                      <th className="py-3 px-4">Samples</th>
                      <th className="py-3 px-4">Flagged At</th>
                      <th className="py-3 px-4 text-right">Admin Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-borderHairline">
                    {hitlCases.map((c) => (
                      <tr key={c.case_id} className="hover:bg-bgSurface/50 transition-colors">
                        <td className="py-4 px-4 font-bold text-white">{c.case_id.slice(0, 12)}…</td>
                        <td className="py-4 px-4 text-amber-400 font-bold">
                          {((c.suspicion_score || 0) * 100).toFixed(1)}%
                        </td>
                        <td className="py-4 px-4 text-textMuted">{c.n_samples}</td>
                        <td className="py-4 px-4 text-textMuted text-[11px]">
                          {c.created_at ? new Date(c.created_at).toLocaleString() : 'N/A'}
                        </td>
                        <td className="py-4 px-4 text-right space-x-2">
                          <button
                            onClick={() => handleResolveHitl(c.case_id, 'approve_quarantine')}
                            className="px-3 py-1.5 rounded-lg bg-redPrimary/20 text-redBright hover:bg-redPrimary hover:text-white border border-redPrimary/40 transition-all font-mono text-[10px] font-bold uppercase"
                          >
                            Approve Quarantine
                          </button>
                          <button
                            onClick={() => handleResolveHitl(c.case_id, 'mark_safe')}
                            className="px-3 py-1.5 rounded-lg bg-emerald-950 text-emerald-400 hover:bg-emerald-600 hover:text-white border border-emerald-500/30 transition-all font-mono text-[10px] font-bold uppercase"
                          >
                            Mark Safe
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Tab Content 3: System Security Audit Feed ────────────────────────────── */}
      {activeTab === 'audit' && (
        <div className="p-6 rounded-2xl bg-bgPanel border border-borderHairline space-y-6">
          <div className="flex items-center justify-between border-b border-borderHairline pb-4">
            <div>
              <h2 className="font-display text-xl font-bold text-white">Centralized System Security Audit Log</h2>
              <p className="font-mono text-xs text-textMuted mt-1">
                Persisted audit records for all user actions, security overrides, scans, and credential updates.
              </p>
            </div>
            <button
              onClick={fetchAuditEvents}
              className="p-2.5 rounded-xl bg-bgVoid border border-borderHairline text-textMuted hover:text-white transition-all font-mono text-xs"
            >
              <RefreshCw className={`w-4 h-4 ${loadingAudit ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {loadingAudit ? (
            <div className="py-16 text-center font-mono text-xs text-textMuted">Loading security audit feed…</div>
          ) : auditEvents.length === 0 ? (
            <div className="py-16 text-center font-mono text-xs text-textMuted">No security audit events recorded yet.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs border-collapse">
                <thead>
                  <tr className="border-b border-borderHairline text-textMuted uppercase text-[10px] tracking-wider bg-bgVoid/50">
                    <th className="py-3.5 px-4">Timestamp</th>
                    <th className="py-3.5 px-4">Actor ID</th>
                    <th className="py-3.5 px-4">Action</th>
                    <th className="py-3.5 px-4">Resource</th>
                    <th className="py-3.5 px-4 text-right">Details</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-borderHairline">
                  {auditEvents.map((evt) => (
                    <tr key={evt.id} className="hover:bg-bgSurface/50 transition-colors">
                      <td className="py-4 px-4 text-textMuted text-[11px]">
                        {evt.created_at ? new Date(evt.created_at).toLocaleString() : 'N/A'}
                      </td>
                      <td className="py-4 px-4 font-bold text-cyan-400">{evt.actor_id}</td>
                      <td className="py-4 px-4 font-bold text-white">
                        <span className="px-2.5 py-1 rounded bg-bgVoid border border-borderHairline">
                          {evt.action}
                        </span>
                      </td>
                      <td className="py-4 px-4 text-textMuted uppercase text-[10px]">{evt.resource_type}</td>
                      <td className="py-4 px-4 text-right">
                        <button
                          onClick={() => setSelectedEvent(evt)}
                          className="p-2 rounded-lg bg-bgPanelRaised hover:bg-redPrimary/20 text-textMuted hover:text-white border border-borderHairline transition-all inline-flex items-center gap-1 text-[11px]"
                        >
                          <Eye className="w-3.5 h-3.5" /> View JSON
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ── Tab Content 4: Global Scan Registry ───────────────────────────────────── */}
      {activeTab === 'registry' && (
        <div className="p-6 rounded-2xl bg-bgPanel border border-borderHairline space-y-6">
          <div className="flex items-center justify-between border-b border-borderHairline pb-4">
            <div>
              <h2 className="font-display text-xl font-bold text-white">Global Asset & Model Scan Registry</h2>
              <p className="font-mono text-xs text-textMuted mt-1">
                Centralized registry of all scanned CSV datasets and Scikit-Learn `.pkl` models across all users.
              </p>
            </div>
            <button
              onClick={fetchScanRegistry}
              className="p-2.5 rounded-xl bg-bgVoid border border-borderHairline text-textMuted hover:text-white transition-all font-mono text-xs"
            >
              <RefreshCw className={`w-4 h-4 ${loadingRegistry ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {loadingRegistry ? (
            <div className="py-16 text-center font-mono text-xs text-textMuted">Loading scan registry…</div>
          ) : scanHistory.length === 0 ? (
            <div className="py-16 text-center font-mono text-xs text-textMuted">No scan records stored in SQLite database.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs border-collapse">
                <thead>
                  <tr className="border-b border-borderHairline text-textMuted uppercase text-[10px] tracking-wider bg-bgVoid/50">
                    <th className="py-3.5 px-4">Asset / Filename</th>
                    <th className="py-3.5 px-4">Source Type</th>
                    <th className="py-3.5 px-4">Verdict</th>
                    <th className="py-3.5 px-4">Suspicion Score</th>
                    <th className="py-3.5 px-4">Scan Date</th>
                    <th className="py-3.5 px-4 text-right">Admin Purge</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-borderHairline">
                  {scanHistory.map((item) => (
                    <tr key={item.id} className="hover:bg-bgSurface/50 transition-colors">
                      <td className="py-4 px-4 font-bold text-white">
                        {item.filename || item.model_filename || item.id.slice(0, 8)}
                      </td>
                      <td className="py-4 px-4 text-textMuted uppercase text-[10px]">{item.source || 'scan'}</td>
                      <td className="py-4 px-4">
                        <span
                          className={`px-2.5 py-1 rounded text-[10px] font-bold uppercase ${
                            item.verdict === 'POISONED' || item.verdict === 'SUSPECT'
                              ? 'bg-redPrimary/20 text-redBright border border-redPrimary/40'
                              : 'bg-emerald-950 text-emerald-400 border border-emerald-500/30'
                          }`}
                        >
                          {item.verdict || 'CLEAN'}
                        </span>
                      </td>
                      <td className="py-4 px-4 font-bold text-white">
                        {item.score !== undefined && item.score !== null ? `${(item.score * 100).toFixed(1)}%` : 'N/A'}
                      </td>
                      <td className="py-4 px-4 text-textMuted text-[11px]">
                        {item.created_at ? new Date(item.created_at).toLocaleString() : 'N/A'}
                      </td>
                      <td className="py-4 px-4 text-right">
                        <button
                          onClick={() => handleDeleteScanRecord(item.id)}
                          className="p-2 rounded-lg bg-bgPanelRaised hover:bg-redPrimary/20 text-textMuted hover:text-redBright border border-borderHairline transition-all"
                          title="Purge Scan Record"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ── Provision User Modal ─────────────────────────────────────────────────── */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
          <div className="w-full max-w-md p-6 rounded-2xl bg-bgPanel border border-borderHairline shadow-2xl space-y-6 relative">
            <button
              onClick={() => setShowCreateModal(false)}
              className="absolute top-4 right-4 text-textMuted hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-redPrimary/20 border border-redPrimary/40 flex items-center justify-center text-redBright">
                <UserPlus className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-display font-bold text-white text-lg">Provision New Platform Account</h3>
                <p className="font-mono text-xs text-textMuted">Create regular user or administrator credentials</p>
              </div>
            </div>

            <form onSubmit={handleCreateUser} className="space-y-4">
              <div>
                <label className="block font-mono text-[10px] font-bold text-textMuted uppercase tracking-wider mb-1">
                  Username
                </label>
                <input
                  type="text"
                  required
                  value={newUsername}
                  onChange={(e) => setNewUsername(e.target.value)}
                  placeholder="Enter username (min 3 chars)"
                  className="w-full p-3 rounded-xl bg-bgVoid border border-borderHairline text-white font-mono text-xs focus:border-redPrimary outline-none"
                />
              </div>

              <div>
                <label className="block font-mono text-[10px] font-bold text-textMuted uppercase tracking-wider mb-1">
                  Password
                </label>
                <input
                  type="password"
                  required
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="Enter password (min 4 chars)"
                  className="w-full p-3 rounded-xl bg-bgVoid border border-borderHairline text-white font-mono text-xs focus:border-redPrimary outline-none"
                />
              </div>

              <div>
                <label className="block font-mono text-[10px] font-bold text-textMuted uppercase tracking-wider mb-1">
                  Role Privilege
                </label>
                <select
                  value={newRole}
                  onChange={(e) => setNewRole(e.target.value)}
                  className="w-full p-3 rounded-xl bg-bgVoid border border-borderHairline text-white font-mono text-xs focus:border-redPrimary outline-none"
                >
                  <option value="user">Regular User (Standard Access)</option>
                  <option value="admin">Administrator (Full Access & Overrides)</option>
                </select>
              </div>

              <div className="flex items-center gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="flex-1 py-3 px-4 rounded-xl bg-bgVoid hover:bg-bgSurface text-textMuted font-mono text-xs font-bold uppercase transition-all"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submittingUser}
                  className="flex-1 py-3 px-4 rounded-xl bg-redPrimary hover:bg-redBright text-white font-mono text-xs font-bold uppercase tracking-wider transition-all shadow-red-glow disabled:opacity-50"
                >
                  {submittingUser ? 'Saving...' : 'Provision User'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ── JSON Audit Detail Modal ──────────────────────────────────────────────── */}
      {selectedEvent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
          <div className="w-full max-w-lg p-6 rounded-2xl bg-bgPanel border border-borderHairline shadow-2xl space-y-4 relative">
            <button
              onClick={() => setSelectedEvent(null)}
              className="absolute top-4 right-4 text-textMuted hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>

            <h3 className="font-display font-bold text-white text-base">Audit Event Details</h3>
            <pre className="p-4 rounded-xl bg-bgVoid border border-borderHairline font-mono text-xs text-cyan-400 overflow-x-auto max-h-80">
              {JSON.stringify(selectedEvent, null, 2)}
            </pre>

            <button
              onClick={() => setSelectedEvent(null)}
              className="w-full py-3 rounded-xl bg-bgVoid hover:bg-bgSurface text-white font-mono text-xs font-bold uppercase"
            >
              Close
            </button>
          </div>
        </div>
      )}

    </div>
  );
}
