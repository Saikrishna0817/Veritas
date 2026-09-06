import { Link } from 'react-router-dom';
import { Lock, Shield, Activity, ExternalLink, Cpu, CheckCircle2 } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="w-full bg-bgVoid border-t border-borderHairline relative z-20 text-textSecondary font-sans">
      {/* Upper Main Footer Grid */}
      <div className="max-w-7xl mx-auto px-6 md:px-12 py-16">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-12">
          
          {/* Brand & Mission (2 cols on lg) */}
          <div className="lg:col-span-2 space-y-6">
            <Link to="/" className="inline-flex items-center gap-3 group">
              <div className="w-9 h-9 rounded-xl bg-redPrimary/10 border border-redPrimary/20 flex items-center justify-center group-hover:bg-redPrimary/20 group-hover:border-redPrimary/40 transition-all">
                <Lock className="w-4 h-4 text-redPrimary group-hover:scale-110 transition-transform" />
              </div>
              <span className="font-display font-bold text-xl tracking-tight text-white group-hover:text-redBright transition-colors">
                SPECTRA
              </span>
            </Link>

            <p className="text-sm text-textMuted leading-relaxed max-w-sm">
              Enterprise AI Security & Data Poisoning Defense Platform. Delivering continuous algorithmic integrity, adversarial data detection, and proxy impact forensics across deep learning pipelines.
            </p>

            {/* Live Telemetry Status Pill */}
            <div className="inline-flex items-center gap-3 px-3.5 py-2 rounded-xl bg-bgPanel border border-borderHairline">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
              </span>
              <div className="flex flex-col">
                <span className="font-mono text-[10px] font-bold text-white uppercase tracking-wider">
                  All Systems Operational
                </span>
                <span className="font-mono text-[9px] text-textMuted">
                  Cluster us-east-spectra • Latency 16ms
                </span>
              </div>
            </div>
          </div>

          {/* Column 1: Core Platform */}
          <div className="space-y-4">
            <h4 className="font-mono text-xs font-bold uppercase tracking-widest text-textPrimary">
              Platform Modules
            </h4>
            <ul className="space-y-2.5 text-sm font-medium">
              <li>
                <Link to="/" className="text-textMuted hover:text-redBright transition-colors">
                  Trust Dashboard
                </Link>
              </li>
              <li>
                <Link to="/upload" className="text-textMuted hover:text-redBright transition-colors">
                  Dataset Upload & Scan
                </Link>
              </li>
              <li>
                <Link to="/model-scan" className="text-textMuted hover:text-redBright transition-colors">
                  Model Weight Scanner
                </Link>
              </li>
              <li>
                <Link to="/forensics" className="text-textMuted hover:text-redBright transition-colors">
                  Spectral Poison Forensics
                </Link>
              </li>
              <li>
                <Link to="/blue-team" className="text-textMuted hover:text-redBright transition-colors">
                  Blue Team SOC Hub
                </Link>
              </li>
              <li>
                <Link to="/reports" className="text-textMuted hover:text-redBright transition-colors">
                  Evidence Reports
                </Link>
              </li>
              <li>
                <Link to="/history" className="text-textMuted hover:text-redBright transition-colors">
                  Audit History
                </Link>
              </li>
            </ul>
          </div>

          {/* Column 2: Governance & Frameworks */}
          <div className="space-y-4">
            <h4 className="font-mono text-xs font-bold uppercase tracking-widest text-textPrimary">
              AI Governance
            </h4>
            <ul className="space-y-2.5 text-sm font-medium">
              <li className="flex items-center gap-2 text-textMuted hover:text-white transition-colors cursor-default">
                <CheckCircle2 className="w-3.5 h-3.5 text-redPrimary shrink-0" />
                <span>NIST AI RMF 1.0 (MAP 1.5)</span>
              </li>
              <li className="flex items-center gap-2 text-textMuted hover:text-white transition-colors cursor-default">
                <CheckCircle2 className="w-3.5 h-3.5 text-redPrimary shrink-0" />
                <span>EU AI Act (Articles 9 & 17)</span>
              </li>
              <li className="flex items-center gap-2 text-textMuted hover:text-white transition-colors cursor-default">
                <CheckCircle2 className="w-3.5 h-3.5 text-redPrimary shrink-0" />
                <span>MITRE ATLAS (AML.T0018)</span>
              </li>
              <li className="flex items-center gap-2 text-textMuted hover:text-white transition-colors cursor-default">
                <CheckCircle2 className="w-3.5 h-3.5 text-redPrimary shrink-0" />
                <span>ISO/IEC 42001 Certified</span>
              </li>
              <li className="flex items-center gap-2 text-textMuted hover:text-white transition-colors cursor-default">
                <CheckCircle2 className="w-3.5 h-3.5 text-redPrimary shrink-0" />
                <span>OWASP Top 10 for LLMs</span>
              </li>
              <li className="flex items-center gap-2 text-textMuted hover:text-white transition-colors cursor-default">
                <CheckCircle2 className="w-3.5 h-3.5 text-redPrimary shrink-0" />
                <span>STIX 2.1 Threat Export</span>
              </li>
            </ul>
          </div>

          {/* Column 3: Security & Operations */}
          <div className="space-y-4">
            <h4 className="font-mono text-xs font-bold uppercase tracking-widest text-textPrimary">
              Security Operations
            </h4>
            <ul className="space-y-2.5 text-sm font-medium">
              <li>
                <span className="text-textMuted hover:text-white transition-colors cursor-default">
                  Multi-Layer Spectral Scoring
                </span>
              </li>
              <li>
                <span className="text-textMuted hover:text-white transition-colors cursor-default">
                  Causal Blast Radius Engine
                </span>
              </li>
              <li>
                <span className="text-textMuted hover:text-white transition-colors cursor-default">
                  Automated Pipeline Quarantine
                </span>
              </li>
              <li>
                <span className="text-textMuted hover:text-white transition-colors cursor-default">
                  Human-in-the-Loop Reviews
                </span>
              </li>
              <li>
                <span className="text-textMuted hover:text-white transition-colors cursor-default">
                  Red Team Infiltration Sandbox
                </span>
              </li>
              <li>
                <Link to="/admin" className="text-redBright hover:underline transition-colors flex items-center gap-1.5 font-mono text-xs">
                  <span>Admin Console</span>
                  <ExternalLink className="w-3 h-3" />
                </Link>
              </li>
            </ul>
          </div>

        </div>
      </div>

      {/* Middle Telemetry Ribbon */}
      <div className="border-t border-b border-borderHairline bg-bgPanel/60 py-3 px-6 md:px-12">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4 font-mono text-[11px] text-textMuted">
          <div className="flex items-center gap-6 flex-wrap">
            <span className="flex items-center gap-2">
              <Cpu className="w-3.5 h-3.5 text-redPrimary" />
              <span>CORE ENGINE: <strong className="text-textPrimary">v2.2-enterprise</strong></span>
            </span>
            <span>ALGORITHM: <strong className="text-textPrimary">L1-L5 Spectral + Counterfactual</strong></span>
            <span>INGESTION: <strong className="text-textPrimary">Kafka / REST / CSV</strong></span>
            <span>CIPHER: <strong className="text-textPrimary">AES-256-GCM / TLS 1.3</strong></span>
          </div>
          <div className="text-textMuted">
            PLATFORM ID: <span className="text-textPrimary">SPECTRA-SEC-9982</span>
          </div>
        </div>
      </div>

      {/* Bottom Copyright & Legal Links */}
      <div className="max-w-7xl mx-auto px-6 md:px-12 py-6 flex flex-col md:flex-row items-center justify-between gap-4 text-xs font-mono text-textMuted">
        <div>
          © {new Date().getFullYear()} SPECTRA Systems Inc. All rights reserved. For authorized analyst & administrator access only.
        </div>
        <div className="flex items-center gap-6 flex-wrap">
          <span className="hover:text-textPrimary transition-colors cursor-pointer">Security Whitepaper</span>
          <span>•</span>
          <span className="hover:text-textPrimary transition-colors cursor-pointer">Privacy Policy</span>
          <span>•</span>
          <span className="hover:text-textPrimary transition-colors cursor-pointer">Terms of Service</span>
          <span>•</span>
          <span className="hover:text-textPrimary transition-colors cursor-pointer">Responsible Disclosure</span>
          <span>•</span>
          <span className="text-redPrimary/90 font-bold">SOC 2 Type II</span>
        </div>
      </div>
    </footer>
  );
}
