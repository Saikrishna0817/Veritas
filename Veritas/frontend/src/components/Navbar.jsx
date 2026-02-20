export default function Navbar({ title = 'AI Trust Forensics' }) {
  return (
    <header className="w-full px-6 py-4 border-b border-border bg-bg2">
      <div className="font-mono text-sm text-text2 tracking-widest uppercase">{title}</div>
    </header>
  );
}

