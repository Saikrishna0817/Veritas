export default function Loader({ label = 'Loading…' }) {
  return (
    <div className="flex items-center gap-2 font-mono text-xs text-text3">
      <span className="inline-block w-2 h-2 rounded-full bg-accent animate-pulse" />
      {label}
    </div>
  );
}

