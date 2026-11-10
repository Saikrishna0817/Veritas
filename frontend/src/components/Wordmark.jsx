/** The mark: rows as overlapping rings on a slide, one of them flagged. */
export function LogoMark({ size = 26 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 26 26" fill="none" aria-hidden>
      {[[13, 8.5], [9, 11], [17, 11], [9, 15.5], [17, 15.5], [13, 18]].map(([x, y]) => (
        <circle key={`${x}-${y}`} cx={x} cy={y} r="5" stroke="#fff" strokeOpacity="0.85" strokeWidth="0.9" />
      ))}
      <circle cx="13" cy="13.2" r="2" fill="#FF6A4D" />
    </svg>
  );
}

export default function Wordmark({ size = 19 }) {
  return (
    <span className="flex items-center gap-2.5">
      <LogoMark />
      <span className="font-display font-medium lowercase text-textPrimary" style={{ fontSize: size }}>veritas</span>
    </span>
  );
}
