// Line icons for the nav, drawn for Veritas on a 24 × 24 grid (stroke and size are set in GlassNav.css). Every stroke
// has pathLength 1 and a dash of 1, so the scene can erase and redraw it by moving its dash offset (1 = erased).
const draw = { pathLength: 1, style: { strokeDasharray: '1' } };

export const ICONS = {
  // Check data: a tray with an arrow going in
  upload: (
    <svg viewBox="0 0 24 24">
      <path d="M4 14.5v3.5a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3.5" {...draw} />
      <path d="M12 15V4" {...draw} />
      <path d="M7.5 8.5 12 4l4.5 4.5" {...draw} />
    </svg>
  ),
  // History: a clock face with a turning-back arrow
  history: (
    <svg viewBox="0 0 24 24">
      <path d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1" {...draw} />
      <path d="M3.5 4.5v4h4" {...draw} />
      <path d="M12 8v4.5l3 2" {...draw} />
    </svg>
  ),
  // Scanner: a viewfinder with a scan line
  scanner: (
    <svg viewBox="0 0 24 24">
      <path d="M4 8.5V5a1 1 0 0 1 1-1h3.5" {...draw} />
      <path d="M15.5 4H19a1 1 0 0 1 1 1v3.5" {...draw} />
      <path d="M20 15.5V19a1 1 0 0 1-1 1h-3.5" {...draw} />
      <path d="M8.5 20H5a1 1 0 0 1-1-1v-3.5" {...draw} />
      <path d="M7.5 12h9" {...draw} />
    </svg>
  ),
  // Reports: a page with a folded corner and two lines
  reports: (
    <svg viewBox="0 0 24 24">
      <path d="M14 3.5H6.5a1 1 0 0 0-1 1v15a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1V8z" {...draw} />
      <path d="M14 3.5V8h4.5" {...draw} />
      <path d="M9 12.5h6" {...draw} />
      <path d="M9 16h6" {...draw} />
    </svg>
  ),
  // Benchmarks: three measured bars on a baseline
  benchmarks: (
    <svg viewBox="0 0 24 24">
      <path d="M4 20h16" {...draw} />
      <path d="M7 16.5V12" {...draw} />
      <path d="M12 16.5V6.5" {...draw} />
      <path d="M17 16.5V9.5" {...draw} />
    </svg>
  ),
  // Admin console: two sliders
  admin: (
    <svg viewBox="0 0 24 24">
      <path d="M4 7.5h9" {...draw} />
      <path d="M19 7.5h1" {...draw} />
      <circle cx="16" cy="7.5" r="2.5" {...draw} />
      <path d="M4 16.5h1" {...draw} />
      <path d="M11 16.5h9" {...draw} />
      <circle cx="8" cy="16.5" r="2.5" {...draw} />
    </svg>
  ),
};
