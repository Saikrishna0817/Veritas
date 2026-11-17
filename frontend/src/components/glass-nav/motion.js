// The bead's motion: a move goes from where the bead is (a snapshot of x and width) to a tab, or appears at a tab,
// or sinks away when no tab is active. state(move, t, tabs) gives everything one frame needs.

export const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
export const lerp = (a, b, u) => a + (b - a) * u;
/** Quintic ease-out. */
export const eOut = (u) => 1 - Math.pow(1 - clamp(u), 5);
/** Cubic ease-in-out. */
export const eIO = (u) => {
  u = clamp(u);
  return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
};

/** A move's length for a jump of n tabs: 1 tab 0.5 s, 2 – 3 tabs 0.6 s, 4+ tabs 0.7 s. Appearing or sinking: 0.4 s. */
export const moveLength = (n) => (n <= 1 ? 0.5 : n <= 3 ? 0.6 : 0.7);
export const FADE = 0.4;
/** The light sweep starts at 90% of a move and lasts this long; the move is over (no more frames) after it. */
export const SWEEP = 0.7;
export const moveEnd = (m) => m.ts + Math.max(m.d * 0.9 + SWEEP, m.d + 0.6);

/**
 * The state at time t (s). A move is { ts, d, from: { x, w } | null, to: index | -1, n }: `from` null means the bead
 * appears at `to`; `to` -1 means it sinks where it is. x, w: the bead's centre and tab width (px); rise 1 = up,
 * ~0.1 = sunk; sx: horizontal stretch; on: presence 0 – 1; tap: [x, age s] of the ripple (age < 0: none); sw: sweep
 * 0 – 1; cur: the target tab (-1 none); draw: how much of the target icon's strokes are drawn.
 */
export function state(m, t, tabs) {
  const k = clamp((t - m.ts) / m.d), u = eIO(k), since = t - m.ts - m.d;
  const bounce = k >= 1 ? 0.05 * Math.exp(-since * 9) * Math.sin(since * 22) : 0;
  const s = { x: 0, w: 0, rise: 1, sx: 1, on: 1, tap: [0, -1], sw: 0, cur: m.to, draw: 1 };

  if (m.to < 0) {                                   // no tab active: the bead sinks where it is and fades out
    Object.assign(s, { x: m.from.x, w: m.from.w, rise: 1 - u, on: 1 - u });
    return s;
  }
  const tab = tabs[m.to];
  if (!m.from) {                                    // nothing was active: the bead rises at the tab
    Object.assign(s, { x: tab.x, w: tab.w, rise: eOut(k) + bounce, on: eOut(k * 1.4) });
  } else {                                          // tab to tab: sink, travel as a stretched wave, resurface
    const dive = Math.sin(Math.PI * Math.min(1, k * 1.08));
    s.x = lerp(m.from.x, tab.x, u);
    s.w = lerp(m.from.w, tab.w, u);
    s.rise = k >= 1 ? 1 + bounce : 1 - 0.9 * Math.pow(dive, 0.7);
    s.sx = 1 + 0.45 * Math.sin(Math.PI * k) * (m.n > 1 ? 1.25 : 1);
  }
  s.tap = [tab.x, t - m.ts];
  s.sw = clamp((t - m.ts - m.d * 0.9) / SWEEP);
  // the target icon's strokes erase during the second quarter, then redraw from the middle on
  s.draw = m.from ? (k < 0.5 ? 1 - eIO((k - 0.25) / 0.25) : eOut((t - m.ts - m.d * 0.5) / 0.6)) : eOut(k / 0.8);
  return s;
}

/** A settled state at tab i (or nothing, i = -1): for the first frame and reduced motion. */
export const still = (i, tabs) => (i < 0
  ? { x: 0, w: 0, rise: 0, sx: 1, on: 0, tap: [0, -1], sw: 0, cur: -1, draw: 1 }
  : { x: tabs[i].x, w: tabs[i].w, rise: 1, sx: 1, on: 1, tap: [0, -1], sw: 0, cur: i, draw: 1 });
