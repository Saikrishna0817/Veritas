// Sizes of the glass nav, in CSS px. The track (the glass bar) is a real element laid out by CSS; the canvas sits over
// it with this much room on each side for the bead, its bulge and the shadow. Everything the shader draws is placed
// from measurements of the track and its tabs (measure()), so the numbers here are the only fixed ones.

/** Below this width the nav is a floating bar at the bottom of the screen (matches the 820 px breakpoint in GlassNav.css). */
export const MOBILE_QUERY = '(max-width: 820px)';

/** Room around the track that the canvas covers: top, right, bottom, left. */
export const PAD = {
  desktop: { t: 14, r: 14, b: 16, l: 14 },
  mobile: { t: 48, r: 16, b: 18, l: 16 },
};

/**
 * The bead's shape for a mode, from the motion state: centre (x, y), half-size (hx, hy) and the smin melt k.
 * Desktop: a capsule behind the active tab that swells 4 px past the track's top and bottom edges when up and thins
 * to a 12 px sliver inside the track when sunk. Mobile: a 22 px round bead that rises 30 px out of the bar's top edge.
 */
export function beadShape(mode, s, L) {
  const tab = { x: s.x, w: s.w };
  if (mode === 'desktop') {
    const hy = 12 + (L.track.hy + 4 - 12) * s.rise;
    const hx = (tab.w / 2 - 6 + 10 * s.rise) * s.sx;
    return { x: tab.x, y: L.track.cy, hx, hy, k: 8 };
  }
  const r = 22, squash = 1 + (s.sx - 1) * 0.35;
  return { x: tab.x, y: L.track.cy - 10 - 30 * s.rise, hx: r * s.sx, hy: r / squash, k: 18 };
}

/** Measures the track and its tabs in canvas px (the canvas starts `pad` above and left of the track). */
export function measure(track, tabs, pad) {
  const tr = track.getBoundingClientRect();
  return {
    w: tr.width + pad.l + pad.r,
    h: tr.height + pad.t + pad.b,
    track: { cx: pad.l + tr.width / 2, cy: pad.t + tr.height / 2, hx: tr.width / 2, hy: tr.height / 2 },
    tabs: tabs.map((el) => {
      const r = el.getBoundingClientRect();
      return { x: pad.l + r.left - tr.left + r.width / 2, w: r.width };
    }),
  };
}
