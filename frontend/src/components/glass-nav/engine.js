import { FADE, moveEnd, moveLength, state, still } from './motion';

/**
 * Drives the scene. It draws only when something changes: a move runs a frame loop until its bounce and sweep are
 * over, then stops; a layout change draws one frame. goTo(i) moves the bead to tab i (-1: no tab), starting from
 * wherever the bead is, even mid-move. Reduced motion: the bead is placed at once, settled.
 */
export function createEngine(scene) {
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const now = () => performance.now() / 1000;
  let L = null, move = null, last = null, cur = -1, raf = 0;

  const frame = () => {
    raf = 0;
    if (!L) return;
    const t = now();
    last = move ? state(move, t, L.tabs) : still(cur, L.tabs);
    scene.draw(last, t);
    if (move && t < moveEnd(move)) raf = requestAnimationFrame(frame);
    else if (move) { move = null; last = still(cur, L.tabs); scene.draw(last, t); }
  };
  const kick = () => { if (!raf) raf = requestAnimationFrame(frame); };

  return {
    layout(next) {
      L = next;
      scene.layout(L);
      if (!move) last = still(cur, L.tabs);
      kick();
    },
    goTo(i) {
      if (i === cur) return;
      cur = i;
      // start from where the bead is now (mid-move included); nothing visible means it appears at the new tab
      const from = last && last.on > 0.05 ? { x: last.x, w: last.w } : null;
      if (reduce || !L || (i < 0 && !from)) {
        move = null;
        if (L) last = still(cur, L.tabs);
        kick();
        return;
      }
      const n = from && i >= 0 ? Math.round(Math.abs(L.tabs[i].x - from.x) / L.tabs[i].w) : 0;
      move = { ts: now(), d: i < 0 || !from ? FADE : moveLength(Math.max(1, n)), from, to: i, n };
      kick();
    },
    stop() { cancelAnimationFrame(raf); raf = 0; },
  };
}
