import { useEffect, useRef, useSyncExternalStore } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { createEngine } from './engine';
import { MOBILE_QUERY, PAD, measure } from './geometry';
import { ICONS } from './icons';
import { createScene } from './scene';
import './GlassNav.css';

const mq = window.matchMedia(MOBILE_QUERY);
const subscribe = (cb) => {
  mq.addEventListener('change', cb);
  return () => mq.removeEventListener('change', cb);
};
const getMode = () => (mq.matches ? 'mobile' : 'desktop');

/** The tab for a path: exact, or a sub-path (/history/42 is History). -1 when none matches (Overview). */
const activeIndex = (links, path) => links.findIndex(({ to }) => path === to || path.startsWith(`${to}/`));

/**
 * The app's main navigation. Real links (NavLink: aria-current, keyboard, middle-click) laid out by CSS in a glass
 * track; a canvas behind them draws the glass and the bead, which follows the current route. Desktop: inline in the
 * top bar. Mobile (≤ 820 px): a floating bar at the bottom of the screen. Without WebGL2 the same links get a plain
 * CSS style (.gn-flat).
 * links: [{ to, label, icon }] where icon is a key of ICONS.
 */
export default function GlassNav({ links }) {
  const mode = useSyncExternalStore(subscribe, getMode);
  const { pathname } = useLocation();
  const active = activeIndex(links, pathname);
  const pad = PAD[mode];
  const key = links.map((l) => l.to).join('|');

  const nav = useRef(null);
  const track = useRef(null);
  const canvas = useRef(null);
  const tabs = useRef([]);
  const icons = useRef([]);
  const labels = useRef([]);
  const engine = useRef(null);

  useEffect(() => {
    const n = links.length;
    const parts = { canvas: canvas.current, mode, tabs: tabs.current.slice(0, n), icons: icons.current.slice(0, n), labels: labels.current.slice(0, n) };
    const root = nav.current;
    let scene;
    try {
      scene = createScene(parts);
    } catch (error) {
      console.warn('Glass nav: WebGL2 unavailable, using the plain style.', error);
      root.classList.add('gn-flat');
      return () => root.classList.remove('gn-flat');
    }
    let dead = false;
    const eng = createEngine(scene);
    engine.current = eng;
    // the element is kept from here: when the nav leaves the page (signing out) React clears the ref before this
    // effect's cleanup runs, and a resize notice arriving in between must not measure null
    const trackEl = track.current;
    const relayout = () => !dead && trackEl.isConnected && eng.layout(measure(trackEl, parts.tabs, PAD[mode]));
    relayout();
    const ro = new ResizeObserver(relayout);
    ro.observe(trackEl);
    document.fonts?.ready.then(relayout);
    const onLost = (e) => {
      e.preventDefault();
      eng.stop();
      root.classList.add('gn-flat');
    };
    parts.canvas.addEventListener('webglcontextlost', onLost);

    return () => {
      dead = true;
      ro.disconnect();
      eng.stop();
      scene.dispose();
      parts.canvas.removeEventListener('webglcontextlost', onLost);
      root.classList.remove('gn-flat');
      engine.current = null;
    };
  }, [mode, key, links.length]);

  // after the scene effect, so a rebuilt scene (mode change) gets the current tab too
  useEffect(() => {
    engine.current?.goTo(active);
  }, [active, mode, key]);

  return (
    <nav ref={nav} className={`gn gn-${mode}`} aria-label="Main">
      <div ref={track} className="gn-track">
        <canvas
          ref={canvas}
          className="gn-canvas"
          aria-hidden="true"
          style={{ left: -pad.l, top: -pad.t, width: `calc(100% + ${pad.l + pad.r}px)`, height: `calc(100% + ${pad.t + pad.b}px)` }}
        />
        {links.map(({ to, label, icon }, i) => (
          <NavLink key={to} to={to} ref={(el) => void (tabs.current[i] = el)} className="gn-tab">
            <span ref={(el) => void (icons.current[i] = el)} className="gn-icon" aria-hidden="true">{ICONS[icon]}</span>
            <span ref={(el) => void (labels.current[i] = el)} className="gn-label">{label}</span>
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
