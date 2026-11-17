import { Component, lazy, Suspense, useEffect, useState } from 'react';

/**
 * The 3D world behind the whole app (sign-in pages included): mounted once, it persists across route changes.
 * three.js and the scene load in their own chunk only after the page's content has rendered and the browser is
 * idle, so the first paint and the forms never wait for WebGL. Without WebGL2, after a lost context or a failed
 * load the layer renders nothing and the static warm background in argilla.css shows instead.
 */
const World = lazy(() => import('./World'));

const hasWebGL2 = () => {
  try {
    return typeof window !== 'undefined' && !!window.WebGL2RenderingContext && !!document.createElement('canvas').getContext('webgl2');
  } catch {
    return false;
  }
};

class Quiet extends Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error) { console.warn('3D scene unavailable, using the static background.', error); }
  render() { return this.state.failed ? null : this.props.children; }
}

export default function SceneLayer() {
  const [ready, setReady] = useState(false);
  const [lost, setLost] = useState(false);

  useEffect(() => {
    if (!hasWebGL2()) return undefined;
    let id;
    const start = () => {
      id = window.requestIdleCallback ? window.requestIdleCallback(() => setReady(true), { timeout: 1500 }) : window.setTimeout(() => setReady(true), 300);
    };
    if (document.readyState === 'complete') start();
    else window.addEventListener('load', start, { once: true });
    return () => {
      window.removeEventListener('load', start);
      if (window.cancelIdleCallback && id) window.cancelIdleCallback(id);
      else window.clearTimeout(id);
    };
  }, []);

  if (!ready || lost) return null;
  return (
    <div className="ar-scene" aria-hidden="true">
      <Quiet>
        <Suspense fallback={null}>
          <World onLost={() => setLost(true)} />
        </Suspense>
      </Quiet>
    </div>
  );
}
