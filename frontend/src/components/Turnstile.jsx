/* eslint-disable react-refresh/only-export-components */
import { useEffect, useRef } from 'react';

/**
 * Cloudflare Turnstile, the human check the API asks for on public forms when VERITAS_TURNSTILE_SECRET is set.
 * Shown only when the build has VITE_TURNSTILE_SITE_KEY; without it nothing renders and no script is loaded.
 *
 *   <Turnstile onToken={setToken} reset={attempt} />
 *
 * onToken gets the token, or '' when it expires or the check fails. Each token works once, so the parent bumps
 * `reset` after every submit and the widget issues a new one.
 */
export const TURNSTILE_SITE_KEY = import.meta.env.VITE_TURNSTILE_SITE_KEY || '';
const SCRIPT = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';

let loading = null;
function loadScript() {
  if (window.turnstile) return Promise.resolve(window.turnstile);
  if (!loading) {
    loading = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = SCRIPT; s.async = true; s.defer = true;
      s.onload = () => resolve(window.turnstile);
      s.onerror = () => { loading = null; reject(new Error('The human check could not be loaded.')); };
      document.head.appendChild(s);
    });
  }
  return loading;
}

export default function Turnstile({ onToken, onError, reset = 0 }) {
  const box = useRef(null);
  const widget = useRef(null);
  const handlers = useRef({ onToken, onError });
  useEffect(() => { handlers.current = { onToken, onError }; });

  useEffect(() => {
    if (!TURNSTILE_SITE_KEY) return undefined;
    let live = true;
    loadScript().then((ts) => {
      if (!live || !box.current) return;
      widget.current = ts.render(box.current, {
        sitekey: TURNSTILE_SITE_KEY,
        theme: 'dark',
        callback: (token) => handlers.current.onToken(token),
        'expired-callback': () => handlers.current.onToken(''),
        'error-callback': () => { handlers.current.onToken(''); handlers.current.onError?.('The human check failed. Try it again.'); },
      });
    }).catch((e) => handlers.current.onError?.(e.message));
    return () => {
      live = false;
      if (widget.current != null) { try { window.turnstile?.remove(widget.current); } catch { /* already gone */ } }
      widget.current = null;
    };
  }, []);

  useEffect(() => {
    if (reset && widget.current != null) { handlers.current.onToken(''); window.turnstile?.reset(widget.current); }
  }, [reset]);

  if (!TURNSTILE_SITE_KEY) return null;
  return <div ref={box} className="in-turnstile" aria-label="Human check" />;
}
