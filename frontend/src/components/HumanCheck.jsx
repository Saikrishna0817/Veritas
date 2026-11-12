import { useEffect, useRef } from 'react';
import { TURNSTILE_SITE_KEY } from '../services/config';

/**
 * Cloudflare Turnstile. The backend checks the token only when VERITAS_TURNSTILE_SECRET is set, so the widget is
 * rendered only when the matching public key is configured: VITE_TURNSTILE_SITE_KEY at build time. Without it this
 * renders nothing and forms send no token.
 */
const SCRIPT = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';

let loading;
function loadScript() {
  if (window.turnstile) return Promise.resolve(window.turnstile);
  loading ??= new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = SCRIPT;
    s.async = true;
    s.onload = () => resolve(window.turnstile);
    s.onerror = () => { loading = null; reject(new Error('The human check could not load. Check your connection and reload.')); };
    document.head.appendChild(s);
  });
  return loading;
}

/** onToken(token | '') fires when the check passes, expires or fails. `resetKey` changes force a fresh check. */
export default function HumanCheck({ onToken, onError, resetKey = 0 }) {
  const box = useRef(null);
  const cb = useRef({ onToken, onError });
  useEffect(() => { cb.current = { onToken, onError }; });

  useEffect(() => {
    if (!TURNSTILE_SITE_KEY) return undefined;
    let id;
    let gone = false;
    loadScript().then((ts) => {
      if (gone || !box.current) return;
      id = ts.render(box.current, {
        sitekey: TURNSTILE_SITE_KEY,
        theme: 'dark',
        callback: (t) => cb.current.onToken(t),
        'expired-callback': () => cb.current.onToken(''),
        'error-callback': () => { cb.current.onToken(''); cb.current.onError?.('The human check failed. Try again.'); },
      });
    }).catch((e) => cb.current.onError?.(e.message));
    return () => {
      gone = true;
      if (id !== undefined) window.turnstile?.remove(id);
    };
  }, [resetKey]);

  if (!TURNSTILE_SITE_KEY) return null;
  return <div ref={box} className="min-h-[65px]" aria-label="Human check" />;
}
