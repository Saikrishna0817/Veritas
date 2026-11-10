/** Public build-time settings (Vite injects VITE_* variables). */

// Cloudflare Turnstile site key. Set it when the backend has VERITAS_TURNSTILE_SECRET; leave it empty otherwise.
export const TURNSTILE_SITE_KEY = import.meta.env.VITE_TURNSTILE_SITE_KEY || '';
