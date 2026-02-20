import { api } from './api';

export const authService = {
  me: () => apiFetchSafe('/auth/me'),
};

async function apiFetchSafe(path) {
  try {
    // `api` doesn't currently expose a generic fetch; keep this minimal.
    const res = await fetch(`${api.BASE_URL}${path}`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

