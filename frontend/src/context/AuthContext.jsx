/* eslint-disable react-refresh/only-export-components */
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { api } from '../services/api';

const AuthContext = createContext(null);

// Only these answers mean "not signed in". Anything else (no answer, a server error, rate limiting) says nothing
// about the session, so the person is not sent to sign in again because of it.
const SIGNED_OUT = new Set([401, 403]);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [sessionError, setSessionError] = useState(null);

  // The session lives in an httpOnly cookie, so ask the server who we are.
  const ask = useCallback(() => api.me()
    .then(({ user: currentUser }) => { setUser(currentUser); setSessionError(null); })
    .catch((e) => { setUser(null); setSessionError(SIGNED_OUT.has(e.status) ? null : e); })
    .finally(() => setLoading(false)), []);
  useEffect(() => { ask(); }, [ask]);              // first check: the initial state already says "loading"
  const check = useCallback(() => { setLoading(true); setSessionError(null); return ask(); }, [ask]);

  const login = useCallback(async (username, password, requiredRole = null) => {
    const result = await api.login(username, password, requiredRole);
    if (result.two_factor_required) return { two_factor_required: true, challenge: result.challenge };   // second step: verifyTwoFactor
    setUser(result.user); setSessionError(null);
    return result.user;
  }, []);

  const verifyTwoFactor = useCallback(async (challenge, code) => {
    const result = await api.verifyTwoFactor(challenge, code);
    setUser(result.user); setSessionError(null);
    return result.user;
  }, []);

  const logout = useCallback(async () => {
    try { await api.logout(); } catch { /* the cookie is cleared server-side; nothing else to do */ }
    setUser(null);
  }, []);

  const isAdmin = user?.role === 'admin';
  const isUser = Boolean(user && user.role !== 'admin');

  const value = useMemo(
    () => ({ user, loading, sessionError, recheck: check, login, verifyTwoFactor, logout, isAdmin, isUser }),
    [user, loading, sessionError, check, login, verifyTwoFactor, logout, isAdmin, isUser]
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}


export function useAuthContext() {
  return useContext(AuthContext);
}
