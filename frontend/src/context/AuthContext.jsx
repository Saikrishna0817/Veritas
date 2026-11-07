/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { api } from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // The session lives in an httpOnly cookie, so ask the server who we are.
  useEffect(() => {
    api.me().then(({ user: currentUser }) => setUser(currentUser)).catch(() => setUser(null)).finally(() => setLoading(false));
  }, []);

  const login = async (username, password, requiredRole = null) => {
    const result = await api.login(username, password, requiredRole);
    setUser(result.user);
    return result.user;
  };

  const logout = async () => {
    try { await api.logout(); } catch { /* the cookie is cleared server-side; nothing else to do */ }
    setUser(null);
  };

  const isAdmin = user?.role === 'admin';
  const isUser = Boolean(user && user.role !== 'admin');

  const value = useMemo(
    () => ({ user, loading, login, logout, isAdmin, isUser }),
    [user, loading, isAdmin, isUser]
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}


export function useAuthContext() {
  return useContext(AuthContext);
}
