/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { api } from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(() => Boolean(localStorage.getItem('veritas_access_token')));

  useEffect(() => {
    if (!localStorage.getItem('veritas_access_token')) {
      return;
    }
    api.me().then(({ user: currentUser }) => setUser(currentUser)).catch(() => {
      localStorage.removeItem('veritas_access_token');
    }).finally(() => setLoading(false));
  }, []);

  const login = async (username, password) => {
    const result = await api.login(username, password);
    // ADR docs/decisions.md records why this internal deployment uses
    // localStorage today and when it must move to cookie-based sessions.
    localStorage.setItem('veritas_access_token', result.access_token);
    setUser(result.user);
  };

  const logout = () => {
    localStorage.removeItem('veritas_access_token');
    setUser(null);
  };

  const value = useMemo(() => ({ user, loading, login, logout }), [user, loading]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuthContext() {
  return useContext(AuthContext);
}
