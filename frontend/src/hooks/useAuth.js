import { useAuthContext } from '../context/AuthContext';

export function useAuth() {
  const ctx = useAuthContext();
  return ctx || { user: null, setUser: () => {} };
}

