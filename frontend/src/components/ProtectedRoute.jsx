import { useAuth } from '../hooks/useAuth';

export default function ProtectedRoute({ children, fallback }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="p-8 text-text3 font-mono">Checking session…</div>;
  return user ? children : fallback;
}
