import { useState } from 'react';
import { useAuth } from '../hooks/useAuth';

export default function LoginPage() {
  const { login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  async function submit(event) {
    event.preventDefault();
    setError('');
    try { await login(username, password); }
    catch (err) { setError(err.message || 'Unable to sign in'); }
  }

  return <main className="min-h-screen flex items-center justify-center bg-bg p-6">
    <form onSubmit={submit} className="w-full max-w-sm space-y-4 rounded-lg border border-border bg-bg2 p-6">
      <h1 className="font-mono text-accent">AI TRUST FORENSICS</h1>
      <input aria-label="Username" className="w-full p-2 bg-surface" value={username} onChange={e => setUsername(e.target.value)} placeholder="Username" required />
      <input aria-label="Password" className="w-full p-2 bg-surface" value={password} onChange={e => setPassword(e.target.value)} placeholder="Password" type="password" required />
      {error && <p className="text-danger text-sm">{error}</p>}
      <button className="w-full p-2 bg-accent text-bg font-mono" type="submit">Sign in</button>
    </form>
  </main>;
}
