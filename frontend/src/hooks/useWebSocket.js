import { useState, useEffect } from 'react';
import { createWebSocket } from '../services/api';

/**
 * WebSocket custom hook with jittered exponential backoff reconnection.
 */
export function useWebSocket() {
  const [events, setEvents] = useState([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let ws;
    let reconnectTimer;
    let disposed = false;
    let attempt = 0;

    const calculateBackoff = () => {
      // Exponential backoff: 1s, 1.5s, 2.25s... up to 30s with ±20% jitter
      const baseDelay = Math.min(30000, 1000 * Math.pow(1.5, attempt));
      const jitter = baseDelay * 0.2 * (Math.random() * 2 - 1);
      return Math.max(1000, Math.round(baseDelay + jitter));
    };

    const connect = () => {
      if (disposed || ws?.readyState === WebSocket.OPEN) return;
      ws = createWebSocket((msg) => {
        setEvents((prev) => [{ ...msg, receivedAt: new Date().toISOString() }, ...prev].slice(0, 50));
      });

      const apiOnOpen = ws.onopen;
      const apiOnClose = ws.onclose;

      ws.onopen = (event) => {
        apiOnOpen?.call(ws, event);
        setConnected(true);
        attempt = 0; // reset retry counter on clean connection
      };

      ws.onclose = (event) => {
        apiOnClose?.call(ws, event);
        setConnected(false);
        if (!disposed) {
          attempt += 1;
          const delay = calculateBackoff();
          reconnectTimer = setTimeout(connect, delay);
        }
      };
    };

    connect();

    return () => {
      disposed = true;
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, []);

  const clearEvents = () => setEvents([]);

  return { events, connected, clearEvents };
}
