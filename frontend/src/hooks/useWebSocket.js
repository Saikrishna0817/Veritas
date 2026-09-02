import { useState, useEffect } from 'react';
import { createWebSocket } from '../services/api';

export function useWebSocket() {
    const [events, setEvents] = useState([]);
    const [connected, setConnected] = useState(false);
    useEffect(() => {
        let ws;
        let reconnectTimer;
        let disposed = false;

        const connect = () => {
            if (disposed || ws?.readyState === WebSocket.OPEN) return;
            ws = createWebSocket((msg) => {
                setEvents(prev => [{ ...msg, receivedAt: new Date().toISOString() }, ...prev].slice(0, 50));
            });
            ws.onopen = () => setConnected(true);
            ws.onclose = () => {
                setConnected(false);
                if (!disposed) reconnectTimer = setTimeout(connect, 3000);
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
