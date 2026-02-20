import { useState, useEffect, useCallback, useRef } from 'react';
import { createWebSocket } from '../api';

export function useWebSocket() {
    const [events, setEvents] = useState([]);
    const [connected, setConnected] = useState(false);
    const wsRef = useRef(null);

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        const ws = createWebSocket((msg) => {
            setEvents(prev => [{ ...msg, receivedAt: new Date().toISOString() }, ...prev].slice(0, 50));
        });

        ws.onopen = () => setConnected(true);
        ws.onclose = () => {
            setConnected(false);
            // Reconnect after 3s
            setTimeout(connect, 3000);
        };
        wsRef.current = ws;
    }, []);

    useEffect(() => {
        connect();
        return () => wsRef.current?.close();
    }, [connect]);

    const clearEvents = () => setEvents([]);

    return { events, connected, clearEvents };
}
