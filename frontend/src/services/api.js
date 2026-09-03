// API client for AI Trust Forensics Platform
export const BASE_URL =
    import.meta.env.VITE_API_BASE_URL || '/api/v1';

function authHeaders() {
    const token = localStorage.getItem('veritas_access_token');
    return token ? { Authorization: `Bearer ${token}` } : {};
}

async function apiFetch(path, { skipAuth = false, ...options } = {}) {
    const res = await fetch(`${BASE_URL}${path}`, {
        headers: { 'Content-Type': 'application/json', ...(skipAuth ? {} : authHeaders()), ...options.headers },
        ...options,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'API error');
    }
    return res.json();
}

async function apiFormData(path, formData) {
    const res = await fetch(`${BASE_URL}${path}`, { method: 'POST', headers: authHeaders(), body: formData });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
}

async function downloadFile(path, filename) {
    const res = await fetch(`${BASE_URL}${path}`, { headers: authHeaders() });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Download failed');
    }
    const url = URL.createObjectURL(await res.blob());
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

export const api = {
    BASE_URL,
    login: (username, password) => apiFetch('/auth/token', {
        method: 'POST',
        skipAuth: true,
        body: JSON.stringify({ username, password }),
    }),
    me: () => apiFetch('/auth/me'),

    // Demo
    runDemo: () => apiFetch('/demo/run', { method: 'POST' }),
    getDemoDataset: () => apiFetch('/datasets/demo'),
    getDemoSamples: (limit = 50, offset = 0, status = null) => {
        const params = new URLSearchParams({ limit, offset });
        if (status) params.append('filter_status', status);
        return apiFetch(`/datasets/demo/samples?${params}`);
    },

    // Detection
    analyzeDataset: (sampleIds = []) =>
        apiFetch('/detect/analyze', { method: 'POST', body: JSON.stringify({ sample_ids: sampleIds }) }),
    getLatestResults: () => apiFetch('/detect/results/latest'),

    // Forensics
    getLatestForensics: (source = 'auto') => apiFetch(`/forensics/latest?source=${source}`),
    getAttackNarrative: (source = 'auto') => apiFetch(`/forensics/narrative?source=${source}`),
    getAttackTimeline: () => apiFetch('/forensics/timeline'),
    getBlastRadius: (source = 'auto') => apiFetch(`/blast-radius/latest?source=${source}`),

    // Trust
    getTrustScore: () => apiFetch('/trust/score'),

    // Defense
    triggerQuarantine: () => apiFetch('/defense/quarantine', { method: 'POST' }),
    getDefenseStatus: () => apiFetch('/defense/status'),
    getPendingReviews: () => apiFetch('/defense/hitl/pending'),
    submitReviewDecision: (caseId, decision, reviewer = 'analyst') =>
        apiFetch('/defense/hitl/decide', {
            method: 'POST',
            body: JSON.stringify({ case_id: caseId, decision, reviewer }),
        }),



    // Federated
    getFederatedClients: () => apiFetch('/federated/clients'),

    // Reports
    generateReport: (source = 'auto') => apiFetch(`/reports/generate?source=${source}`, { method: 'POST' }),

    // CSV Upload Analysis
    uploadCSV: (file) => {
        const formData = new FormData();
        formData.append('file', file);
        return apiFormData('/analyze/upload', formData);
    },
    getLatestUpload: () => apiFetch('/analyze/upload/latest'),

    // Model Scan
    scanModel: (formData) => apiFormData('/analyze/model', formData),
    getModelScanHistory: (limit = 20) => apiFetch(`/analyze/model/history?limit=${limit}`),
    getModelScan: (scanId) => apiFetch(`/analyze/model/${scanId}`),

    // Real Dataset Library
    getRealDatasets: () => apiFetch('/datasets/real'),
    analyzeRealDataset: (name) => apiFetch(`/datasets/real/${name}/analyze`, { method: 'POST' }),
    downloadRealDataset: (name) => downloadFile(`/datasets/real/${name}/download`, `${name}.csv`),

    // History / Persistence
    getHistory: (source = null, limit = 20) => {
        const params = new URLSearchParams({ limit });
        if (source) params.append('source', source);
        return apiFetch(`/history?${params}`);
    },
    getHistoricalResult: (id) => apiFetch(`/history/${id}`),

    // Blue Team SOC
    getBlueTeamStatus: () => apiFetch('/blueteam/status'),
    getBlueTeamIncidents: () => apiFetch('/blueteam/incidents'),
    getBlueTeamResilience: () => apiFetch('/blueteam/resilience'),
    listPlaybooks: () => apiFetch('/blueteam/playbooks'),
    getPlaybook: (attackType) => apiFetch(`/blueteam/playbook/${attackType}`),
};


// WebSocket
export function createWebSocket(onMessage) {
    const configuredUrl = import.meta.env?.VITE_WS_URL;
    const defaultUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/v1/detection-stream`;
    const baseWsUrl = configuredUrl || defaultUrl;
    const token = localStorage.getItem('veritas_access_token');
    const wsUrl = token ? `${baseWsUrl}?access_token=${encodeURIComponent(token)}` : baseWsUrl;
    const ws = new WebSocket(wsUrl);
    ws.onopen = () => {
        console.log('WebSocket connected');
        // Keep-alive ping
        const ping = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping' }));
            } else {
                clearInterval(ping);
            }
        }, 30000);
    };
    ws.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            onMessage(msg);
        } catch {
            return;
        }
    };
    ws.onerror = (e) => console.error('WebSocket error', e);
    ws.onclose = () => console.log('WebSocket disconnected');
    return ws;
}
