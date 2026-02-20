// API client for AI Trust Forensics Platform
export const BASE_URL = import.meta.env?.VITE_API_BASE_URL || 'http://localhost:8001/api/v1';

async function apiFetch(path, options = {}) {
    const res = await fetch(`${BASE_URL}${path}`, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'API error');
    }
    return res.json();
}

async function apiFormData(path, formData) {
    const res = await fetch(`${BASE_URL}${path}`, { method: 'POST', body: formData });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
}

export const api = {
    BASE_URL,

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
    getLatestForensics: () => apiFetch('/forensics/latest'),
    getAttackNarrative: () => apiFetch('/forensics/narrative'),
    getAttackTimeline: () => apiFetch('/forensics/timeline'),
    getBlastRadius: () => apiFetch('/blast-radius/latest'),

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

    // Red Team
    runRedTeam: (attackType) =>
        apiFetch('/redteam/simulate', { method: 'POST', body: JSON.stringify({ attack_type: attackType }) }),
    getRedTeamHistory: () => apiFetch('/redteam/history'),

    // Federated
    getFederatedClients: () => apiFetch('/federated/clients'),

    // Reports
    generateReport: () => apiFetch('/reports/generate', { method: 'POST' }),

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
    const wsUrl = import.meta.env?.VITE_WS_URL || 'ws://localhost:8001/ws/v1/detection-stream';
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
        } catch { }
    };
    ws.onerror = (e) => console.error('WebSocket error', e);
    ws.onclose = () => console.log('WebSocket disconnected');
    return ws;
}

