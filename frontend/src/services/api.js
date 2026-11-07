// API client for AI Trust Forensics Platform
export const BASE_URL =
    import.meta.env.VITE_API_BASE_URL || '/api/v1';

function authHeaders() {
    const token = localStorage.getItem('veritas_access_token');
    return token ? { Authorization: `Bearer ${token}` } : {};
}

// An expired or revoked token makes every call fail with 401. Drop it and send the
// user to the right sign-in page instead of leaving each page to show a raw error.
function handleUnauthorized() {
    localStorage.removeItem('veritas_access_token');
    const loginPath = window.location.pathname.startsWith('/admin') ? '/admin/login' : '/login';
    if (window.location.pathname !== loginPath) window.location.assign(loginPath);
}

async function failIfNotOk(res, fallback, skipAuth = false) {
    if (res.ok) return;
    if (res.status === 401 && !skipAuth) handleUnauthorized();
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || fallback);
}

async function apiFetch(path, { skipAuth = false, headers, ...options } = {}) {
    const res = await fetch(`${BASE_URL}${path}`, {
        ...options,
        headers: { 'Content-Type': 'application/json', ...(skipAuth ? {} : authHeaders()), ...headers },
    });
    await failIfNotOk(res, 'API error', skipAuth);
    return res.json();
}

async function apiFormData(path, formData) {
    const res = await fetch(`${BASE_URL}${path}`, { method: 'POST', headers: authHeaders(), body: formData });
    await failIfNotOk(res, 'Upload failed');
    return res.json();
}

async function downloadFile(path, filename) {
    const res = await fetch(`${BASE_URL}${path}`, { headers: authHeaders() });
    await failIfNotOk(res, 'Download failed');
    const url = URL.createObjectURL(await res.blob());
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

export function createWebSocket(onMessage) {
    const token = localStorage.getItem('veritas_access_token');
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/v1/detection-stream`);
    // The token goes in the first frame, not the URL, so it stays out of access logs.
    ws.onopen = () => ws.send(JSON.stringify({ type: 'auth', token }));
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (onMessage) onMessage(data);
        } catch {
            // ignore
        }
    };
    return ws;
}

export const api = {

    BASE_URL,
    login: (username, password, requiredRole = null) => apiFetch('/auth/token', {
        method: 'POST',
        skipAuth: true,
        body: JSON.stringify({ username, password, required_role: requiredRole }),
    }),

    me: () => apiFetch('/auth/me'),

    // User management (Admin only)
    getUsers: () => apiFetch('/users'),
    createUser: (data) => apiFetch('/users', { method: 'POST', body: JSON.stringify(data) }),
    deleteUser: (userId) => apiFetch(`/users/${userId}`, { method: 'DELETE' }),

    // Demo & Real Datasets
    runDemo: () => apiFetch('/demo/run', { method: 'POST' }),
    getDemoDataset: () => apiFetch('/datasets/demo'),
    getDemoSamples: (limit = 50, offset = 0, status = null) => {
        const params = new URLSearchParams({ limit, offset });
        if (status) params.append('filter_status', status);
        return apiFetch(`/datasets/demo/samples?${params}`);
    },
    getRealDatasets: () => apiFetch('/datasets/real'),
    analyzeRealDataset: (name) => apiFetch(`/datasets/real/${name}/analyze`, { method: 'POST' }),
    downloadRealDataset: (name) => downloadFile(`/datasets/real/${name}/download`, `${name}.csv`),

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

    // Reports
    generateReport: (source = 'auto') => apiFetch(`/reports/generate?source=${source}`, { method: 'POST' }),

    // CSV Upload Analysis
    // opts: { referenceFile?: File, splitColumn?: string, splitCutoff?: string }; default is the first 70% in file order
    uploadCSV: (file, opts = {}) => {
        const formData = new FormData();
        formData.append('file', file);
        if (opts.referenceFile) formData.append('reference_file', opts.referenceFile);
        if (opts.splitColumn && opts.splitCutoff) {
            formData.append('split_column', opts.splitColumn);
            formData.append('split_cutoff', opts.splitCutoff);
        }
        return apiFormData('/analyze/upload', formData);
    },
    getLatestUpload: () => apiFetch('/analyze/upload/latest'),

    // Model Scan
    scanModel: (formData) => apiFormData('/analyze/model', formData),
    uploadModel: (file) => {
        const formData = new FormData();
        formData.append('model_file', file);
        return apiFormData('/analyze/model', formData);
    },
    getModelScanHistory: (limit = 20) => apiFetch(`/analyze/model/history?limit=${limit}`),
    getModelScan: (scanId) => apiFetch(`/analyze/model/${scanId}`),

    // Analysis History & Audit
    getHistory: (source = null, limit = 20) =>
        apiFetch(`/history?limit=${limit}${source ? `&source=${source}` : ''}`),
    getAnalysisHistory: (limit = 20) => apiFetch(`/history?limit=${limit}`),
    getHistoricalResult: (id) => apiFetch(`/history/${id}`),
    deleteAnalysisHistory: (runId) => apiFetch(`/history/${runId}`, { method: 'DELETE' }),
    getAuditEvents: (limit = 50) => apiFetch(`/audit/events?limit=${limit}`),

    // Red Team
    runRedTeamSimulation: (attackType = 'label_flip') =>
        apiFetch('/redteam/simulate', { method: 'POST', body: JSON.stringify({ attack_type: attackType }) }),
    getRedTeamHistory: () => apiFetch('/redteam/history'),

    // Blue Team SOC
    getBlueTeamStatus: () => apiFetch('/blueteam/status'),
    getBlueTeamIncidents: () => apiFetch('/blueteam/incidents'),
    getBlueTeamResilience: () => apiFetch('/blueteam/resilience'),
    listPlaybooks: () => apiFetch('/blueteam/playbooks'),
    getPlaybook: (attackType) => apiFetch(`/blueteam/playbook/${attackType}`),
};


