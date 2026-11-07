// API client for AI Trust Forensics Platform
export const BASE_URL =
    import.meta.env.VITE_API_BASE_URL || '/api/v1';

// Browser sessions are an httpOnly cookie the page cannot read, so there is no token to store (nothing for an
// injected script to steal). State-changing calls echo the CSRF cookie in a header; the server compares the two.
function csrfToken() {
    const match = document.cookie.match(/(?:^|; )veritas_csrf=([^;]*)/);
    return match ? decodeURIComponent(match[1]) : '';
}

function authHeaders(method = 'GET') {
    return method === 'GET' || method === 'HEAD' ? {} : { 'X-CSRF-Token': csrfToken() };
}

// An expired or revoked token makes every call fail with 401. Drop it and send the
// user to the right sign-in page instead of leaving each page to show a raw error.
function handleUnauthorized() {
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
        credentials: 'same-origin',
        ...options,
        headers: { 'Content-Type': 'application/json', ...authHeaders(options.method || 'GET'), ...headers },
    });
    await failIfNotOk(res, 'API error', skipAuth);
    return res.json();
}

async function apiFormData(path, formData) {
    const res = await fetch(`${BASE_URL}${path}`, { method: 'POST', credentials: 'same-origin', headers: authHeaders('POST'), body: formData });
    await failIfNotOk(res, 'Upload failed');
    return res.json();
}

async function downloadFile(path, filename) {
    const res = await fetch(`${BASE_URL}${path}`, { credentials: 'same-origin' });
    await failIfNotOk(res, 'Download failed');
    const url = URL.createObjectURL(await res.blob());
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

export function createWebSocket(onMessage) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // The browser sends the session cookie with the handshake; the server also checks the Origin header.
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/v1/detection-stream`);
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

    me: () => apiFetch('/auth/me', { skipAuth: true }),
    logout: () => apiFetch('/auth/logout', { method: 'POST', skipAuth: true }),
    signup: (username, email, password) => apiFetch('/auth/signup', { method: 'POST', skipAuth: true, body: JSON.stringify({ username, email, password }) }),
    verifyEmail: (token) => apiFetch('/auth/verify-email', { method: 'POST', skipAuth: true, body: JSON.stringify({ token }) }),
    resendVerification: (email) => apiFetch('/auth/resend-verification', { method: 'POST', skipAuth: true, body: JSON.stringify({ email }) }),
    requestPasswordReset: (email) => apiFetch('/auth/password-reset/request', { method: 'POST', skipAuth: true, body: JSON.stringify({ email }) }),
    confirmPasswordReset: (token, new_password) => apiFetch('/auth/password-reset/confirm', { method: 'POST', skipAuth: true, body: JSON.stringify({ token, new_password }) }),
    getUsage: () => apiFetch('/account/usage'),
    deleteMyData: () => apiFetch('/account/data', { method: 'DELETE' }),
    listApiKeys: () => apiFetch('/account/api-keys'),
    createApiKey: (name) => apiFetch('/account/api-keys', { method: 'POST', body: JSON.stringify({ name }) }),
    revokeApiKey: (id) => apiFetch(`/account/api-keys/${id}`, { method: 'DELETE' }),

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

    // Blue Team SOC
    getBlueTeamStatus: () => apiFetch('/blueteam/status'),
    getBlueTeamIncidents: () => apiFetch('/blueteam/incidents'),
    listPlaybooks: () => apiFetch('/blueteam/playbooks'),
    getPlaybook: (attackType) => apiFetch(`/blueteam/playbook/${attackType}`),
};


