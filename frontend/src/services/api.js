// Veritas API client. Every page goes through this file; none calls fetch itself.
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

// How long a request may wait for the server's answer before the page says so. Inline analyses run while the
// request is open, so they get longer; large files go through background jobs instead (services/uploads.js).
export const TIMEOUT_MS = { default: 30_000, analysis: 180_000, download: 120_000 };

// A request that never reached the server (offline, DNS, server down) rejects with a bare TypeError such as
// "Failed to fetch". Give it words a person can act on and mark it, so pages can tell it from a server error.
// A request with no answer within `timeoutMs` is aborted and marked `timeout` the same way.
async function send(url, init, timeoutMs = TIMEOUT_MS.default) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await fetch(url, { ...init, signal: controller.signal });
    } catch (e) {
        const timedOut = controller.signal.aborted;
        const err = new Error(timedOut
            ? `The server did not answer within ${Math.round(timeoutMs / 1000)} seconds. Try again; if it keeps happening, the server may be overloaded.`
            : "Can't reach the Veritas server. Check your connection, then try again.");
        err.status = 0;
        if (timedOut) err.timeout = true; else err.offline = true;
        err.cause = e;
        throw err;
    } finally {
        clearTimeout(timer);
    }
}

async function failIfNotOk(res, fallback, skipAuth = false) {
    if (res.ok) return;
    if (res.status === 401 && !skipAuth) handleUnauthorized();
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    // FastAPI validation errors carry a list in `detail`; show the first message instead of [object Object].
    const detail = Array.isArray(body.detail) ? body.detail[0]?.msg : body.detail;
    const err = new Error(detail || fallback);
    err.status = res.status;
    throw err;
}

async function apiFetch(path, { skipAuth = false, headers, timeoutMs, ...options } = {}) {
    const res = await send(`${BASE_URL}${path}`, {
        credentials: 'same-origin',
        ...options,
        headers: { 'Content-Type': 'application/json', ...authHeaders(options.method || 'GET'), ...headers },
    }, timeoutMs);
    await failIfNotOk(res, 'API error', skipAuth);
    return res.json();
}

async function apiFormData(path, formData, timeoutMs = TIMEOUT_MS.analysis) {
    const res = await send(`${BASE_URL}${path}`, { method: 'POST', credentials: 'same-origin', headers: authHeaders('POST'), body: formData }, timeoutMs);
    await failIfNotOk(res, 'Upload failed');
    return res.json();
}

async function downloadFile(path, filename) {
    const res = await send(`${BASE_URL}${path}`, { credentials: 'same-origin' }, TIMEOUT_MS.download);
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
            // Detection events are { event, data }; control frames such as { type: 'pong' } are not shown.
            if (data?.event && onMessage) onMessage(data);
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
    signup: (username, email, password, turnstileToken) => apiFetch('/auth/signup', { method: 'POST', skipAuth: true, body: JSON.stringify({ username, email, password, turnstile_token: turnstileToken || undefined }) }),
    verifyEmail: (token) => apiFetch('/auth/verify-email', { method: 'POST', skipAuth: true, body: JSON.stringify({ token }) }),
    resendVerification: (email, turnstileToken) => apiFetch('/auth/resend-verification', { method: 'POST', skipAuth: true, body: JSON.stringify({ email, turnstile_token: turnstileToken || undefined }) }),
    requestPasswordReset: (email, turnstileToken) => apiFetch('/auth/password-reset/request', { method: 'POST', skipAuth: true, body: JSON.stringify({ email, turnstile_token: turnstileToken || undefined }) }),
    confirmPasswordReset: (token, new_password) => apiFetch('/auth/password-reset/confirm', { method: 'POST', skipAuth: true, body: JSON.stringify({ token, new_password }) }),
    getUsage: () => apiFetch('/account/usage'),
    deleteMyData: () => apiFetch('/account/data', { method: 'DELETE' }),
    deleteAccount: (password) => apiFetch('/account/delete', { method: 'POST', skipAuth: true, body: JSON.stringify({ password }) }),
    listApiKeys: () => apiFetch('/account/api-keys'),
    createApiKey: (name) => apiFetch('/account/api-keys', { method: 'POST', body: JSON.stringify({ name }) }),
    revokeApiKey: (id) => apiFetch(`/account/api-keys/${id}`, { method: 'DELETE' }),

    // User management (Admin only)
    getUsers: () => apiFetch('/users'),
    createUser: (data) => apiFetch('/users', { method: 'POST', body: JSON.stringify(data) }),
    deleteUser: (userId) => apiFetch(`/users/${userId}`, { method: 'DELETE' }),

    // Demo & Real Datasets
    runDemo: () => apiFetch('/demo/run', { method: 'POST', timeoutMs: TIMEOUT_MS.analysis }),
    getRealDatasets: () => apiFetch('/datasets/real'),
    analyzeRealDataset: (name) => apiFetch(`/datasets/real/${name}/analyze`, { method: 'POST', timeoutMs: TIMEOUT_MS.analysis }),
    downloadRealDataset: (name) => downloadFile(`/datasets/real/${name}/download`, `${name}.csv`),

    // Forensics (the latest result already carries the narrative and blast radius)
    getLatestForensics: (source = 'auto') => apiFetch(`/forensics/latest?source=${source}`),

    // Defense
    triggerQuarantine: () => apiFetch('/defense/quarantine', { method: 'POST' }),
    getDefenseStatus: () => apiFetch('/defense/status'),
    getPendingReviews: () => apiFetch('/defense/hitl/pending'),
    // The server records the signed-in user as the reviewer; the body carries only the case and the decision.
    submitReviewDecision: (caseId, decision) =>
        apiFetch('/defense/hitl/decide', {
            method: 'POST',
            body: JSON.stringify({ case_id: caseId, decision }),
        }),

    // Reports
    generateReport: (source = 'auto') => apiFetch(`/reports/generate?source=${source}`, { method: 'POST', timeoutMs: TIMEOUT_MS.analysis }),

    // CSV Upload Analysis
    // opts: { referenceFile?: File, splitColumn?: string, splitCutoff?: string }; default is the first 70% in file order
    uploadCSV: (file, opts = {}, { background = false } = {}) => {
        const formData = new FormData();
        formData.append('file', file);
        if (opts.referenceFile) formData.append('reference_file', opts.referenceFile);
        if (opts.splitColumn && opts.splitCutoff) {
            formData.append('split_column', opts.splitColumn);
            formData.append('split_cutoff', opts.splitCutoff);
        }
        return apiFormData(`/analyze/upload${background ? '?background=true' : ''}`, formData);
    },
    getLatestUpload: () => apiFetch('/analyze/upload/latest'),

    // Model Scan
    uploadModel: (file, { background = false } = {}) => {
        const formData = new FormData();
        formData.append('model_file', file);
        return apiFormData(`/analyze/model${background ? '?background=true' : ''}`, formData);
    },

    // Background jobs and direct-to-storage uploads
    getJob: (jobId) => apiFetch(`/jobs/${jobId}`),
    presignUpload: (role, filename, size) => apiFetch('/uploads/presign', { method: 'POST', body: JSON.stringify({ role, filename, size }) }),
    completeUpload: (body) => apiFetch('/uploads/complete', { method: 'POST', body: JSON.stringify(body) }),
    getModelScanHistory: (limit = 20) => apiFetch(`/analyze/model/history?limit=${limit}`),
    getModelScan: (scanId) => apiFetch(`/analyze/model/${scanId}`),

    // Analysis History & Audit
    getHistory: (source = null, limit = 20) =>
        apiFetch(`/history?limit=${limit}${source ? `&source=${source}` : ''}`),
    getAnalysisHistory: (limit = 20) => apiFetch(`/history?limit=${limit}`),
    getHistoricalResult: (id) => apiFetch(`/history/${id}`),
    deleteAnalysisHistory: (runId) => apiFetch(`/history/${runId}`, { method: 'DELETE' }),
    getAuditEvents: (limit = 50) => apiFetch(`/audit/events?limit=${limit}`),

    // Blue Team SOC
    getBlueTeamStatus: () => apiFetch('/blueteam/status'),
    getBlueTeamIncidents: () => apiFetch('/blueteam/incidents'),
    listPlaybooks: () => apiFetch('/blueteam/playbooks'),
    getPlaybook: (attackType) => apiFetch(`/blueteam/playbook/${attackType}`),
};


