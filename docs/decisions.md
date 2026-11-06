# Architecture decisions

## 2026-09-02: JWT storage for the current deployment

The frontend stores the short-lived JWT in `localStorage` and sends it as a
Bearer token. This is acceptable only for the current same-origin, internal
analyst deployment: it keeps the API stateless and allows the native WebSocket
client to authenticate without a cookie/CSRF protocol.

This is not a general recommendation for internet-facing deployments. The app
must use a restrictive Content Security Policy and avoid untrusted script
injection. If the product is exposed beyond trusted analysts, migrate to secure
httpOnly, SameSite cookies with explicit CSRF protection and update the
WebSocket authentication mechanism at the same time.

## 2026-11-06: Shared results and WebSocket token handshake

**Results are not isolated per user.** Uploads, scans, history, forensics and reports are
global: any authenticated user can read any analysis, and WebSocket events go to every
connected client. This is acceptable for one trusted analyst team. Before serving several
teams, add an `owner_id` column to `analysis_results`/`model_scans`, set it on save, and
filter every read by it (admins excepted). Background jobs are already owner-scoped.

**WebSocket authentication** uses a first-frame handshake (`{"type":"auth","token":...}`,
5 s timeout) instead of `?access_token=` so tokens never appear in proxy or access logs.
The server closes the socket when the token expires.

**Single worker.** Rate limiters, result caches, the job store and the WebSocket registry
are process-local, so the container runs one uvicorn worker. Moving to several workers or
hosts requires a shared store (e.g. Redis) for those pieces first.
