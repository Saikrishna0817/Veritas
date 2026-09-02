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
