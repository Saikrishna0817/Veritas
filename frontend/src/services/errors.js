/**
 * Sorts a failed API call into what the person can do about it. api.js puts the HTTP `status` on server
 * errors and marks a request that never reached the server with `offline`. The server's own `detail`
 * (err.message) is kept wherever it says something specific.
 *
 * kind: offline | unavailable | rate_limited | forbidden | not_found | failed
 */
export function describeError(err) {
  const status = err?.status;
  const detail = err?.message || '';
  if (err?.offline || status === 502 || status === 504) {
    return { kind: 'offline', title: "Can't reach Veritas", message: 'The server did not answer. Check your connection, then try again.' };
  }
  if (status === 501 || status === 503) {
    return { kind: 'unavailable', title: 'Not available on this server', message: detail || 'This server has not turned this feature on.' };
  }
  if (status === 429) return { kind: 'rate_limited', title: 'Too many requests', message: detail || 'Wait a minute, then try again.' };
  if (status === 403) return { kind: 'forbidden', title: "You don't have access", message: detail || 'Your role cannot open this.' };
  if (status === 404) return { kind: 'not_found', title: 'Nothing here', message: detail || 'It may have been deleted.' };
  return { kind: 'failed', title: 'Something went wrong', message: detail || 'The request failed. Try again.' };
}
