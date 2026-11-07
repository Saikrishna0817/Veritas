// Load test:  docker run --rm -i --network host -e BASE=http://127.0.0.1:5173 -e USER=... -e PASS=... grafana/k6 run - < loadtest/k6.js
// Each virtual user signs in once (cookie session + CSRF), then mixes cheap reads with background dataset uploads.
import http from 'k6/http';
import { check, sleep } from 'k6';

// 429 is the rate limiter / quota doing its job, not a failure of the service.
http.setResponseCallback(http.expectedStatuses(200, 202, 429));

const BASE = __ENV.BASE || 'http://127.0.0.1:5173';
export const options = {
  vus: Number(__ENV.VUS || 5),   // login is limited to 5 attempts/min per IP, so more VUs from one machine need a different IP or a raised limit
  duration: __ENV.DURATION || '30s',
  noCookiesReset: true,   // keep the session cookie across iterations
  thresholds: {
    http_req_failed: ['rate<0.01'],
    'http_req_duration{kind:read}': ['p(95)<500'],
    'http_req_duration{kind:enqueue}': ['p(95)<3000'],
  },
};

function csv() {
  let rows = ['a,b,c,label'];
  for (let i = 0; i < 400; i++) {
    const a = Math.random(), b = Math.random();
    rows.push(`${a.toFixed(4)},${b.toFixed(4)},${(a + b).toFixed(4)},${a + b > 1 ? 1 : 0}`);
  }
  return rows.join('\n');
}

export function setup() { return { csv: csv() }; }

let csrf = null;      // per virtual user: the session cookie lives in the VU's jar, so sign in once

export default function (data) {
  if (csrf === null) {
    const login = http.post(`${BASE}/api/v1/auth/token`, JSON.stringify({ username: __ENV.USER, password: __ENV.PASS }), { headers: { 'Content-Type': 'application/json' } });
    check(login, { 'login ok': (r) => r.status === 200 });
    csrf = (http.cookieJar().cookiesForURL(BASE).veritas_csrf || [''])[0];
  }
  for (let i = 0; i < 5; i++) {
    const h = http.get(`${BASE}/api/v1/history`, { tags: { kind: 'read' } });
    check(h, { 'history ok': (r) => r.status === 200 });
    sleep(0.3);
  }
  const up = http.post(`${BASE}/api/v1/analyze/upload?background=true`, { file: http.file(data.csv, 'load.csv', 'text/csv') },
    { headers: { 'X-CSRF-Token': csrf }, tags: { kind: 'enqueue' } });
  check(up, { 'enqueued (202) or quota (429)': (r) => r.status === 202 || r.status === 429 });
  sleep(1);
}
