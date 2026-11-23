// Load test:  docker run --rm -i --network host -e BASE=http://127.0.0.1:5173 -e USER=... -e PASS=... grafana/k6 run - < loadtest/k6.js
// Each virtual user signs in once (cookie session + CSRF), then mixes cheap reads with background dataset uploads.
// KEYS=keys.json (a JSON list of API keys, one per test account) gives each VU its own account instead, so more VUs
// than the login limit allows can run from one machine:  k6 run -e BASE=http://127.0.0.1:8011 -e KEYS=keys.json -e VUS=20 loadtest/k6.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';

const KEYS = __ENV.KEYS ? JSON.parse(open(__ENV.KEYS)) : null;
const jobSeconds = new Trend('job_seconds');   // upload accepted -> job completed, as the user waits for it

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
const auth = KEYS ? { Authorization: `Bearer ${KEYS[(__VU - 1) % KEYS.length]}` } : {};

export default function (data) {
  if (KEYS) csrf = '';
  if (csrf === null) {
    const login = http.post(`${BASE}/api/v1/auth/token`, JSON.stringify({ username: __ENV.USER, password: __ENV.PASS }), { headers: { 'Content-Type': 'application/json' } });
    check(login, { 'login ok': (r) => r.status === 200 });
    csrf = (http.cookieJar().cookiesForURL(BASE).veritas_csrf || [''])[0];
  }
  for (let i = 0; i < 5; i++) {
    const h = http.get(`${BASE}/api/v1/history`, { headers: auth, tags: { kind: 'read' } });
    check(h, { 'history ok': (r) => r.status === 200 });
    sleep(0.3);
  }
  const up = http.post(`${BASE}/api/v1/analyze/upload?background=true`, { file: http.file(data.csv, 'load.csv', 'text/csv') },
    { headers: { 'X-CSRF-Token': csrf, ...auth }, tags: { kind: 'enqueue' } });
  check(up, { 'enqueued (202) or quota (429)': (r) => r.status === 202 || r.status === 429 });
  if (up.status === 202) {
    const t0 = Date.now();
    for (let i = 0; i < 120; i++) {
      sleep(1);
      const j = http.get(`${BASE}/api/v1/jobs/${up.json('job_id')}`, { headers: auth, tags: { kind: 'poll' } });
      const st = j.status === 200 ? j.json('status') : 'error';
      if (st === 'completed' || st === 'failed' || st === 'error') {
        check(j, { 'job completed': () => st === 'completed' });
        jobSeconds.add((Date.now() - t0) / 1000);
        break;
      }
    }
  }
  sleep(1);
}
