import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { fetchWithRefresh, singleFlight } from './refresh.js';

describe('singleFlight', () => {
  it('shares one run between concurrent callers and runs again afterwards', async () => {
    let runs = 0;
    const refresh = singleFlight(async () => { runs += 1; await new Promise((r) => setTimeout(r, 5)); return true; });
    assert.deepEqual(await Promise.all([refresh(), refresh(), refresh()]), [true, true, true]);
    assert.equal(runs, 1);
    await refresh();
    assert.equal(runs, 2);
  });
});

describe('fetchWithRefresh', () => {
  it('retries once after a successful refresh', async () => {
    const answers = [{ status: 401 }, { status: 200 }];
    let refreshes = 0;
    const res = await fetchWithRefresh(async () => answers.shift(), async () => { refreshes += 1; return true; });
    assert.equal(res.status, 200);
    assert.equal(refreshes, 1);
  });

  it('returns the 401 when the refresh fails, and does not retry', async () => {
    let sends = 0;
    const res = await fetchWithRefresh(async () => { sends += 1; return { status: 401 }; }, async () => false);
    assert.equal(res.status, 401);
    assert.equal(sends, 1);
  });

  it('does not refresh for other statuses', async () => {
    let refreshes = 0;
    await fetchWithRefresh(async () => ({ status: 500 }), async () => { refreshes += 1; return true; });
    assert.equal(refreshes, 0);
  });

  it('concurrent 401s trigger one refresh', async () => {
    let refreshes = 0;
    const refresh = singleFlight(async () => { refreshes += 1; await new Promise((r) => setTimeout(r, 5)); return true; });
    const mk = () => { let first = true; return async () => { const s = first ? 401 : 200; first = false; return { status: s }; }; };
    const results = await Promise.all([fetchWithRefresh(mk(), refresh), fetchWithRefresh(mk(), refresh), fetchWithRefresh(mk(), refresh)]);
    assert.deepEqual(results.map((r) => r.status), [200, 200, 200]);
    assert.equal(refreshes, 1);
  });
});
