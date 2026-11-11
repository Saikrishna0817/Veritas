import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api } from './api';

const ok = (body = {}) => ({ ok: true, status: 200, json: async () => body });
const fail = (status, body, statusText = 'Error') => ({
  ok: false, status, statusText, json: body === undefined ? async () => { throw new SyntaxError('not json'); } : async () => body,
});

let location;
beforeEach(() => {
  location = { pathname: '/upload', protocol: 'http:', host: 'localhost:5173', assign: vi.fn() };
  vi.stubGlobal('window', { location });
  vi.stubGlobal('document', { cookie: 'other=1; veritas_csrf=abc%3D123' });
  vi.stubGlobal('fetch', vi.fn(async () => ok({ fine: true })));
});
afterEach(() => vi.unstubAllGlobals());

const lastCall = () => fetch.mock.calls.at(-1);

describe('requests', () => {
  it('sends reads with the session cookie and no CSRF header', async () => {
    await expect(api.getUsage()).resolves.toEqual({ fine: true });
    const [url, init] = lastCall();
    expect(url).toBe('/api/v1/account/usage');
    expect(init.credentials).toBe('same-origin');
    expect(init.headers).not.toHaveProperty('X-CSRF-Token');
  });

  it('echoes the decoded CSRF cookie on requests that change something', async () => {
    await api.createApiKey('ci');
    const [url, init] = lastCall();
    expect(url).toBe('/api/v1/account/api-keys');
    expect(init.method).toBe('POST');
    expect(init.headers['X-CSRF-Token']).toBe('abc=123');
    expect(JSON.parse(init.body)).toEqual({ name: 'ci' });
  });

  it('sends uploads as form data, leaving the Content-Type to the browser', async () => {
    const file = new Blob(['a,b\n1,2'], { type: 'text/csv' });
    const reference = new Blob(['a,b\n1,2'], { type: 'text/csv' });
    await api.uploadCSV(file, { referenceFile: reference, splitColumn: 'day', splitCutoff: '2026-01-01' }, { background: true });
    const [url, init] = lastCall();
    expect(url).toBe('/api/v1/analyze/upload?background=true');
    expect(init.headers).toEqual({ 'X-CSRF-Token': 'abc=123' });
    expect([...init.body.keys()]).toEqual(['file', 'reference_file', 'split_column', 'split_cutoff']);
  });

  it('only sends a split when both the column and the cutoff are set', async () => {
    await api.uploadCSV(new Blob(['x']), { splitColumn: 'day' });
    const [url, init] = lastCall();
    expect(url).toBe('/api/v1/analyze/upload');
    expect([...init.body.keys()]).toEqual(['file']);
  });
});

describe('errors', () => {
  it('carries the status and the server detail', async () => {
    fetch.mockResolvedValueOnce(fail(429, { detail: 'Daily analyses quota reached' }));
    await expect(api.getUsage()).rejects.toMatchObject({ message: 'Daily analyses quota reached', status: 429 });
  });

  it('shows the first message of a validation error list, not [object Object]', async () => {
    fetch.mockResolvedValueOnce(fail(422, { detail: [{ msg: 'String should have at least 10 characters' }, { msg: 'second' }] }));
    await expect(api.createUser({})).rejects.toThrow('String should have at least 10 characters');
  });

  it('falls back to the status text when the body is not JSON', async () => {
    fetch.mockResolvedValueOnce(fail(502, undefined, 'Bad Gateway'));
    await expect(api.getUsage()).rejects.toMatchObject({ message: 'Bad Gateway', status: 502 });
  });

  it('marks a request that never reached the server, in words a person can act on', async () => {
    fetch.mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await expect(api.getUsage()).rejects.toMatchObject({ offline: true, status: 0, message: expect.stringMatching(/can't reach the veritas server/i) });
  });

  it('gives up on a request with no answer and says it timed out', async () => {
    vi.useFakeTimers();
    try {
      fetch.mockImplementationOnce((url, init) => new Promise((_, reject) => {
        init.signal.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })));
      }));
      const call = expect(api.getUsage()).rejects.toMatchObject({ timeout: true, status: 0, message: expect.stringMatching(/did not answer within 30 seconds/) });
      await vi.advanceTimersByTimeAsync(30_000);
      await call;
    } finally {
      vi.useRealTimers();
    }
  });

  it('gives inline analyses longer than ordinary reads', async () => {
    vi.useFakeTimers();
    try {
      let aborted = false;
      fetch.mockImplementationOnce((url, init) => new Promise(() => { init.signal.addEventListener('abort', () => { aborted = true; }); }));
      api.analyzeRealDataset('adult').catch(() => {});
      await vi.advanceTimersByTimeAsync(60_000);
      expect(aborted).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it('sends only the case and the decision for a review; the server records the reviewer', async () => {
    await api.submitReviewDecision('case-1', 'mark_safe');
    expect(JSON.parse(lastCall()[1].body)).toEqual({ case_id: 'case-1', decision: 'mark_safe' });
  });

  it('sends an expired session to sign-in', async () => {
    fetch.mockResolvedValueOnce(fail(401, { detail: 'Not authenticated' }));
    await expect(api.getHistory()).rejects.toMatchObject({ status: 401 });
    expect(location.assign).toHaveBeenCalledWith('/login');
  });

  it('sends an expired admin session to the admin sign-in', async () => {
    location.pathname = '/admin';
    fetch.mockResolvedValueOnce(fail(401, { detail: 'Not authenticated' }));
    await expect(api.getUsers()).rejects.toThrow();
    expect(location.assign).toHaveBeenCalledWith('/admin/login');
  });

  it('does not redirect for calls that expect to be signed out, such as the session check and sign-in', async () => {
    fetch.mockResolvedValueOnce(fail(401, { detail: 'Not authenticated' }));
    await expect(api.me()).rejects.toThrow();
    fetch.mockResolvedValueOnce(fail(401, { detail: 'Incorrect username or password' }));
    await expect(api.login('a', 'b')).rejects.toThrow('Incorrect username or password');
    expect(location.assign).not.toHaveBeenCalled();
  });

  it('does not reload the sign-in page it is already on', async () => {
    location.pathname = '/login';
    fetch.mockResolvedValueOnce(fail(401, {}));
    await expect(api.getUsage()).rejects.toThrow();
    expect(location.assign).not.toHaveBeenCalled();
  });
});
