import { describe, expect, it } from 'vitest';
import { describeError } from './errors';

const err = (status, message = '', extra = {}) => Object.assign(new Error(message), { status, ...extra });

describe('describeError', () => {
  it('treats an unreachable server and gateway failures as offline', () => {
    expect(describeError(err(0, "Can't reach", { offline: true })).kind).toBe('offline');
    expect(describeError(err(502, 'Bad Gateway')).kind).toBe('offline');
    expect(describeError(err(504, 'Gateway Timeout')).kind).toBe('offline');
  });

  it('keeps the server detail for a feature switched off on this deployment', () => {
    const d = describeError(err(501, 'Direct upload needs S3-compatible storage.'));
    expect(d).toMatchObject({ kind: 'unavailable', message: 'Direct upload needs S3-compatible storage.' });
    expect(describeError(err(503, 'Authentication is not configured')).kind).toBe('unavailable');
  });

  it('names rate limits, missing access and missing items', () => {
    expect(describeError(err(429, 'Too many login attempts. Try again in one minute.'))).toMatchObject({ kind: 'rate_limited', message: 'Too many login attempts. Try again in one minute.' });
    expect(describeError(err(403)).kind).toBe('forbidden');
    expect(describeError(err(404)).kind).toBe('not_found');
  });

  it('falls back to a plain failure, never to offline, for errors without a status', () => {
    expect(describeError(new Error('boom'))).toMatchObject({ kind: 'failed', message: 'boom' });
    expect(describeError(undefined)).toMatchObject({ kind: 'failed' });
    expect(describeError(err(500, 'Internal Server Error')).kind).toBe('failed');
  });
});
