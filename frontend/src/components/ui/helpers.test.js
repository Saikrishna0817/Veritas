import { describe, expect, it } from 'vitest';
import { DATA_VERDICTS, MODEL_VERDICTS, verdictInfo } from './verdicts';
import { orDash, percent } from './format';

describe('verdictInfo', () => {
  it('returns the known entry for a known code', () => {
    expect(verdictInfo('CLEAN')).toBe(DATA_VERDICTS.CLEAN);
    expect(verdictInfo('DANGEROUS', 'model')).toBe(MODEL_VERDICTS.DANGEROUS);
  });

  it('never borrows another verdict for an unknown or missing code', () => {
    const v = verdictInfo('NEW_BACKEND_CODE');
    expect(v.label).toBe('new backend code');
    expect(v.color).toBe('var(--text-muted)');
    expect(Object.values(DATA_VERDICTS)).not.toContain(v);
    expect(verdictInfo(undefined).label).toBe('No verdict');
    expect(verdictInfo('CLEAN', 'model').label).toBe('clean');   // a data code is not a model verdict
  });
});

describe('orDash and percent', () => {
  it('shows a dash for missing values, never a 0', () => {
    expect(orDash(undefined)).toBe('—');
    expect(orDash(null)).toBe('—');
    expect(orDash(Number.NaN)).toBe('—');
    expect(percent(undefined)).toBe('—');
  });

  it('keeps a real zero and formats present values', () => {
    expect(orDash(0)).toBe(0);
    expect(orDash(1234, (n) => n.toLocaleString('en-US'))).toBe('1,234');
    expect(percent(0)).toBe('0%');
    expect(percent(0.426)).toBe('43%');
  });
});
