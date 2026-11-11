/** The backend's verdict codes, with the words and colours the UI uses for them. */
export const DATA_VERDICTS = {
  CONFIRMED_POISONED: { label: 'Confirmed poisoned', short: 'Confirmed', color: 'var(--status-critical)', p: 'p ≤ 1e-4', blurb: 'The incoming rows differ from the trusted reference far beyond what clean batches do. Review the flagged rows before training.' },
  SUSPICIOUS: { label: 'Suspicious', short: 'Suspicious', color: 'var(--status-warn)', p: 'p ≤ 0.01', blurb: 'Stronger evidence than 99% of clean batches. Review the ranked rows first.' },
  LOW_RISK: { label: 'Low risk', short: 'Low risk', color: 'var(--status-info)', p: 'p ≤ 0.05', blurb: 'Weak evidence; about 7% of clean batches reach this level. Spot-check the top rows.' },
  CLEAN: { label: 'Clean', short: 'Clean', color: 'var(--status-safe)', p: 'p > 0.05', blurb: 'Nothing here beyond what clean batches show. Small contamination (~1%) can still hide below this.' },
};

export const MODEL_VERDICTS = {
  DANGEROUS: { label: 'Dangerous', color: 'var(--status-critical)', blurb: 'Loading this file can run code or reach the system. Do not load it.' },
  REVIEW: { label: 'Needs review', color: 'var(--status-warn)', blurb: 'Something could not be verified. A person should look before this file is loaded.' },
  NO_CODE_EXECUTION_FOUND: { label: 'No code execution found', color: 'var(--status-info)', blurb: 'Every import is a known data container. That is not a guarantee of safety.' },
  SAFE_FORMAT: { label: 'Safe format', color: 'var(--status-safe)', blurb: 'This format stores weights only and cannot contain code.' },
};
