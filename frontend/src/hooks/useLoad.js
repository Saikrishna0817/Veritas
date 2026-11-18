import { useCallback, useEffect, useState } from 'react';

/**
 * Loads data once per `load` (pass a useCallback) and lets the page try again after a failure.
 * Returns { data, error, retry }; data and error are null while loading. A result that arrives after the page moved
 * on (or after a retry) is dropped.
 */
export function useLoad(load) {
  const [state, setState] = useState({ data: null, error: null });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let live = true;
    load().then((data) => { if (live) setState({ data, error: null }); })
          .catch((error) => { if (live) setState({ data: null, error }); });
    return () => { live = false; };
  }, [load, attempt]);
  const retry = useCallback(() => { setState({ data: null, error: null }); setAttempt((a) => a + 1); }, []);
  return { ...state, retry };
}
