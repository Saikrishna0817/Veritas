import { useState } from 'react';

/**
 * "Load more" for a list loaded with useLoad. fetchPage(before) resolves to { items, next }; pass the first
 * page's cursor as `first`. Returns the pages loaded after the first, the cursor, and a loader.
 */
export function useMore(fetchPage, first) {
  const [state, setState] = useState({ items: [], next: undefined, busy: false, error: null });
  const next = state.next === undefined ? first : state.next;
  const load = async () => {
    setState((s) => ({ ...s, busy: true, error: null }));
    try {
      const r = await fetchPage(next);
      setState((s) => ({ items: [...s.items, ...r.items], next: r.next, busy: false, error: null }));
    } catch (error) {
      setState((s) => ({ ...s, busy: false, error }));
    }
  };
  return { items: state.items, next, busy: state.busy, error: state.error, load };
}
