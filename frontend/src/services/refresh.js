// Silent session renewal. Kept free of browser globals so `npm test` can run it under node.

// Runs fn once at a time: callers that arrive while it is running share its result instead of starting another.
export function singleFlight(fn) {
    let inFlight = null;
    return () => {
        inFlight ??= Promise.resolve().then(fn).finally(() => { inFlight = null; });
        return inFlight;
    };
}

// Runs the request; on a 401 asks refresh() (true when the session was renewed) and, if so, runs the request once more.
export async function fetchWithRefresh(send, refresh) {
    const res = await send();
    if (res.status !== 401) return res;
    return (await refresh()) ? send() : res;
}
