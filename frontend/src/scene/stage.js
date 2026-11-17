import { useEffect, useSyncExternalStore } from 'react';

/**
 * What the 3D world shows, and where. A page marks an empty part of its own layout as the stage
 * (`useStage(kind, ref, options)`); the world (scene/World.jsx) draws that subject inside the element's on-screen box
 * and follows it on scroll and resize. Without WebGL the element stays empty. Lives in the main chunk: it is tiny and
 * pages use it before three.js has loaded.
 *
 * Kinds: 'sphere' (the keycap sphere at rest, sign-in pages).
 * Options: busy (a request is running: the sphere turns slowly until it ends).
 * pressKey(): one key dips and springs back (the person typed something; which key is random, never the character).
 */
let state = { kind: null, el: null, busy: false, presses: 0 };
const listeners = new Set();
const emit = () => listeners.forEach((l) => l());
const set = (patch) => {
  state = { ...state, ...patch };
  emit();
};

export const stage = {
  subscribe(l) {
    listeners.add(l);
    return () => listeners.delete(l);
  },
  get: () => state,
};

export function pressKey() {
  set({ presses: state.presses + 1 });
}

export function useStageState() {
  return useSyncExternalStore(stage.subscribe, stage.get);
}

export function useStage(kind, ref, { busy = false } = {}) {
  useEffect(() => {
    const el = ref.current;
    set({ kind, el });
    return () => {
      if (stage.get().el === el) set({ kind: null, el: null, busy: false });
    };
  }, [kind, ref]);
  useEffect(() => {
    if (stage.get().busy !== busy) set({ busy });
  }, [busy]);
}
