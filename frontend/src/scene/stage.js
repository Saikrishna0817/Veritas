import { useEffect, useSyncExternalStore } from 'react';

/**
 * What the 3D world shows, and where. A page marks an empty part of its own layout as the stage
 * (`useStage(kind, ref, options)`); the world (scene/World.jsx) draws that subject inside the element's on-screen box
 * and follows it on scroll and resize. Without WebGL the element stays empty. Lives in the main chunk: it is tiny and
 * pages use it before three.js has loaded. One stage at a time: the last one mounted wins.
 *
 * Kinds:
 *  'sphere'  the keycap sphere at rest (sign-in pages, Check data). busy: a request or job is running, so the sphere
 *            turns slowly until it ends; the turning follows the job's real state and never stands for progress.
 *  'keys'    one terracotta key per listed item of a result, most suspicious first, lifted by rank only (never by the
 *            score's size). items: [{ glyph }], at most 24. The key and its table row light up together (hot).
 *            With items: [{ glyph, tint, group, groups }] the keys form clusters (Overview), one per group, glazed
 *            by tint (a token name); hot is then the group. An item may also carry tint and lift of its own (a model
 *            file's imports, glazed and lifted by what the scan found for each).
 * pressKey(): one key of the sphere dips and springs back (the person typed; which key is random, never the character).
 */
export const MAX_KEYS = 24;

let state = { kind: null, el: null, busy: false, items: null, presses: 0, hot: -1 };
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

/** The item under the pointer, from the table or from the keys (-1: none). */
export function setHot(i) {
  if (state.hot !== i) set({ hot: i });
}

export function useStageState() {
  return useSyncExternalStore(stage.subscribe, stage.get);
}

export function useStage(kind, ref, { busy = false, items = null } = {}) {
  useEffect(() => {
    const el = ref.current;
    set({ kind, el, hot: -1 });
    return () => {
      if (stage.get().el === el) set({ kind: null, el: null, busy: false, items: null, hot: -1 });
    };
  }, [kind, ref]);
  useEffect(() => {
    if (stage.get().busy !== busy) set({ busy });
  }, [busy]);
  useEffect(() => {
    set({ items, hot: -1 });
  }, [items]);
}
