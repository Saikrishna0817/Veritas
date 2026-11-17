import * as THREE from 'three';

/** A token from argilla.css, as written there (a hex colour). The scene takes every colour from the token file. */
export const hex = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** A token from argilla.css as a three.js colour (sRGB hex → linear). */
export const token = (name) => new THREE.Color(hex(name));

/** A small seeded random generator: the same layout on every load. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
