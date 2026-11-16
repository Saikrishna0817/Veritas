/**
 * Mutable, non-reactive experience state.
 * GSAP writes the scroll target, the Director smooths it once per frame, every scene system
 * reads from here inside useFrame — no React re-renders per frame.
 */
import { phaseWeights, type Weights } from '../choreography/choreography'

export const store = {
  /** Raw scroll progress from ScrollTrigger (already scrubbed by GSAP). */
  target: 0,
  /** Frame-smoothed raw scroll progress (0..1 over the whole pinned scroll). */
  scroll: 0,
  /** Sphere-story progress (0..1 over the first `phases.storyEnd` of the scroll) — used by the scene. */
  progress: 0,
  /** Third-section anagram progress (0..1). */
  material: 0,
  /** Cursor light: world position on the phrase plane and its presence (0..1). */
  cursor: { x: 0, y: 0, z: 0, presence: 0 },
  /** Phase weights derived from progress (refreshed by the Director). */
  w: phaseWeights(0) as Weights,
  /** Experience clock (seconds). */
  time: 0,
  /** Pointer in normalised device coords; `active` false when outside the window. */
  pointer: { x: 0, y: 0, active: false },
  /** Smoothed pointer for camera parallax. */
  pointerSmooth: { x: 0, y: 0 },
  /** Preloader reveal drawn inside WebGL: 0 = screen fully covered by the preloader colour, 1 = open.
   *  Starts open (the DOM preloader covers the load); the preloader sets 0 when its exit begins. */
  reveal: 1,
  /** True while the pointer is over a 3D key that reacts to it (the custom cursor takes its hover state). */
  pointerOverKey: false,
  /** Frames the post chain has drawn (the preloader waits for the cover to be on screen before swapping). */
  postFrames: 0,
  /** 1 = full quality, 0 = reduced (set by the performance monitor). */
  quality: 1,
}

export type Store = typeof store
