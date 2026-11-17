/**
 * World-space framing helpers shared by the camera, the lights and the tests.
 * The choreography works in sphere-local units; the world is that × sphere.worldScale.
 */
import { config } from '../config'

export const WORLD_SCALE = config.sphere.worldScale

/** Width of the finished phrase in sphere-local units (outer key edges). */
export function phraseLocalWidth(): number {
  const P = config.phrase
  const words = P.text.split(/\s+/)
  const keys = words.reduce((n, w) => n + w.length, 0)
  const keyWidth = 2 * config.sphere.keyScale
  return (keys - 1) * P.pitch + (words.length - 1) * P.wordGap + keyWidth
}

/** World centre of the phrase. */
export function phraseWorldCenter(): [number, number, number] {
  const c = config.phrase.center
  return [c[0] * WORLD_SCALE, c[1] * WORLD_SCALE, c[2] * WORLD_SCALE]
}

/** Camera distance (from the phrase plane) so one keycap spans `keyScreenWidth` of the viewport width. */
export function finalCameraDistance(aspect: number, fov = config.camera.final.fov): number {
  const keyWidth = 2 * config.sphere.keyScale * WORLD_SCALE
  const visibleWidthPerUnit = 2 * Math.tan((fov * Math.PI) / 360) * aspect
  return keyWidth / (config.phrase.keyScreenWidth * visibleWidthPerUnit) + keyWidth * 0.6
}
