/**
 * Keycap overlap test shared by the scripts: each key is two stacked oriented boxes (the skirt and the
 * narrower top), tested with the separating-axis theorem. `grow` inflates the boxes (a safety margin).
 */
import { config } from '../src/config'
import { type Q4, type V3, vrotate, v3, vlen } from '../src/utils/math'

export type Box = { c: V3; ax: [V3, V3, V3]; h: V3 }
const KS = config.sphere.keyScale
const PARTS: { y: number; h: V3 }[] = [
  { y: 0.3 * KS, h: [1.0 * KS, 0.3 * KS, 1.0 * KS] },
  { y: 0.89 * KS, h: [0.86 * KS, 0.3 * KS, 0.86 * KS] },
]

export function keyBoxes(p: V3, q: Q4, grow = 1): Box[] {
  const ax: [V3, V3, V3] = [vrotate(v3(), [1, 0, 0], q), vrotate(v3(), [0, 1, 0], q), vrotate(v3(), [0, 0, 1], q)]
  return PARTS.map((pt) => ({
    c: [p[0] + ax[1][0] * pt.y, p[1] + ax[1][1] * pt.y, p[2] + ax[1][2] * pt.y] as V3,
    ax,
    h: [pt.h[0] * grow, pt.h[1] * grow, pt.h[2] * grow] as V3,
  }))
}

const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

function overlap(A: Box, B: Box) {
  const d: V3 = [B.c[0] - A.c[0], B.c[1] - A.c[1], B.c[2] - A.c[2]]
  if (vlen(d) > 3.8 * KS) return false
  const axes: V3[] = [...A.ax, ...B.ax]
  for (const a of A.ax) for (const b of B.ax) { const c = cross(a, b); if (vlen(c) > 1e-6) axes.push(c) }
  for (const ax of axes) {
    const ra = A.h[0] * Math.abs(dot(A.ax[0], ax)) + A.h[1] * Math.abs(dot(A.ax[1], ax)) + A.h[2] * Math.abs(dot(A.ax[2], ax))
    const rb = B.h[0] * Math.abs(dot(B.ax[0], ax)) + B.h[1] * Math.abs(dot(B.ax[1], ax)) + B.h[2] * Math.abs(dot(B.ax[2], ax))
    if (Math.abs(dot(d, ax)) > ra + rb) return false
  }
  return true
}

export function keysOverlap(A: Box[], B: Box[]) {
  for (const a of A) for (const b of B) if (overlap(a, b)) return true
  return false
}
