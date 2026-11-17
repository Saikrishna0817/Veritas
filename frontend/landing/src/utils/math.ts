/**
 * Small allocation-free math helpers used by the choreography.
 * Kept independent from three.js so the choreography can be unit-tested in Node.
 */

export type V3 = [number, number, number]
export type Q4 = [number, number, number, number] // x y z w

export const clamp = (v: number, a = 0, b = 1) => (v < a ? a : v > b ? b : v)
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t
export const remap = (v: number, a: number, b: number) => clamp((v - a) / (b - a))
export const smoothstep = (a: number, b: number, v: number) => {
  const t = remap(v, a, b)
  return t * t * (3 - 2 * t)
}
export const smootherstep = (a: number, b: number, v: number) => {
  const t = remap(v, a, b)
  return t * t * t * (t * (t * 6 - 15) + 10)
}

export const ease = {
  inCubic: (t: number) => t * t * t,
  outCubic: (t: number) => 1 - Math.pow(1 - t, 3),
  inOutCubic: (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  inQuad: (t: number) => t * t,
  outQuart: (t: number) => 1 - Math.pow(1 - t, 4),
  inOutSine: (t: number) => -(Math.cos(Math.PI * t) - 1) / 2,
  /** Accelerating exit that keeps a little velocity at t=0 (no dead start). */
  exit: (t: number) => 0.18 * t + 0.82 * t * t * t,
}

/** Deterministic PRNG. */
export function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/* ----------------------------- vectors ----------------------------- */

export const v3 = (x = 0, y = 0, z = 0): V3 => [x, y, z]
export const vset = (o: V3, x: number, y: number, z: number) => {
  o[0] = x
  o[1] = y
  o[2] = z
  return o
}
export const vcopy = (o: V3, a: Readonly<V3>) => vset(o, a[0], a[1], a[2])
export const vadd = (o: V3, a: Readonly<V3>, b: Readonly<V3>) => vset(o, a[0] + b[0], a[1] + b[1], a[2] + b[2])
export const vsub = (o: V3, a: Readonly<V3>, b: Readonly<V3>) => vset(o, a[0] - b[0], a[1] - b[1], a[2] - b[2])
export const vscale = (o: V3, a: Readonly<V3>, s: number) => vset(o, a[0] * s, a[1] * s, a[2] * s)
export const vaddScaled = (o: V3, a: Readonly<V3>, b: Readonly<V3>, s: number) =>
  vset(o, a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s)
export const vdot = (a: Readonly<V3>, b: Readonly<V3>) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
export const vlen = (a: Readonly<V3>) => Math.hypot(a[0], a[1], a[2])
export const vcross = (o: V3, a: Readonly<V3>, b: Readonly<V3>) =>
  vset(o, a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
export const vnorm = (o: V3, a: Readonly<V3>) => {
  const l = vlen(a) || 1
  return vset(o, a[0] / l, a[1] / l, a[2] / l)
}
export const vlerp = (o: V3, a: Readonly<V3>, b: Readonly<V3>, t: number) =>
  vset(o, a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)

/** Spherical interpolation between two unit vectors. */
export function vslerpUnit(o: V3, a: Readonly<V3>, b: Readonly<V3>, t: number) {
  const d = clamp(vdot(a, b), -1, 1)
  const th = Math.acos(d)
  if (th < 1e-4) return vlerp(o, a, b, t)
  if (Math.PI - th < 1e-3) {
    // antipodal: rotate around an arbitrary perpendicular axis
    const ax: V3 = Math.abs(a[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]
    const p = vnorm(v3(), vcross(v3(), a, ax))
    const c = Math.cos(th * t)
    const s = Math.sin(th * t)
    return vset(o, a[0] * c + p[0] * s, a[1] * c + p[1] * s, a[2] * c + p[2] * s)
  }
  const s = Math.sin(th)
  const wa = Math.sin((1 - t) * th) / s
  const wb = Math.sin(t * th) / s
  return vset(o, a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb, a[2] * wa + b[2] * wb)
}

/** Cubic bezier. */
export function vbezier(o: V3, p0: Readonly<V3>, p1: Readonly<V3>, p2: Readonly<V3>, p3: Readonly<V3>, t: number) {
  const u = 1 - t
  const a = u * u * u
  const b = 3 * u * u * t
  const c = 3 * u * t * t
  const d = t * t * t
  return vset(
    o,
    a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
    a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
    a * p0[2] + b * p1[2] + c * p2[2] + d * p3[2],
  )
}

/* ---------------------------- quaternions --------------------------- */

export const q4 = (): Q4 => [0, 0, 0, 1]
export const qset = (o: Q4, x: number, y: number, z: number, w: number) => {
  o[0] = x
  o[1] = y
  o[2] = z
  o[3] = w
  return o
}
export const qcopy = (o: Q4, a: Readonly<Q4>) => qset(o, a[0], a[1], a[2], a[3])

export function qaxisAngle(o: Q4, axis: Readonly<V3>, angle: number) {
  const h = angle / 2
  const s = Math.sin(h)
  const l = vlen(axis) || 1
  return qset(o, (axis[0] / l) * s, (axis[1] / l) * s, (axis[2] / l) * s, Math.cos(h))
}

/** o = a * b (apply b first, then a). */
export function qmul(o: Q4, a: Readonly<Q4>, b: Readonly<Q4>) {
  const [ax, ay, az, aw] = a
  const [bx, by, bz, bw] = b
  return qset(
    o,
    ax * bw + aw * bx + ay * bz - az * by,
    ay * bw + aw * by + az * bx - ax * bz,
    az * bw + aw * bz + ax * by - ay * bx,
    aw * bw - ax * bx - ay * by - az * bz,
  )
}

export function qslerp(o: Q4, a: Readonly<Q4>, b: Readonly<Q4>, t: number) {
  let [bx, by, bz, bw] = b
  let cos = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw
  if (cos < 0) {
    cos = -cos
    bx = -bx
    by = -by
    bz = -bz
    bw = -bw
  }
  let k0: number
  let k1: number
  if (cos > 0.9995) {
    k0 = 1 - t
    k1 = t
  } else {
    const th = Math.acos(cos)
    const s = Math.sin(th)
    k0 = Math.sin((1 - t) * th) / s
    k1 = Math.sin(t * th) / s
  }
  qset(o, a[0] * k0 + bx * k1, a[1] * k0 + by * k1, a[2] * k0 + bz * k1, a[3] * k0 + bw * k1)
  const l = Math.hypot(o[0], o[1], o[2], o[3]) || 1
  return qset(o, o[0] / l, o[1] / l, o[2] / l, o[3] / l)
}

/** Rotate vector by quaternion. */
export function vrotate(o: V3, v: Readonly<V3>, q: Readonly<Q4>) {
  const [x, y, z] = v
  const [qx, qy, qz, qw] = q
  const ix = qw * x + qy * z - qz * y
  const iy = qw * y + qz * x - qx * z
  const iz = qw * z + qx * y - qy * x
  const iw = -qx * x - qy * y - qz * z
  return vset(
    o,
    ix * qw + iw * -qx + iy * -qz - iz * -qy,
    iy * qw + iw * -qy + iz * -qx - ix * -qz,
    iz * qw + iw * -qz + ix * -qy - iy * -qx,
  )
}

/** Quaternion from an orthonormal basis given as columns (X, Y, Z). */
export function qfromBasis(o: Q4, X: Readonly<V3>, Y: Readonly<V3>, Z: Readonly<V3>) {
  const m11 = X[0], m12 = Y[0], m13 = Z[0]
  const m21 = X[1], m22 = Y[1], m23 = Z[1]
  const m31 = X[2], m32 = Y[2], m33 = Z[2]
  const tr = m11 + m22 + m33
  if (tr > 0) {
    const s = 0.5 / Math.sqrt(tr + 1)
    return qset(o, (m32 - m23) * s, (m13 - m31) * s, (m21 - m12) * s, 0.25 / s)
  } else if (m11 > m22 && m11 > m33) {
    const s = 2 * Math.sqrt(1 + m11 - m22 - m33)
    return qset(o, 0.25 * s, (m12 + m21) / s, (m13 + m31) / s, (m32 - m23) / s)
  } else if (m22 > m33) {
    const s = 2 * Math.sqrt(1 + m22 - m11 - m33)
    return qset(o, (m12 + m21) / s, 0.25 * s, (m23 + m32) / s, (m13 - m31) / s)
  } else {
    const s = 2 * Math.sqrt(1 + m33 - m11 - m22)
    return qset(o, (m13 + m31) / s, (m23 + m32) / s, 0.25 * s, (m21 - m12) / s)
  }
}
