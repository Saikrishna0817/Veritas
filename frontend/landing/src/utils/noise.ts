/**
 * Seeded 2D simplex noise (Gustavson / Stefan Gustavson's reference, public domain)
 * plus fractal helpers used by the CPU terrain generator.
 */
import { mulberry32 } from './math'

const F2 = 0.5 * (Math.sqrt(3) - 1)
const G2 = (3 - Math.sqrt(3)) / 6
const GRAD = [
  [1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1],
  [0.7071, 0.7071], [-0.7071, 0.7071], [0.7071, -0.7071], [-0.7071, -0.7071],
]

export function createNoise2D(seed: number) {
  const rnd = mulberry32(seed)
  const p = new Uint8Array(256)
  for (let i = 0; i < 256; i++) p[i] = i
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1))
    const t = p[i]
    p[i] = p[j]
    p[j] = t
  }
  const perm = new Uint8Array(512)
  const permMod = new Uint8Array(512)
  for (let i = 0; i < 512; i++) {
    perm[i] = p[i & 255]
    permMod[i] = perm[i] % GRAD.length
  }

  return function noise2D(xin: number, yin: number): number {
    const s = (xin + yin) * F2
    const i = Math.floor(xin + s)
    const j = Math.floor(yin + s)
    const t = (i + j) * G2
    const x0 = xin - (i - t)
    const y0 = yin - (j - t)
    const i1 = x0 > y0 ? 1 : 0
    const j1 = x0 > y0 ? 0 : 1
    const x1 = x0 - i1 + G2
    const y1 = y0 - j1 + G2
    const x2 = x0 - 1 + 2 * G2
    const y2 = y0 - 1 + 2 * G2
    const ii = i & 255
    const jj = j & 255
    let n0 = 0
    let n1 = 0
    let n2 = 0
    let t0 = 0.5 - x0 * x0 - y0 * y0
    if (t0 >= 0) {
      const g = GRAD[permMod[ii + perm[jj]]]
      t0 *= t0
      n0 = t0 * t0 * (g[0] * x0 + g[1] * y0)
    }
    let t1 = 0.5 - x1 * x1 - y1 * y1
    if (t1 >= 0) {
      const g = GRAD[permMod[ii + i1 + perm[jj + j1]]]
      t1 *= t1
      n1 = t1 * t1 * (g[0] * x1 + g[1] * y1)
    }
    let t2 = 0.5 - x2 * x2 - y2 * y2
    if (t2 >= 0) {
      const g = GRAD[permMod[ii + 1 + perm[jj + 1]]]
      t2 *= t2
      n2 = t2 * t2 * (g[0] * x2 + g[1] * y2)
    }
    return 70 * (n0 + n1 + n2) // ~[-1, 1]
  }
}

export type Noise2D = ReturnType<typeof createNoise2D>

export function fbm(n: Noise2D, x: number, y: number, octaves = 5, lac = 2.02, gain = 0.5) {
  let sum = 0
  let amp = 0.5
  let f = 1
  for (let o = 0; o < octaves; o++) {
    sum += amp * n(x * f, y * f)
    f *= lac
    amp *= gain
  }
  return sum
}

/** Ridged multifractal: sharp crests, eroded valleys. Returns ~[0, 1]. */
export function ridged(n: Noise2D, x: number, y: number, octaves = 7, lac = 2.07, gain = 0.52) {
  let sum = 0
  let amp = 0.5
  let f = 1
  let weight = 1
  let norm = 0
  for (let o = 0; o < octaves; o++) {
    let v = 1 - Math.abs(n(x * f + o * 17.3, y * f - o * 9.1))
    v *= v
    v *= weight
    weight = Math.min(1, Math.max(0, v * 1.6))
    sum += v * amp
    norm += amp
    f *= lac
    amp *= gain
  }
  return sum / norm
}
