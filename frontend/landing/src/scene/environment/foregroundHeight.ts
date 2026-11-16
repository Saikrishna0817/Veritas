/**
 * Procedural foreground massif (pure, deterministic): one snowy peak on a square patch, built from the
 * same noise as the valley (terrainHeight.ts) — a peaked base shape, ridged multifractal crests with
 * domain warping, and radial spurs running down from the summit — plus its erosion / detail map.
 *
 * Patch: `extent` × `extent` units centred on the origin, sampled on a `size` × `size` grid.
 * The map is greyscale: bright = settled snow and scree, dark = exposed rock in gullies and on steep faces.
 */
import { createNoise2D, fbm, ridged } from '../../utils/noise'
import { smoothstep } from '../../utils/math'

export const FOREGROUND = {
  seed: 23,
  /** Grid resolution of the heightfield (vertices per side). */
  size: 362,
  /** Patch size in model units (the placements in config.terrain.foreground scale it). */
  extent: 1000,
  /** Summit, slightly off centre (model units). */
  peak: [-30, 10] as [number, number],
  /** Height range in model units (0 = the foot; the placements sink the copies below the valley floor). */
  low: 0,
  high: 300,
  /** Erosion map resolution (pixels per side). */
  detailSize: 1024,
}

export function createForegroundHeight(seed = FOREGROUND.seed) {
  const n1 = createNoise2D(seed)
  const n2 = createNoise2D(seed + 101)
  const n3 = createNoise2D(seed + 202)
  const F = FOREGROUND
  const half = F.extent / 2

  return function height(x: number, z: number): number {
    // domain warp: crests and couloirs wander instead of running straight
    const wx = x + 60 * fbm(n2, x * 0.003, z * 0.003, 3)
    const wz = z + 60 * fbm(n2, x * 0.003 + 7.7, z * 0.003 - 3.1, 3)
    const dx = wx - F.peak[0]
    const dz = wz - F.peak[1]
    const r = Math.hypot(dx, dz) / half
    // base: a broad dome with massive shoulders (the crests below give it its peaks)
    const cone = Math.exp(-Math.pow(r / 0.8, 2.1))
    // a long main ridge runs through the summit, falling away towards both ends
    const along = (dx * 0.8 - dz * 0.6) / half
    const across = (dx * 0.6 + dz * 0.8) / half
    const ridge = Math.exp(-Math.pow(across / 0.16, 2)) * Math.exp(-Math.pow(along / 0.95, 2))
    // spurs radiate from the summit: noise sampled along the direction away from it
    const ang = Math.atan2(dz, dx)
    const spur = ridged(n3, Math.cos(ang) * 2.2 + r * 0.5, Math.sin(ang) * 2.2 - r * 0.5, 5)
    const crest = ridged(n1, wx * 0.0028, wz * 0.0028, 7)
    let h = 0.08 + cone * 0.52 + ridge * 0.05 + (crest - 0.42) * 0.3 * (0.5 + 0.5 * cone) + (spur - 0.45) * 0.26 * smoothstep(0.04, 0.35, r) * cone
    // fine rock breakup
    h += 0.025 * fbm(n2, x * 0.02, z * 0.02, 3)
    return F.low + (F.high - F.low) * Math.max(-0.05, h)
  }
}

/** Heights on the patch grid, row-major [iz][ix], x and z from −extent/2 to +extent/2. */
export function sampleForeground(height = createForegroundHeight()) {
  const F = FOREGROUND
  const n = F.size
  const out = new Float32Array(n * n)
  for (let iz = 0; iz < n; iz++) {
    const z = -F.extent / 2 + (iz / (n - 1)) * F.extent
    for (let ix = 0; ix < n; ix++) out[iz * n + ix] = height(-F.extent / 2 + (ix / (n - 1)) * F.extent, z)
  }
  return out
}

/**
 * Erosion map from the height grid: gullies (concave, steep) go dark, ridges and gentle slopes hold snow;
 * thin streaks run down the fall line, as melt water and rockfall would carve them.
 */
export function buildForegroundDetail(heights: Float32Array, seed = FOREGROUND.seed) {
  const F = FOREGROUND
  const n = F.size
  const S = F.detailSize
  const cell = F.extent / (n - 1)
  const streak = createNoise2D(seed + 303)
  const grain = createNoise2D(seed + 404)
  const H = (ix: number, iz: number) => heights[Math.min(n - 1, Math.max(0, iz)) * n + Math.min(n - 1, Math.max(0, ix))]
  // bilinear height and its gradient / curvature at a grid position
  const out = new Uint8Array(S * S)
  for (let py = 0; py < S; py++) {
    const gz = (py / (S - 1)) * (n - 1)
    const iz = Math.floor(gz)
    const fz = gz - iz
    for (let px = 0; px < S; px++) {
      const gx = (px / (S - 1)) * (n - 1)
      const ix = Math.floor(gx)
      const fx = gx - ix
      const h00 = H(ix, iz), h10 = H(ix + 1, iz), h01 = H(ix, iz + 1), h11 = H(ix + 1, iz + 1)
      const sx = ((h10 - h00) * (1 - fz) + (h11 - h01) * fz) / cell
      const sz = ((h01 - h00) * (1 - fx) + (h11 - h10) * fx) / cell
      const slope = Math.hypot(sx, sz)
      const ci = Math.round(gx)
      const cj = Math.round(gz)
      // curvature over ±4 cells: the valley-scale form, not the fine rock breakup
      const lap = (H(ci + 4, cj) + H(ci - 4, cj) + H(ci, cj + 4) + H(ci, cj - 4) - 4 * H(ci, cj)) / (16 * cell * cell)
      // streak coordinates: stretched along the fall line (u across it, v down it)
      const wx = (px / S) * F.extent
      const wz = (py / S) * F.extent
      const len = slope > 1e-4 ? 1 / slope : 0
      const ax = sx * len
      const az = sz * len
      const u = (wx * -az + wz * ax) * 0.09
      const v = (wx * ax + wz * az) * 0.006
      const s = 1 - Math.abs(fbm(streak, u, v, 3))
      const g = fbm(grain, wx * 0.05, wz * 0.05, 3)
      // darkness: steep faces shed snow, gullies (concave) are bare rock, streaks follow the fall line
      let dark = smoothstep(0.5, 1.4, slope) * 0.38
      dark += smoothstep(0.0015, 0.01, lap) * 0.36
      dark -= smoothstep(-0.002, -0.01, lap) * 0.1
      dark += Math.pow(s, 6) * 0.36 * smoothstep(0.15, 0.6, slope)
      dark += g * 0.08
      const snow = 0.8 - 0.62 * Math.min(1, Math.max(0, dark))
      out[py * S + px] = Math.round(255 * Math.min(1, Math.max(0, snow)))
    }
  }
  return out
}
