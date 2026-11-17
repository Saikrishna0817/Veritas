/**
 * Procedural mountain valley heightfield (pure, deterministic — also used by the preview script).
 *
 * Shape language (camera at z≈168 looking towards -z):
 *  - a deep V-shaped valley along -z, meandering slightly;
 *  - two massive dark flanks close to camera that frame the lower corners;
 *  - the valley opens with distance and ends in a high, snow-dusted range lost in fog.
 * Ridged multifractal noise with domain warping gives crests, couloirs and rock buttresses.
 */
import { createNoise2D, fbm, ridged } from '../../utils/noise'
import { smoothstep, lerp } from '../../utils/math'

export function createTerrainHeight(seed: number, floor = -78) {
  const n1 = createNoise2D(seed)
  const n2 = createNoise2D(seed + 101)
  const n3 = createNoise2D(seed + 202)

  return function height(x: number, z: number): number {
    // distance into the valley (0 at camera, grows towards -z)
    const d = Math.max(0, 170 - z)

    // meandering valley axis
    const axis = 22 * Math.sin(z * 0.0042 + 0.6) + 9 * Math.sin(z * 0.013 + 2.1)
    // domain warp for organic ridges
    const wx = x + 38 * fbm(n2, x * 0.0035, z * 0.0035, 3)
    const wz = z + 38 * fbm(n2, x * 0.0035 + 7.7, z * 0.0035 - 3.1, 3)

    // valley half-width at the floor widens with distance
    const halfW = lerp(10, 190, smoothstep(0, 1150, d))
    const ax = Math.abs(x - axis)
    const over = Math.max(0, ax - halfW)

    // flank: steep near, gentler and higher far away
    const steep = lerp(0.72, 0.2, smoothstep(60, 650, d))
    let h = Math.pow(over, 1.08) * steep

    // rocky detail scaled by how high we are on the flank
    const r = ridged(n1, wx * 0.0062, wz * 0.0062, 7)
    const flank = smoothstep(0, 60, over)
    h += (r - 0.35) * lerp(14, 62, flank) * (0.6 + 0.4 * smoothstep(0, 600, d))

    // secondary spurs that step down into the valley (buttresses)
    const spurs = ridged(n3, wx * 0.017, wz * 0.009, 4)
    h += spurs * 22 * smoothstep(halfW * 0.4, halfW + 90, ax) * (1 - smoothstep(500, 1100, d))

    // distant high range closing the valley
    const far = smoothstep(560, 950, d)
    h += far * (25 + 105 * ridged(n1, wx * 0.0028 + 11, wz * 0.0028, 7)) * (0.55 + 0.45 * smoothstep(0, 260, ax + 60))

    // limit near flanks so the frame stays open towards the sky
    // soft ceiling (keeps flanks massive but below the sphere's horizon)
  const ceil = lerp(175, 85, smoothstep(100, 600, d)) + far * 40
  if (h > ceil * 0.7) h = ceil * 0.7 + (ceil * 0.3) * Math.tanh((h - ceil * 0.7) / (ceil * 0.3))

    // gentle floor undulation
    h += 6 * fbm(n3, x * 0.01, z * 0.01, 3)

    return floor + h
  }
}
