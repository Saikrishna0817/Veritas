/**
 * The whole scroll transformation as a pure function:
 *   pose(key) = f(progress, ring angles, time, mouse push)
 *
 * Everything that must be reversible depends on scroll progress only. The only time-integrated
 * quantity is each ring's spin angle, which never affects the *shape* (a ring is rotationally
 * symmetric), so scrolling back always restores a perfect sphere.
 */
import { config } from '../config'
import type { KeyDef, Layout, LetterDef } from './layout'
import {
  type Q4, type V3,
  clamp, ease, q4, qaxisAngle, qcopy, qmul, qslerp, remap, smootherstep, smoothstep,
  v3, vadd, vaddScaled, vbezier, vcopy, vrotate, vscale, vset, vsub,
} from '../utils/math'

export interface Weights {
  activation: number
  decoding: number
  detach: number
  disassembly: number
  glow: number
  environment: number
  reconstruction: number
  final: number
  mouse: number
}

export function phaseWeights(p: number): Weights {
  const ph = config.phases
  const a = smoothstep(ph.activation[0], ph.activation[1], p)
  const d = smootherstep(ph.decoding[0], ph.decoding[1], p)
  const glowUp = 0.55 * smoothstep(ph.activation[0] + 0.01, ph.activation[1], p) + 0.45 * smoothstep(ph.decoding[0], ph.decoding[1] - 0.04, p)
  const glowDown = 1 - smoothstep(ph.disassembly[0] + 0.04, ph.disassembly[1] + 0.03, p)
  return {
    activation: a,
    decoding: d,
    detach: smoothstep(ph.decoding[0] + 0.09, ph.decoding[1] + 0.01, p),
    disassembly: remap(p, ph.disassembly[0], ph.disassembly[1]),
    glow: glowUp * glowDown,
    environment: smoothstep(ph.disassembly[0] + 0.05, ph.reconstruction[0] + 0.14, p),
    reconstruction: remap(p, ph.reconstruction[0], ph.reconstruction[1]),
    final: smoothstep(ph.reconstruction[1] - 0.04, ph.final[1] - 0.02, p),
    mouse: 1 - smoothstep(config.mouse.fadeOut[0], config.mouse.fadeOut[1], p),
  }
}

/** Ring spin speed (rad/s) on top of the global idle rotation. */
export function ringSpinSpeed(layout: Layout, ring: number, w: Weights): number {
  const C = config.choreography
  const r = layout.rings[ring]
  const base = C.activationSpin * w.activation + (C.decodingSpin - C.activationSpin) * w.decoding
  const fall = (C.decodingSpin - C.disassemblySpin) * smoothstep(0, 0.6, w.disassembly)
  return r.dir * r.spinMult * Math.max(0, base - fall)
}

export function exitProgress(layout: Layout, ring: number, p: number): number {
  const r = layout.rings[ring]
  const start = config.phases.disassembly[0] + r.exitDelay
  return remap(p, start, start + config.choreography.exitDuration)
}

export interface FrameInput {
  p: number
  time: number
  /** Global idle yaw and bob of the whole sphere. */
  idleYaw: number
  bob: number
  tiltX: number
  tiltZ: number
  /** Accumulated spin per ring. */
  ringAngle: ArrayLike<number>
  /** Mouse push per key (units along the key normal). */
  push: ArrayLike<number>
  /** Third-section anagram progress (0..1). */
  material?: number
}

/* scratch */
const _yaw = q4()
const _tilt = q4()
const _g = q4()
const _tmpQ = q4()
const _tmpQ2 = q4()
const _lp = v3()
const _center = v3()
const _p0 = v3()
const _q0 = q4()
const _anchor = v3()
const _slot = v3()
const _slotQ = q4()
const _anchorQ = q4()
const _c1 = v3()
const _c2 = v3()
const AXIS_Y: V3 = [0, 1, 0]
const AXIS_X: V3 = [1, 0, 0]
const AXIS_Z: V3 = [0, 0, 1]

/** Global sphere transform (idle yaw + subtle wobble). */
function globalQuat(out: Q4, f: FrameInput) {
  qaxisAngle(_yaw, AXIS_Y, f.idleYaw)
  qaxisAngle(_tilt, AXIS_X, f.tiltX)
  qmul(out, _yaw, _tilt)
  qaxisAngle(_tilt, AXIS_Z, f.tiltZ)
  return qmul(out, out, _tilt)
}

/** Sphere centre in sphere-local units (world centre / worldScale), including the idle bob. */
export function sphereCenter(out: V3, f: FrameInput) {
  const c = config.sphere.center
  const k = 1 / config.sphere.worldScale
  return vset(out, c[0] * k, c[1] * k + f.bob, c[2] * k)
}

/**
 * Pose of a key while it belongs to the sphere / its ring (phases 1–3).
 * Writes world position (key base centre) and orientation.
 */
export function ringPose(layout: Layout, k: KeyDef, f: FrameInput, w: Weights, outPos: V3, outQ: Q4) {
  const S = config.sphere
  const C = config.choreography
  const r = layout.rings[k.ring]
  const R = S.radius

  // radial distance: base radius + micro breathing + mouse + controlled detachment
  const micro = S.microAmplitude * Math.sin(f.time * 0.9 + k.phase) * (1 - w.disassembly)
  const rho = R + micro + (f.push[k.id] || 0) + k.detach * C.detachDistance * w.detach

  // ring spin
  qaxisAngle(_tmpQ, AXIS_Y, f.ringAngle[k.ring])
  vscale(_lp, k.dir, rho)
  vrotate(_lp, _lp, _tmpQ)
  qmul(outQ, _tmpQ, k.quat)

  // relative to ring centre so the ring can tilt/expand about itself
  const cy = R * Math.sin(r.lat)
  _lp[1] -= cy

  // decoding: ring tilt
  // small polar rings barely move off-axis; wide rings carry the mechanical drift
  const cl = Math.cos(r.lat)
  const offAxis = cl * cl
  qaxisAngle(_tmpQ, r.tiltAxis, r.tilt * C.ringTilt * w.decoding * offAxis)
  vrotate(_lp, _lp, _tmpQ)
  qmul(outQ, _tmpQ, outQ)

  // disassembly: the ring keeps its circle and spin, grows in radius and climbs/sinks along a cone
  const s = exitProgress(layout, k.ring, f.p)
  if (s > 0) {
    const dr = C.exitGrowth * ease.exit(s) * (1 + 0.04 * k.scatter)
    const hl = Math.hypot(_lp[0], _lp[2])
    if (hl > 1e-4) {
      _lp[0] += (_lp[0] / hl) * dr
      _lp[2] += (_lp[2] / hl) * dr
    }
    _lp[1] += r.exitSlope * C.exitGrowth * ease.exit(s)
    // a gentle tumble once the gaps have opened up
    qaxisAngle(_tmpQ, k.tumbleAxis, k.tumble * ease.inCubic(s))
    qmul(outQ, outQ, _tmpQ)
  }

  // back to sphere space: ring centre + decoding spread + drift
  const mid = (layout.rings.length - 1) / 2
  _lp[1] += cy + (k.ring - mid) * C.ringSpread * w.decoding
  vaddScaled(_lp, _lp, r.drift, C.ringDrift * w.decoding * offAxis)

  // global transform
  globalQuat(_g, f)
  vrotate(_lp, _lp, _g)
  qmul(outQ, _g, outQ)
  sphereCenter(_center, f)
  vadd(outPos, _center, _lp)

}

/** Final slot pose including the weightless float. */
function slotPose(L: LetterDef, f: FrameInput, w: Weights, outPos: V3, outQ: Q4, slot: Readonly<V3> = L.slot) {
  const C = config.choreography
  const fl = w.final
  const t = f.time
  vset(
    outPos,
    slot[0],
    slot[1] + Math.sin(t * 0.55 + L.floatPhase[0]) * C.floatAmplitude * fl,
    slot[2] + Math.sin(t * 0.41 + L.floatPhase[1]) * C.floatDepth * fl,
  )
  // slow, independent wobble on all three axes (two detuned sines per axis → never looks looped)
  const wob = (i: number, f1: number, f2: number) =>
    (Math.sin(t * f1 + L.floatPhase[i]) * 0.7 + Math.sin(t * f2 + L.floatPhase[(i + 1) % 3] * 1.7) * 0.3) * L.wobble[i] * fl
  qaxisAngle(_tmpQ2, AXIS_X, wob(0, 0.47, 0.83))
  qmul(outQ, _tmpQ2, L.slotQuat)
  qaxisAngle(_tmpQ2, AXIS_Y, wob(1, 0.36, 0.71))
  qmul(outQ, _tmpQ2, outQ)
  qaxisAngle(_tmpQ2, AXIS_Z, wob(2, 0.29, 0.61))
  qmul(outQ, _tmpQ2, outQ)
}

export function computePose(layout: Layout, keyId: number, f: FrameInput, w: Weights, outPos: V3, outQ: Q4) {
  const k = layout.keys[keyId]
  ringPose(layout, k, f, w, outPos, outQ)
  if (k.letter < 0) return

  const C = config.choreography
  const L = layout.letters[k.letter]
  // phrase keys sink one after another from their ring into the hollow centre (a straight path
  // through the empty interior — they never cross the shell or the leaving rings)
  const span = C.holdWindow[1] - C.holdWindow[0]
  const h0 = C.holdWindow[0] + L.holdOrder * span * 0.75
  const hold = smootherstep(h0, h0 + span * 0.25, f.p)
  if (hold <= 0) return

  const t = f.time
  vset(
    _anchor,
    L.anchor[0] + Math.sin(t * 0.31 + L.floatPhase[0]) * 0.25,
    L.anchor[1] + Math.sin(t * 0.43 + L.floatPhase[1]) * 0.22 + f.bob,
    L.anchor[2] + Math.sin(t * 0.27 + L.floatPhase[2]) * 0.3,
  )
  qaxisAngle(_tmpQ2, L.swayAxis, Math.sin(t * 0.5 + L.floatPhase[1]) * 0.05)
  qmul(_anchorQ, _tmpQ2, L.anchorQuat)

  vcopy(_p0, outPos)
  qcopy(_q0, outQ)
  // enter the hollow radially (radial paths never cross near the shell), then settle onto the
  // anchor along the depth axis, from the side the key came from — never sliding through the
  // neighbours already hovering in the anchor plane
  sphereCenter(_center, f)
  vsub(_c1, _p0, _center)
  vscale(_c1, _c1, 0.5)
  vadd(_c1, _c1, _center)
  const side = _p0[2] - _center[2] >= 0 ? 1 : -1
  vset(_c2, _anchor[0], _anchor[1], _anchor[2] + side * 7.5)
  vbezier(outPos, _p0, _c1, _c2, _anchor, ease.inOutCubic(hold))
  qslerp(outQ, _q0, _anchorQ, ease.inOutCubic(hold))

  const rec = remap(f.p, L.start, L.start + C.reconstructDuration)
  if (rec <= 0) return

  // self-assembly: a soft forward arc into the slot, with depth/orientation offsets en route
  slotPose(L, f, w, _slot, _slotQ)
  const e = ease.inOutCubic(rec)
  // neighbours fly in separate depth lanes, so their arcs can overlap on screen but never in space
  const lane = L.lane * 4.2
  vset(_c1, _anchor[0] + L.lift[0], _anchor[1] + L.lift[1], _anchor[2] + 6 + lane)
  vset(_c2, _slot[0] - L.lift[0] * 0.5, _slot[1] + L.lift[1] * 0.2, _slot[2] + 5 + lane)
  vbezier(outPos, _anchor, _c1, _c2, _slot, e)
  qslerp(_tmpQ, _anchorQ, _slotQ, ease.outCubic(clamp(rec * 1.08)))
  qaxisAngle(_tmpQ2, L.swayAxis, Math.sin(Math.PI * rec) * (1 - rec) * 0.32)
  qmul(outQ, _tmpQ2, _tmpQ)

  const m = f.material ?? 0
  if (m <= 0) return
  materialPose(L, f, w, m, outPos, outQ)
}

const _mFrom = v3()
const _mTo = v3()
const _mQ = q4()

/**
 * Third section: MAKE IT REAL re-spells itself as the anagram. Each used key travels to its new
 * slot in its own depth lane while making one full turn (the hollow underside flashes by);
 * unused keys tip over and fall into the dark.
 */
function materialPose(L: LetterDef, f: FrameInput, w: Weights, m: number, outPos: V3, outQ: Q4) {
  vcopy(_mFrom, outPos)
  qcopy(_mQ, outQ)
  if (!L.materialSlot) {
    // first sink back out of the composition plane, then drop — never through the other keys
    const k = remap(m, 0, 0.45)
    const back = ease.outCubic(remap(k, 0, 0.45))
    // they do not fall like stones: a slow, accelerating sink into the dark
    const drop = ease.inQuad(remap(k, 0.2, 1))
    vset(outPos, _mFrom[0] + L.fall[0] * drop, _mFrom[1] + L.fall[1] * drop, _mFrom[2] + L.fall[2] * back)
    qaxisAngle(_tmpQ2, AXIS_X, -drop * 2.2)
    qmul(outQ, _tmpQ2, _mQ)
    return
  }
  const k = remap(m, L.materialStart, L.materialStart + 0.5)
  if (k <= 0) return
  slotPose(L, f, w, _mTo, _tmpQ, L.materialSlot)
  // 1) step into the depth lane, 2) travel (turning), 3) step back into the plane
  const inLane = smoothstep(0, 0.2, k) * (1 - smoothstep(0.8, 1, k))
  // weightless: a sine ease (no hard push-off, no braking), a slow float of the whole path
  // and a lazy tumble about the key's own sway axis that peaks mid-flight and dies out on arrival
  const e = ease.inOutSine(remap(k, 0.14, 0.86))
  const air = Math.sin(Math.PI * e)
  const drift = Math.sin(L.floatPhase[0] + e * 5.2) * 0.35 * air
  vset(
    outPos,
    _mFrom[0] + (_mTo[0] - _mFrom[0]) * e,
    _mFrom[1] + (_mTo[1] - _mFrom[1]) * e + air * 1.5 + drift,
    _mFrom[2] + (_mTo[2] - _mFrom[2]) * e + L.materialLane * inLane,
  )
  qslerp(outQ, _mQ, _tmpQ, e)
  qaxisAngle(_tmpQ2, L.swayAxis, air * air * 0.26 * Math.sin(L.floatPhase[1] + e * 3.1))
  qmul(outQ, _tmpQ2, outQ)
  if (L.flip !== 0) {
    // one full turn about the key's own centre while it travels (its hollow back flashes by)
    // each key starts and ends its turn at its own moment → an asynchronous ripple of turns
    const t0 = 0.22 + (L.floatPhase[2] / 6.28) * 0.08
    const a = L.flip * Math.PI * 2 * ease.inOutSine(remap(k, t0, t0 + 0.5))
    qaxisAngle(_tmpQ2, AXIS_Y, a)
    const hz = config.sphere.keyScale * 1.19 * 0.5 // half height: base → centre along the facing axis
    // rotate about the centre instead of the base: shift by (R·c − c)
    vset(_c1, 0, 0, hz)
    vrotate(_c2, _c1, _tmpQ2)
    outPos[0] -= _c2[0] - _c1[0]
    outPos[2] -= _c2[2] - _c1[2]
    qmul(outQ, _tmpQ2, outQ)
  }
}
