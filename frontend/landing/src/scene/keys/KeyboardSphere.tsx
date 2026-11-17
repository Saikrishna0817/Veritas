/**
 * The keyboard sphere: 77 instanced keycaps in independent rings.
 *  - 2 InstancedMeshes, 1 shared material → 2 draw calls (+2 for shadows) for every key.
 *  - Poses come from the pure choreography (scroll-driven, reversible).
 *  - Mouse: exact ray–sphere hit, gaussian falloff over the surface, per-key springs whose stiffness
 *    drops with distance → neighbours respond later and softer.
 */
import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { config } from '../../config'
import { store } from '../../state/store'
import { buildLayout } from '../../choreography/layout'
import { computePose, exitProgress, ringSpinSpeed, sphereCenter, type FrameInput } from '../../choreography/choreography'
import { WORLD_SCALE } from '../../choreography/framing'
import { type Q4, type V3, q4, v3 } from '../../utils/math'
import { useKeycapGeometries } from './useKeycapGeometry'
import { createCharacterAtlas } from './CharacterAtlas'
import { createKeycapMaterial } from './KeycapMaterials'
import { Core } from './Core'

export function KeyboardSphere({ onCore }: { onCore?: (mesh: THREE.Mesh | null) => void }) {
  const { full } = useKeycapGeometries()
  const gl = useThree((s) => s.gl)
  const layout = useMemo(() => buildLayout(), [])
  const atlas = useMemo(() => createCharacterAtlas(2048, config.keycap.legendWeight), [])
  const { material, uniforms } = useMemo(
    () =>
      createKeycapMaterial(atlas, {
        wireColor: config.wireframe.color,
        wireIntensity: config.wireframe.intensity,
        wireDensity: config.wireframe.density,
      }),
    [atlas],
  )

  const geos = useMemo(() => {
    const glyphs = (ids: number[]) =>
      new THREE.InstancedBufferAttribute(Float32Array.from(ids.map((id) => layout.keys[id].glyph)), 1)
    // each key's own colour: its ring's glaze, a scattered accent, or (the ten phrase keys) the phrase colour
    const K = config.keycap
    const tints = (ids: number[], isPhrase: boolean) => {
      const out = new Float32Array(ids.length * 3)
      const c = new THREE.Color()
      ids.forEach((id, i) => {
        const key = layout.keys[id]
        let hex = K.ringColors[key.ring] ?? K.color
        if (isPhrase) hex = K.phraseColor || hex
        else {
          // the key's own seed picks the accent, so the colourway is the same on every load
          let r = key.seed
          for (const [accent, share] of K.accents) {
            if ((r -= share) < 0) {
              hex = accent
              break
            }
          }
        }
        c.set(hex)
        out.set([c.r, c.g, c.b], i * 3)
      })
      return new THREE.InstancedBufferAttribute(out, 3)
    }
    const sphere = full.clone()
    sphere.setAttribute('aGlyph', glyphs(layout.sphereKeys))
    sphere.setAttribute('aTint', tints(layout.sphereKeys, false))
    const phrase = full.clone()
    phrase.setAttribute('aGlyph', glyphs(layout.phraseKeys))
    phrase.setAttribute('aTint', tints(layout.phraseKeys, true))
    // per-key x-ray amount (0 = plastic, 1 = dark glass + wireframe); sphere keys never use it
    sphere.setAttribute('aXray', new THREE.InstancedBufferAttribute(new Float32Array(layout.sphereKeys.length), 1))
    const xray = new THREE.InstancedBufferAttribute(new Float32Array(layout.phraseKeys.length), 1)
    xray.setUsage(THREE.DynamicDrawUsage)
    phrase.setAttribute('aXray', xray)
    return { sphere, phrase, xray }
  }, [full, layout])

  useEffect(() => {
    atlas.anisotropy = Math.min(8, gl.capabilities.getMaxAnisotropy())
  }, [atlas, gl])

  useEffect(
    () => () => {
      geos.sphere.dispose()
      geos.phrase.dispose()
      material.dispose()
      atlas.dispose()
    },
    [geos, material, atlas],
  )

  const sphereRef = useRef<THREE.InstancedMesh>(null)
  const phraseRef = useRef<THREE.InstancedMesh>(null)
  const xrayLevel = useMemo(() => new Float32Array(layout.phraseKeys.length), [layout])

  const sim = useMemo(() => {
    const n = layout.keys.length
    // instance slot of each key inside its InstancedMesh
    const slot = new Int32Array(n)
    layout.sphereKeys.forEach((id, i) => (slot[id] = i))
    layout.phraseKeys.forEach((id, i) => (slot[id] = i))
    return {
      slot,
      ringAngle: new Float64Array(layout.rings.length),
      push: new Float32Array(n),
      goal: new Float32Array(n),
      // phrase keys' response to the cursor light: smoothed rotation vector (x, y) + lift
      lean: new Float32Array(n * 3),
      vel: new Float32Array(n),
      dirs: new Float32Array(n * 3),
      idleYaw: 0,
      frame: { p: 0, time: 0, idleYaw: 0, bob: 0, tiltX: 0, tiltZ: 0, ringAngle: [], push: [] } as FrameInput,
      pos: v3() as V3,
      quat: q4() as Q4,
      m: new THREE.Matrix4(),
      tp: new THREE.Vector3(),
      tq: new THREE.Quaternion(),
      tq2: new THREE.Quaternion(),
      keyScale: new THREE.Vector3().setScalar(WORLD_SCALE * config.sphere.keyScale),
      zero: new THREE.Vector3(0, 0, 0),
      center: v3() as V3,
      ray: new THREE.Raycaster(),
      ndc: new THREE.Vector2(),
      hit: new THREE.Vector3(),
      hitDir: new THREE.Vector3(),
      hasHit: false,
      hadHit: false,
      hitRaw: new THREE.Vector3(),
      bounds: new THREE.Sphere(),
      axis: new THREE.Vector3(),
      kd: new THREE.Vector3(),
      kp: new THREE.Vector3(),
    }
  }, [layout])

  useEffect(() => {
    for (const mesh of [sphereRef.current, phraseRef.current]) {
      if (!mesh) continue
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    }
  }, [])

  useFrame(({ camera }, rawDt) => {
    const dt = Math.min(rawDt, 1 / 20)
    const S = config.sphere
    const M = config.mouse
    const w = store.w
    const t = store.time
    const f = sim.frame
    uniforms.uKeyTime.value = t

    // ---- time-integrated state (never affects shape → reversible) ----
    sim.idleYaw += dt * S.idleRotationSpeed * (1 + 0.6 * w.decoding)
    for (let r = 0; r < layout.rings.length; r++) sim.ringAngle[r] += dt * ringSpinSpeed(layout, r, w)

    f.p = store.progress
    f.time = t
    f.idleYaw = sim.idleYaw
    f.bob = S.bobAmplitude * Math.sin(t * S.bobSpeed * 2)
    f.tiltX = 0.035 * Math.sin(t * 0.21)
    f.tiltZ = 0.028 * Math.sin(t * 0.17 + 1.3)
    f.ringAngle = sim.ringAngle
    f.push = sim.push
    f.material = store.material

    // ---- mouse: exact ray / sphere intersection ----
    sim.hasHit = false
    if (w.mouse > 0.001 && store.pointer.active) {
      sim.ndc.set(store.pointer.x, store.pointer.y)
      sim.ray.setFromCamera(sim.ndc, camera)
      sphereCenter(sim.center, f)
      sim.bounds.center.set(sim.center[0], sim.center[1], sim.center[2]).multiplyScalar(WORLD_SCALE)
      sim.bounds.radius = (layout.outerRadius + 0.6) * WORLD_SCALE
      if (sim.ray.ray.intersectSphere(sim.bounds, sim.hit)) {
        sim.hitRaw.copy(sim.hit).sub(sim.bounds.center).normalize()
        // the influence centre glides after the cursor: hand jitter on one key never steers it
        if (!sim.hadHit) sim.hitDir.copy(sim.hitRaw)
        else sim.hitDir.lerp(sim.hitRaw, 1 - Math.exp(-dt / M.hitSmoothing)).normalize()
        sim.hasHit = true
      }
    }
    sim.hadHit = sim.hasHit

    // ---- springs ----
    const sigma2 = M.radius * M.radius
    for (let id = 0; id < layout.keys.length; id++) {
      let target = 0
      let infl = 0
      if (sim.hasHit) {
        const dx = sim.dirs[id * 3]
        const dy = sim.dirs[id * 3 + 1]
        const dz = sim.dirs[id * 3 + 2]
        const cos = dx * sim.hitDir.x + dy * sim.hitDir.y + dz * sim.hitDir.z
        const ang = Math.acos(Math.min(1, Math.max(-1, cos)))
        infl = Math.exp(-(ang * ang) / sigma2)
        target = M.strength * infl * w.mouse
      }
      // delayed goal: nearest keys respond first, the push then travels outwards; release is slower
      const tau = (M.responseDelay + M.neighbourDelay * (1 - infl)) * (target < sim.goal[id] ? M.releaseSlowdown : 1)
      sim.goal[id] += (target - sim.goal[id]) * (1 - Math.exp(-dt / Math.max(tau, 1e-3)))
      const k = M.stiffness * (1 - M.neighbourSoftness * (1 - infl))
      const c = M.damping * Math.sqrt(k / M.stiffness)
      sim.vel[id] += (k * (sim.goal[id] - sim.push[id]) - c * sim.vel[id]) * dt
      sim.push[id] += sim.vel[id] * dt
      if (w.mouse <= 0.001 && Math.abs(sim.push[id]) < 1e-4 && Math.abs(sim.vel[id]) < 1e-4) {
        sim.push[id] = 0
        sim.vel[id] = 0
        sim.goal[id] = 0
      }
    }

    // ---- poses ----
    const sphereMesh = sphereRef.current
    const phraseMesh = phraseRef.current
    if (!sphereMesh || !phraseMesh) return
    sphereCenter(sim.center, f)
    const [cx, cy, cz] = sim.center
    for (let id = 0; id < layout.keys.length; id++) {
      computePose(layout, id, f, w, sim.pos, sim.quat)
      const [px, py, pz] = sim.pos
      sim.tp.set(px * WORLD_SCALE, py * WORLD_SCALE, pz * WORLD_SCALE)
      sim.tq.set(sim.quat[0], sim.quat[1], sim.quat[2], sim.quat[3])

      // direction from the centre (used by next frame's mouse falloff)
      sim.kd.set(px - cx, py - cy, pz - cz).normalize()
      sim.dirs[id * 3] = sim.kd.x
      sim.dirs[id * 3 + 1] = sim.kd.y
      sim.dirs[id * 3 + 2] = sim.kd.z

      // tiny tilt away from the cursor, proportional to the push
      const push = sim.push[id]
      if (sim.hasHit && push > 1e-3) {
        sim.axis.crossVectors(sim.hitDir, sim.kd)
        const l = sim.axis.length()
        if (l > 1e-4) {
          sim.axis.multiplyScalar(1 / l)
          sim.tq2.setFromAxisAngle(sim.axis, (M.tilt * push) / M.strength)
          sim.tq.premultiply(sim.tq2)
        }
      }

      const isPhrase = layout.keys[id].letter >= 0
      // a ring that has fully left the frame is collapsed — nothing lingers in the final void
      const gone = !isPhrase && exitProgress(layout, layout.keys[id].ring, f.p) >= 0.999
      // phrase keys lean towards the cursor light and lift a little (small, never enough to touch)
      if (isPhrase) {
        const CL = config.cursorLight
        const pres = store.cursor.presence
        const dxw = store.cursor.x - sim.tp.x
        const dyw = store.cursor.y - sim.tp.y
        const d = Math.hypot(dxw, dyw)
        const R = CL.keyRadius * WORLD_SCALE
        const infl = pres * Math.exp(-(d * d) / (R * R))
        const near = Math.min(1, d / (R * 0.25))
        const ang = ((CL.keyTiltDeg * Math.PI) / 180) * infl * near
        // rotation axis z × dir = (−dy, dx, 0): turns the legend face towards the light
        const tx = d > 1e-4 ? (-dyw / d) * ang : 0
        const ty = d > 1e-4 ? (dxw / d) * ang : 0
        const tl = CL.keyLift * WORLD_SCALE * infl
        const kf = 1 - Math.exp(-dt * 5)
        const o = id * 3
        sim.lean[o] += (tx - sim.lean[o]) * kf
        sim.lean[o + 1] += (ty - sim.lean[o + 1]) * kf
        sim.lean[o + 2] += (tl - sim.lean[o + 2]) * kf
        const la = Math.hypot(sim.lean[o], sim.lean[o + 1])
        if (la > 1e-5) {
          sim.axis.set(sim.lean[o] / la, sim.lean[o + 1] / la, 0)
          sim.tq2.setFromAxisAngle(sim.axis, la)
          sim.tq.premultiply(sim.tq2)
        }
        sim.tp.z += sim.lean[o + 2]
      }
      sim.m.compose(sim.tp, sim.tq, gone ? sim.zero : sim.keyScale)
      ;(isPhrase ? phraseMesh : sphereMesh).setMatrixAt(sim.slot[id], sim.m)
    }
    sphereMesh.instanceMatrix.needsUpdate = true
    phraseMesh.instanceMatrix.needsUpdate = true

    // ---- x-ray under the cursor (third section only): fast in, slow fade back ----
    const W = config.wireframe
    const gate = Math.min(1, Math.max(0, (store.material - W.from) / (1 - W.from)))
    const pres = store.cursor.presence * gate
    const R = W.radius * WORLD_SCALE
    let over = false
    for (let i = 0; i < layout.phraseKeys.length; i++) {
      phraseMesh.getMatrixAt(i, sim.m)
      sim.kp.setFromMatrixPosition(sim.m)
      const d = Math.hypot(store.cursor.x - sim.kp.x, store.cursor.y - sim.kp.y)
      const x = Math.min(1, Math.max(0, (R - d) / (R * (1 - W.core))))
      const target = pres * x * x * (3 - 2 * x)
      if (target > 0.5) over = true
      const cur = xrayLevel[i]
      const tau = target > cur ? W.inTime : W.outTime
      const v = cur + (target - cur) * (1 - Math.exp(-dt / tau))
      xrayLevel[i] = v < 1e-3 && target === 0 ? 0 : v
    }
    store.pointerOverKey = over
    geos.xray.array.set(xrayLevel)
    geos.xray.needsUpdate = true
  })

  return (
    <group>
      <instancedMesh
        ref={sphereRef}
        args={[geos.sphere, material, layout.sphereKeys.length]}
        castShadow
        receiveShadow
        frustumCulled={false}
      />
      <instancedMesh
        ref={phraseRef}
        args={[geos.phrase, material, layout.phraseKeys.length]}
        castShadow
        receiveShadow
        frustumCulled={false}
      />

      <Core onReady={onCore} />
    </group>
  )
}
