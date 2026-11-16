/**
 * Cinematic camera: scroll-blended keyframes (dolly-in, framing correction, a slight perspective
 * change at the disassembly, a frontal final composition), plus cursor parallax and a breath of
 * idle drift. No shake, no FOV jumps — every parameter is smoothly blended.
 */
import { useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { config } from '../config'
import { store } from '../state/store'
import { smootherstep } from '../utils/math'
import { finalCameraDistance, phraseWorldCenter } from '../choreography/framing'

const v = (a: readonly number[]) => new THREE.Vector3(a[0], a[1], a[2])

export function CameraRig() {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera
  const C = config.camera
  const k = useMemo(
    () => ({
      p0: v(C.hero.position), t0: v(C.hero.target),
      p1: v(C.decode.position), t1: v(C.decode.target),
      p2: v(C.disassembly.position), t2: v(C.disassembly.target),
      p3: new THREE.Vector3(), t3: v(phraseWorldCenter()),
      pos: new THREE.Vector3(),
      tgt: new THREE.Vector3(),
    }),
    [C],
  )

  useFrame((_, rawDt) => {
    const dt = Math.min(rawDt, 1 / 20)
    const p = store.progress
    const ph = config.phases
    const a = smootherstep(ph.activation[0] + 0.05, ph.decoding[1], p)
    const b = smootherstep(ph.decoding[1] - 0.05, ph.disassembly[1], p)
    const c = smootherstep(ph.disassembly[1] - 0.06, ph.final[0] + 0.04, p)
    // final framing adapts to the viewport: the phrase always fills the same share of its width
    k.p3.copy(k.t3)
    k.p3.z += finalCameraDistance(camera.aspect, C.final.fov)

    k.pos.copy(k.p0).lerp(k.p1, a).lerp(k.p2, b).lerp(k.p3, c)
    k.tgt.copy(k.t0).lerp(k.t1, a).lerp(k.t2, b).lerp(k.t3, c)
    const fov = THREE.MathUtils.lerp(THREE.MathUtils.lerp(C.fov, C.disassembly.fov, b), C.final.fov, c)

    // cursor parallax with inertia
    const ps = store.pointerSmooth
    const damp = 1 - Math.exp(-C.parallaxDamping * dt)
    ps.x += ((store.pointer.active ? store.pointer.x : 0) - ps.x) * damp
    ps.y += ((store.pointer.active ? store.pointer.y : 0) - ps.y) * damp
    const par = 1 - 0.45 * store.w.final
    const t = store.time
    k.pos.x += ps.x * C.parallax[0] * par + Math.sin(t * 0.13) * C.idle
    k.pos.y += ps.y * C.parallax[1] * par + Math.sin(t * 0.17 + 1.1) * C.idle * 0.6
    k.tgt.x += ps.x * C.parallax[0] * 0.12
    k.tgt.y += ps.y * C.parallax[1] * 0.12

    camera.position.copy(k.pos)
    camera.lookAt(k.tgt)
    if (Math.abs(camera.fov - fov) > 1e-4) {
      camera.fov = fov
      camera.updateProjectionMatrix()
    }
  })
  return null
}
