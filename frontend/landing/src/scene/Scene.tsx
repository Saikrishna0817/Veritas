import { useCallback, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import type * as THREE from 'three'
import { store } from '../state/store'
import { phaseWeights } from '../choreography/choreography'
import { config } from '../config'
import { CameraRig } from './CameraRig'
import { Lighting } from './Lighting'
import { Sky } from './environment/Sky'
import { MountainTerrain } from './environment/MountainTerrain'
import { ForegroundMountains } from './environment/ForegroundMountains'
import { Atmosphere } from './environment/Atmosphere'
import { KeyboardSphere } from './keys/KeyboardSphere'
import { PostFX } from './PostFX'
import { CursorLight } from './cursor/CursorLight'
import { SafeBoundary } from './SafeBoundary'
import { demo, demoProgress } from '../scroll/demo'

/**
 * The anagram does not follow the scroll 1:1: a critically damped follow with a speed cap gives the
 * keys inertia — they ease out of the phrase, drift, and settle softly, as if in zero gravity —
 * and stretches a fast scroll into a slower, deliberate rebuild (≈ +1–1.5 s).
 */
let matVel = 0
function followMaterial(cur: number, target: number, dt: number) {
  const M = config.anagramMotion
  const omega = 2 / M.smoothTime
  const x = omega * dt
  const exp = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x)
  const maxChange = M.maxSpeed * M.smoothTime
  const change = Math.max(-maxChange, Math.min(maxChange, cur - target))
  const tgt = cur - change
  const temp = (matVel + omega * change) * dt
  matVel = (matVel - omega * temp) * exp
  let out = tgt + (change + temp) * exp
  if (target - cur > 0 === out > target) {
    out = target
    matVel = 0
  }
  if (Math.abs(out - target) < 1e-5 && Math.abs(matVel) < 1e-4) {
    out = target
    matVel = 0
  }
  return Math.min(1, Math.max(0, out))
}

/** Runs first every frame: advances the clock and smooths scroll progress into phase weights. */
function Director() {
  useFrame((_, rawDt) => {
    const dt = Math.min(rawDt, 1 / 20)
    store.time += dt
    if (demo.tl) {
      demo.t = demo.start < 0 ? 0 : store.time - demo.start
      demo.tl.progress(demo.still >= 0 ? demo.still : demoProgress(demo.t))
    }
    // GSAP already scrubs; this extra critically-damped follow removes wheel micro-steps
    const k = 1 - Math.exp(-9 * dt)
    store.scroll += (store.target - store.scroll) * k
    if (Math.abs(store.target - store.scroll) < 1e-5) store.scroll = store.target
    const ph = config.phases
    store.progress = Math.min(1, store.scroll / ph.storyEnd)
    const mTarget = Math.min(1, Math.max(0, (store.scroll - ph.material[0]) / (ph.material[1] - ph.material[0])))
    store.material = followMaterial(store.material, mTarget, dt)
    if (demo.still >= 0) {
      // a still stands where it was asked to, with no easing towards it
      store.scroll = store.target
      store.progress = Math.min(1, store.scroll / ph.storyEnd)
      store.material = mTarget
    }
    store.w = phaseWeights(store.progress)
  }, -100)
  return null
}

export function Scene() {
  const [sun, setSun] = useState<THREE.Mesh | null>(null)
  const onCore = useCallback((m: THREE.Mesh | null) => setSun(m), [])
  return (
    <>
      <Director />
      <CameraRig />
      <Lighting />
      <Sky />
      <MountainTerrain />
      <SafeBoundary name="ForegroundMountains">
        <ForegroundMountains />
      </SafeBoundary>
      <Atmosphere />
      <CursorLight />
      <KeyboardSphere onCore={onCore} />
      <PostFX sun={sun} />
    </>
  )
}
