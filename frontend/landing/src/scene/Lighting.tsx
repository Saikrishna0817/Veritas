/**
 * Lighting rig.
 *  Hero: cold sky (hemisphere) + high back/top key with soft shadows (rim on the sphere, deep
 *  shadows between keys) + faint frontal fill + a studio-free environment built from lightformers
 *  (soft reflections on plastic, no HDR download).
 *  Final: the key light glides to a soft upper-left frontal position so the phrase reads as a
 *  sculptural object.
 */
import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Environment, Lightformer } from '@react-three/drei'
import * as THREE from 'three'
import { config } from '../config'
import { store } from '../state/store'
import { smoothstep } from '../utils/math'
import { WORLD_SCALE, phraseWorldCenter } from '../choreography/framing'

export function Lighting() {
  const L = config.lighting
  const key = useRef<THREE.DirectionalLight>(null)
  const fill = useRef<THREE.DirectionalLight>(null)
  const hemi = useRef<THREE.HemisphereLight>(null)

  const tmp = useMemo(
    () => ({
      heroDir: new THREE.Vector3(...L.key.position).normalize(),
      finalDir: new THREE.Vector3(...L.finalKey.position).sub(new THREE.Vector3(...phraseWorldCenter())).normalize(),
      dir: new THREE.Vector3(),
      focus: new THREE.Vector3(),
      sphere: new THREE.Vector3(...config.sphere.center),
      phrase: new THREE.Vector3(...phraseWorldCenter()),
      heroCol: new THREE.Color(L.key.color),
      finalCol: new THREE.Color(L.finalKey.color),
    }),
    [L],
  )

  useFrame(({ scene }) => {
    const w = store.w
    const k = key.current
    if (!k) return
    const toFinal = smoothstep(0.62, 0.92, store.progress)

    // shadow frustum follows the subject
    tmp.focus.lerpVectors(tmp.sphere, tmp.phrase, smoothstep(0.66, 0.86, store.progress))
    tmp.dir.copy(tmp.heroDir).lerp(tmp.finalDir, toFinal).normalize()
    k.position.copy(tmp.focus).addScaledVector(tmp.dir, 260)
    k.target.position.copy(tmp.focus)
    k.target.updateMatrixWorld()
    k.color.copy(tmp.heroCol).lerp(tmp.finalCol, toFinal)
    k.intensity = THREE.MathUtils.lerp(L.key.intensity, L.finalKey.intensity, toFinal)

    if (fill.current) fill.current.intensity = L.fill.intensity * (1 - 0.6 * w.environment)
    if (hemi.current) hemi.current.intensity = L.hemisphere.intensity * (1 - 0.75 * w.environment)
    // reflections calm down in the void, but never vanish — the plastic must still read as plastic
    scene.environmentIntensity = L.environmentIntensity * (1 - 0.35 * w.environment)
  })

  // shadow frustum covers the sphere (and later the phrase) at world scale
  const s = 30 * WORLD_SCALE
  return (
    <>
      <hemisphereLight ref={hemi} args={[L.hemisphere.sky, L.hemisphere.ground, L.hemisphere.intensity]} />
      <directionalLight
        ref={key}
        color={L.key.color}
        intensity={L.key.intensity}
        position={L.key.position as unknown as [number, number, number]}
        castShadow
        shadow-mapSize={[L.shadowMapSize, L.shadowMapSize]}
        shadow-bias={-0.0004}
        shadow-normalBias={0.12}
        shadow-radius={4}
        shadow-camera-left={-s}
        shadow-camera-right={s}
        shadow-camera-top={s}
        shadow-camera-bottom={-s}
        shadow-camera-near={120}
        shadow-camera-far={420}
      />
      <directionalLight
        ref={fill}
        color={L.fill.color}
        intensity={L.fill.intensity}
        position={L.fill.position as unknown as [number, number, number]}
      />
      <Environment resolution={256} frames={1} environmentIntensity={L.environmentIntensity}>
        {/* broad cold sky panel overhead */}
        <Lightformer form="rect" intensity={1.6} color={L.env.sky} scale={[60, 30, 1]} position={[0, 30, -10]} rotation={[Math.PI / 2, 0, 0]} />
        {/* bright breach behind/above (matches the sky glow) */}
        <Lightformer form="circle" intensity={3.4} color={L.env.breach} scale={14} position={[0, 8, -40]} />
        {/* dim teal horizon ring */}
        <Lightformer form="ring" intensity={0.5} color={L.env.horizon} scale={80} position={[0, -6, 0]} rotation={[Math.PI / 2, 0, 0]} />
        {/* soft frontal card for gentle highlights on the legends' faces */}
        <Lightformer form="rect" intensity={0.55} color={L.env.front} scale={[24, 10, 1]} position={[-18, 10, 40]} />
      </Environment>
    </>
  )
}
