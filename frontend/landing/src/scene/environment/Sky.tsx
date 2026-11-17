/**
 * Sky dome: cold, almost monochrome gradient whose horizon equals the fog colour (seamless with the
 * fogged terrain), a soft bright breach above the sphere, faint high veils — and a final crossfade to
 * the near-black void with a barely-there cold gradient.
 */
import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { config } from '../../config'
import { store } from '../../state/store'
import { skyFrag as frag, skyVert as vert } from '../../../../shared/argilla/glsl/sky.glsl'





const _size = new THREE.Vector2()

export function Sky() {
  const mesh = useRef<THREE.Mesh>(null)
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: vert,
        fragmentShader: frag,
        side: THREE.BackSide,
        depthWrite: false,
        uniforms: {
          uHorizon: { value: new THREE.Color(config.atmosphere.fogColor) },
          uZenith: { value: new THREE.Color(config.atmosphere.skyZenith) },
          uGlow: { value: new THREE.Color(config.atmosphere.skyGlow) },
          // the luminous breach sits right behind the sphere, like the reference
          uGlowDir: { value: new THREE.Vector3(...config.sky.glowDir) },
          uGlowStrength: { value: config.sky.glow },
          uZenithAt: { value: config.sky.zenithAt },
          uVoidTop: { value: new THREE.Color(config.sky.voidTop) },
          uVoidBottom: { value: new THREE.Color(config.atmosphere.voidColor) },
          uFade: { value: 0 },
          uTime: { value: 0 },
          uStars: { value: 0 },
          uAurora: { value: 0 },
          uViewH: { value: 900 },
          // top darkening: starts at uAuroraShade.x of the frame height, full at .y; strength
          uAuroraShade: { value: new THREE.Vector2(config.aurora.shadeFrom, 1.0) },
          uAuroraTop: { value: config.aurora.shadeStrength },
          uAuroraSpeed: { value: config.aurora.speed },
          uAuroraShift: { value: 0 },
          uAuroraLow: { value: new THREE.Color(config.aurora.low) },
          uAuroraHigh: { value: new THREE.Color(config.aurora.high) },
          uAuroraEdge: { value: new THREE.Color(config.aurora.edge) },
          uAuroraGain: { value: config.aurora.intensity },
          uStarColor: { value: new THREE.Vector3(...config.sky.stars) },
          uSunSize: { value: config.sky.sun.size },
          uSunBands: { value: config.sky.sun.bands },
          uSunIntensity: { value: config.sky.sun.intensity },
          uSunTop: { value: new THREE.Color(config.sky.sun.top) },
          uSunBottom: { value: new THREE.Color(config.sky.sun.bottom) },
        },
      }),
    [],
  )
  useEffect(() => () => material.dispose(), [material])

  useFrame(({ camera, scene, gl, size }) => {
    mesh.current?.position.copy(camera.position)
    const u = material.uniforms
    u.uViewH.value = gl.getDrawingBufferSize(_size).y
    // vertical offset in screen pixels → sky-space units at the current field of view
    const fov = (camera as THREE.PerspectiveCamera).fov ?? 40
    u.uAuroraShift.value = (config.aurora.offsetPx / Math.max(1, size.height)) * 2 * Math.tan(THREE.MathUtils.degToRad(fov) / 2)
    u.uTime.value = store.time
    u.uFade.value = store.w.environment
    // a few stars already with the aurora, the full sky once MATERIAL forms
    u.uStars.value = Math.max(0.45 * store.w.final, Math.min(1, Math.max(0, (store.material - 0.1) / 0.75)))
    u.uAurora.value = store.w.final
    // the breach brightens a little while the core is active
    u.uGlowStrength.value = config.sky.glow * (1 + 0.6 * store.w.glow)
    if (scene.fog) (u.uHorizon.value as THREE.Color).copy(scene.fog.color)
  })

  return (
    <mesh ref={mesh} material={material} renderOrder={-10} frustumCulled={false}>
      <sphereGeometry args={[3000, 48, 24]} />
    </mesh>
  )
}
