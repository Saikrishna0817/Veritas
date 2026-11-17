/**
 * Atmosphere: everything that makes the air visible.
 *  - scene fog (FogExp2 + patched height fog), animated towards the void;
 *  - layered valley fog sheets (soft fbm, drifting) — local fog banks at different depths;
 *  - light shafts around the sphere (Y-billboarded, additive, streaked) that swell with the core;
 *  - suspended dust catching the light, brighter near the core.
 * All are cheap screen-space-friendly imitations of volumetrics — no raymarching.
 */
import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { config } from '../../config'
import { store } from '../../state/store'
import { glslNoise } from '../shaders/noise.glsl'
import { dustFrag, dustVert } from '../../../../shared/argilla/glsl/dust.glsl'
import { mulberry32 } from '../../utils/math'
import { WORLD_SCALE } from '../../choreography/framing'

/* ------------------------------------------------------------------ */
/* Fog                                                                 */
/* ------------------------------------------------------------------ */

export function FogController() {
  const scene = useThree((s) => s.scene)
  const cols = useMemo(
    () => ({
      fog: new THREE.Color(config.atmosphere.fogColor),
      void: new THREE.Color(config.atmosphere.voidColor),
    }),
    [],
  )
  useEffect(() => {
    const fog = new THREE.FogExp2(cols.fog.getHex(), config.atmosphere.fogDensity)
    fog.color.copy(cols.fog)
    scene.fog = fog
    return () => {
      scene.fog = null
    }
  }, [scene, cols])

  useFrame(() => {
    const fog = scene.fog as THREE.FogExp2 | null
    if (!fog) return
    const e = store.w.environment
    fog.color.copy(cols.fog).lerp(cols.void, e)
    // a touch denser while the valley dissolves, then thinner so the phrase stays crisp
    fog.density = config.atmosphere.fogDensity * (1 + 0.8 * Math.sin(Math.PI * e)) * (1 - 0.5 * store.w.final)
  })
  return null
}

/* ------------------------------------------------------------------ */
/* Fog sheets                                                          */
/* ------------------------------------------------------------------ */

const sheetVert = /* glsl */ `
varying vec2 vUv;
varying float vDepth;
void main() {
  vUv = uv;
  vec4 mv = modelViewMatrix * vec4( position, 1.0 );
  vDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}`

const sheetFrag = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
uniform float uTime;
uniform float uSeed;
uniform float uScale;
varying vec2 vUv;
varying float vDepth;
${glslNoise}
void main() {
  vec2 p = vec2( vUv.x * uScale, vUv.y * uScale * 0.32 );
  float t = uTime * 0.012;
  float n = kcFbm( p + vec2( t + uSeed, -t * 0.4 ) );
  n = mix( n, kcFbm( p * 2.1 - vec2( t * 1.7, uSeed ) ), 0.35 );
  float body = smoothstep( 0.28, 0.82, n );
  float vert = smoothstep( 0.0, 0.38, vUv.y ) * ( 1.0 - smoothstep( 0.42, 1.0, vUv.y ) );
  float horiz = smoothstep( 0.0, 0.18, vUv.x ) * ( 1.0 - smoothstep( 0.82, 1.0, vUv.x ) );
  float nearFade = smoothstep( 6.0, 40.0, vDepth );
  float a = body * vert * horiz * nearFade * uOpacity;
  gl_FragColor = vec4( uColor, a );
}`

const SHEETS = [
  { z: 72, y: -58, w: 520, h: 70, o: 0.16, s: 4.0 },
  { z: -70, y: -48, w: 760, h: 110, o: 0.3, s: 4.8 },
  { z: -240, y: -44, w: 1150, h: 150, o: 0.34, s: 5.6 },
  { z: -480, y: -30, w: 1650, h: 210, o: 0.36, s: 6.4 },
  { z: -820, y: -6, w: 2300, h: 300, o: 0.3, s: 7.0 },
]

function FogSheets() {
  const color = useMemo(() => new THREE.Color(config.atmosphere.fogColor).multiplyScalar(1.18), [])
  const mats = useMemo(
    () =>
      SHEETS.map(
        (s, i) =>
          new THREE.ShaderMaterial({
            vertexShader: sheetVert,
            fragmentShader: sheetFrag,
            transparent: true,
            depthWrite: false,
            uniforms: {
              uColor: { value: color.clone() },
              uOpacity: { value: s.o },
              uTime: { value: 0 },
              uSeed: { value: i * 13.7 },
              uScale: { value: s.s },
            },
          }),
      ),
    [color],
  )
  useEffect(() => () => mats.forEach((m) => m.dispose()), [mats])

  useFrame(({ scene }) => {
    const fade = 1 - store.w.environment
    const fogCol = scene.fog?.color
    mats.forEach((m, i) => {
      m.uniforms.uTime.value = store.time
      m.uniforms.uOpacity.value = SHEETS[i].o * fade
      if (fogCol) (m.uniforms.uColor.value as THREE.Color).copy(fogCol).multiplyScalar(1.18)
    })
  })

  return (
    <group>
      {SHEETS.map((s, i) => (
        <mesh key={i} position={[0, s.y, s.z]} material={mats[i]} renderOrder={5 + (SHEETS.length - i)} frustumCulled={false}>
          <planeGeometry args={[s.w, s.h]} />
        </mesh>
      ))}
    </group>
  )
}

/* ------------------------------------------------------------------ */
/* Light shafts                                                        */
/* ------------------------------------------------------------------ */

const shaftFrag = /* glsl */ `
uniform vec3 uColor;
uniform float uIntensity;
uniform float uTime;
uniform float uSeed;
varying vec2 vUv;
varying float vDepth;
${glslNoise}
void main() {
  float x = vUv.x;
  float y = vUv.y;
  float streak = kcFbm( vec2( x * 7.0 + uSeed, y * 0.7 - uTime * 0.018 ) );
  streak = smoothstep( 0.35, 0.85, streak );
  float across = pow( sin( 3.14159 * x ), 2.2 );
  float along = smoothstep( 0.0, 0.42, y ) * ( 1.0 - smoothstep( 0.9, 1.0, y ) );
  float a = streak * across * along * uIntensity;
  gl_FragColor = vec4( uColor * a, 1.0 );
}`

// sphere-local layout, scaled with the sphere system
const SHAFTS = [
  { x: -16, z: -18, w: 10, tilt: 0.12 },
  { x: -5, z: -26, w: 16, tilt: 0.08 },
  { x: 6, z: -22, w: 9, tilt: 0.1 },
  { x: 14, z: -30, w: 18, tilt: 0.05 },
  { x: -1, z: -40, w: 26, tilt: 0.09 },
].map((s) => ({ x: s.x * WORLD_SCALE, z: s.z * WORLD_SCALE, w: s.w * WORLD_SCALE, tilt: s.tilt }))

function LightShafts() {
  const group = useRef<THREE.Group>(null)
  const color = useMemo(() => new THREE.Color(config.atmosphere.shaftColor), [])
  const mats = useMemo(
    () =>
      SHAFTS.map(
        (_, i) =>
          new THREE.ShaderMaterial({
            vertexShader: sheetVert,
            fragmentShader: shaftFrag,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            uniforms: {
              uColor: { value: color.clone() },
              uIntensity: { value: 0 },
              uTime: { value: 0 },
              uSeed: { value: i * 7.31 },
            },
          }),
      ),
    [color],
  )
  useEffect(() => () => mats.forEach((m) => m.dispose()), [mats])

  useFrame(({ camera }) => {
    const w = store.w
    const I = config.atmosphere.shaftIntensity * (1 + 1.8 * w.glow) * (1 - w.environment)
    const g = group.current
    if (!g) return
    g.children.forEach((child, i) => {
      // billboard around Y so the shafts never show their thin edge
      const dx = camera.position.x - child.position.x
      const dz = camera.position.z - child.position.z
      child.rotation.set(0, Math.atan2(dx, dz), SHAFTS[i].tilt)
      const m = mats[i]
      m.uniforms.uTime.value = store.time
      m.uniforms.uIntensity.value = I
    })
    g.visible = I > 0.0005
  })

  return (
    <group ref={group}>
      {SHAFTS.map((s, i) => (
        <mesh key={i} position={[s.x, 60, s.z]} material={mats[i]} renderOrder={20} frustumCulled={false}>
          <planeGeometry args={[s.w, 260]} />
        </mesh>
      ))}
    </group>
  )
}

/* ------------------------------------------------------------------ */
/* Dust                                                                */
/* ------------------------------------------------------------------ */





function Dust() {
  const gl = useThree((s) => s.gl)
  const geometry = useMemo(() => {
    const n = config.atmosphere.dustCount
    const rnd = mulberry32(31)
    const pos = new Float32Array(n * 3)
    const seed = new Float32Array(n)
    for (let i = 0; i < n; i++) {
      pos[i * 3] = (rnd() - 0.5) * 320
      pos[i * 3 + 1] = (rnd() - 0.5) * 170
      pos[i * 3 + 2] = -110 + rnd() * 260
      seed[i] = rnd()
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1))
    return g
  }, [])
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: dustVert,
        fragmentShader: dustFrag,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uTime: { value: 0 },
          uSize: { value: config.atmosphere.dust.size },
          uRise: { value: config.atmosphere.dust.rise },
          uPixelRatio: { value: 1 },
          uCenter: { value: new THREE.Vector3(...config.sphere.center) },
          uCore: { value: 0 },
          uColor: { value: new THREE.Color(config.atmosphere.dustColor) },
          uGlowColor: { value: new THREE.Color(config.lighting.core.color) },
          uOpacity: { value: 0.5 },
        },
      }),
    [],
  )
  useEffect(
    () => () => {
      geometry.dispose()
      material.dispose()
    },
    [geometry, material],
  )
  useFrame(() => {
    const u = material.uniforms
    u.uTime.value = store.time
    u.uPixelRatio.value = gl.getPixelRatio()
    u.uCore.value = store.w.glow
    // dust lingers into the final void, but thinner
    u.uOpacity.value = 0.5 * (1 - 0.65 * store.w.environment)
  })
  return <points geometry={geometry} material={material} frustumCulled={false} renderOrder={30} />
}

/* ------------------------------------------------------------------ */

export function Atmosphere() {
  return (
    <>
      <FogController />
      <FogSheets />
      <LightShafts />
      <Dust />
    </>
  )
}
