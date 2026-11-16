/**
 * Second & third sections: the cursor becomes a small sphere of cold light.
 *  - a real PointLight rides with it, so the keycaps catch light as it passes;
 *  - a glowing orb (HDR, picked up by bloom) marks the source;
 *  - a smoky light trail is simulated in a half-resolution ping-pong buffer: every frame the previous
 *    state is advected by curl noise (rising, swirling), slightly blurred and faded, and a soft splat is
 *    painted along the orb's last screen segment. The post chain (TrailEffect) composites it behind the keys.
 * Active only once the phrase has formed (store.w.final), fades with the cursor leaving the window.
 */
import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { config } from '../../config'
import { store } from '../../state/store'
import { trailState } from '../../state/trail'
import { WORLD_SCALE, phraseWorldCenter } from '../../choreography/framing'


const quadVert = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4( position.xy, 0.0, 1.0 ); }`

const simFrag = /* glsl */ `
precision highp float;
uniform sampler2D uPrev;
uniform vec2 uTexel;
uniform vec2 uA;
uniform vec2 uB;
uniform float uAspect;
uniform float uDt;
uniform float uTime;
uniform float uSplat;
uniform float uRadius;
varying vec2 vUv;

float th( vec2 p ) { return fract( sin( dot( p, vec2( 127.1, 311.7 ) ) ) * 43758.5453 ); }
float tn( vec2 p ) {
  vec2 i = floor( p ), f = fract( p );
  vec2 u = f * f * ( 3.0 - 2.0 * f );
  return mix( mix( th( i ), th( i + vec2( 1, 0 ) ), u.x ), mix( th( i + vec2( 0, 1 ) ), th( i + vec2( 1, 1 ) ), u.x ), u.y );
}
vec2 curl( vec2 p ) {
  float e = 0.05;
  float a = tn( p + vec2( 0.0, e ) ), b = tn( p - vec2( 0.0, e ) );
  float c = tn( p + vec2( e, 0.0 ) ), d = tn( p - vec2( e, 0.0 ) );
  return vec2( a - b, -( c - d ) ) / ( 2.0 * e );
}

void main() {
  vec2 q = vUv * vec2( uAspect, 1.0 );
  // smoke drifts upward and swirls
  // three octaves of curl: large slow swirls, medium eddies, fine turbulence
  vec2 vel = curl( q * 2.4 + vec2( 0.0, uTime * 0.07 ) ) * 0.0017
    + curl( q * 6.5 - uTime * 0.13 ) * 0.0009
    + curl( q * 15.0 + vec2( uTime * 0.2, -uTime * 0.1 ) ) * 0.00045
    + vec2( 0.0, 0.0007 );
  vec2 uv = vUv - vel * uDt * 60.0;
  float c = texture2D( uPrev, uv ).r * 0.6
    + ( texture2D( uPrev, uv + vec2( uTexel.x, 0.0 ) ).r + texture2D( uPrev, uv - vec2( uTexel.x, 0.0 ) ).r
      + texture2D( uPrev, uv + vec2( 0.0, uTexel.y ) ).r + texture2D( uPrev, uv - vec2( 0.0, uTexel.y ) ).r ) * 0.1;
  c *= exp( -uDt * 0.95 );

  // soft capsule splat along the orb's last screen segment
  vec2 a = uA * vec2( uAspect, 1.0 );
  vec2 b = uB * vec2( uAspect, 1.0 );
  vec2 pa = q - a, ba = b - a;
  float h = clamp( dot( pa, ba ) / max( dot( ba, ba ), 1e-6 ), 0.0, 1.0 );
  float d = length( pa - ba * h );
  c += exp( -d * d / ( uRadius * uRadius ) ) * uSplat;

  gl_FragColor = vec4( min( c, 3.0 ), 0.0, 0.0, 1.0 );
}`

const orbFrag = /* glsl */ `
uniform vec3 uColor;
uniform float uIntensity;
varying vec2 vUv;
void main() {
  float r = length( vUv - 0.5 ) * 2.0;
  float core = exp( -r * r * 60.0 ) * 3.5;
  float halo = exp( -r * 4.5 ) * ( 1.0 - smoothstep( 0.75, 1.0, r ) );
  gl_FragColor = vec4( uColor * ( core + halo * 0.6 ) * uIntensity, 1.0 );
}`

const orbVert = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 ); }`

export function CursorLight() {
  const gl = useThree((s) => s.gl)
  const size = useThree((s) => s.size)
  const light = useRef<THREE.PointLight>(null)
  const orb = useRef<THREE.Mesh>(null)
  const C = config.cursorLight

  const sim = useMemo(() => {
    const opts = {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      depthBuffer: false,
      stencilBuffer: false,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      wrapS: THREE.ClampToEdgeWrapping,
      wrapT: THREE.ClampToEdgeWrapping,
    } as const
    const a = new THREE.WebGLRenderTarget(4, 4, opts)
    const b = new THREE.WebGLRenderTarget(4, 4, opts)
    const material = new THREE.ShaderMaterial({
      vertexShader: quadVert,
      fragmentShader: simFrag,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        uPrev: { value: null },
        uTexel: { value: new THREE.Vector2() },
        uA: { value: new THREE.Vector2(0.5, 0.5) },
        uB: { value: new THREE.Vector2(0.5, 0.5) },
        uAspect: { value: 1 },
        uDt: { value: 1 / 60 },
        uTime: { value: 0 },
        uSplat: { value: 0 },
        uRadius: { value: C.trailRadius },
      },
    })
    const scene = new THREE.Scene()
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    return { a, b, material, scene, camera, flip: false }
  }, [C.trailRadius])

  const orbMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: orbVert,
        fragmentShader: orbFrag,
        uniforms: { uColor: { value: new THREE.Color(C.color) }, uIntensity: { value: 0 } },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        toneMapped: false,
      }),
    [C.color],
  )

  const tmp = useMemo(
    () => ({
      ndc: new THREE.Vector2(),
      ray: new THREE.Raycaster(),
      plane: new THREE.Plane(new THREE.Vector3(0, 0, 1), 0),
      hit: new THREE.Vector3(),
      pos: new THREE.Vector3(),
      proj: new THREE.Vector3(),
      lastUv: new THREE.Vector2(0.5, 0.5),
      started: false,
      presence: 0,
    }),
    [],
  )

  useEffect(() => {
    const w = Math.max(4, Math.round(size.width * 0.5))
    const h = Math.max(4, Math.round(size.height * 0.5))
    sim.a.setSize(w, h)
    sim.b.setSize(w, h)
    sim.material.uniforms.uTexel.value.set(1 / w, 1 / h)
    sim.material.uniforms.uAspect.value = w / h
  }, [size, sim])

  useEffect(
    () => () => {
      sim.a.dispose()
      sim.b.dispose()
      sim.material.dispose()
      orbMat.dispose()
      trailState.texture = null
    },
    [sim, orbMat],
  )

  useFrame(({ camera }, rawDt) => {
    const dt = Math.min(rawDt, 1 / 20)
    const want = store.w.final * (store.pointer.active ? 1 : 0)
    tmp.presence += (want - tmp.presence) * (1 - Math.exp(-dt * 3))
    const presence = tmp.presence

    // cursor ray → plane in front of the phrase (the light floats just above the keycaps)
    const pc = phraseWorldCenter()
    const lightZ = pc[2] + C.depthOffset * WORLD_SCALE
    tmp.plane.constant = -lightZ
    tmp.ndc.set(store.pointer.x, store.pointer.y)
    tmp.ray.setFromCamera(tmp.ndc, camera)
    if (tmp.ray.ray.intersectPlane(tmp.plane, tmp.hit)) {
      if (!tmp.started) {
        tmp.pos.copy(tmp.hit)
        tmp.started = true
      }
      tmp.pos.lerp(tmp.hit, 1 - Math.exp(-dt * C.follow))
    }
    store.cursor.x = tmp.pos.x
    store.cursor.y = tmp.pos.y
    store.cursor.z = pc[2]
    store.cursor.presence = presence
    trailState.presence = Math.max(trailState.presence * Math.exp(-dt * 0.4), presence)

    if (light.current) {
      light.current.position.copy(tmp.pos)
      light.current.intensity = C.intensity * presence
      light.current.visible = presence > 0.002
    }
    if (orb.current) {
      orb.current.position.copy(tmp.pos)
      orb.current.quaternion.copy(camera.quaternion)
      orbMat.uniforms.uIntensity.value = presence
      orb.current.visible = presence > 0.002
    }

    // ---- trail simulation (ping-pong) ----
    tmp.proj.copy(tmp.pos).project(camera)
    const u = tmp.proj.x * 0.5 + 0.5
    const v = tmp.proj.y * 0.5 + 0.5
    const U = sim.material.uniforms
    const speed = Math.hypot(u - tmp.lastUv.x, v - tmp.lastUv.y) / Math.max(dt, 1e-3)
    U.uA.value.copy(tmp.lastUv)
    U.uB.value.set(u, v)
    U.uDt.value = dt
    U.uTime.value = store.time
    // a moving light paints a richer trail; a resting one keeps a quiet ember
    U.uSplat.value = presence * C.trailStrength * (0.18 + Math.min(1, speed * 1.6)) * dt * 60
    const read = sim.flip ? sim.b : sim.a
    const write = sim.flip ? sim.a : sim.b
    U.uPrev.value = read.texture
    const prevTarget = gl.getRenderTarget()
    gl.setRenderTarget(write)
    gl.render(sim.scene, sim.camera)
    gl.setRenderTarget(prevTarget)
    sim.flip = !sim.flip
    trailState.texture = write.texture
    tmp.lastUv.set(u, v)
  }, -1)

  const orbSize = C.orbSize * WORLD_SCALE
  return (
    <group>
      <pointLight ref={light} color={C.color} intensity={0} distance={C.distance * WORLD_SCALE} decay={2} visible={false} />
      <mesh ref={orb} material={orbMat} visible={false} renderOrder={40} frustumCulled={false}>
        <planeGeometry args={[orbSize, orbSize]} />
      </mesh>
    </group>
  )
}
