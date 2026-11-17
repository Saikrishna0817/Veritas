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
import { glslNoise } from '../shaders/noise.glsl'

const vert = /* glsl */ `
varying vec3 vDir;
void main() {
  vec4 wp = modelMatrix * vec4( position, 1.0 );
  vDir = wp.xyz - cameraPosition;
  gl_Position = projectionMatrix * viewMatrix * wp;
  gl_Position.z = gl_Position.w; // pin to the far plane
}`

const frag = /* glsl */ `
uniform vec3 uHorizon;
uniform vec3 uZenith;
uniform vec3 uGlow;
uniform vec3 uGlowDir;
uniform float uGlowStrength;
uniform float uZenithAt;
uniform vec3 uVoidTop;
uniform vec3 uVoidBottom;
uniform float uFade;
uniform float uTime;
uniform float uStars;
uniform float uAurora;
uniform float uViewH;
uniform float uAuroraTop;
uniform vec2 uAuroraShade;
uniform float uAuroraSpeed;
uniform float uAuroraShift;
uniform vec3 uAuroraLow;
uniform vec3 uAuroraHigh;
uniform vec3 uAuroraEdge;
uniform float uAuroraGain;
uniform vec3 uStarColor;
uniform float uSunSize;
uniform float uSunBands;
uniform float uSunIntensity;
uniform vec3 uSunTop;
uniform vec3 uSunBottom;
varying vec3 vDir;
${glslNoise}

// One aurora curtain in screen-ish sky space: a meandering lower edge, a long soft glow above it,
// fine vertical rays and slow folds. q = forward-projected sky coords (x across, y up).
float auroraCurtain( vec2 q, float t, float seed ) {
  float x = q.x;
  float edge = 0.28 * sin( x * 1.25 + seed + t * 0.06 )
    + 0.09 * sin( x * 2.9 - t * 0.09 + seed * 2.3 )
    + ( kcFbm( vec2( x * 0.8 + seed, t * 0.035 ) ) - 0.5 ) * 0.42;
  float d = q.y - edge;
  float body = smoothstep( -0.035, 0.015, d ) * exp( -max( d, 0.0 ) * 3.0 );
  float rays = kcNoise( vec2( x * 22.0 + t * 0.25 + seed * 7.0, d * 1.2 - t * 0.04 ) );
  rays = 0.45 + 0.55 * rays * rays;
  float folds = 0.6 + 0.4 * kcNoise( vec2( x * 4.5 - t * 0.12, seed + t * 0.02 ) );
  return body * rays * folds;
}

void main() {
  vec3 d = normalize( vDir );
  float h = d.y;
  vec3 col = mix( uHorizon, uZenith, smoothstep( -0.02, uZenithAt, h ) );
  col = mix( col, uHorizon * 0.8, smoothstep( 0.0, -0.35, h ) );

  // breach of light above the object
  float g = max( dot( d, normalize( uGlowDir ) ), 0.0 );
  col += uGlow * ( pow( g, 3.0 ) * 0.28 + pow( g, 12.0 ) * 0.4 + pow( g, 60.0 ) * 0.32 ) * uGlowStrength;

  // faint high veils
  vec2 uv = d.xz / ( abs( d.y ) + 0.28 );
  float veil = kcFbm( uv * 1.3 + vec2( uTime * 0.006, uTime * 0.002 ) );
  col *= 1.0 + ( veil - 0.5 ) * 0.16 * smoothstep( 0.0, 0.35, h );

  // a drawn sun (a look's own: a low evening sun, or a banded arcade sun), in the plane facing uGlowDir
  if ( uSunSize > 0.0 ) {
    vec3 sd = normalize( uGlowDir );
    float dz = dot( d, sd );
    if ( dz > 0.0 ) {
      vec3 sx = normalize( cross( vec3( 0.0, 1.0, 0.0 ), sd ) );
      vec3 sy = cross( sd, sx );
      vec2 sp = vec2( dot( d, sx ), dot( d, sy ) ) / ( dz * uSunSize ); // the rim is at length 1
      float r = length( sp );
      float aa = fwidth( r ) * 1.5;
      float disc = 1.0 - smoothstep( 1.0 - aa, 1.0 + aa, r );
      float cut = 1.0;
      if ( uSunBands > 0.0 && sp.y < 0.2 ) {
        // gaps across the lower part, each wider than the one above
        float k = clamp( ( 0.2 - sp.y ) / 1.2, 0.0, 1.0 );
        float s = fract( k * uSunBands );
        float gap = 0.14 + 0.5 * k;
        float ay = fwidth( sp.y ) / 1.2 * uSunBands * 1.5;
        cut = smoothstep( gap - ay, gap + ay, s ) * ( 1.0 - smoothstep( 1.0 - 2.0 * ay, 1.0, s ) );
      }
      vec3 sc = mix( uSunBottom, uSunTop, smoothstep( -0.9, 0.8, sp.y ) ) * uSunIntensity;
      col = mix( col, sc, disc * cut );
    }
  }

  vec3 voidCol = mix( uVoidBottom, uVoidTop, smoothstep( -0.25, 0.7, h ) );
  col = mix( col, voidCol, uFade );

  // northern lights (sections 2–3): two curtains in the scene's blue, cyan-white at the lower edge
  if ( uAurora > 0.001 && d.z < -0.02 ) {
    vec2 q = d.xy / -d.z;
    q = vec2( q.x * 2.4, ( q.y - 0.14 + uAuroraShift ) * 2.4 );
    float t = uTime * uAuroraSpeed;
    float a1 = auroraCurtain( q, t, 0.0 );
    float a2 = auroraCurtain( q * vec2( 0.8, 1.0 ) + vec2( 1.7, -0.32 ), t * 0.8, 3.7 ) * 0.65;
    float a = a1 + a2;
    // colour by height above the edge: bright edge → deep blue body → faint violet top
    float h1 = clamp( q.y + 0.3, 0.0, 1.0 );
    vec3 tint = mix( uAuroraEdge, uAuroraLow, smoothstep( 0.0, 0.35, h1 ) );
    tint = mix( tint, uAuroraHigh, smoothstep( 0.45, 1.0, h1 ) );
    col += tint * a * uAurora * uAuroraGain * 0.85;
    // brighter cyan-white seam right at the lower edge, the signature of real aurora
    col += uAuroraEdge * pow( a1, 3.0 ) * uAurora * uAuroraGain * 0.5;
  }
  // night closing in from the top of the frame (screen space, like the design frame): the curtains
  // dissolve into a deep dark band instead of running into the edge
  if ( uAurora > 0.001 ) {
    float sy = gl_FragCoord.y / uViewH;
    float shade = smoothstep( uAuroraShade.x, uAuroraShade.y, sy );
    col = mix( col, uVoidTop * 0.6, shade * shade * ( 3.0 - 2.0 * shade ) * uAuroraTop * uAurora );
  }

  // twinkling stars in the dark (third section): a jittered star per sparse grid cell
  if ( uStars > 0.001 && d.z < -0.05 ) {
    vec2 sp = d.xy / -d.z * 150.0;
    vec2 cell = floor( sp );
    vec2 fr = fract( sp );
    float h0 = kcHash12( cell );
    if ( h0 < 0.11 ) {
      vec2 sPos = vec2( kcHash12( cell + 1.7 ), kcHash12( cell + 3.1 ) ) * 0.8 + 0.1;
      float sz = mix( 0.035, 0.11, pow( kcHash12( cell + 5.3 ), 3.0 ) );
      float rate = 0.6 + 2.6 * kcHash12( cell + 7.7 );
      float tw = 0.5 + 0.5 * sin( uTime * rate + h0 * 90.0 );
      tw = tw * tw;
      float mag = 0.25 + 0.75 * kcHash12( cell + 9.1 );
      float star = smoothstep( sz, 0.0, length( fr - sPos ) ) * tw * mag;
      // the reveal is staggered per star, so they light up one by one
      float reveal = smoothstep( kcHash12( cell + 11.3 ) * 0.7, kcHash12( cell + 11.3 ) * 0.7 + 0.3, uStars );
      col += uStarColor * star * reveal * 1.6;
    }
  }

  // sub-LSB dither against banding in the dark gradients
  col += ( kcHash12( gl_FragCoord.xy + fract( uTime ) * 61.0 ) - 0.5 ) / 512.0;
  gl_FragColor = vec4( max( col, 0.0 ), 1.0 );
}`

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
