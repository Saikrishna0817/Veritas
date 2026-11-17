/**
 * Postprocessing (HDR half-float pipeline, no MSAA — on some GPUs multisampled composer targets
 * output a blank frame; edges are antialiased with SMAA at the end of the chain instead):
 *   [god rays from the core] → soft bloom (only HDR emitters pass the threshold) → ACES filmic
 *   → cinematic grade (cool shadows, gentle desaturation, fine grain, restrained vignette).
 */
import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { EffectComposer, Bloom, ToneMapping, GodRays, SMAA } from '@react-three/postprocessing'
import { BlendFunction, Effect, EffectAttribute, KernelSize, ToneMappingMode, type BloomEffect } from 'postprocessing'
import { trailState } from '../state/trail'
import * as THREE from 'three'
import { config } from '../config'
import { store } from '../state/store'

const gradeFrag = /* glsl */ `
uniform float uGrain;
uniform float uVignette;
uniform float uTime;
uniform float uSaturation;
uniform vec3 uShadowTint;
uniform vec3 uHighlightTint;
uniform float uReveal;
uniform vec3 uCover;
uniform float uScanlines;

float gradeHash( vec2 p ) {
  vec3 p3 = fract( vec3( p.xyx ) * 0.1031 );
  p3 += dot( p3, p3.yzx + 33.33 );
  return fract( ( p3.x + p3.y ) * p3.z );
}

void mainImage( const in vec4 inputColor, const in vec2 uv, out vec4 outputColor ) {
  vec3 c = inputColor.rgb;
  float l = dot( c, vec3( 0.2126, 0.7152, 0.0722 ) );
  c = mix( vec3( l ), c, uSaturation );
  // split toning: cold lift in the shadows, neutral-cool highlights
  c += uShadowTint * ( 1.0 - smoothstep( 0.0, 0.3, l ) );
  c *= mix( vec3( 1.0 ), uHighlightTint, smoothstep( 0.45, 1.0, l ) );
  // restrained vignette
  vec2 q = ( uv - 0.5 ) * vec2( resolution.x / resolution.y, 1.0 );
  c *= 1.0 - uVignette * smoothstep( 0.35, 1.05, length( q ) );
  // luminance-weighted film grain (fresh every frame)
  float g = gradeHash( uv * resolution + fract( uTime * 7.31 ) * 917.0 ) - 0.5;
  c += g * uGrain * ( 0.25 + 0.75 * ( 1.0 - l ) ) * 0.25;
  c = max( c, 0.0 );
  // a look's scanlines (two device pixels on, two off)
  if ( uScanlines > 0.0 ) c *= 1.0 - uScanlines * ( 0.5 + 0.5 * sin( uv.y * resolution.y * 1.5708 ) );
  // preloader exit, drawn in the same frame as the scene: everything outside an opening circle is the
  // preloader colour (no grain, so it matches the DOM layer it replaces pixel for pixel)
  if ( uReveal < 1.0 ) {
    float R = uReveal * ( 0.5 * length( resolution ) + 4.0 );
    float d = length( ( uv - 0.5 ) * resolution );
    c = mix( c, uCover, smoothstep( R - 1.0, R + 1.0, d ) );
  }
  outputColor = vec4( c, inputColor.a );
}`

class CinematicGradeEffect extends Effect {
  constructor() {
    super('CinematicGradeEffect', gradeFrag, {
      blendFunction: BlendFunction.NORMAL,
      uniforms: new Map<string, THREE.Uniform>([
        ['uGrain', new THREE.Uniform(config.post.grain)],
        ['uVignette', new THREE.Uniform(config.post.vignette)],
        ['uTime', new THREE.Uniform(0)],
        ['uSaturation', new THREE.Uniform(config.post.saturation)],
        ['uShadowTint', new THREE.Uniform(new THREE.Vector3(...config.post.shadowTint))],
        ['uHighlightTint', new THREE.Uniform(new THREE.Vector3(...config.post.highlightTint))],
        ['uScanlines', new THREE.Uniform(config.post.scanlines)],
        ['uReveal', new THREE.Uniform(0)],
        // linear value of the preloader's CSS colour (three converts the sRGB hex)
        ['uCover', new THREE.Uniform(new THREE.Color(config.preloader.color))],
      ]),
    })
  }
  update(_renderer: THREE.WebGLRenderer, _input: THREE.WebGLRenderTarget, deltaTime?: number) {
    const u = this.uniforms.get('uTime')
    if (u) u.value += deltaTime ?? 0.016
    this.uniforms.get('uReveal')!.value = store.reveal
    store.postFrames++
  }
}

const trailFrag = /* glsl */ `
uniform sampler2D uTrail;
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform float uIntensity;
uniform float uTrailTime;

float trailHash( vec2 p ) { return fract( sin( dot( p, vec2( 41.3, 289.1 ) ) ) * 45758.5453 ); }
float trailNoise( vec2 p ) {
  vec2 i = floor( p ), f = fract( p );
  vec2 u = f * f * ( 3.0 - 2.0 * f );
  return mix( mix( trailHash( i ), trailHash( i + vec2( 1, 0 ) ), u.x ), mix( trailHash( i + vec2( 0, 1 ) ), trailHash( i + vec2( 1, 1 ) ), u.x ), u.y );
}

void mainImage( const in vec4 inputColor, const in vec2 uv, const in float depth, out vec4 outputColor ) {
  float t = texture2D( uTrail, uv ).r;
  // break the density into wisps so it reads as smoke, not a blur
  vec2 q = uv * vec2( aspect, 1.0 );
  vec2 warp = vec2( trailNoise( q * 3.0 + uTrailTime * 0.05 ), trailNoise( q * 3.0 - uTrailTime * 0.04 + 7.1 ) ) - 0.5;
  vec2 w = q + warp * 0.12;
  float n = trailNoise( w * 5.0 + vec2( 0.0, -uTrailTime * 0.09 ) ) * 0.55
    + trailNoise( w * 12.0 + uTrailTime * 0.07 ) * 0.3
    + trailNoise( w * 26.0 - uTrailTime * 0.12 ) * 0.15;
  float smoke = t * ( 0.3 + 1.4 * n * n * 1.6 );
  vec3 col = mix( uColorA, uColorB, smoothstep( 0.25, 1.6, t ) ) * smoke * uIntensity;
  // the smoke hangs behind the keycaps: background pixels (far plane) take it fully, keys only a veil
  float behind = step( 0.99995, depth );
  col *= mix( 0.22, 1.0, behind );
  outputColor = vec4( inputColor.rgb + col, inputColor.a );
}`

class TrailEffect extends Effect {
  constructor() {
    const C = config.cursorLight
    super('KeycapTrailEffect', trailFrag, {
      blendFunction: BlendFunction.NORMAL,
      attributes: EffectAttribute.DEPTH,
      uniforms: new Map<string, THREE.Uniform>([
        ['uTrail', new THREE.Uniform(null)],
        ['uColorA', new THREE.Uniform(new THREE.Color(C.trailColor))],
        ['uColorB', new THREE.Uniform(new THREE.Color(C.trailCore))],
        ['uIntensity', new THREE.Uniform(0)],
        ['uTrailTime', new THREE.Uniform(0)],
      ]),
    })
  }
  update() {
    this.uniforms.get('uTrail')!.value = trailState.texture
    this.uniforms.get('uIntensity')!.value = config.cursorLight.trailIntensity * Math.min(1, trailState.presence * 4)
    this.uniforms.get('uTrailTime')!.value = store.time
  }
}

export function PostFX({ sun }: { sun: THREE.Mesh | null }) {
  const P = config.post
  const bloom = useRef<BloomEffect>(null)
  const grade = useMemo(() => new CinematicGradeEffect(), [])
  const trail = useMemo(() => new TrailEffect(), [])
  useEffect(
    () => () => {
      grade.dispose()
      trail.dispose()
    },
    [grade, trail],
  )

  useFrame(() => {
    const w = store.w
    if (bloom.current) bloom.current.intensity = P.bloomIntensity * (0.55 + 0.9 * w.glow) * (1 - 0.35 * w.final)
  })

  const common = (
    <>
      <primitive object={trail} />
      <Bloom
        ref={bloom}
        mipmapBlur
        intensity={P.bloomIntensity}
        luminanceThreshold={P.bloomThreshold}
        luminanceSmoothing={P.bloomSmoothing}
        radius={0.72}
      />
      <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
      <primitive object={grade} />
      <SMAA />
    </>
  )

  // Mount the composer exactly once: switching between a no-god-rays and a god-rays composer re-creates
  // every render target (a blank frame). Until the core light exists the scene renders without post.
  if (P.godRays && !sun) return null
  if (P.godRays && sun) {
    return (
      <EffectComposer multisampling={0} frameBufferType={THREE.HalfFloatType} enableNormalPass={false}>
        <GodRays
          sun={sun}
          blendFunction={BlendFunction.SCREEN}
          samples={40}
          density={0.95}
          decay={0.93}
          weight={0.42}
          exposure={0.42}
          clampMax={1}
          kernelSize={KernelSize.SMALL}
          blur
        />
        {common}
      </EffectComposer>
    )
  }
  return (
    <EffectComposer multisampling={0} frameBufferType={THREE.HalfFloatType} enableNormalPass={false}>
      {common}
    </EffectComposer>
  )
}
