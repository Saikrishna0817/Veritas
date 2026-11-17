/**
 * Global fog upgrade: exponential distance fog (scene.fog = FogExp2) + analytic height fog.
 *
 * Patches three's shared fog chunks once at startup, so every built-in material
 * (terrain, keys, …) gets the same atmospheric model with zero per-material work.
 * Height fog integrates density(y) = a·e^(-b·(y - base)) along the view ray — no raymarching.
 * The fog colour and density stay animatable through scene.fog.
 */
import * as THREE from 'three'
import { config } from '../../config'

let patched = false

export function patchFogChunks() {
  if (patched) return
  patched = true
  const hf = config.atmosphere.heightFog
  const ff = config.atmosphere.farFog
  const f = (v: number) => v.toFixed(6)

  THREE.ShaderChunk.fog_pars_vertex = /* glsl */ `
#ifdef USE_FOG
  varying float vFogDepth;
  varying vec3 vFogWorldPos;
#endif`

  THREE.ShaderChunk.fog_vertex = /* glsl */ `
#ifdef USE_FOG
  vFogDepth = - mvPosition.z;
  vFogWorldPos = ( inverse( viewMatrix ) * vec4( mvPosition.xyz, 1.0 ) ).xyz;
#endif`

  THREE.ShaderChunk.fog_pars_fragment = /* glsl */ `
#ifdef USE_FOG
  uniform vec3 fogColor;
  varying float vFogDepth;
  varying vec3 vFogWorldPos;
  #ifdef FOG_EXP2
    uniform float fogDensity;
  #else
    uniform float fogNear;
    uniform float fogFar;
  #endif
  float kcHeightFog( vec3 camPos, vec3 wp ) {
    const float a = ${f(hf.density)};
    const float b = ${f(hf.falloff)};
    vec3 ray = wp - camPos;
    float L = length( ray );
    float y0 = camPos.y - (${f(hf.base)});
    float dy = ray.y;
    float e0 = exp( -b * y0 );
    float integral = abs( dy ) < 1e-3
      ? a * L * e0
      : a * L * ( e0 - exp( -b * ( y0 + dy ) ) ) / ( b * dy );
    return 1.0 - exp( -max( integral, 0.0 ) );
  }
#endif`

  THREE.ShaderChunk.fog_fragment = /* glsl */ `
#ifdef USE_FOG
  #ifdef FOG_EXP2
    float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
  #else
    float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
  #endif
  float kcH = kcHeightFog( cameraPosition, vFogWorldPos );
  // far haze: everything beyond the sphere's depth gradually dissolves into the air
  float kcFar = smoothstep( ${f(ff.start)}, ${f(ff.end)}, vFogDepth ) * ${f(ff.max)};
  fogFactor = 1.0 - ( 1.0 - fogFactor ) * ( 1.0 - kcH ) * ( 1.0 - kcFar );
  gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`
}
