/**
 * Shared by the landing page and the app (frontend/shared/argilla).
 * Keycap material: physically-based, slightly matte off-white ABS/PBT plastic,
 * with legends printed from the glyph atlas through the model's own cap UVs.
 *
 * - The glyph is placed only where aCap = 1 (dished top surface) — never on the walls.
 * - Glyph local coords come from the cap UV (planar, isotropic → no stretching).
 * - Ink is dark grey and a touch glossier than the plastic, like pad-printed legends.
 *
 * A look (config.keycap) can change the plastic itself, give every key its own colour (aTint, per instance;
 * the ink flips to the light one on dark keys) and backlight the keys: glowing legends and glowing skirts,
 * in one colour or a colour that runs across the sphere.
 */
import * as THREE from 'three'
import { ATLAS_GRID } from './glyphs'

/** The keycap look (the landing page: config.keycap with its look's overrides). */
export interface KeycapLook {
  color: string
  ink: string
  inkAlt: string
  roughness: number
  specularIntensity: number
  clearcoat: number
  clearcoatRoughness: number
  envMapIntensity: number
  inkRoughness: number
  glow: { colors: [string, string, string]; legend: number; side: number; speed: number; scale: number; flicker: number }
  xrayBody: string
  xrayGlow: string
}

export interface KeycapMaterialOptions {
  color?: THREE.ColorRepresentation
  ink?: THREE.ColorRepresentation
  /** Fraction of the cap width covered by a glyph cell. */
  glyphScale?: number
  wireColor?: THREE.ColorRepresentation
  wireIntensity?: number
  /** Wire lattice cells per model unit. */
  wireDensity?: number
}

/** Cap UV width measured on the source model (u range 0.382…0.617). */
const CAP_UV_WIDTH = 0.235

export function createKeycapMaterial(atlas: THREE.Texture, K: KeycapLook, cacheKey: string, opts: KeycapMaterialOptions = {}) {
  // the key's own colour comes per instance (aTint), so the material itself is white
  const material = new THREE.MeshPhysicalMaterial({
    color: '#ffffff',
    roughness: K.roughness,
    metalness: 0,
    specularIntensity: K.specularIntensity,
    clearcoat: K.clearcoat,
    clearcoatRoughness: K.clearcoatRoughness,
    envMapIntensity: K.envMapIntensity,
  })

  const uniforms = {
    uAtlas: { value: atlas },
    uInk: { value: new THREE.Color(opts.ink ?? K.ink) },
    uInkAlt: { value: new THREE.Color(K.inkAlt) },
    uGlyphSpan: { value: CAP_UV_WIDTH * (opts.glyphScale ?? 0.64) },
    uInkRoughness: { value: K.inkRoughness },
    uXrayBody: { value: new THREE.Color(K.xrayBody) },
    uXrayGlow: { value: new THREE.Color(K.xrayGlow).multiplyScalar(0.25) },
    uGlowA: { value: new THREE.Color(K.glow.colors[0]) },
    uGlowB: { value: new THREE.Color(K.glow.colors[1]) },
    uGlowC: { value: new THREE.Color(K.glow.colors[2]) },
    uGlowLegend: { value: K.glow.legend },
    uGlowSide: { value: K.glow.side },
    uGlowSpeed: { value: K.glow.speed },
    uGlowScale: { value: K.glow.scale },
    uGlowFlicker: { value: K.glow.flicker },
    uKeyTime: { value: 0 },
    uXrayWire: { value: new THREE.Color(opts.wireColor ?? '#3d7bff').multiplyScalar(opts.wireIntensity ?? 1.4) },
    uXrayDensity: { value: opts.wireDensity ?? 6 },
  }
  const G = ATLAS_GRID.toFixed(1)

  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms)

    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        /* glsl */ `#include <common>
attribute float aGlyph;
attribute float aCap;
attribute vec2 aKeyUv;
attribute float aXray;
attribute vec3 aTint;
varying float vXray;
varying vec3 vTint;
varying vec3 vKeyOrigin;
varying float vKeySeed;
varying vec3 vLocalPos;
varying vec3 vLocalNrm;
varying float vGlyph;
varying float vCap;
varying vec2 vKeyUv;`,
      )
      .replace(
        '#include <begin_vertex>',
        /* glsl */ `#include <begin_vertex>
vGlyph = aGlyph;
vCap = aCap;
vKeyUv = aKeyUv;
vXray = aXray;
vTint = aTint;
// the key's own origin in the world and a number of its own: one glow colour and one rhythm per key
#ifdef USE_INSTANCING
vKeyOrigin = ( modelMatrix * instanceMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
vKeySeed = fract( sin( float( gl_InstanceID ) * 12.9898 + aGlyph * 4.137 ) * 43758.5453 );
#else
vKeyOrigin = modelMatrix[ 3 ].xyz;
vKeySeed = 0.0;
#endif
vLocalPos = position;
vLocalNrm = normal;`,
      )

    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        /* glsl */ `#include <common>
uniform sampler2D uAtlas;
uniform vec3 uInk;
uniform float uGlyphSpan;
uniform float uInkRoughness;
uniform vec3 uXrayBody;
uniform vec3 uXrayGlow;
uniform vec3 uXrayWire;
uniform float uXrayDensity;
uniform vec3 uInkAlt;
uniform vec3 uGlowA;
uniform vec3 uGlowB;
uniform vec3 uGlowC;
uniform float uGlowLegend;
uniform float uGlowSide;
uniform float uGlowSpeed;
uniform float uGlowScale;
uniform float uGlowFlicker;
uniform float uKeyTime;
varying float vXray;
varying vec3 vTint;
varying vec3 vKeyOrigin;
varying float vKeySeed;
varying vec3 vLocalPos;

// backlight colour of this key: a three-stop loop (A → B → C → A) that runs across the sphere and in time,
// dimmed on the key's own slow rhythm
vec3 keycapGlow() {
  float h = fract( dot( vKeyOrigin.xy, vec2( 0.8, 0.45 ) ) * uGlowScale + uKeyTime * uGlowSpeed ) * 3.0;
  vec3 c = h < 1.0 ? mix( uGlowA, uGlowB, h ) : h < 2.0 ? mix( uGlowB, uGlowC, h - 1.0 ) : mix( uGlowC, uGlowA, h - 2.0 );
  float dim = 0.5 + 0.5 * sin( uKeyTime * ( 0.9 + vKeySeed * 1.7 ) + vKeySeed * 40.0 );
  return c * ( 1.0 - uGlowFlicker * dim );
}
varying vec3 vLocalNrm;
varying float vGlyph;
varying float vCap;
varying vec2 vKeyUv;

// blue wireframe lattice in the key's own space (triplanar: on each face only the two in-plane axes
// draw lines), anti-aliased with screen derivatives — the key's "inner structure" under x-ray
float keycapWire() {
  vec3 p = vLocalPos * uXrayDensity;
  vec3 g = abs( fract( p - 0.5 ) - 0.5 ) / max( fwidth( p ), vec3( 1e-4 ) );
  vec3 l = 1.0 - clamp( g * 0.9, 0.0, 1.0 ); // ~1 px lines per axis
  vec3 n = abs( normalize( vLocalNrm ) );
  vec3 w = smoothstep( 0.35, 0.9, n );
  float onX = max( l.y, l.z ) * w.x; // face facing ±x shows y/z lines
  float onY = max( l.x, l.z ) * w.y;
  float onZ = max( l.x, l.y ) * w.z;
  // silhouette-ish edges where the faces bend
  float edge = 1.0 - smoothstep( 0.0, 0.25, abs( max( max( n.x, n.y ), n.z ) - 0.72 ) );
  return clamp( max( max( onX, onY ), max( onZ, edge * 0.6 ) ), 0.0, 1.0 );
}

float keycapInk() {
  // cap UV → glyph cell coords (x right, y up on the printed face)
  vec2 g = vec2( 0.5 + ( vKeyUv.x - 0.5 ) / uGlyphSpan, 0.5 - ( vKeyUv.y - 0.778 ) / uGlyphSpan );
  vec2 inside2 = step( vec2( 0.0 ), g ) * step( g, vec2( 1.0 ) );
  float inside = inside2.x * inside2.y;
  vec2 gc = clamp( g, 0.002, 0.998 );
  float id = floor( vGlyph + 0.5 );
  float col = mod( id, ${G} );
  float row = floor( id / ${G} );
  vec2 auv = vec2( ( col + gc.x ) / ${G}, ( ${G} - row - 1.0 + gc.y ) / ${G} );
  float cov = texture2D( uAtlas, auv ).r;
  return cov * inside * smoothstep( 0.55, 0.95, vCap );
}`,
      )
      .replace(
        '#include <map_fragment>',
        /* glsl */ `#include <map_fragment>
diffuseColor.rgb *= vTint;
float keyInk = keycapInk();
// dark keys take the light ink
diffuseColor.rgb = mix( diffuseColor.rgb, dot( vTint, vec3( 0.2126, 0.7152, 0.0722 ) ) > 0.2 ? uInk : uInkAlt, keyInk );
// x-ray (third section, under the cursor): the plastic turns to dark glass so the blue wireframe
// drawn on top reads as the key's inner structure
diffuseColor.rgb = mix( diffuseColor.rgb, uXrayBody, vXray * 0.9 );`,
      )
      .replace(
        '#include <roughnessmap_fragment>',
        /* glsl */ `#include <roughnessmap_fragment>
roughnessFactor = mix( roughnessFactor, uInkRoughness, keyInk );`,
      )
      .replace(
        '#include <emissivemap_fragment>',
        /* glsl */ `#include <emissivemap_fragment>
if ( vXray > 0.002 ) totalEmissiveRadiance += ( uXrayGlow + uXrayWire * keycapWire() ) * vXray;
if ( uGlowLegend > 0.0 || uGlowSide > 0.0 ) {
  // the walls glow most at the skirt and not at all on the cap
  float wall = ( 1.0 - smoothstep( 0.05, 0.6, vCap ) ) * smoothstep( 0.95, 0.0, vLocalPos.y );
  totalEmissiveRadiance += keycapGlow() * ( uGlowLegend * keyInk + uGlowSide * wall );
}`,
      )
      .replace(
        '#include <dithering_fragment>',
        /* glsl */ `// never let a NaN/Inf pixel reach bloom (one bad pixel would flash the whole frame)
if ( any( isnan( gl_FragColor.rgb ) ) || any( isinf( gl_FragColor.rgb ) ) ) gl_FragColor.rgb = vec3( 0.0 );
#include <dithering_fragment>`,
      )
  }
  material.customProgramCacheKey = () => `keycap-keycap-v4-${cacheKey}`

  return { material, uniforms }
}
