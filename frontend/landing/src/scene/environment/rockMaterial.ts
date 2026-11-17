/**
 * Shared rock/snow material for every mountain in the scene (procedural valley + foreground massifs),
 * so they all sit in one palette and one light: slope/altitude snow, strata, fragment micro relief,
 * optional erosion detail texture, and the dissolve into the final void.
 *
 * A look can change what the mountains are made of (config.terrain.surface): how much of the settled layer
 * there is and where it settles, how strongly the strata show, and two emissive surfaces — lava (a river on
 * the valley floor and veins up the slopes) and a neon grid draped over the relief.
 */
import * as THREE from 'three'
import { config } from '../../config'
import { look } from '../../looks'
import { glslNoise } from '../shaders/noise.glsl'

export interface RockMaterialOptions {
  /** Greyscale erosion/detail map (planar UV). */
  detailMap?: THREE.Texture
  /** How strongly the detail map is overlaid onto the palette (0..1, overlay blend). */
  detailStrength?: number
  /** World-space scale of the procedural micro relief. */
  reliefScale?: number
  /** Strength of the micro relief. */
  relief?: number
  /** Altitude band where snow can settle (world Y). */
  snowLine?: [number, number]
  key: string
}

export function createRockMaterial(opts: RockMaterialOptions) {
  const A = config.atmosphere
  const S = config.terrain.surface
  const uniforms = {
    uFade: { value: 0 },
    uVoid: { value: new THREE.Color(A.voidColor) },
    uRockA: { value: new THREE.Color(A.rock.dark) },
    uRockB: { value: new THREE.Color(A.rock.light) },
    uSnow: { value: new THREE.Color(A.rock.snow) },
    uDetail: { value: opts.detailMap ?? null },
    uDetailStrength: { value: opts.detailStrength ?? 0.85 },
    uStrata: { value: S.strata },
    uSnowAmount: { value: S.snow },
    uSnowLow: { value: S.snowLow },
    uRockTime: { value: 0 },
    uLavaColor: { value: new THREE.Color(S.lava.color) },
    uLavaIntensity: { value: S.lava.intensity },
    uLavaLevel: { value: new THREE.Vector2(S.lava.level[0], S.lava.level[1]) },
    uLavaVeins: { value: S.lava.veins },
    uGridLow: { value: new THREE.Color(S.grid.low) },
    uGridHigh: { value: new THREE.Color(S.grid.high) },
    uGridIntensity: { value: S.grid.intensity },
    uGridCell: { value: S.grid.cell },
    uGridContour: { value: S.grid.contour },
  }
  const snow = opts.snowLine ?? [-40, 25]
  const relief = (opts.relief ?? 2.2).toFixed(3)
  const reliefScale = (opts.reliefScale ?? 0.09).toFixed(4)
  const useDetail = !!opts.detailMap
  const useLava = S.lava.intensity > 0
  const useGrid = S.grid.intensity > 0

  const material = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.92, metalness: 0, envMapIntensity: 0.35 })
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms)
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
varying vec3 vTWorldPos;
varying vec3 vTWorldNormal;
${useDetail ? 'attribute vec2 aRockUv;\nvarying vec2 vTUv;' : ''}`,
      )
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
vTWorldPos = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;
vTWorldNormal = normalize( mat3( modelMatrix ) * objectNormal );
${useDetail ? 'vTUv = aRockUv;' : ''}`,
      )
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
uniform float uFade;
uniform vec3 uVoid;
uniform vec3 uRockA;
uniform vec3 uRockB;
uniform vec3 uSnow;
uniform sampler2D uDetail;
uniform float uDetailStrength;
uniform float uStrata;
uniform float uSnowAmount;
uniform float uSnowLow;
uniform float uRockTime;
uniform vec3 uLavaColor;
uniform float uLavaIntensity;
uniform vec2 uLavaLevel;
uniform float uLavaVeins;
uniform vec3 uGridLow;
uniform vec3 uGridHigh;
uniform float uGridIntensity;
uniform float uGridCell;
uniform float uGridContour;
varying vec3 vTWorldPos;
varying vec3 vTWorldNormal;
${useDetail ? 'varying vec2 vTUv;' : ''}
${glslNoise}
float terrainSnow;
vec3 terrainGlow;
`,
      )
      .replace(
        '#include <map_fragment>',
        `#include <map_fragment>
{
  vec3 wp = vTWorldPos;
  vec3 wn = normalize( vTWorldNormal );
  float n1 = kcFbm( wp.xz * 0.018 );
  float n2 = kcNoise( wp.xz * 0.11 + wp.y * 0.05 );
  float strata = 0.5 + 0.5 * sin( wp.y * 0.21 + n1 * 6.0 );
  float detail = ${useDetail ? 'texture2D( uDetail, vTUv ).r' : '0.5'};
  vec3 rock = mix( uRockA, uRockB, clamp( n1 * ( 1.0 - uStrata ) + strata * uStrata + n2 * 0.12 + ( detail - 0.5 ) * 0.5, 0.0, 1.0 ) );
  float alt = smoothstep( ${snow[0].toFixed(1)}, ${snow[1].toFixed(1)}, wp.y + ( n1 - 0.5 ) * 40.0 );
  // a look can let the layer settle on the low ground instead (sand in a desert)
  alt = mix( alt, 1.0 - alt, uSnowLow );
  float slopeSnow = smoothstep( 0.62, 0.86, wn.y + ( n2 - 0.5 ) * 0.22 );
  ${useDetail ? '// the erosion map: bright = settled snow/scree, dark = exposed rock gullies\n  slopeSnow = mix( slopeSnow, smoothstep( 0.6, 0.9, detail ), 0.35 ) * smoothstep( 0.35, 0.6, wn.y );' : ''}
  terrainSnow = slopeSnow * alt * uSnowAmount;
  vec3 base = mix( rock, uSnow, terrainSnow );
  ${useDetail ? `// the asset's own texture, overlay-blended into the scene palette
  vec3 ov = mix( 2.0 * base * detail, 1.0 - 2.0 * ( 1.0 - base ) * ( 1.0 - detail ), step( 0.5, base ) );
  base = mix( base, clamp( ov, 0.0, 1.0 ), uDetailStrength );` : ''}
  terrainGlow = vec3( 0.0 );
  ${useLava ? `{
    // a river of lava on the lowest ground: dark crust, a network of bright cracks between its plates
    // (two scales, drifting against each other) and a few open pools…
    float river = 1.0 - smoothstep( uLavaLevel.x, uLavaLevel.y, wp.y + ( n2 - 0.5 ) * 6.0 );
    vec2 flow = vec2( uRockTime * 0.02, uRockTime * 0.008 );
    float c1 = 1.0 - abs( kcNoise( wp.xz * 0.085 + flow ) * 2.0 - 1.0 );
    float c2 = 1.0 - abs( kcNoise( wp.xz * 0.21 - flow * 1.7 ) * 2.0 - 1.0 );
    float cracks = pow( c1, 6.0 ) + 0.6 * pow( c2, 8.0 );
    float pools = smoothstep( 0.5, 0.75, kcFbm( wp.xz * 0.03 + flow * 0.5 ) );
    // …and thin veins up the slopes (an iso-line of the noise), dying out with height and distance
    float vein = 1.0 - smoothstep( 0.0, 0.03, abs( kcFbm3( wp.xz * 0.028 + 3.7 ) - 0.44 ) );
    vein *= 1.0 - smoothstep( uLavaLevel.y, uLavaLevel.y + 90.0, wp.y );
    vein *= 1.0 - smoothstep( 250.0, 800.0, length( wp - cameraPosition ) );
    float pulse = 0.8 + 0.2 * sin( uRockTime * 0.7 + n1 * 18.0 );
    float lava = river * ( 0.07 + 1.3 * cracks + 1.1 * pools ) + vein * uLavaVeins * ( 1.0 - river );
    terrainGlow += uLavaColor * lava * pulse * uLavaIntensity;
    base *= 1.0 - 0.8 * river;
  }` : ''}
  ${useGrid ? `{
    // thin lines, faded out where they would get closer than a few pixels: a grid on the ground plane…
    float lines = 0.0;
    if ( uGridCell > 0.0 ) {
      vec2 gp = wp.xz / uGridCell;
      vec2 gw = max( fwidth( gp ), vec2( 1e-4 ) );
      vec2 gl2 = 1.0 - smoothstep( vec2( 0.0 ), gw * 1.4, abs( fract( gp - 0.5 ) - 0.5 ) );
      lines = max( gl2.x, gl2.y ) * ( 1.0 - smoothstep( 0.1, 0.5, max( gw.x, gw.y ) ) );
    }
    // …and contour lines, one every uGridContour units of height
    if ( uGridContour > 0.0 ) {
      float cp = wp.y / uGridContour;
      float cw = max( fwidth( cp ), 1e-4 );
      float cl = 1.0 - smoothstep( 0.0, cw * 1.5, abs( fract( cp - 0.5 ) - 0.5 ) );
      lines = max( lines, cl * ( 1.0 - smoothstep( 0.25, 0.7, cw ) ) );
    }
    terrainGlow += mix( uGridLow, uGridHigh, smoothstep( -70.0, 40.0, wp.y ) ) * lines * uGridIntensity;
  }` : ''}
  diffuseColor.rgb = base;
}`,
      )
      .replace(
        '#include <roughnessmap_fragment>',
        `#include <roughnessmap_fragment>
roughnessFactor = mix( roughnessFactor, 0.7, terrainSnow );`,
      )
      .replace(
        '#include <emissivemap_fragment>',
        `#include <emissivemap_fragment>
totalEmissiveRadiance += terrainGlow;`,
      )
      .replace(
        '#include <normal_fragment_maps>',
        `#include <normal_fragment_maps>
{
  float dist = length( vTWorldPos - cameraPosition );
  float fade = 1.0 - smoothstep( 120.0, 900.0, dist );
  if ( fade > 0.001 ) {
    vec2 p = vTWorldPos.xz * ${reliefScale};
    float e = 0.35;
    float h0 = kcFbm3( p );
    float hx = kcFbm3( p + vec2( e, 0.0 ) );
    float hz = kcFbm3( p + vec2( 0.0, e ) );
    vec3 gw = vec3( -( hx - h0 ), 0.0, -( hz - h0 ) ) * ( ${relief} * fade );
    normal = normalize( normal + ( viewMatrix * vec4( gw, 0.0 ) ).xyz );
  }
}`,
      )
      .replace(
        '#include <dithering_fragment>',
        `gl_FragColor.rgb = mix( gl_FragColor.rgb, uVoid, uFade );
if ( any( isnan( gl_FragColor.rgb ) ) || any( isinf( gl_FragColor.rgb ) ) ) gl_FragColor.rgb = uVoid;
#include <dithering_fragment>`,
      )
  }
  material.customProgramCacheKey = () => `keycap-rock-${opts.key}-${look.id}`
  return { material, uniforms }
}
