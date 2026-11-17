/** The sky dome of the landing page (frontend/landing, Sky.tsx): horizon to zenith, the breach of light, a drawn sun,
 *  the dissolve into the void, the aurora curtains, the stars. Shared with the app. */
import { glslNoise } from './noise.glsl'

export const skyVert = /* glsl */ `
varying vec3 vDir;
void main() {
  vec4 wp = modelMatrix * vec4( position, 1.0 );
  vDir = wp.xyz - cameraPosition;
  gl_Position = projectionMatrix * viewMatrix * wp;
  gl_Position.z = gl_Position.w; // pin to the far plane
}`

export const skyFrag = /* glsl */ `
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
