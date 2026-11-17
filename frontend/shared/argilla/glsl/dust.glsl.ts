/** Suspended dust (landing page, Atmosphere.tsx): drifting points, brighter near the core. Shared with the app. */
export const dustVert = /* glsl */ `
attribute float aSeed;
uniform float uTime;
uniform float uSize;
uniform float uPixelRatio;
uniform vec3 uCenter;
uniform float uCore;
uniform float uRise;
varying float vAlpha;
varying float vGlow;
void main() {
  vec3 p = position;
  // a look can make the dust rise (sparks): each mote climbs at its own pace and wraps around
  p.y = mod( p.y + 85.0 + uTime * uRise * ( 0.4 + aSeed ), 170.0 ) - 85.0;
  float t = uTime * 0.06;
  p += vec3( sin( t + aSeed * 6.28 ) * 2.2, sin( t * 0.7 + aSeed * 11.0 ) * 1.4, cos( t * 0.8 + aSeed * 3.1 ) * 2.0 );
  vec4 mv = modelViewMatrix * vec4( p, 1.0 );
  float depth = -mv.z;
  gl_Position = projectionMatrix * mv;
  gl_PointSize = uSize * ( 0.6 + aSeed ) * uPixelRatio * ( 220.0 / max( depth, 1.0 ) );
  vAlpha = smoothstep( 8.0, 40.0, depth ) * ( 1.0 - smoothstep( 260.0, 420.0, depth ) );
  vGlow = uCore * exp( -length( p - uCenter ) / 38.0 );
}`

export const dustFrag = /* glsl */ `
uniform vec3 uColor;
uniform vec3 uGlowColor;
uniform float uOpacity;
varying float vAlpha;
varying float vGlow;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = length( c );
  float a = smoothstep( 0.5, 0.0, d );
  a *= a;
  vec3 col = uColor + uGlowColor * vGlow * 3.0;
  gl_FragColor = vec4( col * a * vAlpha * uOpacity, 1.0 );
}`
