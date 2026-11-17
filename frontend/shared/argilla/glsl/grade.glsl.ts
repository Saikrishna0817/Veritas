/** The cinematic colour grade (landing page, PostFX.tsx): saturation, split toning, vignette, grain, scanlines and the
 *  preloader's opening circle. A postprocessing Effect fragment (mainImage). Shared with the app. */
export const gradeFrag = /* glsl */ `
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
