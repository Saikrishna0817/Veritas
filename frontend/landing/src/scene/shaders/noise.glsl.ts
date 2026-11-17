/** Shared GLSL helpers (value noise / fbm / hash) — cheap and band-free enough for atmosphere. */
export const glslNoise = /* glsl */ `
float kcHash12( vec2 p ) {
  vec3 p3 = fract( vec3( p.xyx ) * 0.1031 );
  p3 += dot( p3, p3.yzx + 33.33 );
  return fract( ( p3.x + p3.y ) * p3.z );
}
float kcNoise( vec2 p ) {
  vec2 i = floor( p );
  vec2 f = fract( p );
  vec2 u = f * f * f * ( f * ( f * 6.0 - 15.0 ) + 10.0 );
  float a = kcHash12( i );
  float b = kcHash12( i + vec2( 1.0, 0.0 ) );
  float c = kcHash12( i + vec2( 0.0, 1.0 ) );
  float d = kcHash12( i + vec2( 1.0, 1.0 ) );
  return mix( mix( a, b, u.x ), mix( c, d, u.x ), u.y );
}
float kcFbm( vec2 p ) {
  float s = 0.0;
  float a = 0.5;
  mat2 r = mat2( 0.8, -0.6, 0.6, 0.8 );
  for ( int i = 0; i < 5; i++ ) {
    s += a * kcNoise( p );
    p = r * p * 2.03 + 11.7;
    a *= 0.5;
  }
  return s;
}
float kcFbm3( vec2 p ) {
  float s = 0.0;
  float a = 0.5;
  mat2 r = mat2( 0.8, -0.6, 0.6, 0.8 );
  for ( int i = 0; i < 3; i++ ) {
    s += a * kcNoise( p );
    p = r * p * 2.03 + 11.7;
    a *= 0.5;
  }
  return s;
}
`
