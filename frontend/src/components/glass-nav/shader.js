/** A full-screen triangle. */
export const VS = `#version 300 es
in vec2 a;void main(){gl_Position=vec4(a,0.,1.);}`;

/**
 * The glass nav in one pass, in canvas CSS px (p, y down, origin top-left). The track (a rounded box) melted with the
 * bead (a rounded box, smin), drawn as clear glass on the light paper theme: a body that is lighter at the top, a cool
 * thick edge, the bead raised and lit from inside by the indigo light, a white rim lit from up-left over a darker
 * hairline below, the bright neck where bead and track fuse, a glint on the bead, the tap ripple, one light sweep,
 * and a soft shadow outside that turns indigo near the bead. Colours are linear RGB; the output is sRGB, premultiplied.
 * Squares are sq(x), not pow(x, 2.), which is undefined for x < 0 in GLSL.
 * Uniforms: uRes canvas device px; uDpr device px per CSS px; uT time (s, dither only); uTrack, uBead (cx, cy, hx, hy);
 * uMix (smin k, presence, rise, sweep); uTap (x, y, age s, < 0 none); uTop, uBase, uLight, uDeep, uRule, uInk colours.
 */
export const FS = `#version 300 es
precision highp float;
uniform vec2 uRes;uniform float uDpr,uT;uniform vec4 uTrack,uBead,uMix;uniform vec3 uTap;
uniform vec3 uTop,uBase,uLight,uDeep,uRule,uInk;
out vec4 o;
float sq(float x){return x*x;}
float h21(vec2 p){p=fract(p*vec2(123.34,456.21));p+=dot(p,p+45.32);return fract(p.x*p.y);}
float sdBox(vec2 p,vec2 b,float r){vec2 q=abs(p)-b+r;return length(max(q,0.))+min(max(q.x,q.y),0.)-r;}
float smin(float a,float b,float k){float h=clamp(.5+.5*(b-a)/k,0.,1.);return mix(b,a,h)-k*h*(1.-h);}
float bar(vec2 p){return sdBox(p-uTrack.xy,uTrack.zw,uTrack.w);}
float bead(vec2 p){vec2 h=uBead.zw*uMix.y;if(min(h.x,h.y)<.5)return 1e4;return sdBox(p-uBead.xy,h,min(h.x,h.y));}
float shape(vec2 p){return smin(bar(p),bead(p),uMix.x);}
vec3 enc(vec3 c){c=clamp(c,0.,1.);return mix(c*12.92,1.055*pow(c,vec3(1./2.4))-.055,step(.0031308,c));}
void main(){
 vec2 p=vec2(gl_FragCoord.x,uRes.y-gl_FragCoord.y)/uDpr;
 float d=shape(p);float e=1.;
 vec2 g=vec2(shape(p+vec2(e,0.))-d,shape(p+vec2(0.,e))-d)/e;
 float inM=smoothstep(.75,-.75,d);
 float b=bead(p);
 float on=uMix.y,rise=uMix.z;
 // shadow outside the glass, a little below it, turning indigo near the bead
 float ds=shape(p-vec2(0.,2.5));
 float near=exp(-max(b,0.)/12.)*on;
 float shA=(exp(-max(ds,0.)/7.)*.09+exp(-max(ds,0.)/2.)*.05)*(1.+.6*near)*(1.-inM);
 vec3 shc=mix(uInk,uDeep,.6*near);
 // the glass body: lighter at the top of the track, the bead raised and lighter still
 float ty=clamp((p.y-(uTrack.y-uTrack.w))/(2.*uTrack.w),0.,1.);
 vec3 glass=mix(uTop,uBase,smoothstep(.15,1.,ty));
 float inB=smoothstep(1.,-1.,b)*on;
 glass=mix(glass,uTop,.85*inB);
 // thick glass at the edge looks cool and a touch darker
 float thick=exp(min(d,0.)/4.);
 glass=mix(glass,uLight,.3*thick);glass*=1.-.03*thick;
 // the light inside the bead, sitting a little low, and the colour pooling at its lower inner edge
 vec2 q=(p-uBead.xy-vec2(0.,uBead.w*.2))/max(uBead.zw*on,vec2(1.));
 float glow=exp(-dot(q,q)*1.6)*on;
 glass=mix(glass,uLight,.85*glow*(.55+.45*rise));
 float pool=exp(min(b,0.)/3.)*inB*smoothstep(uBead.y-uBead.w*.2,uBead.y+uBead.w,p.y);
 glass=mix(glass,uDeep,.22*pool);
 // the neck where bead and track melt together catches the light
 float neck=clamp((min(bar(p),b)-d)/uMix.x*2.,0.,1.)*on;
 glass=mix(glass,vec3(1.),.45*neck);
 // rim: white where it faces the light (up-left), a darker hairline where it faces away
 float lit=clamp(dot(normalize(g+1e-4),normalize(vec2(-.35,-.94))),0.,1.);
 float rim=exp(-sq((d+1.1)/.8));
 glass=mix(glass,vec3(1.),.85*rim*pow(lit,1.2));
 float line=exp(-sq((d+.2)/.65));
 glass=mix(glass,uRule,.7*line*(1.-.8*lit));
 // a glint on the bead's upper left
 vec2 s=(p-(uBead.xy+uBead.zw*on*vec2(-.45,-.55)))/max(uBead.zw*on*vec2(.22,.16),vec2(1.));
 glass=mix(glass,vec3(1.),.7*exp(-dot(s,s))*on*rise);
 // tap ripple from the new tab
 if(uTap.z>=0.){float r=length(p-uTap.xy);glass=mix(glass,uDeep,.22*exp(-sq((r-6.-uTap.z*110.)/2.2))*max(0.,1.-uTap.z/.5));}
 // one soft light sweep across the track after landing
 float sx=uTrack.x-uTrack.z+2.*uTrack.z*uMix.w+(p.y-uTrack.y)*.7;
 glass=mix(glass,vec3(1.),.5*exp(-sq((p.x-sx)/22.))*sin(3.14159*uMix.w));
 glass+=(h21(p)-.5)*.006;
 float a=inM+shA;
 vec3 c=enc(glass)*inM+enc(shc)*shA;
 c+=(h21(p+fract(uT*7.))-.5)/255.*a;
 o=vec4(c,a);}`;
