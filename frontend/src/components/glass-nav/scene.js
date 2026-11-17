import { beadShape } from './geometry';
import { clamp } from './motion';
import { FS, VS } from './shader';

/**
 * The palette, linear RGB, from the ARGILLA tokens (styles/argilla.css): dark warm glass over the void, the bead lit
 * by the warm light. top = --accent-wash (#36231a), base = --sheet (#24160f), light = the warm light dimmed inside the
 * bead (#7a4520, so the cream label on it keeps 6.5:1), deep = --warm-core (#ffb060: the ripple and the light pooling
 * at the bead's edge), rule = --rule (#4a3428), ink = a near-black shadow (#050302). Change the tokens and these together.
 */
export const COLOURS = {
  uTop: [0.0369, 0.0168, 0.0103],
  uBase: [0.0176, 0.008, 0.0048],
  uLight: [0.1946, 0.0595, 0.0144],
  uDeep: [1, 0.4342, 0.117],
  uRule: [0.0685, 0.0343, 0.0212],
  uInk: [0.0015, 0.0009, 0.0006],
};
const UNIFORMS = ['uRes', 'uDpr', 'uT', 'uTrack', 'uBead', 'uMix', 'uTap', ...Object.keys(COLOURS)];

/**
 * Sets up WebGL2 on the canvas (throws if it isn't there) and returns layout(L) and draw(state, t). draw() also moves
 * the DOM icons and labels: on desktop the tab the bead sits on darkens; on mobile its icon rides up into the bead
 * and its label fades in from a blur where the icon was.
 */
export function createScene({ canvas, mode, icons, labels, tabs }) {
  const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: true, alpha: true });
  if (!gl) throw new Error('WebGL2 is not available');
  const DPR = Math.min(2, window.devicePixelRatio || 1);
  let L = null;

  const compile = (type, src) => {
    const x = gl.createShader(type);
    gl.shaderSource(x, src);
    gl.compileShader(x);
    if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x));
    return x;
  };
  const vs = compile(gl.VERTEX_SHADER, VS);
  const fs = compile(gl.FRAGMENT_SHADER, FS);
  const pr = gl.createProgram();
  gl.attachShader(pr, vs);
  gl.attachShader(pr, fs);
  gl.linkProgram(pr);
  if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(pr));
  gl.useProgram(pr);
  const vb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vb);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const al = gl.getAttribLocation(pr, 'a');
  gl.enableVertexAttribArray(al);
  gl.vertexAttribPointer(al, 2, gl.FLOAT, false, 0, 0);
  const U = Object.fromEntries(UNIFORMS.map((n) => [n, gl.getUniformLocation(pr, n)]));
  Object.entries(COLOURS).forEach(([n, v]) => gl.uniform3fv(U[n], v));

  // every stroke of every icon can be erased and redrawn: a dash as long as the whole path (pathLength 1)
  const strokes = icons.map((el) => [...el.querySelectorAll('[pathLength]')]);

  const layout = (next) => {
    L = next;
    canvas.width = Math.round(L.w * DPR);
    canvas.height = Math.round(L.h * DPR);
  };

  const draw = (s, t) => {
    if (!L) return;
    const b = beadShape(mode, s, L);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform2f(U.uRes, canvas.width, canvas.height);
    gl.uniform1f(U.uDpr, DPR);
    gl.uniform1f(U.uT, t);
    gl.uniform4f(U.uTrack, L.track.cx, L.track.cy, L.track.hx, L.track.hy);
    gl.uniform4f(U.uBead, b.x, b.y, b.hx, b.hy);
    gl.uniform4f(U.uMix, b.k, s.on, clamp(s.rise), s.sw);
    gl.uniform3f(U.uTap, s.tap[0], L.track.cy, s.tap[1]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    L.tabs.forEach((tab, i) => {
      // w: how close the bead is to this tab; a: how much this tab is "up"
      const w = Math.pow(clamp(1 - Math.abs(s.x - tab.x) / Math.max(48, tab.w * 0.75)), 1.5),
        a = w * clamp(s.rise) * s.on;
      tabs[i].style.setProperty('--gn-a', a.toFixed(3));
      const dr = i === s.cur ? s.draw : 1;
      strokes[i].forEach((q) => (q.style.strokeDashoffset = String(1 - dr)));
      if (mode === 'mobile') {
        icons[i].style.transform = `translateY(${(-34 * a).toFixed(2)}px) scale(${(1 + 0.1 * a).toFixed(3)})`;
        const la = Math.pow(clamp((a - 0.55) / 0.45), 1.5);
        labels[i].style.opacity = String(la);
        labels[i].style.transform = `translateY(${(6 * (1 - la)).toFixed(2)}px)`;
        labels[i].style.filter = `blur(${((1 - la) * 4).toFixed(2)}px)`;
      } else {
        icons[i].style.transform = '';
        labels[i].style.opacity = labels[i].style.transform = labels[i].style.filter = '';
      }
    });
  };

  const dispose = () => {
    gl.deleteBuffer(vb);
    gl.deleteProgram(pr);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
  };

  return { layout, draw, dispose };
}
