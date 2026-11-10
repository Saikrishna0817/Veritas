import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * The batch as a specimen on a glass slide. Each point is one incoming row; a UV scan line sweeps
 * across it and rows behind the line rise by their row score, flagged rows fluorescing chartreuse.
 *
 * props:
 *  scores   number[]  per-row evidence (-log10 p); omitted → a seeded illustrative specimen
 *  flagged  number[]  indices of flagged rows
 *  cut      number    score at which a row counts as flagged (for the illustrative specimen)
 *  loop     boolean   keep sweeping (idle/hero) instead of a single pass
 */
const MAX_POINTS = 6000;

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function illustrative(n = 2400) {
  const rnd = mulberry32(7);
  const scores = [];
  const flagged = [];
  for (let i = 0; i < n; i++) {
    const u = rnd();
    // mostly clean rows near 0, a 5% poisoned tail with high evidence
    const poisoned = u < 0.05;
    const s = poisoned ? 2 + rnd() * 3.2 : Math.max(0, -Math.log10(1 - rnd() * 0.97) * 0.8);
    scores.push(s);
    if (s >= 2) flagged.push(i);
  }
  return { scores, flagged };
}

const vertex = /* glsl */ `
  attribute float aScore;
  attribute float aFlag;
  uniform float uTime;
  uniform float uSweep;
  uniform float uLift;
  uniform float uPixel;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec3 p = position;
    float passed = smoothstep(0.0, 0.35, uSweep - p.x);
    float band = exp(-pow((uSweep - p.x) * 7.0, 2.0));
    p.y += aScore * uLift * passed * mix(0.18, 1.0, aFlag);
    p.y += sin(uTime * 0.7 + p.x * 3.1 + p.z * 2.3) * 0.012;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float size = mix(5.5, 12.0, aFlag * passed) + band * 6.0;
    gl_PointSize = max(size * uPixel * (5.0 / -mv.z), 2.0);
    vec3 clean = vec3(0.73, 0.64, 1.0);
    vec3 bone = vec3(1.0, 1.0, 1.0);
    vec3 uv = vec3(1.0, 0.40, 0.28);
    vec3 base = mix(clean, bone, 0.25 + 0.3 * smoothstep(0.0, 1.5, aScore));
    vColor = mix(base, uv, aFlag * passed) + band * vec3(0.45, 0.3, 0.7);
    vAlpha = mix(0.8, 1.0, max(aFlag * passed, band));
  }
`;

const fragment = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = length(c);
    if (d > 0.5) discard;
    float core = smoothstep(0.5, 0.0, d);
    gl_FragColor = vec4(vColor * (0.75 + 0.6 * core), vAlpha * smoothstep(0.0, 0.6, core));
  }
`;

/** Pull the camera back on narrow canvases so the whole slide stays in frame. */
function FitCamera() {
  const { camera, size } = useThree();
  useEffect(() => {
    const aspect = size.width / Math.max(size.height, 1);
    const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
    const dist = Math.max(3.3, 2.15 / (Math.tan(halfFov) * aspect));
    camera.position.set(0, dist * 0.36, dist);
    camera.lookAt(0, 0.15, 0);
    camera.updateProjectionMatrix();
  }, [camera, size]);
  return null;
}

function Cloud({ scores, flagged, loop, reduced }) {
  const mat = useRef();
  const group = useRef();
  const start = useRef(null);

  const geometry = useMemo(() => {
    const n = scores.length;
    const flagSet = new Set(flagged);
    // keep every flagged row, sample the rest down to the budget
    let idx = [...Array(n).keys()];
    if (n > MAX_POINTS) {
      const rest = idx.filter((i) => !flagSet.has(i));
      const keep = MAX_POINTS - Math.min(flagSet.size, MAX_POINTS);
      const step = rest.length / Math.max(keep, 1);
      idx = [...flagSet].slice(0, MAX_POINTS).concat(Array.from({ length: keep }, (_, k) => rest[Math.floor(k * step)]));
    }
    const m = idx.length;
    const pos = new Float32Array(m * 3);
    const aScore = new Float32Array(m);
    const aFlag = new Float32Array(m);
    const max = Math.max(4, ...idx.map((i) => scores[i] || 0));
    const golden = Math.PI * (3 - Math.sqrt(5));
    idx.forEach((rowIdx, k) => {
      // rows laid out on an elliptical slide, file order spiralling outward
      const r = Math.sqrt((k + 0.5) / m);
      const a = k * golden;
      pos[k * 3] = Math.cos(a) * r * 1.65;
      pos[k * 3 + 1] = 0;
      pos[k * 3 + 2] = Math.sin(a) * r * 0.95;
      aScore[k] = Math.min(scores[rowIdx] || 0, max) / max * 4;
      aFlag[k] = flagSet.has(rowIdx) ? 1 : 0;
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aScore', new THREE.BufferAttribute(aScore, 1));
    g.setAttribute('aFlag', new THREE.BufferAttribute(aFlag, 1));
    return g;
  }, [scores, flagged]);

  const uniforms = useMemo(() => ({
    uTime: { value: 0 },
    uSweep: { value: reduced ? 3 : -2.2 },
    uLift: { value: 0.22 },
    uPixel: { value: Math.min(window.devicePixelRatio || 1, 2) },
  }), [reduced]);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (start.current === null) start.current = t;
    const u = mat.current?.uniforms;
    if (!u) return;
    u.uTime.value = t;
    if (!reduced) {
      const e = t - start.current;
      u.uSweep.value = loop ? -2.2 + ((e * 0.55) % 5.6) : Math.min(3, -2.2 + e * 1.1);
    }
    if (group.current) {
      const { x, y } = state.pointer;
      group.current.rotation.y = THREE.MathUtils.lerp(group.current.rotation.y, x * 0.35 + (reduced ? 0 : Math.sin(t * 0.15) * 0.15), 0.05);
      group.current.rotation.x = THREE.MathUtils.lerp(group.current.rotation.x, 0.42 - y * 0.12, 0.05);
    }
  });

  return (
    <group ref={group} rotation={[0.42, 0, 0]}>
      {/* the glass slide */}
      <mesh position={[0, -0.02, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[3.9, 2.4]} />
        <meshBasicMaterial color="#B9A3FF" transparent opacity={0.05} depthWrite={false} />
      </mesh>
      <lineSegments position={[0, -0.02, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <edgesGeometry args={[new THREE.PlaneGeometry(3.9, 2.4)]} />
        <lineBasicMaterial color="#B9A3FF" transparent opacity={0.45} />
      </lineSegments>
      <gridHelper args={[3.9, 26, '#3a2f52', '#221d2c']} position={[0, -0.025, 0]} scale={[1, 1, 0.615]} />
      <points geometry={geometry}>
        <shaderMaterial
          ref={mat}
          vertexShader={vertex}
          fragmentShader={fragment}
          uniforms={uniforms}
          transparent
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </points>
    </group>
  );
}

export default function Specimen3D({ scores, flagged, loop = false, className = '' }) {
  const data = useMemo(() => (scores?.length ? { scores, flagged: flagged || [] } : illustrative()), [scores, flagged]);
  const reduced = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  return (
    <div className={className} aria-hidden="true">
      <Canvas
        dpr={[1, 2]}
        camera={{ position: [0, 1.35, 3.6], fov: 42 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        frameloop={reduced ? 'demand' : 'always'}
      >
        <FitCamera />
        <Cloud scores={data.scores} flagged={data.flagged} loop={loop} reduced={reduced} />
      </Canvas>
    </div>
  );
}
