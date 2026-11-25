import { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { token } from '../scene/tokens';
import { useFrameCap } from '../../shared/argilla/frameCap';

/**
 * The batch's real row scores as a point cloud: one point per row, laid out in file order on a disc, standing higher
 * the higher its score. Flagged rows are terracotta, the rest cream. Nothing here is invented: at most MAX_POINTS
 * rows are drawn (every flagged row, the rest sampled evenly), and each point's height is its row score.
 * Still under prefers-reduced-motion; otherwise it turns slowly while on screen.
 */
const MAX_POINTS = 6000;
const GOLDEN = Math.PI * (3 - Math.sqrt(5));

function useCloud(scores, flagged) {
  return useMemo(() => {
    const flagSet = new Set(flagged);
    const n = scores.length;
    const rest = [];
    for (let i = 0; i < n; i++) if (!flagSet.has(i)) rest.push(i);
    const hot = [...flagSet].filter((i) => i < n).slice(0, MAX_POINTS);
    const room = MAX_POINTS - hot.length;
    const step = Math.max(1, rest.length / Math.max(room, 1));
    const cool = Array.from({ length: Math.min(room, rest.length) }, (_, k) => rest[Math.floor(k * step)]);
    let max = 1e-6;
    for (const s of scores) if (s > max) max = s;
    const build = (rows) => {
      const pos = new Float32Array(rows.length * 3);
      rows.forEach((row, k) => {
        const r = Math.sqrt((row + 0.5) / n) * 1.6;
        const a = row * GOLDEN;
        pos.set([Math.cos(a) * r, (Math.max(scores[row], 0) / max) * 1.6, Math.sin(a) * r], k * 3);
      });
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      return g;
    };
    return { hot: build(hot), cool: build(cool) };
  }, [scores, flagged]);
}

function Cloud({ scores, flagged, still, onScreen }) {
  const group = useRef();
  const { hot, cool } = useCloud(scores, flagged);
  const colors = useMemo(() => ({ hot: token('--terracotta'), cool: token('--cream'), rule: token('--rule') }), []);
  const { camera } = useThree();
  useEffect(() => { camera.position.set(0, 2.1, 4.2); camera.lookAt(0, 0.7, 0); }, [camera]);
  useFrame((_, dt) => { if (!still && onScreen && group.current) group.current.rotation.y += dt * 0.15; });
  return (
    <group ref={group}>
      <gridHelper args={[3.4, 12, colors.rule, colors.rule]} />
      <points geometry={cool}><pointsMaterial color={colors.cool} size={0.028} transparent opacity={0.6} depthWrite={false} /></points>
      <points geometry={hot}><pointsMaterial color={colors.hot} size={0.06} /></points>
    </group>
  );
}

// Turning: drawn about 60 times a second at most (FrameCap), and only while the cloud is on screen; off screen the
// last frame stays on the canvas and nothing is drawn (it used to redraw every display refresh, visible or not).
function Turning({ onScreen }) {
  useFrameCap(onScreen);
  return null;
}

export default function ScoreCloud({ scores, flagged }) {
  const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const [onScreen, setOnScreen] = useState(true);
  const box = useRef(null);
  useEffect(() => {
    const el = box.current;
    if (!el || typeof IntersectionObserver === 'undefined') return undefined;
    const io = new IntersectionObserver(([e]) => setOnScreen(e.isIntersecting));
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <div ref={box} style={{ height: '100%' }}>
      <Canvas dpr={[1, 2]} camera={{ fov: 40 }} frameloop={still ? 'demand' : 'never'} gl={{ alpha: true }}>
        {!still && <Turning onScreen={onScreen} />}
        <Cloud scores={scores} flagged={flagged} still={still} onScreen={onScreen} />
      </Canvas>
    </div>
  );
}
