import { useEffect, useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { setHot, stage, useStageState } from './stage';
import { token } from './tokens';
import { ease, KEY_SCALE, KeyLight, still, useKeycapGeometry, useKeycapMaterial } from './keycaps';

/**
 * A result's listed items as keys: one terracotta key per item, most suspicious first (left to right, then the next
 * row), at most 24. A key stands higher the higher its item is ranked: the rank only, never the size of the score,
 * so nothing here is a measurement of its own. The legend is the first letter of the item's top column (rows) or of
 * its class (files). The key under the pointer and its table row light up together.
 *
 * Clusters (Overview): a few keys per data type spelling its short name, glazed with the colour of that type's
 * latest verdict (the same colour as its chip); a type never checked leaves its slot empty.
 *
 * It draws while keys rise into place (once, about a second) and while a hovered key moves; otherwise no frames.
 */

const PITCH_X = 4.7;
const PITCH_Z = 5.4;
const PER_ROW = 8;
const TILT = 0.95;          // the plane of keys leans towards the viewer, like a keyboard on a desk seen from above
const RISE = 0.6;           // seconds per key
const STAGGER = 0.035;      // seconds between keys
const HOT_LIFT = 1.6;

// keys stand on the plane (+Y out of it), legends upright (−Z of the key points away from the viewer)
const QUAT = new THREE.Quaternion();

function Keys({ items }) {
  const base = useKeycapGeometry();
  const material = useKeycapMaterial();
  const { camera, size, invalidate } = useThree();
  const { hot } = useStageState();
  const n = items.length;
  const rows = Math.ceil(n / PER_ROW);
  // clusters (Overview): items carry group (a slot of a 2-column grid, `groups` slots in all) and their own glaze
  const clustered = items[0].group != null;
  const idOf = (i) => (clustered ? items[i].group : i);

  const geometry = useMemo(() => {
    const g = base.clone();
    const tint = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) tint.set(token(items[i].tint || '--terracotta').toArray(), i * 3);
    g.setAttribute('aGlyph', new THREE.InstancedBufferAttribute(Float32Array.from(items.map((it) => it.glyph)), 1));
    g.setAttribute('aTint', new THREE.InstancedBufferAttribute(tint, 3));
    // the hovered key takes the landing page's x-ray glaze (dark glass, warm inner light)
    const xray = new THREE.InstancedBufferAttribute(new Float32Array(n), 1);
    xray.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('aXray', xray);
    return g;
  }, [base, items, n]);
  useEffect(() => () => geometry.dispose(), [geometry]);

  const group = useRef();
  const mesh = useRef();
  const sim = useMemo(() => {
    let home;
    let wide;
    let deep;
    if (clustered) {
      const slots = items[0].groups || 4;
      const slotRows = Math.ceil(slots / 2);
      const per = Math.max(...items.map((it) => items.filter((o) => o.group === it.group).length));
      const slotW = per * PITCH_X + 7;
      home = items.map((it) => {
        const mates = items.filter((o) => o.group === it.group);
        const k = mates.indexOf(it);
        const gx = (it.group % 2) - 0.5;
        const gy = Math.floor(it.group / 2) - (slotRows - 1) / 2;
        return new THREE.Vector3(gx * slotW + (k - (mates.length - 1) / 2) * PITCH_X, 0, gy * PITCH_Z * 1.6);
      });
      wide = 2 * slotW;
      deep = slotRows * PITCH_Z * 1.6;
    } else {
      home = items.map((_, i) => {
        const r = Math.floor(i / PER_ROW);
        const c = i % PER_ROW;
        const inRow = Math.min(PER_ROW, n - r * PER_ROW);
        return new THREE.Vector3((c - (inRow - 1) / 2) * PITCH_X, 0, (r - (rows - 1) / 2) * PITCH_Z);
      });
      wide = Math.min(PER_ROW, n) * PITCH_X;
      deep = rows * PITCH_Z;
    }
    return {
      home,
      wide,
      deep,
      lift: new Float32Array(n),
      t: still() ? 1e3 : 0,
      m: new THREE.Matrix4(),
      p: new THREE.Vector3(),
      s: new THREE.Vector3().setScalar(KEY_SCALE),
      w: new THREE.Vector3(),
      hot: -1,
    };
  }, [items, n, rows, clustered]);

  // rows: the rank only (most suspicious highest); clusters: all level
  // an item may carry its own lift (a model file's imports: by what the scan says about each)
  const rankLift = (i) => (items[i].lift != null ? items[i].lift : clustered ? 0.5 : 0.4 + 2.6 * (1 - i / Math.max(1, n - 1)));

  useEffect(() => {
    sim.hot = hot;
    const x = geometry.getAttribute('aXray');
    x.array.fill(0);
    for (let i = 0; i < n; i++) if (hot >= 0 && idOf(i) === hot) x.array[i] = 0.55;
    x.needsUpdate = true;
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hot, sim, geometry, n, invalidate]);

  // the key under the pointer (only over the empty page, never through text, tables or controls)
  useEffect(() => {
    let mine = false;
    const on = (e) => {
      const g = group.current;
      const el = stage.get().el;
      if (!g || !el || !g.visible || e.pointerType === 'touch') return;
      if (e.target?.closest?.('a, button, input, select, textarea, label, summary, table, [role="button"]')) return;
      const unit = size.height / (2 * camera.position.z * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
      const reach = PITCH_X * 0.5 * g.scale.x * unit;
      let best = -1;
      let bestD = reach;
      for (let i = 0; i < n; i++) {
        sim.w.copy(sim.home[i]).setY(sim.lift[i] + 1).applyMatrix4(g.matrixWorld).project(camera);
        const d = Math.hypot((sim.w.x * 0.5 + 0.5) * size.width - e.clientX, (0.5 - sim.w.y * 0.5) * size.height - e.clientY);
        if (d < bestD) { best = i; bestD = d; }
      }
      if (best >= 0) { mine = true; setHot(idOf(best)); }
      else if (mine) { mine = false; setHot(-1); }
    };
    window.addEventListener('pointermove', on, { passive: true });
    return () => window.removeEventListener('pointermove', on);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera, size, n, sim]);

  useEffect(() => {
    const on = () => invalidate();
    window.addEventListener('scroll', on, { passive: true, capture: true });
    return () => window.removeEventListener('scroll', on, { capture: true });
  }, [invalidate]);

  useFrame((_, rawDt) => {
    const g = group.current;
    const el = stage.get().el;
    if (!g || !mesh.current || !el) return;
    const box = el.getBoundingClientRect();
    g.visible = box.width > 40 && box.height > 40; // the stage is hidden on phones
    if (!g.visible) return;

    const dt = Math.min(rawDt, 0.5); // real time: a slow device must not stretch the rise
    sim.t += dt;
    const unit = (2 * camera.position.z * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / Math.max(1, size.height);
    const wide = sim.wide;
    const deep = sim.deep * Math.cos(TILT) + 4;
    // fit the stage, but a key is never wider than ~72 px (a file with three imports gets three keys, not boulders)
    const scale = Math.min((box.width * 0.94 * unit) / wide, (box.height * 0.9 * unit) / deep, (72 * unit) / (KEY_SCALE * 2));
    g.scale.setScalar(scale);
    g.position.set((box.left + box.width / 2 - size.width / 2) * unit, -(box.top + box.height / 2 - size.height / 2) * unit, 0);
    g.rotation.set(TILT, 0, 0);

    let moving = false;
    const k = 1 - Math.exp(-dt * 12);
    for (let i = 0; i < n; i++) {
      const rise = ease(Math.min(1, Math.max(0, (sim.t - i * STAGGER) / RISE)));
      if (rise < 1) moving = true;
      const target = rise * (rankLift(i) + (sim.hot >= 0 && idOf(i) === sim.hot ? HOT_LIFT : 0));
      const next = still() ? target : sim.lift[i] + (target - sim.lift[i]) * k;
      if (Math.abs(target - next) > 1e-3) moving = true;
      sim.lift[i] = Math.abs(target - next) > 1e-3 ? next : target;
      sim.p.copy(sim.home[i]).setY(sim.lift[i]);
      sim.m.compose(sim.p, QUAT, sim.s);
      mesh.current.setMatrixAt(i, sim.m);
    }
    mesh.current.instanceMatrix.needsUpdate = true;
    if (moving) invalidate();
  });

  return (
    <group ref={group}>
      <instancedMesh ref={mesh} args={[geometry, material, n]} frustumCulled={false} />
    </group>
  );
}

export default function ResultKeys() {
  const { items } = useStageState();
  if (!items?.length) return null;
  return (
    <>
      <KeyLight fill={2.4} />
      <Keys key={items.length} items={items} />
    </>
  );
}
