import { useEffect, useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { RANDOM_POOL, glyphIndex } from '../../shared/argilla/keys/glyphs';
import { stage, useStageValue } from './stage';
import { mulberry32, token } from './tokens';
import { ease, KEY_H, KEY_SCALE, KeyLight, still, useKeycapGeometry, useKeycapMaterial } from './keycaps';

/**
 * The keycap sphere of the landing page's opening, at rest, drawn inside the page's stage element (sign-in pages).
 * Same model, legends and plastic as the landing page (frontend/shared/argilla/keys), glazed in bands from the
 * tokens. It is decoration: it never shows a number or a state of the service.
 *
 * It draws a frame only while something moves: it turns in once when it appears (2.4 s), one key dips when the
 * person types, and it turns slowly while a request is running. Otherwise the page requests no frames.
 * Phones and prefers-reduced-motion get one still frame.
 */

const R = 13.2;          // sphere radius at the keycap base (key units: 1 = half a keycap)
const PITCH = 4.64;      // arc distance between key centres along a ring
const RINGS = 9;         // including the two single-key poles
const INTRO = 2.4;       // seconds

function buildKeys() {
  const rnd = mulberry32(7);
  const keys = [];
  for (let i = 0; i < RINGS; i++) {
    const polar = Math.PI * (1 - i / (RINGS - 1)); // ring 0 = south pole
    // keys counted along the ring's pole-side edge (half a key, 2 units, towards the nearer pole), so they never touch
    const edge = R * Math.sin(Math.max(0, Math.min(polar, Math.PI - polar) - 2 / R));
    const n = R * Math.sin(polar) < 0.5 ? 1 : Math.max(1, Math.floor((2 * Math.PI * edge) / PITCH));
    const off = rnd() * Math.PI * 2;
    for (let j = 0; j < n; j++) {
      const az = off + (j * 2 * Math.PI) / n;
      keys.push({
        ring: i,
        dir: new THREE.Vector3(Math.sin(polar) * Math.sin(az), Math.cos(polar), Math.sin(polar) * Math.cos(az)),
        az,
        glyph: glyphIndex(RANDOM_POOL[Math.floor(rnd() * RANDOM_POOL.length)]),
        seed: rnd(),
      });
    }
  }
  return keys;
}

/** The legends read upright: the key's −Z (legend up) points north along the sphere, its +Y out of the sphere. */
function keyBasis(dir, az) {
  const up = new THREE.Vector3(0, 1, 0);
  let north = up.clone().addScaledVector(dir, -dir.dot(up));
  if (north.lengthSq() < 1e-6) north = new THREE.Vector3(-Math.sin(az), 0, -Math.cos(az)); // the poles
  north.normalize();
  const y = dir.clone();
  const z = north.clone().negate();
  const x = new THREE.Vector3().crossVectors(y, z);
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
}

function Sphere() {
  const base = useKeycapGeometry();
  const { camera, size, invalidate } = useThree();
  const busy = useStageValue('busy');
  const presses = useStageValue('presses');

  const keys = useMemo(() => buildKeys(), []);
  const material = useKeycapMaterial();
  const geometry = useMemo(() => {
    const g = base.clone();
    const bands = ['--glaze-umber', '--glaze-umber', '--glaze-sand', '--glaze-cream', '--glaze-cream', '--glaze-cream', '--glaze-sage', '--glaze-green', '--glaze-green'].map((n) => token(n));
    const accent = token('--mustard');
    const tint = new Float32Array(keys.length * 3);
    keys.forEach((k, i) => tint.set((k.seed < 0.07 ? accent : bands[k.ring]).toArray(), i * 3));
    g.setAttribute('aGlyph', new THREE.InstancedBufferAttribute(Float32Array.from(keys.map((k) => k.glyph)), 1));
    g.setAttribute('aTint', new THREE.InstancedBufferAttribute(tint, 3));
    g.setAttribute('aXray', new THREE.InstancedBufferAttribute(new Float32Array(keys.length), 1));
    return g;
  }, [base, keys]);
  useEffect(() => () => geometry.dispose(), [geometry]);

  const group = useRef();
  const mesh = useRef();
  const sim = useMemo(
    () => ({
      quats: keys.map((k) => keyBasis(k.dir, k.az)),
      push: new Float32Array(keys.length),
      vel: new Float32Array(keys.length),
      intro: still() ? 1 : 0,
      spin: 0,
      m: new THREE.Matrix4(),
      p: new THREE.Vector3(),
      s: new THREE.Vector3().setScalar(KEY_SCALE),
      w: new THREE.Vector3(),
      rnd: mulberry32(19),
    }),
    [keys],
  );

  const writeKey = (i) => {
    sim.p.copy(keys[i].dir).multiplyScalar(R - sim.push[i]);
    sim.m.compose(sim.p, sim.quats[i], sim.s);
    mesh.current.setMatrixAt(i, sim.m);
  };

  useEffect(() => {
    if (!mesh.current) return;
    for (let i = 0; i < keys.length; i++) writeKey(i);
    mesh.current.instanceMatrix.needsUpdate = true;
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keys, invalidate]);

  // a keystroke: one key facing the viewer dips and springs back (random, never the character typed)
  useEffect(() => {
    if (!presses || still() || !group.current) return;
    const front = [];
    for (let i = 0; i < keys.length; i++) {
      sim.w.copy(keys[i].dir).applyQuaternion(group.current.quaternion);
      if (sim.w.z > 0.55) front.push(i);
    }
    if (!front.length) return;
    sim.vel[front[Math.floor(sim.rnd() * front.length)]] -= 9;
    invalidate();
  }, [presses, keys, sim, invalidate]);

  // follow the stage element on scroll and resize (one frame each)
  useEffect(() => {
    const on = () => invalidate();
    window.addEventListener('scroll', on, { passive: true, capture: true });
    window.addEventListener('resize', on);
    return () => {
      window.removeEventListener('scroll', on, { capture: true });
      window.removeEventListener('resize', on);
    };
  }, [invalidate]);
  useEffect(() => { invalidate(); }, [busy, invalidate]);

  useFrame((_, rawDt) => {
    const g = group.current;
    const el = stage.get().el;
    if (!g || !mesh.current || !el) return;
    const calm = still();
    let moving = false;

    // place the sphere in the stage's box (world units at the z = 0 plane)
    const box = el.getBoundingClientRect();
    const unit = (2 * camera.position.z * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / Math.max(1, size.height);
    const fit = (Math.min(box.width, box.height) / 2) * 0.86 * unit;
    g.position.set((box.left + box.width / 2 - size.width / 2) * unit, -(box.top + box.height / 2 - size.height / 2) * unit, 0);

    if (sim.intro < 1) {
      sim.intro = Math.min(1, sim.intro + Math.min(rawDt, 0.5) / INTRO); // real time: a slow device must not stretch it
      moving = true;
    }
    if (busy && !calm) {
      sim.spin += Math.min(rawDt, 0.5) * 0.35;
      moving = true;
    }
    const e = ease(sim.intro);
    g.scale.setScalar((fit / (R + KEY_H * KEY_SCALE)) * (0.94 + 0.06 * e));
    g.rotation.set(0.22, -0.9 * (1 - e) + sim.spin + 0.35, -0.08);

    // key springs
    const steps = Math.max(1, Math.round(Math.min(rawDt, 0.5) * 120));
    const h = 1 / 120;
    let touched = false;
    for (let i = 0; i < keys.length; i++) {
      if (sim.push[i] === 0 && sim.vel[i] === 0) continue;
      // fixed 1/120 s substeps of real time: stable, and as quick on a slow device as on a fast one
      for (let t = 0; t < steps; t++) {
        sim.vel[i] += (-90 * sim.push[i] - 11 * sim.vel[i]) * h;
        sim.push[i] += sim.vel[i] * h;
      }
      if (Math.abs(sim.push[i]) < 1e-3 && Math.abs(sim.vel[i]) < 1e-3) sim.push[i] = sim.vel[i] = 0;
      else moving = true;
      writeKey(i);
      touched = true;
    }
    if (touched) mesh.current.instanceMatrix.needsUpdate = true;
    if (moving) invalidate();
  });

  return (
    <group ref={group}>
      <instancedMesh ref={mesh} args={[geometry, material, keys.length]} frustumCulled={false} />
    </group>
  );
}

export default function KeySphere() {
  return (
    <>
      <KeyLight />
      <Sphere />
    </>
  );
}
