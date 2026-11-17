import { useEffect, useMemo } from 'react';
import { useThree } from '@react-three/fiber';
import { Environment, Lightformer, useGLTF } from '@react-three/drei';
import { KEY_MODEL_URL, prepareKeycap } from '../../shared/argilla/keys/keycapGeometry';
import { createCharacterAtlas } from '../../shared/argilla/keys/characterAtlas';
import { createKeycapMaterial } from '../../shared/argilla/keys/keycapMaterial';
import { hex, token } from './tokens';

/* eslint-disable react-refresh/only-export-components -- hooks, helpers and the light, shared by the keycap subjects */
/**
 * What every keycap subject shares (the sphere, a result's keys): the landing page's keycap model, its legends and
 * its plastic (frontend/shared/argilla/keys), and its warm light. Colours come from the tokens in argilla.css.
 */

export const KEY_SCALE = 2;     // keycap model (2 wide) → 4 key units wide (1 unit = half a keycap)
export const KEY_H = 1.187;     // keycap model height

/** Phones and prefers-reduced-motion get still frames. */
export const still = () =>
  window.matchMedia('(prefers-reduced-motion: reduce)').matches || window.matchMedia('(max-width: 720px), (pointer: coarse)').matches;

export const ease = (t) => 1 - Math.pow(1 - t, 3);

/** A fresh copy of the prepared keycap geometry; the subject adds its per-key attributes (aGlyph, aTint, aXray). */
export function useKeycapGeometry() {
  const gltf = useGLTF(KEY_MODEL_URL);
  return useMemo(() => prepareKeycap(gltf.scene), [gltf.scene]);
}

export function useKeycapMaterial() {
  const gl = useThree((s) => s.gl);
  const atlas = useMemo(() => createCharacterAtlas(1024, 500), []);
  const { material } = useMemo(
    () =>
      createKeycapMaterial(
        atlas,
        {
          color: hex('--glaze-cream'),
          ink: hex('--key-ink'),
          inkAlt: hex('--key-ink-alt'),
          roughness: 0.6,
          specularIntensity: 0.5,
          clearcoat: 0.1,
          clearcoatRoughness: 0.5,
          envMapIntensity: 0.9,
          inkRoughness: 0.5,
          glow: { colors: [hex('--glaze-cream'), hex('--glaze-cream'), hex('--glaze-cream')], legend: 0, side: 0, speed: 0, scale: 0.005, flicker: 0 }, // backlight off
          xrayBody: hex('--xray-body'),
          xrayGlow: hex('--xray-glow'),
        },
        'app',
        { wireColor: hex('--warm-light') },
      ),
    [atlas],
  );

  useEffect(() => {
    atlas.anisotropy = Math.min(8, gl.capabilities.getMaxAnisotropy());
  }, [atlas, gl]);
  useEffect(() => () => { material.dispose(); atlas.dispose(); }, [material, atlas]);
  return material;
}

/** The landing page's light on the keys (its warm look): a lavender sky, a warm key light from behind, a soft fill. */
export function KeyLight({ fill = 1.4 }) {
  return (
    <>
      <hemisphereLight args={[token('--light-sky'), token('--light-ground'), 0.95]} />
      <directionalLight color={token('--light-key')} intensity={3} position={[95, 46, -70]} />
      <directionalLight color={token('--light-fill')} intensity={fill} position={[-60, 30, 130]} />
      <Environment resolution={128} frames={1} environmentIntensity={0.8}>
        <Lightformer form="rect" intensity={1.6} color={token('--env-sky')} scale={[60, 30, 1]} position={[0, 30, -10]} rotation={[Math.PI / 2, 0, 0]} />
        <Lightformer form="circle" intensity={3.4} color={token('--env-breach')} scale={14} position={[0, 8, -40]} />
        <Lightformer form="ring" intensity={0.5} color={token('--env-horizon')} scale={80} position={[0, -6, 0]} rotation={[Math.PI / 2, 0, 0]} />
        <Lightformer form="rect" intensity={0.55} color={token('--env-front')} scale={[24, 10, 1]} position={[-18, 10, 40]} />
      </Environment>
    </>
  );
}
