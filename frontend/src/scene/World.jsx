import { lazy, Suspense, useEffect, useMemo } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { EffectComposer, ToneMapping } from '@react-three/postprocessing';
import { BlendFunction, Effect, ToneMappingMode } from 'postprocessing';
import * as THREE from 'three';
import { skyFrag, skyVert } from '../../shared/argilla/glsl/sky.glsl';
import { dustFrag, dustVert } from '../../shared/argilla/glsl/dust.glsl';
import { gradeFrag } from '../../shared/argilla/glsl/grade.glsl';
import { mulberry32, token } from './tokens';
import { useStageValue } from './stage';

/** The page's own subject (the keycap sphere, a result's keys); its own chunk, loaded when needed. */
const KeySphere = lazy(() => import('./KeySphere'));
const ResultKeys = lazy(() => import('./ResultKeys'));

/**
 * The world behind the app: the landing page's final state — the void, the rose and amber curtains, the stars and
 * the dust — through its colour grade. Same shaders as the landing page (frontend/shared/argilla), colours from the
 * tokens in argilla.css. It renders on demand only: one frame on mount and on resize, nothing while the page is idle
 * or hidden. Resolution is capped at 1.5×. A page adds its own subject through its stage (scene/stage.js).
 */

/** The curtains are frozen at a calm moment of their own animation (the landing page's clock at ~40 s). */
const STILL_TIME = 40;

function Sky() {
  const { camera, gl, size } = useThree();
  const material = useMemo(() => {
    const star = token('--star');
    return new THREE.ShaderMaterial({
      vertexShader: skyVert,
      fragmentShader: skyFrag,
      side: THREE.BackSide,
      depthWrite: false,
      uniforms: {
        uHorizon: { value: token('--sky-horizon') },
        uZenith: { value: token('--sky-zenith') },
        uGlow: { value: token('--sky-glow') },
        uGlowDir: { value: new THREE.Vector3(0.34, 0.085, -1) },
        uGlowStrength: { value: 0.7 },
        uZenithAt: { value: 0.34 },
        uVoidTop: { value: token('--void-top') },
        uVoidBottom: { value: token('--void') },
        uFade: { value: 1 },                     // fully dissolved into the void, as at the end of the landing page
        uTime: { value: STILL_TIME },
        uStars: { value: 1 },
        uAurora: { value: 1 },
        uViewH: { value: 900 },
        uAuroraShade: { value: new THREE.Vector2(0.78, 1) },
        uAuroraTop: { value: 0.85 },
        uAuroraSpeed: { value: 1.2 },
        uAuroraShift: { value: 0 },
        uAuroraLow: { value: token('--aurora-low') },
        uAuroraHigh: { value: token('--aurora-high') },
        uAuroraEdge: { value: token('--aurora-edge') },
        uAuroraGain: { value: 0.5 },
        uStarColor: { value: new THREE.Vector3(star.r, star.g, star.b) },
        uSunSize: { value: 0 },
        uSunBands: { value: 0 },
        uSunIntensity: { value: 1 },
        uSunTop: { value: new THREE.Color(1, 1, 1) },
        uSunBottom: { value: new THREE.Color(1, 1, 1) },
      },
    });
  }, []);
  useEffect(() => () => material.dispose(), [material]);

  // The sky never changes between resizes (the camera stands still, the curtains are frozen), so it is drawn once per
  // size into a half-float texture, the same format the composer draws into, and shown as the scene background. A frame
  // that only moves a key, follows a scroll or dips a keystroke no longer re-runs the sky shader for every pixel; the
  // background copies the texture pixel for pixel, so the picture is the same.
  const sky = useMemo(() => {
    const skyScene = new THREE.Scene();
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(3000, 48, 24), material);
    mesh.frustumCulled = false;
    skyScene.add(mesh);
    const target = new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.HalfFloatType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      generateMipmaps: false,
      depthBuffer: false,
    });
    return { scene: skyScene, mesh, target };
  }, [material]);
  useEffect(() => () => { sky.mesh.geometry.dispose(); sky.target.dispose(); }, [sky]);

  // the curtains sit 150 screen px lower than the shader's default, at any viewport size (as on the landing page)
  const invalidate = useThree((s) => s.invalidate);
  const scene = useThree((s) => s.scene);
  const dpr = useThree((s) => s.viewport.dpr);
  useEffect(() => {
    const buffer = gl.getDrawingBufferSize(new THREE.Vector2());
    const u = material.uniforms;
    u.uViewH.value = buffer.y;
    u.uAuroraShift.value = (150 / Math.max(1, size.height)) * 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    sky.target.setSize(buffer.x, buffer.y);
    sky.mesh.position.copy(camera.position);
    const previous = gl.getRenderTarget();
    gl.setRenderTarget(sky.target);
    gl.render(sky.scene, camera);
    gl.setRenderTarget(previous);
    scene.background = sky.target.texture;
    invalidate();
  }, [material, sky, scene, gl, size, dpr, camera, invalidate]);
  useEffect(() => () => { scene.background = null; }, [scene]);

  return null;
}

function Dust() {
  const gl = useThree((s) => s.gl);
  const { geometry, material } = useMemo(() => {
    const n = 900;
    const rnd = mulberry32(31);
    const pos = new Float32Array(n * 3);
    const seed = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      pos[i * 3] = (rnd() - 0.5) * 320;
      pos[i * 3 + 1] = (rnd() - 0.5) * 170;
      pos[i * 3 + 2] = -110 + rnd() * 260;
      seed[i] = rnd();
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    const m = new THREE.ShaderMaterial({
      vertexShader: dustVert,
      fragmentShader: dustFrag,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: STILL_TIME },
        uSize: { value: 2 },
        uRise: { value: 0 },
        uPixelRatio: { value: gl.getPixelRatio() },
        uCenter: { value: new THREE.Vector3() },
        uCore: { value: 0 },
        uColor: { value: token('--dust') },
        uGlowColor: { value: token('--warm-core') },
        uOpacity: { value: 0.5 * 0.35 },          // the landing page's dust in the final void (thinner)
      },
    });
    return { geometry: g, material: m };
  }, [gl]);
  useEffect(() => () => { geometry.dispose(); material.dispose(); }, [geometry, material]);
  return <points geometry={geometry} material={material} frustumCulled={false} renderOrder={30} />;
}

class GradeEffect extends Effect {
  constructor() {
    super('ArgillaGrade', gradeFrag, {
      blendFunction: BlendFunction.NORMAL,
      uniforms: new Map([
        ['uGrain', new THREE.Uniform(0.07)],
        ['uVignette', new THREE.Uniform(0.45)],
        ['uTime', new THREE.Uniform(0)],
        ['uSaturation', new THREE.Uniform(1.02)],
        ['uShadowTint', new THREE.Uniform(new THREE.Vector3(0.01, 0.004, 0.006))],
        ['uHighlightTint', new THREE.Uniform(new THREE.Vector3(1.02, 1.0, 0.96))],
        ['uScanlines', new THREE.Uniform(0)],
        ['uReveal', new THREE.Uniform(1)],
        ['uCover', new THREE.Uniform(token('--bg'))],
      ]),
    });
  }
}

function Subject() {
  const kind = useStageValue('kind');
  if (kind === 'sphere') return <Suspense fallback={null}><KeySphere /></Suspense>;
  if (kind === 'keys') return <Suspense fallback={null}><ResultKeys /></Suspense>;
  return null;
}

function Post() {
  const grade = useMemo(() => new GradeEffect(), []);
  useEffect(() => () => grade.dispose(), [grade]);
  return (
    <EffectComposer multisampling={0} frameBufferType={THREE.HalfFloatType} enableNormalPass={false}>
      <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
      <primitive object={grade} />
    </EffectComposer>
  );
}

export default function World({ onLost }) {
  return (
    <Canvas
      className="ar-canvas"
      frameloop="demand"
      dpr={[1, 1.5]}
      flat
      gl={{ antialias: false, alpha: false, stencil: false, powerPreference: 'low-power' }}
      camera={{ fov: 32, near: 0.5, far: 5000, position: [0, 0, 160] }}
      onCreated={({ gl }) => {
        gl.setClearColor(token('--void'), 1);
        document.documentElement.classList.add('has-3d');
        gl.domElement.addEventListener('webglcontextlost', (e) => { e.preventDefault(); document.documentElement.classList.remove('has-3d'); onLost?.(); }, { once: true });
      }}
    >
      <Sky />
      <Dust />
      <Subject />
      <Post />
    </Canvas>
  );
}
