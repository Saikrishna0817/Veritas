import { useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Environment, MeshDistortMaterial } from '@react-three/drei';
import * as THREE from 'three';

// ── Core Metallic Sphere ───────────────────────────────────────────────
function CoreSphere({ intensity }) {
  const meshRef = useRef();

  useFrame((state) => {
    if (!meshRef.current) return;
    const t = state.clock.getElapsedTime();
    const speed = intensity > 1.0 ? 1.2 : 0.35;
    meshRef.current.rotation.y = t * 0.15 * speed;
    meshRef.current.rotation.x = Math.sin(t * 0.08) * 0.2;
  });

  return (
    <mesh ref={meshRef} castShadow>
      <sphereGeometry args={[1.4, 128, 128]} />
      <MeshDistortMaterial
        color="#b0b0b8"
        metalness={1.0}
        roughness={0.15}
        envMapIntensity={2.0}
        distort={0.08}
        speed={1.5}
      />
    </mesh>
  );
}

// ── Thick Orbital Ribbon ───────────────────────────────────────────────
// Creates a wide, flat, metallic ribbon orbiting around the sphere
function OrbitalRibbon({ radius, width, tiltX, tiltZ, speed, color, metalness = 0.9, roughness = 0.18 }) {
  const meshRef = useRef();

  // Create a custom ribbon geometry: a torus with an elliptical cross-section (wide + thin)
  const geometry = useMemo(() => {
    const segments = 200;
    const shape = new THREE.Shape();

    // Flat ribbon cross-section: wide but thin
    const halfW = width / 2;
    const halfH = width * 0.12; // very thin compared to width
    shape.moveTo(-halfW, -halfH);
    shape.lineTo(halfW, -halfH);
    shape.quadraticCurveTo(halfW + halfH, 0, halfW, halfH);
    shape.lineTo(-halfW, halfH);
    shape.quadraticCurveTo(-halfW - halfH, 0, -halfW, -halfH);

    // Extrude along a circular path
    const path = new THREE.CurvePath();
    const curve = new THREE.EllipseCurve(0, 0, radius, radius, 0, Math.PI * 2, false, 0);
    path.add(curve);

    const extrudeSettings = {
      steps: segments,
      bevelEnabled: false,
      extrudePath: new THREE.CatmullRomCurve3(
        Array.from({ length: segments + 1 }, (_, i) => {
          const angle = (i / segments) * Math.PI * 2;
          return new THREE.Vector3(
            Math.cos(angle) * radius,
            0,
            Math.sin(angle) * radius
          );
        }),
        true
      ),
    };

    const geo = new THREE.ExtrudeGeometry(shape, extrudeSettings);
    geo.computeVertexNormals();
    return geo;
  }, [radius, width]);

  useFrame((state) => {
    if (!meshRef.current) return;
    const t = state.clock.getElapsedTime();
    meshRef.current.rotation.y = t * speed;
  });

  return (
    <mesh ref={meshRef} geometry={geometry} rotation={[tiltX, 0, tiltZ]} castShadow>
      <meshStandardMaterial
        color={color}
        metalness={metalness}
        roughness={roughness}
        envMapIntensity={1.8}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

// ── Simpler Torus Ribbon (fallback if ExtrudeGeometry has issues) ──────
function TorusRibbon({ radius, tube, tiltX, tiltZ, speed, color, emissiveColor, emissiveIntensity = 0.3 }) {
  const meshRef = useRef();

  useFrame((state) => {
    if (!meshRef.current) return;
    const t = state.clock.getElapsedTime();
    meshRef.current.rotation.y = t * speed;
  });

  return (
    <mesh ref={meshRef} rotation={[tiltX, 0, tiltZ]} castShadow>
      <torusGeometry args={[radius, tube, 32, 256]} />
      <meshStandardMaterial
        color={color}
        metalness={0.95}
        roughness={0.12}
        envMapIntensity={2.5}
        emissive={emissiveColor || color}
        emissiveIntensity={emissiveIntensity}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

// ── Particle field for depth ───────────────────────────────────────────
function ParticleField({ count = 200 }) {
  const points = useRef();
  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const xSeed = Math.sin(i * 12.9898 + 1.0) * 43758.5453;
      const ySeed = Math.sin(i * 78.233 + 2.0) * 43758.5453;
      const zSeed = Math.sin(i * 45.164 + 3.0) * 43758.5453;
      arr[i * 3] = ((xSeed - Math.floor(xSeed)) - 0.5) * 30;
      arr[i * 3 + 1] = ((ySeed - Math.floor(ySeed)) - 0.5) * 30;
      arr[i * 3 + 2] = ((zSeed - Math.floor(zSeed)) - 0.5) * 30;
    }
    return arr;
  }, [count]);

  useFrame((state) => {
    if (!points.current) return;
    points.current.rotation.y = state.clock.getElapsedTime() * 0.02;
    points.current.rotation.x = state.clock.getElapsedTime() * 0.01;
  });

  return (
    <points ref={points}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.04}
        color="#E4242B"
        transparent
        opacity={0.4}
        sizeAttenuation
      />
    </points>
  );
}

// ── Camera Parallax Rig ────────────────────────────────────────────────
function CameraRig() {
  const { camera } = useThree();
  const mouseRef = useRef({ x: 0, y: 0 });
  const vec = useMemo(() => new THREE.Vector3(), []);

  useFrame((state) => {
    // Use state.pointer instead of deprecated mouse
    const px = state.pointer.x || 0;
    const py = state.pointer.y || 0;
    mouseRef.current.x += (px - mouseRef.current.x) * 0.05;
    mouseRef.current.y += (py - mouseRef.current.y) * 0.05;
    camera.position.lerp(
      vec.set(mouseRef.current.x * 1.5, mouseRef.current.y * 1.0, 11),
      0.03
    );
    camera.lookAt(0, 0, 0);
  });

  return null;
}

// ── Main Component ─────────────────────────────────────────────────────
export default function Tactile3DHero({ intensity = 1.0, className = '' }) {
  return (
    <div className={`w-full h-full min-h-[300px] ${className}`} style={{ background: 'transparent' }}>
      <Canvas
        camera={{ position: [0, 0, 11], fov: 45 }}
        gl={{ antialias: true, alpha: true, premultipliedAlpha: false, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.2 }}
        dpr={[1, 1.5]}
        style={{ background: 'transparent' }}
        onCreated={({ gl, scene }) => {
          gl.setClearColor(0x000000, 0);
          scene.background = null;
        }}
      >
        {/* Environment map for reflections only — no visible background */}
        <Environment preset="city" background={false} />

        {/* Lighting — rich enough to make metallic materials shine */}
        <ambientLight intensity={0.15} />
        <directionalLight position={[5, 5, 5]} intensity={1.5} color="#ffffff" />
        <pointLight position={[-4, 2, 3]} color="#E4242B" intensity={40 * intensity} distance={20} decay={2} />
        <pointLight position={[4, -2, 3]} color="#FF3B3B" intensity={25 * intensity} distance={20} decay={2} />
        <pointLight position={[0, 4, -3]} color="#ffffff" intensity={15} distance={15} decay={2} />
        <spotLight position={[0, -5, 5]} angle={0.5} penumbra={1} intensity={20} color="#FF6030" distance={25} decay={2} />

        <CameraRig />
        <ParticleField count={150} />

        <group scale={1.15}>
          {/* Central metallic sphere */}
          <CoreSphere intensity={intensity} />

          {/* Ribbon 1: Gold/orange — largest, tilted horizontally */}
          <TorusRibbon
            radius={2.4}
            tube={0.09}
            tiltX={Math.PI * 0.1}
            tiltZ={Math.PI * 0.05}
            speed={0.18}
            color="#D4880F"
            emissiveColor="#FF8C00"
            emissiveIntensity={0.2}
          />

          {/* Ribbon 2: Silver/white — medium orbit, steep tilt */}
          <TorusRibbon
            radius={2.1}
            tube={0.07}
            tiltX={Math.PI * 0.42}
            tiltZ={Math.PI * -0.15}
            speed={-0.25}
            color="#C0C0C8"
            emissiveColor="#ffffff"
            emissiveIntensity={0.15}
          />

          {/* Ribbon 3: Red/crimson — inner orbit, opposite tilt */}
          <TorusRibbon
            radius={1.85}
            tube={0.06}
            tiltX={Math.PI * -0.3}
            tiltZ={Math.PI * 0.25}
            speed={0.3}
            color="#E4242B"
            emissiveColor="#FF3B3B"
            emissiveIntensity={0.4}
          />

          {/* Ribbon 4: Subtle thin outer silver ring */}
          <TorusRibbon
            radius={2.8}
            tube={0.025}
            tiltX={Math.PI * 0.55}
            tiltZ={Math.PI * 0.1}
            speed={-0.12}
            color="#888890"
            emissiveColor="#aaaaaa"
            emissiveIntensity={0.1}
          />
        </group>

        {/* NOTE: EffectComposer removed — it renders to its own opaque framebuffer
            which destroys the alpha channel, making the canvas background visible */}
      </Canvas>
    </div>
  );
}

