import { useEffect, useRef } from 'react';
import * as THREE from 'three';

/**
 * Signature 3D Visual Element — "Soft Protection" Aesthetic
 *
 * Renders an abstract fluid 3D shield/orb object in Three.js with:
 * - Smooth auto-rotation + cursor parallax tilt (lerped mouse reaction)
 * - Soft-shaded physical material (Electric Blue / Cyan / Soft Violet lighting)
 * - Surrounding fine particle/dot-grid mesh
 * - Diffused ambient glow bleeding into the background
 */
export default function SoftProtection3D({ className = '', height = '320px', interactive = true }) {
  const containerRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || 400;
    const h = container.clientHeight || 320;

    // ── Scene, Camera, Renderer ──────────────────────────────────────────────
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / h, 0.1, 100);
    camera.position.z = 6;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(width, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // ── 3D Hero Shield/Orb Object ────────────────────────────────────────────
    // Complex organic geometry (Icosahedron with detail for smooth volume)
    const geometry = new THREE.IcosahedronGeometry(1.6, 64);
    
    // Deform vertices slightly for an organic, fluid volume
    const pos = geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const z = pos.getZ(i);
      const dist = Math.sqrt(x * x + y * y + z * z);
      const wave = Math.sin(x * 3 + y * 2) * 0.08 + Math.cos(z * 4) * 0.06;
      pos.setXYZ(i, (x / dist) * (1.6 + wave), (y / dist) * (1.6 + wave), (z / dist) * (1.6 + wave));
    }
    geometry.computeVertexNormals();

    // Material with soft-shaded metallic depth
    const material = new THREE.MeshStandardMaterial({
      color: 0x1a243b,
      metalness: 0.4,
      roughness: 0.25,
      wireframe: false,
      flatShading: false,
    });

    const heroMesh = new THREE.Mesh(geometry, material);
    scene.add(heroMesh);

    // Outer subtle wireframe shell for technical precision feel
    const wireGeo = new THREE.IcosahedronGeometry(1.75, 8);
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x3d7fff,
      wireframe: true,
      transparent: true,
      opacity: 0.12,
    });
    const wireMesh = new THREE.Mesh(wireGeo, wireMat);
    scene.add(wireMesh);

    // ── Fine Background Particle Mesh ─────────────────────────────────────────
    const particleCount = 180;
    const particleGeo = new THREE.BufferGeometry();
    const particlePos = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i += 3) {
      particlePos[i] = (Math.random() - 0.5) * 12;
      particlePos[i + 1] = (Math.random() - 0.5) * 12;
      particlePos[i + 2] = (Math.random() - 0.5) * 8 - 2;
    }
    particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePos, 3));

    const particleMat = new THREE.PointsMaterial({
      color: 0x4de8ff,
      size: 0.035,
      transparent: true,
      opacity: 0.35,
    });
    const particles = new THREE.Points(particleGeo, particleMat);
    scene.add(particles);

    // ── Soft Ambient & Key Lights ─────────────────────────────────────────────
    const ambientLight = new THREE.AmbientLight(0x0d111a, 2.5);
    scene.add(ambientLight);

    // Electric Blue Primary Key Light
    const keyLight = new THREE.DirectionalLight(0x3d7fff, 3.5);
    keyLight.position.set(4, 4, 5);
    scene.add(keyLight);

    // Cyan Secondary Rim Light
    const cyanLight = new THREE.DirectionalLight(0x4de8ff, 2.8);
    cyanLight.position.set(-4, -2, 3);
    scene.add(cyanLight);

    // Soft Violet Back Glow Light
    const violetLight = new THREE.PointLight(0x7c6cff, 4, 10);
    violetLight.position.set(0, 0, -2);
    scene.add(violetLight);

    // ── Cursor Mouse Parallax ────────────────────────────────────────────────
    let mouseX = 0;
    let mouseY = 0;
    let targetX = 0;
    let targetY = 0;

    const handleMouseMove = (e) => {
      if (!interactive) return;
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left - rect.width / 2;
      const y = e.clientY - rect.top - rect.height / 2;
      mouseX = (x / (rect.width / 2)) * 0.25; // max ~10° rotation
      mouseY = (y / (rect.height / 2)) * 0.25;
    };

    window.addEventListener('mousemove', handleMouseMove);

    // ── Animation Loop ───────────────────────────────────────────────────────
    let animationId;
    let clock = new THREE.Clock();

    const animate = () => {
      animationId = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Smooth auto-rotation
      heroMesh.rotation.y = elapsedTime * 0.15;
      heroMesh.rotation.x = Math.sin(elapsedTime * 0.1) * 0.08;
      wireMesh.rotation.y = -elapsedTime * 0.1;

      // Parallax mouse tilt lerp
      targetX += (mouseX - targetX) * 0.05;
      targetY += (mouseY - targetY) * 0.05;
      scene.rotation.y = targetX;
      scene.rotation.x = -targetY;

      // Subtle particle float
      particles.rotation.y = elapsedTime * 0.03;

      renderer.render(scene, camera);
    };

    animate();

    // ── Resize Listener ──────────────────────────────────────────────────────
    const handleResize = () => {
      if (!container) return;
      const newW = container.clientWidth;
      const newH = container.clientHeight;
      camera.aspect = newW / newH;
      camera.updateProjectionMatrix();
      renderer.setSize(newW, newH);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      geometry.dispose();
      material.dispose();
      wireGeo.dispose();
      wireMat.dispose();
      particleGeo.dispose();
      particleMat.dispose();
      renderer.dispose();
    };
  }, [interactive]);

  return (
    <div className={`relative flex items-center justify-center ${className}`} style={{ height }}>
      {/* Soft Ambient Glow Bleeding into Background */}
      <div
        className="absolute inset-0 pointer-events-none rounded-full blur-[80px] opacity-30"
        style={{
          background: 'radial-gradient(circle at 50% 50%, #3D7FFF 0%, #4DE8FF 40%, transparent 70%)',
        }}
      />
      {/* Three.js Canvas Container */}
      <div ref={containerRef} className="w-full h-full relative z-10 cursor-grab active:cursor-grabbing" />
    </div>
  );
}
