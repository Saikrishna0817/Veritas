import { useEffect, useRef } from 'react';
import * as THREE from 'three';

export default function Tactile3DHero({ intensity = 1.0, className = '' }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);

    // Use a ref to keep track of the latest intensity for the animation loop
    const intensityRef = useRef(intensity);
    useEffect(() => {
      intensityRef.current = intensity;
    }, [intensity]);

  useEffect(() => {
    if (!mountRef.current) return;

    // Setup
    const w = mountRef.current.clientWidth;
    const h = mountRef.current.clientHeight;
    
    const scene = new THREE.Scene();
    scene.background = null; 
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
    camera.position.set(0, 0, 5);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    mountRef.current.appendChild(renderer.domElement);

    const geometry = new THREE.TorusKnotGeometry(1.2, 0.4, 256, 32);
    
    const material = new THREE.MeshStandardMaterial({
      color: 0x222222,
      roughness: 0.1,
      metalness: 1.0,
      envMapIntensity: 2.0,
    });

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const ambientLight = new THREE.AmbientLight(0xfff5e6, 0.5); 
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
    keyLight.position.set(5, 5, 5);
    scene.add(keyLight);

    const rimLight = new THREE.PointLight(0xe8622c, 8.0, 10);
    rimLight.position.set(-3, -2, -2);
    scene.add(rimLight);
    
    const rimLight2 = new THREE.PointLight(0xe8622c, 4.0, 10);
    rimLight2.position.set(3, 2, -2);
    scene.add(rimLight2);

    let frameId;
    const clock = new THREE.Clock();

    const animate = () => {
      frameId = requestAnimationFrame(animate);
      const time = clock.getElapsedTime();
      
      const currentInt = intensityRef.current;
      const speed = currentInt > 1.0 ? 1.5 : 0.5;

      mesh.rotation.y = time * 0.2 * speed;
      mesh.rotation.x = time * 0.1 * speed;
      
      rimLight.intensity = (8.0 + Math.sin(time * 3) * 2.0) * currentInt;
      rimLight2.intensity = 4.0 * currentInt;

      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return;
      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(frameId);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    };
  }, []); // Run once on mount

  // Update light intensity when analystMode changes
  useEffect(() => {
    if (sceneRef.current) {
      sceneRef.current.children.forEach(child => {
        if (child instanceof THREE.PointLight && child.color.getHex() === 0xe8622c) {
           // We dynamically update it in the animate loop using the outer scope `intensity` closure, 
           // but since the setup effect runs once, we'd need a ref to pass the latest intensity to the loop.
           // For simplicity, we can let it be handled or just use a ref.
        }
      });
    }
  }, [intensity]);

  return <div ref={mountRef} className={`w-full h-full min-h-[300px] ${className}`} />;
}
