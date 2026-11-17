/**
 * Canvas + renderer setup.
 *  - sRGB output, ACES filmic applied in the post chain (renderer itself is `flat`: NoToneMapping);
 *  - soft PCF shadows (keys only), DPR capped once at start (never switched at runtime).
 */
import { Suspense, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import * as THREE from 'three'
import { config } from '../config'
import { store } from '../state/store'
import { patchFogChunks } from './environment/fogChunks'
import { Scene } from './Scene'
import styles from '../App.module.css'

patchFogChunks()

export function Experience() {
  const [dpr] = useState<number>(() => Math.min(window.devicePixelRatio || 1, config.performance.dpr[1]))

  return (
    <Canvas
      className={styles.webgl}
      dpr={dpr}
      flat
      shadows="soft"
      gl={{ antialias: false, powerPreference: 'high-performance', stencil: false, alpha: false, depth: true }}
      camera={{
        fov: config.camera.fov,
        near: config.camera.near,
        far: config.camera.far,
        position: config.camera.hero.position as unknown as [number, number, number],
      }}
      onCreated={(state) => {
        const { gl } = state
        // dev-only handle for debugging from the console: window.__keycap.scene / .camera / .gl
        if (import.meta.env.DEV) (window as unknown as { __keycap: unknown }).__keycap = { ...state, get: state.get, store }
        gl.outputColorSpace = THREE.SRGBColorSpace
        gl.toneMappingExposure = config.lighting.exposure
        gl.setClearColor(new THREE.Color(config.atmosphere.voidColor), 1)
      }}
    >
      {/* No runtime quality switching: changing the pixel ratio mid-session re-creates the drawing buffer and
          the post targets, which shows up as a black frame (it used to fire right as the preloader left). */}
      <Suspense fallback={null}>
        <Scene />
      </Suspense>
    </Canvas>
  )
}
