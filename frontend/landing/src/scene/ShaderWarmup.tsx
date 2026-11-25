/**
 * Compiles every scene shader before the first frame, off the main thread where the driver allows it.
 *
 * Without this, three.js compiles each material the first time it is drawn and then waits for the result
 * (getProgramParameter), which froze the page for 1–3 s at a time while the preloader was animating. compileAsync
 * starts all compiles at once and polls KHR_parallel_shader_compile instead of blocking, so the preloader keeps moving.
 * The programs are the same ones the first frames would have built, so nothing on screen changes.
 *
 * Mounted inside the scene's Suspense boundary, after <Scene />: it runs once every model and texture has loaded.
 * The canvas stays on frameloop="never" until onReady is called (Experience.tsx).
 */
import { useEffect } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'

/**
 * Waits for the compile and decides when the scene may start drawing.
 * compile: resolves once every program is linked, or rejects if the driver failed one.
 * start: lets the scene draw its first frame; call it exactly once.
 */
export async function startWhenCompiled(compile: Promise<unknown>, start: () => void): Promise<void> {
  // Wait for the compile, but never longer than MAX_WAIT_MS: on a slow driver the preloader then moves on and the
  // remaining shaders compile on first draw, as they always did. A failed compile starts the scene too, so three.js
  // reports the shader error its usual way instead of the page staying blank.
  let timer = 0
  const limit = new Promise<void>((resolve) => (timer = window.setTimeout(resolve, MAX_WAIT_MS)))
  try {
    await Promise.race([compile, limit])
  } catch {
    // reported by three.js when the scene draws the failing material
  } finally {
    window.clearTimeout(timer)
    start()
  }
}

/** Longest the first frame waits for shaders (ms). The measured freezes added up to about 6 s, so this covers them. */
const MAX_WAIT_MS = 6000

export function ShaderWarmup({ onReady }: { onReady: () => void }) {
  const gl = useThree((s) => s.gl)
  const scene = useThree((s) => s.scene)
  const camera = useThree((s) => s.camera)

  useEffect(() => {
    let done = false
    const start = () => {
      if (done) return
      done = true
      onReady()
    }
    // The scene is never drawn to the screen: the post chain renders it into an offscreen target, and three.js
    // builds a different program variant for that (linear output, no tone mapping). Compile with a target bound,
    // or the warmed-up programs go unused and the real ones still compile on the first frame.
    const target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType })
    const previous = gl.getRenderTarget()
    gl.setRenderTarget(target)
    const compile = gl.compileAsync(scene, camera) // the programs are created synchronously here, with `target` bound
    gl.setRenderTarget(previous)
    void startWhenCompiled(compile, start).finally(() => target.dispose())
    return () => {
      done = true
    }
  }, [gl, scene, camera, onReady])

  return null
}
