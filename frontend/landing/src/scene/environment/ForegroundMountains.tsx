/**
 * Foreground massifs, procedural (foregroundHeight.ts): one snowy peak with its erosion map, generated
 * once at load. One geometry is placed twice — rotated, mirrored and scaled differently — so the copies
 * read as two different mountains. The rim of the patch sinks smoothly below the valley floor (no hard
 * edges), and both copies use the shared rock material, palette, fog and dissolve.
 */
import { useEffect, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { config } from '../../config'
import { store } from '../../state/store'
import { createRockMaterial } from './rockMaterial'
import { FOREGROUND, buildForegroundDetail, sampleForeground } from './foregroundHeight'

function buildGeometry(heights: Float32Array) {
  const n = FOREGROUND.size
  const half = FOREGROUND.extent / 2
  const pos = new Float32Array(n * n * 3)
  const uv = new Float32Array(n * n * 2)
  for (let iz = 0; iz < n; iz++) {
    const tz = iz / (n - 1)
    for (let ix = 0; ix < n; ix++) {
      const tx = ix / (n - 1)
      const h = heights[iz * n + ix]
      // soft rim: the patch border sinks below the valley floor so no edge is ever visible
      const edge = Math.min(tx, 1 - tx, tz, 1 - tz)
      const mask = THREE.MathUtils.smoothstep(edge, 0.0, 0.14)
      const k = iz * n + ix
      const xRaw = -half + tx * FOREGROUND.extent
      const zRaw = -half + tz * FOREGROUND.extent
      pos[k * 3] = xRaw
      pos[k * 3 + 1] = h * mask - (1 - mask) * 60
      pos[k * 3 + 2] = -zRaw
      uv[k * 2] = tx
      uv[k * 2 + 1] = tz
    }
  }
  const idx = new Uint32Array((n - 1) * (n - 1) * 6)
  let o = 0
  for (let j = 0; j < n - 1; j++) {
    for (let i = 0; i < n - 1; i++) {
      const a = j * n + i
      const b = a + 1
      const c = a + n
      const d = c + 1
      idx[o++] = a
      idx[o++] = b
      idx[o++] = c
      idx[o++] = b
      idx[o++] = d
      idx[o++] = c
    }
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  g.setAttribute('aRockUv', new THREE.BufferAttribute(uv, 2))
  g.setIndex(new THREE.BufferAttribute(idx, 1))
  g.computeVertexNormals()
  g.computeBoundingSphere()
  return g
}

export function ForegroundMountains() {
  const gl = useThree((s) => s.gl)
  const heights = useMemo(() => sampleForeground(), [])
  const geometry = useMemo(() => buildGeometry(heights), [heights])

  const { material, uniforms, detail } = useMemo(() => {
    const S = FOREGROUND.detailSize
    const detail = new THREE.DataTexture(buildForegroundDetail(heights), S, S, THREE.RedFormat, THREE.UnsignedByteType)
    detail.colorSpace = THREE.NoColorSpace
    detail.flipY = false
    detail.wrapS = detail.wrapT = THREE.ClampToEdgeWrapping
    detail.magFilter = THREE.LinearFilter
    detail.minFilter = THREE.LinearMipmapLinearFilter
    detail.generateMipmaps = true
    detail.anisotropy = Math.min(8, gl.capabilities.getMaxAnisotropy())
    detail.needsUpdate = true
    return { detail, ...createRockMaterial({ key: 'foreground', detailMap: detail, detailStrength: 0.85, relief: 1.2, reliefScale: 0.14, snowLine: [-70, 10] }) }
  }, [heights, gl])

  useEffect(
    () => () => {
      geometry.dispose()
      material.dispose()
      detail.dispose()
    },
    [geometry, material, detail],
  )

  useFrame(() => {
    uniforms.uFade.value = store.w.environment
    uniforms.uRockTime.value = store.time
  })

  return (
    <group>
      {config.terrain.foreground.map((f, i) => (
        <mesh
          key={i}
          geometry={geometry}
          material={material}
          position={f.position}
          rotation={[0, f.rotationY, 0]}
          scale={f.scale}
          frustumCulled={false}
        />
      ))}
    </group>
  )
}
