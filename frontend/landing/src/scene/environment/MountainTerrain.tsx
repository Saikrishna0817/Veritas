/**
 * Mountain valley: a CPU-generated heightfield (ridged multifractal + domain warp, see terrainHeight.ts)
 * on a non-uniform grid (dense near the camera and the valley axis, sparse far away),
 * shaded with a slope/altitude rock–snow shader and fragment-level micro relief.
 */
import { useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { config } from '../../config'
import { store } from '../../state/store'
import { createTerrainHeight } from './terrainHeight'
import { createRockMaterial } from './rockMaterial'

function buildTerrainGeometry() {
  const T = config.terrain
  const height = createTerrainHeight(T.seed, T.floor)
  const nx = T.segmentsX + 1
  const nz = T.segmentsZ + 1
  const pos = new Float32Array(nx * nz * 3)
  for (let j = 0; j < nz; j++) {
    const tz = j / (nz - 1)
    const z = T.zNear - (T.zNear - T.zFar) * Math.pow(tz, 1.75)
    for (let i = 0; i < nx; i++) {
      const u = (i / (nx - 1)) * 2 - 1
      const x = Math.sign(u) * Math.pow(Math.abs(u), 1.45) * (T.width / 2)
      const k = (j * nx + i) * 3
      pos[k] = x
      pos[k + 1] = height(x, z)
      pos[k + 2] = z
    }
  }
  const idx = new Uint32Array((nx - 1) * (nz - 1) * 6)
  let o = 0
  for (let j = 0; j < nz - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = j * nx + i
      const b = a + 1
      const c = a + nx
      const d = c + 1
      // winding so the surface faces +Y (z decreases with j)
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
  g.setIndex(new THREE.BufferAttribute(idx, 1))
  g.computeVertexNormals()
  g.computeBoundingSphere()
  return g
}

export function MountainTerrain() {
  const geometry = useMemo(() => buildTerrainGeometry(), [])

  const { material, uniforms } = useMemo(() => createRockMaterial({ key: 'valley' }), [])

  useEffect(
    () => () => {
      geometry.dispose()
      material.dispose()
    },
    [geometry, material],
  )

  useFrame(() => {
    // the valley dissolves into darkness during the disassembly
    uniforms.uFade.value = store.w.environment
    uniforms.uRockTime.value = store.time
  })

  return <mesh geometry={geometry} material={material} frustumCulled={false} />
}
