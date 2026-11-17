/**
 * Shared by the landing page and the app (frontend/shared/argilla).
 * Prepares the keycap model (key-3d.gltf) and prepares it for instancing.
 *
 * Source analysis (Cinema 4D export):
 *  - one mesh "Key model", 5095 verts / 8836 tris (≈4418 quads), node rotated 180° about Y;
 *  - footprint 2.0 × 2.0, height 1.187, pivot at the base centre;
 *  - a real keycap: hollow skirt + cross stem inside (3220 tris never visible on the sphere);
 *  - the dished top cap is UV-mapped planar: u = 0.5 + 0.1632·x, v = 0.778 + 0.1622·z (after the node
 *    transform), i.e. legend image-up = −Z, image-right = +X.
 *
 * Every key (the 67 of the sphere and the 10 of the phrase) uses this model: with ~80 large keys the
 * stem, hollow skirt and exact smooth normals are worth their triangles.
 */
import * as THREE from 'three'
import keyModelUrl from '../models/key-3d.gltf?url'

/** The keycap model (self-contained glTF: buffers are data: URIs). */
export const KEY_MODEL_URL: string = keyModelUrl

/** Cap legend mapping — keep in sync with the shader in KeycapMaterial. */
export const CAP_UV_CENTER = new THREE.Vector2(0.5, 0.778)

export function prepareKeycap(root: THREE.Object3D): THREE.BufferGeometry {
  root.updateMatrixWorld(true)
  const found: THREE.Mesh[] = []
  root.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) found.push(o as THREE.Mesh)
  })
  if (!found.length) throw new Error('Keycap GLTF contains no mesh')
  const mesh = found[0]
  const g = mesh.geometry.clone()
  g.applyMatrix4(mesh.matrixWorld) // bake the node transform (180° about Y)

  g.computeBoundingBox()
  const bb = g.boundingBox!
  g.translate(-(bb.min.x + bb.max.x) / 2, -bb.min.y, -(bb.min.z + bb.max.z) / 2)

  // note: the GLTF stores POSITION/NORMAL/UV interleaved (byteStride 32) → always use getX/getY
  const pos = g.getAttribute('position')
  const nor = g.getAttribute('normal')
  const uv = g.getAttribute('uv')
  if (!uv) throw new Error('Keycap GLTF has no TEXCOORD_0')

  // cap mask: up-facing vertices of the dished top surface only (never the side walls)
  const cap = new Float32Array(pos.count)
  for (let i = 0; i < pos.count; i++) cap[i] = nor.getY(i) > 0.97 && pos.getY(i) > 1.09 ? 1 : 0
  g.setAttribute('aCap', new THREE.BufferAttribute(cap, 1))
  const keyUv = new Float32Array(uv.count * 2)
  for (let i = 0; i < uv.count; i++) {
    keyUv[i * 2] = uv.getX(i)
    keyUv[i * 2 + 1] = uv.getY(i)
  }
  g.setAttribute('aKeyUv', new THREE.BufferAttribute(keyUv, 2))

  g.computeBoundingSphere()
  return g
}
