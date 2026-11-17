/** The keycap material with this page's look (the material itself: shared/argilla/keys/keycapMaterial.ts). */
import type * as THREE from 'three'
import { config } from '../../config'
import { look } from '../../looks'
import { createKeycapMaterial as create, type KeycapMaterialOptions } from '../../../../shared/argilla/keys/keycapMaterial'

export type { KeycapMaterialOptions }

export function createKeycapMaterial(atlas: THREE.Texture, opts: KeycapMaterialOptions = {}) {
  return create(atlas, config.keycap, look.id, opts)
}
