import type * as THREE from 'three'

/** The cursor light's smoke trail, shared between CursorLight (writes) and the post chain (reads). */
export const trailState: { texture: THREE.Texture | null; presence: number } = { texture: null, presence: 0 }
