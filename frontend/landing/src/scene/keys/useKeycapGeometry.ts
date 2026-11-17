/** Loads the keycap model for instancing (the model and its preparation: shared/argilla/keys/keycapGeometry.ts). */
import { useMemo } from 'react'
import { useGLTF } from '@react-three/drei'
import { KEY_MODEL_URL, prepareKeycap } from '../../../../shared/argilla/keys/keycapGeometry'

export { CAP_UV_CENTER, KEY_MODEL_URL } from '../../../../shared/argilla/keys/keycapGeometry'

export function useKeycapGeometries() {
  const full = useGLTF(KEY_MODEL_URL)
  return useMemo(() => {
    const f = prepareKeycap(full.scene)
    return { full: f }
  }, [full.scene])
}

useGLTF.preload(KEY_MODEL_URL)
