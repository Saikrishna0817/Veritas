/**
 * Dynamic glyph atlas: every legend is rasterised once into an 8×8 grid on a canvas.
 * Keycaps pick their cell through a per-instance glyph id (see KeycapMaterial).
 * White glyph on black → sampled as a coverage mask, so ink colour stays a shader uniform.
 */
import * as THREE from 'three'
import { ATLAS_GRID, GLYPHS } from '../../choreography/glyphs'

export const ATLAS_FONT = '"Inter Variable", "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif'

export function createCharacterAtlas(size = 2048, weight = 500): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const ctx = canvas.getContext('2d')!
  const cell = size / ATLAS_GRID

  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.NoColorSpace // data mask, not colour
  tex.generateMipmaps = true
  tex.minFilter = THREE.LinearMipmapLinearFilter
  tex.magFilter = THREE.LinearFilter
  tex.anisotropy = 8
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping

  const draw = () => {
    ctx.clearRect(0, 0, size, size)
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, size, size)
    ctx.fillStyle = '#fff'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'alphabetic'
    ctx.font = `${weight} ${Math.round(cell * 0.6)}px ${ATLAS_FONT}`
    for (let i = 0; i < GLYPHS.length; i++) {
      const ch = GLYPHS[i]
      const col = i % ATLAS_GRID
      const row = Math.floor(i / ATLAS_GRID)
      const m = ctx.measureText(ch)
      // optical centring on the real ink box (cap height for letters, full box for symbols)
      const asc = m.actualBoundingBoxAscent
      const desc = m.actualBoundingBoxDescent
      const cx = (col + 0.5) * cell + (m.actualBoundingBoxLeft - m.actualBoundingBoxRight) / 2
      const cy = (row + 0.5) * cell + (asc - desc) / 2
      ctx.fillText(ch, cx, cy)
    }
    tex.needsUpdate = true
  }

  draw()
  // redraw once the web font is ready (first draw may use a fallback face)
  if (typeof document !== 'undefined' && 'fonts' in document) {
    document.fonts
      .load(`${weight} 64px "Inter Variable"`)
      .then(() => document.fonts.ready)
      .then(draw)
      .catch(() => undefined)
  }
  return tex
}
