/**
 * Character set baked into the dynamic glyph atlas.
 * The index of a character in this string is the glyph id stored per instance.
 */
export const GLYPHS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789←→↑↓/?!@#&*+-=%$~;:.,<>'

/** Atlas grid (cells per side). 8×8 = 64 cells ≥ GLYPHS.length. */
export const ATLAS_GRID = 8

export function glyphIndex(ch: string): number {
  const i = GLYPHS.indexOf(ch.toUpperCase())
  if (i < 0) throw new Error(`Glyph "${ch}" is not in the atlas`)
  return i
}

/** Random legend pool for the sphere keys: mostly letters, some digits/symbols, like a real keyboard. */
export const RANDOM_POOL = 'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789←→↑↓/?!@#&*+-=%$~;:.,<>'
