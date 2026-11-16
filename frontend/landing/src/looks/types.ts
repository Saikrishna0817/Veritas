import type { Config } from '../config'
import type { Content } from '../content'

/** Objects become optional key by key; numbers, strings and arrays are replaced whole. */
export type DeepPartial<T> = T extends readonly unknown[] ? T : T extends object ? { [K in keyof T]?: DeepPartial<T[K]> } : T

/**
 * A look: one art direction of the same scene and choreography.
 * It never changes the motion: only what things are made of, what the world around them is, and the words.
 */
export interface Look {
  id: string
  /** Its number in the series and its name. */
  no: string
  name: string
  /** One line on the mood. */
  mood: string
  /** The interface copy. */
  content: Content
  /** CSS custom properties set on :root (styles/tokens.css → "look inputs" and the colours). */
  tokens: Record<string, string>
  /** Overrides of config.ts. */
  config: DeepPartial<Config>
}
