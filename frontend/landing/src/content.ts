import { look } from './looks'

/** Every line of interface copy. It lives in the look (src/looks/veritas.ts → content);
 *  the keycap phrase and its anagram live there too, in config.phrase. */
export interface Content {
  brand: string
  label: string
  /** The link into the app, top right. */
  signIn: string
  /** The six phases of the indicator: [number, name]. */
  phases: [string, string][]
  hero: { title: string; text: string; hint: string }
  final: { text: string; line1: string; line2: string }
  material: { text: string; title: string; restart: string }
  loading: string
}

export const content: Content = look.content
