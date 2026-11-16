/**
 * This project's look. `config.ts`, `content.ts` and `styles/tokens.css` hold the scene's defaults;
 * `veritas.ts` holds what this look changes: its copy, its interface tokens and its overrides of the config.
 */
import type { Look } from './types'
import { veritas } from './veritas'

export const look: Look = veritas
