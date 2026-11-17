/**
 * Title choreography: each section's H1 plays its Focus Reveal when the section is on screen and
 * reverses when it leaves. Driven from the scroll timeline (syncTitles) plus the loader (setTitlesReady),
 * so the hero title waits until the scene is actually visible.
 */
import { config } from '../config'
import { store } from '../state/store'
import { demo } from '../scroll/demo'

export type TitleGroup = 'hero' | 'final' | 'material'
interface Ctl {
  play(): void
  reverse(): void
}

const groups: Record<TitleGroup, Set<Ctl>> = { hero: new Set(), final: new Set(), material: new Set() }
const shown: Record<TitleGroup, boolean> = { hero: false, final: false, material: false }
let ready = false
let last = 0

export function registerTitle(group: TitleGroup, ctl: Ctl) {
  groups[group].add(ctl)
  if (shown[group]) ctl.play()
  return () => {
    groups[group].delete(ctl)
  }
}

/** Visibility windows in raw scroll progress — match the UI fades in useScrollTimeline. */
function windows(p: number): Record<TitleGroup, boolean> {
  const ph = config.phases
  const k = ph.storyEnd
  return {
    hero: p < 0.7 * k,
    final: p >= 0.895 * k && p < ph.material[0] - 0.02,
    material: p >= ph.material[1] - 0.015,
  }
}

export function syncTitles(p: number) {
  last = p
  if (!ready) return
  const vis = windows(p)
  ;(Object.keys(groups) as TitleGroup[]).forEach((g) => {
    if (vis[g] === shown[g]) return
    shown[g] = vis[g]
    groups[g].forEach((c) => (vis[g] ? c.play() : c.reverse()))
  })
}

export function setTitlesReady() {
  if (ready) return
  ready = true
  if (demo.on) {
    demo.start = store.time
    ;(window as unknown as { __keycapDemo: typeof demo }).__keycapDemo = demo
  }
  // a still shows its own section only (the hero's title never plays on the way there)
  syncTitles(demo.still >= 0 ? demo.still : last)
}
