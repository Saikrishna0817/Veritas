/**
 * ScrollController: one pinned, scrubbed GSAP timeline drives everything.
 *  - a proxy tween 0→1 feeds store.target (the 3D scene reads it every frame);
 *  - the DOM interface is animated on the same timeline, so UI and 3D can never drift apart;
 *  - fully reversible: scrolling back plays the same timeline backwards.
 */
import { useLayoutEffect, type RefObject } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { ScrollToPlugin } from 'gsap/ScrollToPlugin'
import { config } from '../config'
import { store } from '../state/store'
import { syncTitles } from '../ui/titles'
import { content } from '../content'
import { demo } from './demo'

gsap.registerPlugin(ScrollTrigger, ScrollToPlugin)

const PHASES = content.phases

export function useScrollTimeline(stage: RefObject<HTMLElement | null>) {
  useLayoutEffect(() => {
    const el = stage.current
    if (!el) return
    if ('scrollRestoration' in history) history.scrollRestoration = 'manual'
    window.scrollTo(0, 0)

    const ph = config.phases
    // sphere phases live in the first `storyEnd` of the scroll; the anagram follows
    const k = ph.storyEnd
    const bounds = [ph.activation[1] * k, ph.decoding[1] * k, ph.disassembly[1] * k, ph.reconstruction[1] * k, ph.material[0], 1.01]
    const num = el.querySelector<HTMLElement>('[data-ui="phase-num"]')
    const name = el.querySelector<HTMLElement>('[data-ui="phase-name"]')
    const fill = el.querySelector<HTMLElement>('[data-ui="phase-fill"]')
    let lastPhase = -1
    const s2From = 0.895 * ph.storyEnd
    const s2To = ph.material[0] - 0.02
    const magnets = [
      { from: s2From - 0.04, to: s2To, at: (s2From + s2To) / 2 }, // section 2 — MAKE IT REAL
      { from: ph.material[1] - 0.03, to: 1.0001, at: 1 }, // section 3 — MATERIAL
    ]

    // a still follows the wall clock even when frames are rare (a background tab), so its titles finish
    if (demo.still >= 0) gsap.ticker.lagSmoothing(0)

    const ctx = gsap.context(() => {
      const proxy = { p: 0 }
      // ?demo=1: no scrolling — the take drives the same timeline from the scene clock (scroll/demo.ts)
      const tl = gsap.timeline({
        defaults: { ease: 'none' },
        paused: demo.on,
        scrollTrigger: demo.on ? undefined : {
          trigger: el,
          start: 'top top',
          end: () => `+=${window.innerHeight * config.scroll.lengthScreens}`,
          pin: true,
          scrub: config.scroll.scrub,
          // magnets: let go inside a section and it settles at its resting point, so a single wheel flick
          // never skips the copy; outside the zones the scroll stays free
          snap: {
            snapTo: (v: number) => {
              for (const m of magnets) if (v > m.from && v < m.to) return m.at
              return v
            },
            duration: { min: config.scroll.magnetDuration[0], max: config.scroll.magnetDuration[1] },
            delay: 0.12,
            ease: 'power2.inOut',
            inertia: false,
          },
          anticipatePin: 1,
          invalidateOnRefresh: true,
        },
      })

      tl.to(
        proxy,
        {
          p: 1,
          duration: 1,
          onUpdate: () => {
            store.target = proxy.p
            syncTitles(proxy.p)
            if (fill) fill.style.transform = `scaleY(${proxy.p.toFixed(4)})`
            const i = Math.max(0, bounds.findIndex((b) => proxy.p < b))
            if (i !== lastPhase) {
              lastPhase = i
              if (num) num.textContent = PHASES[i][0]
              if (name) name.textContent = PHASES[i][1]
            }
          },
        },
        0,
      )

      // Hero UI
      tl.to('[data-ui="hint"]', { autoAlpha: 0, y: 8, duration: 0.04 }, 0.01)
      tl.to('[data-ui="hero"] [data-fade]', { autoAlpha: 0, y: -6, duration: 0.05, stagger: 0.008 }, 0.7 * k)
      // Section titles and body copy reveal themselves (Focus Reveal / Fade Up, see ui/titles.ts).
      // Section 3 extras — once MATERIAL has settled
      tl.fromTo('[data-ui="material"] [data-fade]', { autoAlpha: 0, y: 10 }, { autoAlpha: 1, y: 0, duration: 0.04, stagger: 0.009 }, ph.material[1] - 0.015)
      if (demo.on) demo.tl = tl
    }, el)

    // fonts / layout can shift the pin geometry once loaded
    const onLoad = () => ScrollTrigger.refresh()
    window.addEventListener('load', onLoad)
    document.fonts?.ready.then(onLoad).catch(() => undefined)

    return () => {
      window.removeEventListener('load', onLoad)
      demo.tl = null
      ctx.revert()
    }
  }, [stage])
}

/** Smooth return to the hero; the scrubbed timeline plays the whole transformation in reverse. */
export function restartExperience() {
  gsap.to(window, { scrollTo: { y: 0, autoKill: false }, duration: config.scroll.restartDuration, ease: 'power2.inOut' })
}
