/**
 * Focus Reveal — ported 1:1 from Text Motion Studio (src/engine/effects/focus-reveal.ts),
 * exported settings: focus-reveal.json. A rack-focus pull: a plane of focus sweeps across the line,
 * characters behind it are sharp, characters ahead of it are blurred, slightly enlarged and dim.
 *
 * Differences from the studio export, on purpose:
 *  - typeface/size/leading come from the layout (the extended display face, 96px, the design's line positions);
 *    the per-character positions are re-measured from the real rendered font (and on resize / font load);
 *  - instead of a one-shot load trigger, each title plays when its section appears and reverses when it
 *    leaves (see ui/titles.ts), so scrolling back and forth always shows the right state.
 */
import { Fragment, useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import gsap from 'gsap'
import { config } from '../config'
import { registerTitle, type TitleGroup } from './titles'

interface Props {
  text: string
  group: TitleGroup
  /** Extra delay after the group is triggered (s) — e.g. the second line of a two-part title. */
  delay?: number
  className?: string
}

export function FocusReveal({ text, group, delay = 0, className }: Props) {
  const host = useRef<HTMLSpanElement>(null)
  const lines = useMemo(() => text.split('\n').map((l) => l.split(' ')), [text])

  // the animation state lives outside React: one proxy value drives every character
  const engine = useRef<{
    chars: HTMLElement[]
    pos: number[]
    state: { f: number }
    tween: gsap.core.Tween | null
    lastKey: string
    render: () => void
  } | null>(null)

  useLayoutEffect(() => {
    const el = host.current
    if (!el) return
    const A = config.titles.focusReveal
    const reduced = typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const chars = Array.from(el.querySelectorAll<HTMLElement>('.fr-char'))
    const e = {
      chars,
      pos: [] as number[],
      state: { f: 0 },
      tween: null as gsap.core.Tween | null,
      lastKey: '',
      render: () => {},
    }

    const measure = () => {
      const width = el.clientWidth || 1
      const reverse = A.direction === 'left'
      e.pos = chars.map((c) => {
        const x = (c.offsetLeft + c.offsetWidth / 2) / width
        return reverse ? 1 - x : x
      })
      e.lastKey = ''
      e.render()
    }

    e.render = () => {
      const em = parseFloat(getComputedStyle(el).fontSize) || 16
      const soft = 0.12 + 0.5 * A.intensity // depth of field, as a share of the width
      const maxBlur = Math.max(0.01, A.blur) * em
      // the focus plane travels from before the first character to past the last one
      const fx = -soft + e.state.f * (1 + 2 * soft)
      const key = `${fx.toFixed(4)}:${em}`
      if (key === e.lastKey) return
      e.lastKey = key
      chars.forEach((c, i) => {
        const ahead = Math.min(1, Math.max(0, (e.pos[i] - fx) / soft))
        const k = ahead * ahead * (3 - 2 * ahead)
        if (reduced) {
          c.style.opacity = String(1 - k)
          return
        }
        c.style.filter = k > 0.002 ? `blur(${(k * maxBlur).toFixed(2)}px)` : ''
        c.style.opacity = String(1 - k * (1 - A.opacity))
        c.style.transform = k > 0.002 ? `scale(${(1 + (A.scale - 1) * k).toFixed(4)})` : ''
      })
    }

    measure()
    e.tween = gsap.to(e.state, {
      f: 1,
      duration: reduced ? 0.2 : A.duration,
      delay: reduced ? 0 : A.delay + delay,
      ease: reduced ? 'none' : `${A.ease}.${A.easeType}`,
      paused: true,
      onUpdate: e.render,
    })
    engine.current = e

    const ro = new ResizeObserver(measure)
    ro.observe(el)
    document.fonts?.ready.then(measure).catch(() => undefined)

    const unregister = registerTitle(group, {
      play: () => e.tween?.play(),
      reverse: () => e.tween?.reverse(),
    })
    return () => {
      unregister()
      ro.disconnect()
      e.tween?.kill()
      engine.current = null
    }
  }, [lines, group, delay])

  useEffect(
    () => () => {
      engine.current?.tween?.kill()
    },
    [],
  )

  return (
    <span ref={host} className={`fr ${className ?? ''}`}>
      <span className="fr-sr">{text.replace(/\n/g, ' ')}</span>
      {lines.map((words, li) => (
        <span className="fr-line" key={li} aria-hidden>
          {words.map((w, wi) => (
            <Fragment key={wi}>
              <span className="fr-word">
                {Array.from(w).map((ch, ci) => (
                  <span className="fr-char" key={ci}>
                    {ch}
                  </span>
                ))}
              </span>
              {wi < words.length - 1 ? ' ' : null}
            </Fragment>
          ))}
        </span>
      ))}
    </span>
  )
}
