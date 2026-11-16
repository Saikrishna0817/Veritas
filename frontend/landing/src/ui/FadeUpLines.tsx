/**
 * Fade Up — ported from Text Motion Studio (engine/core.ts splitText + effects/fade-up.ts), exported settings
 * "h2_fade-up-prompt.md": the text is split into the lines the browser actually laid out, and each line rises
 * 0.6em into place while it fades in from blur(0.1em), 0.8 s, power3.out, 0.12 s apart.
 *
 * Adapted to this layout: our font/size/justified 176px column; played when its section appears and reversed when
 * it leaves (ui/titles.ts). Lines are re-measured on resize and when fonts load, keeping the current time.
 * Justification survives the split: every line but the last is justified to the column edges.
 */
import { useLayoutEffect, useRef } from 'react'
import gsap from 'gsap'
import { config } from '../config'
import { registerTitle, type TitleGroup } from './titles'

interface Props {
  text: string
  group: TitleGroup
  delay?: number
  className?: string
}

function el(tag: string, cls: string, text?: string) {
  const n = document.createElement(tag)
  n.className = cls
  if (text !== undefined) n.textContent = text
  return n
}

export function FadeUpLines({ text, group, delay = 0, className }: Props) {
  const host = useRef<HTMLParagraphElement>(null)

  useLayoutEffect(() => {
    const node = host.current
    if (!node) return
    const A = config.titles.fadeUp
    const reduced = typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let tl: gsap.core.Timeline | null = null
    let lastWidth = -1
    let wantForward = false

    const build = () => {
      const keep = tl ? tl.time() : 0
      tl?.kill()
      node.textContent = ''
      const sr = el('span', 'ul-sr', text)
      const visual = el('span', 'ul-visual')
      visual.setAttribute('aria-hidden', 'true')
      node.append(sr, visual)
      const words: HTMLElement[] = []
      text.split(/(\s+)/).forEach((tok) => {
        if (!tok) return
        if (/^\s+$/.test(tok)) {
          visual.appendChild(document.createTextNode(' '))
          return
        }
        const w = el('span', 'ul-word', tok)
        words.push(w)
        visual.appendChild(w)
      })
      // group words by the row the browser put them on
      const rows: HTMLElement[][] = []
      let lastTop = Number.NaN
      words.forEach((w) => {
        const top = w.offsetTop
        if (!rows.length || Math.abs(top - lastTop) > 1) {
          rows.push([w])
          lastTop = top
        } else rows[rows.length - 1].push(w)
      })
      visual.textContent = ''
      const inners: HTMLElement[] = []
      rows.forEach((row) => {
        const line = el('span', 'ul-line')
        const inner = el('span', 'ul-line-inner')
        row.forEach((w, i) => {
          if (i > 0) inner.appendChild(document.createTextNode(' '))
          inner.appendChild(w)
        })
        line.appendChild(inner)
        visual.appendChild(line)
        inners.push(inner)
      })
      lastWidth = node.clientWidth

      const em = parseFloat(getComputedStyle(node).fontSize) || 14
      tl = gsap.timeline({ paused: true })
      if (reduced) {
        inners.forEach((i) => (i.style.opacity = '1'))
        tl.fromTo(visual, { opacity: 0 }, { opacity: 1, duration: 0.2, ease: 'none' })
      } else {
        tl.fromTo(
          inners,
          { y: A.distance * em, opacity: A.opacity, filter: `blur(${(A.blur * em).toFixed(2)}px)` },
          { y: 0, opacity: 1, filter: 'blur(0px)', duration: A.duration, ease: A.ease, stagger: A.stagger },
          A.delay + delay,
        )
      }
      tl.time(Math.min(keep, tl.duration()))
      if (wantForward && tl.time() < tl.duration()) tl.play()
      else if (!wantForward && tl.time() > 0) tl.reverse()
      node.setAttribute('data-ul-ready', '')
    }

    build()
    const ro = new ResizeObserver(() => {
      if (Math.abs(node.clientWidth - lastWidth) > 0.5) build()
    })
    ro.observe(node)
    const fonts = document.fonts
    const onFonts = () => build()
    fonts?.addEventListener?.('loadingdone', onFonts)

    const unregister = registerTitle(group, {
      play: () => {
        wantForward = true
        tl?.play()
      },
      reverse: () => {
        wantForward = false
        tl?.reverse()
      },
    })
    return () => {
      unregister()
      ro.disconnect()
      fonts?.removeEventListener?.('loadingdone', onFonts)
      tl?.kill()
      node.removeAttribute('data-ul-ready')
      node.textContent = text
    }
  }, [text, group, delay])

  return (
    <p ref={host} className={`ul ${className ?? ''}`}>
      {text}
    </p>
  )
}
