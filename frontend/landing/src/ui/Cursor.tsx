/**
 * Custom cursor: a 1 px white ring with a centred dot.
 *  - Default 24 px ring / 8 px dot, Hover 32 px ring / 12 px dot; the change is a smooth size tween.
 *  - Hover = over a link/button (or anything marked data-cursor="hover") or over a reacting 3D key.
 *  - Follows the pointer 1:1 (no lag: this is the pointer itself), hidden on touch devices and when
 *    the pointer leaves the window. The native cursor is hidden only on fine pointers (global.css).
 */
import { useEffect, useRef } from 'react'
import { store } from '../state/store'
import styles from './Cursor.module.css'

const HOVER_SELECTOR = 'a, button, [role="button"], [data-cursor="hover"]'

export function Cursor() {
  const el = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const node = el.current
    if (!node || !window.matchMedia('(pointer: fine)').matches) return
    document.documentElement.classList.add('has-cursor')
    let x = -100
    let y = -100
    let domHover = false
    let shown = false
    let hover = false
    let raf = 0

    const onMove = (e: PointerEvent) => {
      if (e.pointerType !== 'mouse' && e.pointerType !== 'pen') return
      x = e.clientX
      y = e.clientY
      domHover = !!(e.target as Element | null)?.closest?.(HOVER_SELECTOR)
      if (!shown) {
        shown = true
        node.classList.add(styles.visible)
      }
    }
    const onLeave = () => {
      shown = false
      node.classList.remove(styles.visible)
    }
    const frame = () => {
      node.style.transform = `translate3d(${x}px, ${y}px, 0)`
      const h = domHover || store.pointerOverKey
      if (h !== hover) {
        hover = h
        node.classList.toggle(styles.hover, h)
      }
      raf = requestAnimationFrame(frame)
    }
    window.addEventListener('pointermove', onMove, { passive: true })
    document.documentElement.addEventListener('pointerleave', onLeave)
    raf = requestAnimationFrame(frame)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('pointermove', onMove)
      document.documentElement.removeEventListener('pointerleave', onLeave)
      document.documentElement.classList.remove('has-cursor')
    }
  }, [])

  return (
    <div className={styles.cursor} ref={el} aria-hidden>
      <span className={styles.ring} />
      <span className={styles.dot} />
    </div>
  )
}
