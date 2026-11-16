import { useEffect, useRef } from 'react'
import { Experience } from './scene/Experience'
import { Interface } from './ui/Interface'
import { useScrollTimeline } from './scroll/useScrollTimeline'
import { store } from './state/store'
import styles from './App.module.css'

export default function App() {
  const stage = useRef<HTMLDivElement>(null)
  useScrollTimeline(stage)

  useEffect(() => {
    const move = (e: PointerEvent) => {
      store.pointer.x = (e.clientX / window.innerWidth) * 2 - 1
      store.pointer.y = -(e.clientY / window.innerHeight) * 2 + 1
      store.pointer.active = true
    }
    const leave = () => (store.pointer.active = false)
    window.addEventListener('pointermove', move, { passive: true })
    document.documentElement.addEventListener('pointerleave', leave)
    window.addEventListener('blur', leave)
    return () => {
      window.removeEventListener('pointermove', move)
      document.documentElement.removeEventListener('pointerleave', leave)
      window.removeEventListener('blur', leave)
    }
  }, [])

  return (
    <main>
      <div className={styles.stage} ref={stage}>
        <Experience />
        <Interface />
      </div>
    </main>
  )
}
