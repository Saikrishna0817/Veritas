/**
 * Shared code (frontend/shared/argilla); used by the app's point cloud (src/components/ScoreCloud.jsx).
 * A frame cap for a react-three-fiber canvas with frameloop="never": it draws on every Nth display refresh, the largest
 * N that keeps at least 60 frames a second, so the pacing stays even. 60, 75 and 90 Hz displays draw every refresh
 * (as before); 120 and 144 Hz every second one (60 and 72 fps); 240 Hz every fourth (60 fps). Every animation advances by
 * the real time between frames, so nothing moves faster or slower, it is only drawn less often on fast displays.
 * Nothing runs while `active` is false or while the page is hidden (no animation frames are delivered then).
 */
import { useEffect } from 'react'
import { useThree } from '@react-three/fiber'

const SAMPLES = 24

/** Display refreshes per drawn frame for a refresh rate in Hz. */
export const refreshDivisor = (hz: number) => Math.max(1, Math.floor(hz / 60 + 0.08))

export function useFrameCap(active = true) {
  const advance = useThree((s) => s.advance)
  const clock = useThree((s) => s.clock)

  useEffect(() => {
    if (!active) return undefined
    // with frameloop="never" the timestamps passed to advance() are the clock: keep three's clock from restarting
    // itself inside each frame (it would reset elapsedTime and turn the next delta into the page's age)
    clock.autoStart = false
    clock.running = false
    let raf = 0
    let prev = 0
    let skip = 0
    let divisor = 1
    const gaps: number[] = []
    // the canvas clock continues from where it stands: the first frame's delta is 0 (as a started three.js clock's
    // first delta), not the page's age
    let offset = -1

    const tick = (t: number) => {
      raf = requestAnimationFrame(tick)
      if (prev) {
        const gap = t - prev
        // a gap over 100 ms is a stall or a hidden page, not the refresh rate
        if (gap < 100) {
          gaps.push(gap)
          if (gaps.length > SAMPLES) gaps.shift()
          if (gaps.length === SAMPLES) {
            const sorted = [...gaps].sort((a, b) => a - b)
            divisor = refreshDivisor(1000 / sorted[SAMPLES >> 1])
          }
        }
      }
      prev = t
      if (++skip < divisor) return
      skip = 0
      const now = t / 1000
      if (offset < 0) offset = now - clock.elapsedTime
      advance(now - offset)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [active, advance, clock])
}

/** The same as a component, for JSX scenes: <FrameCap /> inside the <Canvas frameloop="never">. */
export function FrameCap({ active = true }: { active?: boolean }) {
  useFrameCap(active)
  return null
}
