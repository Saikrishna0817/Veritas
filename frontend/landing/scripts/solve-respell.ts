/**
 * Finds a collision-free re-spelling for the phrase → anagram (run: npm run solve:respell).
 *
 * layout.ts places the travelling keys by rule (the long crossing in front, the "train" behind, short hops
 * in shallow lanes). Those rules were written for MAKE IT REAL → MATERIAL; another pair of words can send two
 * keys through the same space. This script searches the three numbers each travelling key has —
 * its depth lane, the moment it sets off and the direction of its turn — until no two keys touch at any
 * moment (with a margin around every key), keeping the lanes as shallow as it can.
 * Paste the printed line into the look's config: `phrase: { respell: [...] }`.
 */
declare const process: { env: Record<string, string | undefined> }

import { buildLayout } from '../src/choreography/layout'
import { computePose, phaseWeights, type FrameInput } from '../src/choreography/choreography'
import { config } from '../src/config'
import { mulberry32, q4, v3 } from '../src/utils/math'
import { keyBoxes, keysOverlap } from './sat'

const L = buildLayout()
const N = L.keys.length
const letters = L.letters
const travellers = letters.map((l, i) => (l.materialSlot ? i : -1)).filter((i) => i >= 0)
const ringAngle = new Float64Array(L.rings.length)
const push = new Float32Array(N)
const pos = letters.map(() => v3())
const quat = letters.map(() => q4())
const w = phaseWeights(1)

/** Sum over the rebuild of the pairs of keys that touch (0 = clean). */
function overlaps(steps: number, t0: number, grow: number) {
  let n = 0
  for (let s = 0; s <= steps; s++) {
    const f: FrameInput = { p: 1, time: t0 + s / 60, idleYaw: 0, bob: 0, tiltX: 0, tiltZ: 0, ringAngle, push, material: s / steps }
    letters.forEach((lt, i) => computePose(L, lt.key, f, w, pos[i], quat[i]))
    const bx = pos.map((p, i) => keyBoxes(p, quat[i], grow))
    for (let i = 0; i < bx.length; i++) for (let j = i + 1; j < bx.length; j++) if (keysOverlap(bx[i], bx[j])) n++
  }
  return n
}
const depth = () => travellers.reduce((a, i) => a + Math.abs(letters[i].materialLane), 0)
// the keys float while they wait (each on its own phase), so the same rebuild is checked at several moments
const MOMENTS = [3, 41, 77, 131]
const MARGIN = 1.06
const cost = (steps = 220) => MOMENTS.reduce((a, t) => a + overlaps(steps, t, MARGIN) * 100, 0) + depth()

const LANES = Array.from({ length: 29 }, (_, i) => (i - 14) * 0.9) // −12.6 … 12.6
const STARTS = Array.from({ length: 20 }, (_, i) => 0.04 + i * 0.02) // 0.04 … 0.42

type State = [number, number, number][]
const read = (): State => travellers.map((i) => [letters[i].materialLane, letters[i].materialStart, letters[i].flip])
const write = (s: State) => travellers.forEach((i, k) => ([letters[i].materialLane, letters[i].materialStart, letters[i].flip] = s[k]))

let best: State = read()
let bestCost = cost()
console.log(`${config.phrase.text} → ${config.phrase.anagram}: the rules give cost ${bestCost.toFixed(1)} (overlaps ×100 + lane depth)`)

const tries = Number(process.env.TRIES || 8)
for (let run = 0; run < tries && bestCost >= 100; run++) {
  const rnd = mulberry32(101 + run * 7)
  let cur: State = best.map((r) => [...r] as [number, number, number])
  write(cur)
  let curCost = cost()
  const iters = Number(process.env.ITERS || 1800)
  for (let it = 0; it < iters; it++) {
    const temp = 60 * (1 - it / iters)
    const next: State = cur.map((r) => [...r] as [number, number, number])
    const k = Math.floor(rnd() * next.length)
    const what = rnd()
    if (what < 0.5) next[k][0] = LANES[Math.floor(rnd() * LANES.length)]
    else if (what < 0.85) next[k][1] = STARTS[Math.floor(rnd() * STARTS.length)]
    else next[k][2] = -next[k][2] || 1
    write(next)
    const c = cost()
    if (c < curCost || rnd() < Math.exp((curCost - c) / Math.max(temp, 1e-3))) {
      cur = next
      curCost = c
      if (c < bestCost) {
        best = next.map((r) => [...r] as [number, number, number])
        bestCost = c
      }
    }
  }
  console.log(`run ${run + 1}: best cost ${bestCost.toFixed(1)}`)
}

// second pass: with the collisions gone, pull the lanes towards the plane one key at a time
write(best)
if (bestCost < 100) {
  for (let pass = 0; pass < 3; pass++) {
    best.forEach((r) => {
      for (const lane of [...LANES].sort((a, b) => Math.abs(a) - Math.abs(b))) {
        if (Math.abs(lane) >= Math.abs(r[0])) break
        const keep = r[0]
        r[0] = lane
        write(best)
        if (cost() >= 100) r[0] = keep
        else break
      }
    })
  }
  write(best)
}

// the verdict, at a finer step, three moments of the float, and without the margin too
const CHECK = [7, 23, 37, 51, 69, 88, 104, 127, 143, 166]
const fine = CHECK.map((t) => overlaps(600, t, MARGIN))
const exact = CHECK.map((t) => overlaps(600, t, 1))
console.log(`overlapping pairs over the rebuild: with the margin ${fine.join(' / ')}, exact ${exact.join(' / ')}; lane depth ${depth().toFixed(1)}`)
const table: ([number, number, number] | null)[] = letters.map(() => null)
travellers.forEach((i, k) => (table[i] = [Number(best[k][0].toFixed(2)), Number(best[k][1].toFixed(2)), best[k][2]]))
console.log(letters.map((l, i) => `${l.char}${table[i] ? '' : '×'}`).join(' '))
console.log(`respell: ${JSON.stringify(table.map((r) => r ?? [0, 0, 0])).replace(/,/g, ', ')},`)
