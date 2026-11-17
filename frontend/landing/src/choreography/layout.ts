/**
 * Structured layout of the keyboard sphere: horizontal rings of keycaps.
 *
 * Conventions (sphere-local space, origin = sphere centre; world = local × sphere.worldScale):
 *  - The keycap origin is the centre of its base; local +Y is the key's top (outward).
 *  - Local +Z on the cap points "south" along the meridian, so legends read upright from the front.
 *    (Matches the source GLTF after its node transform: the legend's image-bottom lies along local +Z.)
 *  - lon = 0 faces +Z (the camera).
 */
import { config } from '../config'
import { RANDOM_POOL, glyphIndex } from './glyphs'
import { type Q4, type V3, mulberry32, qfromBasis, qaxisAngle, qmul, q4, v3, vcross, vnorm, vset } from '../utils/math'

export interface RingDef {
  index: number
  lat: number
  count: number
  /** Spin direction (+1/-1) and individual speed multiplier. */
  dir: number
  spinMult: number
  /** Decoding: drift offset (sphere-local) and tilt axis/angle. */
  drift: V3
  tiltAxis: V3
  tilt: number
  /** Disassembly: vertical climb per unit of radial growth (signed), and start delay. */
  exitSlope: number
  exitDelay: number
}

export interface KeyDef {
  id: number
  ring: number
  index: number
  /** Rest direction (unit, sphere-local, before ring spin) and rest orientation. */
  dir: V3
  quat: Q4
  glyph: number
  /** Index in the phrase (0..n-1) or -1 for sphere-only keys. */
  letter: number
  seed: number
  phase: number
  /** 0 = stays, >0 = detaches during decoding (distance multiplier). */
  detach: number
  tumbleAxis: V3
  tumble: number
  /** Extra radial drift while leaving (0..1). */
  scatter: number
}

export interface LetterDef {
  key: number
  char: string
  anchor: V3
  anchorQuat: Q4
  slot: V3
  slotQuat: Q4
  /** Reconstruction start (progress). */
  start: number
  /** 0..1 order in which the key sinks into the centre. */
  holdOrder: number
  /** Depth lane used while flying into the phrase (alternates between neighbours). */
  lane: number
  lift: V3
  swayAxis: V3
  floatPhase: V3
  /** Per-axis wobble amplitude (radians). */
  wobble: V3
  /** Third section: slot in the anagram row, or null when the key is not used and falls away. */
  materialSlot: V3 | null
  materialStart: number
  /** Depth lane used while re-spelling. */
  materialLane: number
  /** Full turn direction (+1/-1) while re-spelling; fall direction for unused keys. */
  flip: number
  fall: V3
}

export interface Layout {
  rings: RingDef[]
  keys: KeyDef[]
  letters: LetterDef[]
  /** Key ids of sphere-only keys and of phrase keys (two instanced meshes of the same geometry). */
  sphereKeys: number[]
  phraseKeys: number[]
  /** Key half-width / height in sphere-local units. */
  keyHalf: number
  keyHeight: number
  outerRadius: number
}

/** Rest orientation of a key whose top points along n, legend upright (south = local +Z). */
export function restQuat(out: Q4, lat: number, lon: number, poleTwist = 0): Q4 {
  const Y = v3(Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon))
  let Z: V3
  if (Math.abs(Math.cos(lat)) < 1e-6) {
    Z = v3(Math.sin(poleTwist), 0, Math.cos(poleTwist))
    if (Y[1] < 0) Z = v3(Math.sin(poleTwist), 0, -Math.cos(poleTwist))
  } else {
    // north tangent: d/dlat of Y; south = -north
    Z = v3(Math.sin(lat) * Math.sin(lon), -Math.cos(lat), Math.sin(lat) * Math.cos(lon))
  }
  const X = vnorm(v3(), vcross(v3(), Y, Z))
  return qfromBasis(out, X, Y, Z)
}

/**
 * Final composition (from the art-direction screenshot), in keycap-pitch units, y up:
 *   M A K E   I T
 *      R E A L
 * Slightly irregular on purpose — it should read as placed by hand, not typeset.
 */
const PHRASE_LAYOUT: [number, number][] = [
  [-3.37, 0.67], [-2.27, 0.75], [-0.97, 0.77], [0.1, 0.65], [2.1, 0.79], [3.31, 0.67],
  [-1.82, -0.75], [-0.55, -0.71], [0.59, -0.75], [1.8, -0.78],
]
/** Keycap pitch of that layout in sphere-local units (a key is 4.0 wide). */
const LAYOUT_UNIT = 4.15
/** Hover anchors inside the hollow sphere are the same composition, compressed (paths then only expand).
 *  Horizontal compression is stronger (the hollow is round); neighbours are kept apart in depth. */
const ANCHOR_COMPRESSION: [number, number] = [0.66, 0.86]
const ANCHOR_DEPTH = 1.85
/** Row membership → alternating depth planes inside each row (neighbours are > a keycap depth apart). */
const ROW_INDEX = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]

const wob = (rnd: () => number) => {
  const [a, b] = config.choreography.floatRotationDeg
  return ((a + (b - a) * rnd()) * Math.PI) / 180
}

/** Sinking order of the phrase keys (interleaved, never left-to-right). */
const HOLD_ORDER = [3, 7, 0, 5, 9, 2, 6, 1, 8, 4]

export function buildLayout(): Layout {
  const S = config.sphere
  const C = config.choreography
  const rnd = mulberry32(S.seed * 9973 + 17)
  const R = S.radius
  const n = S.rings
  const keyHalf = S.keyScale
  const keyHeight = 1.19 * S.keyScale
  const mid = (n - 1) / 2

  // rings above the camera's eye height climb, rings below sink (camera height in sphere-local units)
  const camY = config.camera.hero.position[1] / S.worldScale

  /* ---------------- rings ---------------- */
  const rings: RingDef[] = []
  for (let i = 0; i < n; i++) {
    const lat = -Math.PI / 2 + (i * Math.PI) / (n - 1)
    let count = 1
    if (i !== 0 && i !== n - 1) {
      // circumference measured at the key's pole-side base edge, where neighbours are closest
      const inner = Math.abs(lat) + Math.atan(keyHalf / R)
      count = Math.max(3, Math.floor((2 * Math.PI * R * Math.cos(inner)) / S.pitch))
    }
    const restY = R * Math.sin(lat) + (i - mid) * C.ringSpread
    const up = restY > camY
    const slope = (up ? 1 : -1) * (C.exitSlopeBase + C.exitSlopeGain * Math.abs(Math.sin(lat)))
    const tiltAxis = vnorm(v3(), v3(rnd() - 0.5, 0, rnd() - 0.5))
    rings.push({
      index: i,
      lat,
      count,
      dir: i % 2 === 0 ? 1 : -1,
      spinMult: 0.7 + rnd() * 0.6,
      drift: v3((rnd() - 0.5) * 2, 0, (rnd() - 0.5) * 2),
      tiltAxis,
      tilt: rnd() * 0.6 + 0.4,
      exitSlope: slope,
      // outer rings leave first, the ones around the equator last → stacking order is preserved
      exitDelay: (mid - Math.abs(i - mid)) * C.exitStagger,
    })
  }

  /* ---------------- phrase key picks (from the rings around the equator) ---------------- */
  const phrase = config.phrase.text.replace(/\s+/g, '')
  const m = Math.round(mid)
  const pickRings = [m - 1, m + 1, m, m - 1, m + 1, m, m - 2, m + 2, m, m - 1]
  const picks = new Map<string, number>()
  // spread the picks of each ring evenly around it, so phrase keys never start side by side
  const perRing = new Map<number, number[]>()
  phrase.split('').forEach((_, li) => {
    const ri = pickRings[li % pickRings.length]
    perRing.set(ri, [...(perRing.get(ri) ?? []), li])
  })
  perRing.forEach((lis, ri) => {
    const r = rings[ri]
    const offset = (ri * 0.29) % 1
    lis.forEach((li, q) => {
      let j = Math.floor(((q + offset) / lis.length) * r.count) % r.count
      while (picks.has(`${r.index}:${j}`)) j = (j + 1) % r.count
      picks.set(`${r.index}:${j}`, li)
    })
  })

  /* ---------------- keys ---------------- */
  const keys: KeyDef[] = []
  for (const r of rings) {
    const phase = r.index % 2 === 0 ? 0 : Math.PI / r.count // brick pattern
    for (let j = 0; j < r.count; j++) {
      const lon = phase + (j * 2 * Math.PI) / r.count
      const isPole = r.count === 1
      const quat = restQuat(q4(), r.lat, lon, isPole ? Math.PI / 4 : 0)
      const dir = v3(Math.cos(r.lat) * Math.sin(lon), Math.sin(r.lat), Math.cos(r.lat) * Math.cos(lon))
      if (isPole) vset(dir, 0, Math.sign(r.lat), 0)
      const letter = picks.get(`${r.index}:${j}`) ?? -1
      const glyph =
        letter >= 0 ? glyphIndex(phrase[letter]) : glyphIndex(RANDOM_POOL[Math.floor(rnd() * RANDOM_POOL.length)])
      const detach = letter < 0 && rnd() < C.detachShare ? 0.45 + rnd() * 0.55 : 0
      keys.push({
        id: keys.length,
        ring: r.index,
        index: j,
        dir,
        quat,
        glyph,
        letter,
        seed: rnd(),
        phase: rnd() * Math.PI * 2,
        detach,
        tumbleAxis: vnorm(v3(), v3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5)),
        tumble: (0.35 + rnd() * 0.65) * C.exitTumble,
        scatter: rnd(),
      })
    }
  }

  /* ---------------- phrase targets ---------------- */
  const P = config.phrase
  const slotQuat = qfromBasis(q4(), [1, 0, 0], [0, 0, 1], [0, -1, 0]) // top → camera, legend upright
  const letters: LetterDef[] = []
  const letterKeys = keys.filter((k) => k.letter >= 0).sort((a, b) => a.letter - b.letter)
  letterKeys.forEach((k, li) => {
    const [lx, ly] = PHRASE_LAYOUT[li % PHRASE_LAYOUT.length]
    const parity = ROW_INDEX[li % ROW_INDEX.length] % 2 ? -1 : 1
    // neighbours in a row sit in alternating depth planes (more than a keycap apart)
    const anchor = v3(lx * LAYOUT_UNIT * ANCHOR_COMPRESSION[0], ly * LAYOUT_UNIT * ANCHOR_COMPRESSION[1], parity * ANCHOR_DEPTH)
    // anchors face the camera with only a slight, individual tilt
    const tilt = qaxisAngle(q4(), vnorm(v3(), v3(rnd() - 0.5, rnd() - 0.5, 0)), 0.03 + rnd() * 0.05)
    const anchorQuat = qmul(q4(), tilt, slotQuat)
    const slot = v3(
      P.center[0] + lx * LAYOUT_UNIT,
      P.center[1] + ly * LAYOUT_UNIT,
      P.center[2] + parity * P.depthJitter * (0.5 + 0.5 * rnd()),
    )
    const centerOut = Math.min(1, Math.hypot(lx / 3.4, ly / 0.8) / 1.41)
    // outer letters set off first so they clear the way for the inner ones (paths fan outwards)
    const order = 0.85 * (1 - centerOut) + 0.15 * rnd()
    letters.push({
      key: k.id,
      char: phrase[li],
      anchor,
      anchorQuat,
      slot,
      slotQuat: [slotQuat[0], slotQuat[1], slotQuat[2], slotQuat[3]],
      start: C.reconstructStart + order * C.reconstructStagger,
      holdOrder: HOLD_ORDER[li % HOLD_ORDER.length] / 9,
      // arc bends outward from the composition centre so neighbouring paths never cross
      lift: v3(Math.sign(lx || 1) * (0.6 + rnd() * 1.2), Math.sign(ly) * (0.6 + rnd() * 1.2), 0),
      lane: parity,
      swayAxis: vnorm(v3(), v3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5)),
      floatPhase: v3(rnd() * 6.28, rnd() * 6.28, rnd() * 6.28),
      wobble: v3(wob(rnd), wob(rnd), wob(rnd)),
      materialSlot: null,
      materialStart: 0,
      materialLane: 0,
      flip: 1,
      fall: v3(),
    })
  })

  /* ---------------- third section: anagram ---------------- */
  const word = config.phrase.anagram
  const used = new Set<number>()
  const wordSlots: (number | null)[] = letters.map(() => null)
  word.split('').forEach((ch, wi) => {
    // prefer the occurrence closest in reading order so paths stay short
    let best = -1
    letters.forEach((L, li) => {
      if (used.has(li) || L.char !== ch) return
      if (best < 0 || Math.abs(li - wi) < Math.abs(best - wi)) best = li
    })
    if (best < 0) throw new Error(`Anagram letter "${ch}" is not available in the phrase`)
    used.add(best)
    wordSlots[best] = wi
  })
  letters.forEach((L, li) => {
    const wi = wordSlots[li]
    if (wi === null) {
      L.materialSlot = null
      L.materialStart = 0
      L.fall = v3((L.slot[0] > 0 ? 1 : -1) * (4 + rnd() * 4), -70 - rnd() * 20, -26 - rnd() * 10)
      return
    }
    L.materialSlot = v3(P.center[0] + (wi - (word.length - 1) / 2) * P.pitch, P.center[1] + 0.8, P.center[2])
  })
  // choreograph the re-spelling so no two travelling keys ever share space:
  //  - short hops stay near the plane in alternating shallow lanes,
  //  - the long crossing (e.g. T sweeping across) flies in front of everything,
  //  - keys shifting the same way move as one synchronised "train" in a back lane.
  const travellers = letters.filter((L) => L.materialSlot)
  const dx = (L: LetterDef) => L.materialSlot![0] - L.slot[0]
  const longest = travellers.reduce((a, b) => (Math.abs(dx(b)) > Math.abs(dx(a)) ? b : a))
  let shallow = 0
  for (const L of travellers) {
    const d = dx(L)
    L.flip = d >= 0 ? 1 : -1
    if (L === longest) {
      L.materialLane = 10
      L.materialStart = 0.08
    } else if (Math.abs(d) > 2 * P.pitch * 0.75) {
      L.materialLane = -9
      L.materialStart = 0.2
    } else {
      // short hops turn too, in alternating directions and lanes (lanes are > a turning key apart)
      L.flip = shallow % 2 ? -1 : 1
      L.materialLane = shallow++ % 2 ? -2.7 : 2.7
      L.materialStart = 0.26 + rnd() * 0.1
    }
  }

  // a look's own re-spelling (config.phrase.respell, found by scripts/solve-respell.ts) replaces the rules
  letters.forEach((L, li) => {
    const r = P.respell[li]
    if (!r || !L.materialSlot) return
    L.materialLane = r[0]
    L.materialStart = r[1]
    L.flip = r[2]
  })

  return {
    rings,
    keys,
    letters,
    sphereKeys: keys.filter((k) => k.letter < 0).map((k) => k.id),
    phraseKeys: letterKeys.map((k) => k.id),
    keyHalf,
    keyHeight,
    outerRadius: R + keyHeight,
  }
}
