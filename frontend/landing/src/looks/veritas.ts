import type { Look } from './types'

/**
 * Veritas — the public landing page, in the ARGILLA look: matte ceramic keys glazed in bands (the ten phrase keys
 * are the terracotta ones), red sandstone with pale sand on the low ground, a low sun to the right, warm key light
 * against a violet fill. Light, unhurried type in sentence case.
 * The sphere is the dataset; TRUE OR LIES → OUTLIERS is the question Veritas answers and what it hands back.
 */
export const veritas: Look = {
  id: 'veritas',
  no: '001',
  name: 'Veritas',
  mood: 'Desert evening · glazed ceramic · calm and exact',
  content: {
    brand: 'VERITAS',
    label: 'TRAINING DATA, CHECKED — 2026',
    signIn: 'Sign in ↗',
    phases: [
      ['01', '— Collected'],
      ['02', '— Measured'],
      ['03', '— Separated'],
      ['04', '— Ranked'],
      ['05', '— True or lies'],
      ['06', '— Outliers'],
    ],
    hero: {
      title: 'Check the data\nbefore you train.',
      text: 'Seventy-seven keys, like the rows of a dataset. A few of them do not belong. Veritas shows you where to look.',
      hint: 'Scroll to check ↓',
    },
    final: {
      text: 'Ten keys leave the set and line up to be read. Each one comes with the check that flagged it, and how strongly.',
      line1: 'Evidence before a verdict.',
      line2: 'A verdict is not proof.',
    },
    material: {
      text: 'The same keys, read again. Two fall away: a result should say plainly what it cannot tell you.',
      title: 'Point at what to fix.',
      restart: 'Back to the start ↗',
    },
    loading: 'Checking',
  },
  tokens: {
    '--bg': '#1b0f0a',
    '--ink': '#f6ead8',
    '--white': '#fff6ea',
    '--shade': 'rgba(58, 20, 10, 0.55)',
    '--shade-0': 'rgba(58, 20, 10, 0)',
    '--display-stretch': '100%',
    '--display-weight': '300',
    '--track-display': '-0.045em',
    '--display-vw': '7.2vw',
    '--display-max': '104px',
    '--display-phone-max': '52px',
    '--title-em': '10.1',
    '--final-line-bottom-d': '126px',
    '--final-text-bottom-d': '272px',
    '--logo-weight': '400',
  },
  config: {
    phrase: {
      text: 'TRUE OR LIES',
      anagram: 'OUTLIERS',
      // from scripts/solve-respell.ts (build step 3): no two keys touch while the word is re-spelled
      respell: [[-6.3, 0.32, -1], [0, 0, 0], [0, 0.06, -1], [-0.9, 0.42, -1], [4.5, 0.26, -1], [0, 0.2, 1], [-3.6, 0.22, -1], [3.6, 0.36, 1], [0, 0, 0], [5.4, 0.22, -1]],
    },
    keycap: {
      color: '#efe6d6',
      ink: '#3a2a22',
      inkAlt: '#f6ecda',
      roughness: 0.6,
      specularIntensity: 0.5,
      clearcoat: 0.1,
      clearcoatRoughness: 0.5,
      envMapIntensity: 0.9,
      inkRoughness: 0.5,
      legendWeight: 500,
      // south pole → north pole: raw umber foot, sand, three cream rings, sage, a deep green glaze on top
      ringColors: ['#5a3a2a', '#5a3a2a', '#d9c4a2', '#efe6d6', '#efe6d6', '#efe6d6', '#93a283', '#35534b', '#35534b'],
      accents: [['#dba53f', 0.07]],
      phraseColor: '#c4532e',
      xrayBody: '#1f0e08',
      xrayGlow: '#6a2a12',
    },
    wireframe: { color: '#ffb070' },
    aurora: { low: '#ff7a4a', high: '#7a3a8a', edge: '#ffe0b0', intensity: 0.5 },
    sky: {
      voidTop: '#1a1019',
      glowDir: [0.34, 0.085, -1],
      glow: 0.7,
      zenithAt: 0.34,
      stars: [1.0, 0.9, 0.78],
      sun: { size: 0.036, top: '#ffe2a0', bottom: '#ff9a4a', bands: 0, intensity: 1.7 },
    },
    preloader: { color: '#1b0f0a' },
    cursorLight: { color: '#ffd2a0', trailColor: '#c8703a', trailCore: '#fff0d8' },
    lighting: {
      hemisphere: { sky: '#bda6cc', ground: '#5a2a18', intensity: 0.95 },
      key: { color: '#ffc48a', intensity: 3.0, position: [95, 46, -70] },
      fill: { color: '#ffdcb8', intensity: 0.5, position: [-60, 30, 130] },
      finalKey: { color: '#fff0dc', intensity: 2.2 },
      core: { color: '#ffb060' },
      env: { sky: '#ffd9b8', breach: '#ffe6c0', horizon: '#b8603a', front: '#fff4e8' },
    },
    atmosphere: {
      fogColor: '#c0704a',
      voidColor: '#0d0806',
      skyZenith: '#3d3a63',
      skyGlow: '#ffc890',
      shaftColor: '#ffd6a0',
      shaftIntensity: 0.09,
      dustColor: '#e6a878',
      rock: { dark: '#3a1a12', light: '#a04e2c', snow: '#e2b48a' },
    },
    terrain: { surface: { strata: 0.55, snow: 1, snowLow: 1 } },
    post: { grain: 0.07, vignette: 0.45, saturation: 1.02, shadowTint: [0.01, 0.004, 0.006], highlightTint: [1.02, 1.0, 0.96] },
  },
}
