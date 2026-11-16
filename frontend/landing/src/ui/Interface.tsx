/**
 * Interface: a minimal Swiss layout over the scene (1440 × 900 design).
 * H1 titles: Focus Reveal; body copy: line-by-line Fade Up — both play/reverse with their section (ui/titles.ts).
 * Other reveals/hides are driven by the scroll timeline (useScrollTimeline).
 *  - header (wordmark + label) and the phase indicator are sticky across the whole experience;
 *  - hero layer and final layer cross-fade with the 3D transformation.
 */
import { restartExperience } from '../scroll/useScrollTimeline'
import { FocusReveal } from './FocusReveal'
import { FadeUpLines } from './FadeUpLines'
import { Preloader } from './Preloader'
import { Cursor } from './Cursor'
import { config } from '../config'
import { content } from '../content'
import styles from './Interface.module.css'

/** Where "Sign in" goes: the app's sign-in page (same site by default; set VITE_APP_URL when it lives elsewhere). */
const APP_URL: string = import.meta.env.VITE_APP_URL ?? '/login'

export function Interface() {
  const C = content
  return (
    <div className={styles.ui} data-loading="">
      {/* sticky header */}
      <header className={styles.header}>
        <span className={styles.logo}>{C.brand}</span>
        <span className={styles.headerEnd}>
          <span className="t-label">{C.label}</span>
          <a className={`t-label ${styles.signIn}`} href={APP_URL}>
            {C.signIn}
          </a>
        </span>
      </header>

      {/* sticky phase indicator */}
      <aside className={styles.phase} aria-hidden>
        <span className={styles.track}>
          <span className={styles.fill} data-ui="phase-fill" />
        </span>
        <span className={`${styles.phaseLabel} t-micro`}>
          <span data-ui="phase-num">{C.phases[0][0]}</span>
          <span className={styles.phaseName} data-ui="phase-name">
            {C.phases[0][1]}
          </span>
        </span>
      </aside>

      {/* hero */}
      <div className={styles.layer} data-ui="hero">
        <div className={styles.heroShade} data-fade />
        <h1 className={`t-display ${styles.heroTitle}`}>
          <FocusReveal text={C.hero.title} group="hero" className="fr--stack" />
        </h1>
        <FadeUpLines className={`t-body ${styles.heroText}`} group="hero" delay={config.titles.bodyDelay} text={C.hero.text} />
        <span className={`t-label ${styles.hint}`} data-ui="hint">
          {C.hero.hint}
        </span>
      </div>

      {/* second section */}
      <div className={styles.layer} data-ui="final">
        <FadeUpLines className={`t-body ${styles.finalText}`} group="final" delay={config.titles.bodyDelay} text={C.final.text} />
        <h2 className={`t-display ${styles.finalLine1}`}>
          <FocusReveal text={C.final.line1} group="final" />
        </h2>
        <h2 className={`t-display ${styles.finalLine2}`}>
          <FocusReveal text={C.final.line2} group="final" delay={config.titles.secondLineDelay} />
        </h2>
      </div>

      {/* third section — the anagram */}
      <div className={styles.layer} data-ui="material">
        <FadeUpLines className={`t-body ${styles.materialText}`} group="material" delay={config.titles.bodyDelay} text={C.material.text} />
        <h2 className={`t-display ${styles.materialTitle}`}>
          <FocusReveal text={C.material.title} group="material" />
        </h2>
        <button type="button" className={`t-label ${styles.restart}`} data-fade onClick={restartExperience}>
          {C.material.restart}
        </button>
      </div>
      <Preloader />
      <Cursor />
    </div>
  )
}
