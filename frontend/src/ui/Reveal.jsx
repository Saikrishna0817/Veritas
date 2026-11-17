import { FocusReveal } from '../../shared/argilla/ui/FocusReveal';
import { FadeUpLines } from '../../shared/argilla/ui/FadeUpLines';

/**
 * The landing page's title and intro reveals in the app. A page's title pulls into focus and its intro rises line
 * by line when the page mounts; data (tables, meters, numbers) never uses them. Shorter than on the landing page,
 * where they play once per scroll section, because here people move between pages to work.
 * prefers-reduced-motion: a plain 0.2 s fade, no blur (handled inside the shared components).
 */
const FOCUS = { duration: 1.1, delay: 0.05, ease: 'sine', easeType: 'inOut', direction: 'right', scale: 1.06, opacity: 0, blur: 0.18, intensity: 0.6 };
const FADE_UP = { duration: 0.6, delay: 0.25, stagger: 0.08, ease: 'power3.out', distance: 0.5, opacity: 0, blur: 0.08 };
/** In the app every reveal plays once, as soon as it mounts. */
const playOnMount = (_group, ctl) => {
  ctl.play();
  return () => {};
};

export function PageTitle({ children, className = '' }) {
  return (
    <h1 className={`in-h1 ${className}`}>
      <FocusReveal text={String(children)} group="page" settings={FOCUS} register={playOnMount} />
    </h1>
  );
}

export function PageIntro({ children, className = '' }) {
  return <FadeUpLines className={`in-lede ${className}`} text={String(children)} group="page" settings={FADE_UP} register={playOnMount} />;
}
