import { useRef } from 'react';
import Brand from './Brand';
import { PageIntro, PageTitle } from '../ui/Reveal';
import { pressKey, useStage } from '../scene/stage';

/**
 * The frame of every sign-in page: the wordmark, a title that pulls into focus, a short intro, the form, and beside
 * it (above it on phones) the stage where the 3D world sets the keycap sphere at rest. Typing in the form dips one
 * random key; a running request turns the sphere slowly. Decoration only: the sphere never reflects whether a
 * password is right.
 */
export default function AuthFrame({ title, intro, busy = false, children }) {
  const stageRef = useRef(null);
  useStage('sphere', stageRef, { busy });
  return (
    <main className="instrument in-auth">
      <div className="in-auth-box" onInput={pressKey}>
        <Brand />
        <PageTitle className="in-auth-title">{title}</PageTitle>
        {intro && <PageIntro className="in-hint">{intro}</PageIntro>}
        {children}
      </div>
      <div className="in-auth-stage" ref={stageRef} aria-hidden="true" />
    </main>
  );
}
