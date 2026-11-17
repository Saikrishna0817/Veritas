/** The landing page's Focus Reveal: the shared component with this page's settings and its scroll-driven groups. */
import { FocusReveal as SharedFocusReveal } from '../../../shared/argilla/ui/FocusReveal'
import type { RevealControl } from '../../../shared/argilla/ui/reveal'
import { config } from '../config'
import { registerTitle, type TitleGroup } from './titles'

interface Props {
  text: string
  group: TitleGroup
  /** Extra delay after the group is triggered (s) — e.g. the second line of a two-part title. */
  delay?: number
  className?: string
}

const register = (group: string, ctl: RevealControl) => registerTitle(group as TitleGroup, ctl)

export function FocusReveal(props: Props) {
  return <SharedFocusReveal {...props} settings={config.titles.focusReveal} register={register} />
}
