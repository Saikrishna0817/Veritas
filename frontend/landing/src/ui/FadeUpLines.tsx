/** The landing page's Fade Up: the shared component with this page's settings and its scroll-driven groups. */
import { FadeUpLines as SharedFadeUpLines } from '../../../shared/argilla/ui/FadeUpLines'
import type { RevealControl } from '../../../shared/argilla/ui/reveal'
import { config } from '../config'
import { registerTitle, type TitleGroup } from './titles'

interface Props {
  text: string
  group: TitleGroup
  delay?: number
  className?: string
}

const register = (group: string, ctl: RevealControl) => registerTitle(group as TitleGroup, ctl)

export function FadeUpLines(props: Props) {
  return <SharedFadeUpLines {...props} settings={config.titles.fadeUp} register={register} />
}
