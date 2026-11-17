/** The landing page's cursor: the shared ring, which also grows over a reacting 3D key. */
import { Cursor as SharedCursor } from '../../../shared/argilla/ui/Cursor'
import { store } from '../state/store'

const overKey = () => store.pointerOverKey

export function Cursor() {
  return <SharedCursor isHover={overKey} />
}
