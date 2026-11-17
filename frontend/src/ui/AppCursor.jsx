import { Cursor as SharedCursor } from '../../shared/argilla/ui/Cursor';

/** Where the ring cursor shows: only over the 3D world (empty page areas), never over text, tables, links or inputs. */
const VOID = 'html, body, #root, .in-shell, .in-content, .in-main, .in-page, .in-auth, .in-legal, .in-section, .ar-scene, .in-page-head, .in-review-head, .in-head-stage, .in-keys-stage, .in-auth-stage';
const overVoid = (target) => !!target?.matches?.(VOID);

export default function AppCursor() {
  return <SharedCursor active={overVoid} />;
}
