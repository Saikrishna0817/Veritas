/** What a section does with a reveal: play it when it appears, reverse it when it leaves. */
export interface RevealControl {
  play(): void
  reverse(): void
}
/** Registers a reveal with its group; returns the unregister function. */
export type RegisterReveal = (group: string, ctl: RevealControl) => () => void
