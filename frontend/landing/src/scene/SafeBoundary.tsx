/**
 * Keeps one failing scene system (e.g. a missing asset) from taking the whole experience down.
 */
import { Component, type ReactNode } from 'react'

export class SafeBoundary extends Component<{ name: string; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  componentDidCatch(error: unknown) {
    console.error(`[keycap sphere] ${this.props.name} failed and was skipped:`, error)
  }
  render() {
    return this.state.failed ? null : this.props.children
  }
}
