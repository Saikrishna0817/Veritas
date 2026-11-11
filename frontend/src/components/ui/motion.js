/**
 * Motion tokens. One easing everywhere: the reference's slow expo-out, which starts fast and settles
 * long. framer-motion's MotionConfig (reducedMotion="user") and the reduced-motion rule in index.css
 * collapse all of it when the visitor asks for less motion.
 */
export const EASE = [0.19, 1, 0.22, 1];

/** Seconds. fast: route fade, hover-scale things. base: dialogs, panels. slow: in-view reveals. grow: bars. */
export const DURATION = { fast: 0.25, base: 0.45, slow: 0.8, grow: 1.2 };

/** The white highlight that slides between Segmented and Tabs options. */
export const SPRING = { type: 'spring', stiffness: 520, damping: 42 };
