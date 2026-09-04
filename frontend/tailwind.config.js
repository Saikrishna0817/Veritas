/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Base surfaces
        bgVoid: 'var(--bg-void)',
        bgSurface: 'var(--bg-surface)',
        bgPanel: 'var(--bg-panel)',
        bgPanelRaised: 'var(--bg-panel-raised)',
        borderHairline: 'var(--border-hairline)',
        
        // Brand red
        redPrimary: 'var(--red-primary)',
        redBright: 'var(--red-bright)',
        redDim: 'var(--red-dim)',
        redGlow: 'var(--red-glow)',
        
        // Semantic status
        statusSafe: 'var(--status-safe)',
        statusWarn: 'var(--status-warn)',
        statusCritical: 'var(--status-critical)',
        
        // Text
        textPrimary: 'var(--text-primary)',
        textSecondary: 'var(--text-secondary)',
        textMuted: 'var(--text-muted)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        display: ['Space Grotesk', 'system-ui', 'sans-serif'],
      },
      animation: {
        'marquee': 'marquee 25s linear infinite',
      },
      keyframes: {
        marquee: {
          '0%': { transform: 'translateX(0%)' },
          '100%': { transform: 'translateX(-100%)' },
        }
      },
      boxShadow: {
        'red-glow': '0 0 40px var(--red-glow)',
      },
    },
  },
  plugins: [],
}
