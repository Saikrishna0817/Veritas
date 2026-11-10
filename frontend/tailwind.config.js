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
        bgVoid: '#080708',
        bgSurface: '#0C0B0C',
        bgPanel: '#141314',
        bgPanelRaised: '#1A1A1A',
        borderHairline: 'var(--border-hairline)',
        
        // Brand red
        redPrimary: '#B9A3FF',
        redBright: '#D3C6FF',
        redDim: '#2A2140',
        redGlow: 'var(--red-glow)',
        
        // Semantic status
        statusSafe: '#6EE7A0',
        statusWarn: '#FFB86B',
        statusCritical: '#FF2327',
        
        // Text
        textPrimary: '#FFFFFF',
        textSecondary: '#CFCED0',
        textMuted: '#9A989C',
        uv: '#DDB5FC',
        flag: '#FF6A4D',
        violet: '#B9A3FF',
        statusInfo: '#9DB8FF',
        borderStrong: 'var(--border-strong)',
        ink: { 950: '#080708', 900: '#0C0B0C', 850: '#141314', 800: '#1A1A1A', 700: '#2C2C2C', 600: '#4A494B' },
        bone: '#FFFFFF',
      },
      fontFamily: {
        sans: ['Wix Madefor Display', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        display: ['Wix Madefor Display', 'system-ui', 'sans-serif'],
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
        'red-glow': '0 10px 30px -8px var(--red-glow)',
        'uv': '0 14px 40px -14px rgba(185,163,255,0.55)',
        'lift': '0 18px 40px -18px rgba(0,0,0,0.9)',
      },
    },
  },
  plugins: [],
}
