/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Light Theme Base
        cream: '#F3EEE4',
        creamLighter: '#F7F3EA',
        
        // Dark Panels
        slateDark: '#0D0D0F',
        slateLighter: '#121214',
        
        // Outer Frame
        frameBlack: '#0A0A0A',
        
        // Accents
        burntOrange: '#E8622C',
        softYellow: '#F2E85C',
        
        // Text
        textDark: '#141414',
        textLight: '#F5F5F5',
        textMuted: '#6B7280', // standard gray for secondary text on light
        textMutedDark: '#9CA3AF', // standard gray for secondary text on dark
        
        // Keep some standard legacy colours temporarily if needed by other components, mapping to new ones
        bg: '#F3EEE4', // default bg
        text1: '#141414',
        text2: '#6B7280',
        text3: '#9CA3AF',
        accent: '#E8622C',
        accentCyan: '#E8622C', 
        accentViolet: '#F2E85C',
        surface: '#F7F3EA',
        surface2: '#F7F3EA',
        border: 'rgba(0, 0, 0, 0.1)',
        danger: '#ef4444',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
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
    },
  },
  plugins: [],
}
