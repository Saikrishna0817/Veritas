/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: '#050a0f',
        bg2: '#080f18',
        bg3: '#0c1622',
        surface: '#0f1c2e',
        surface2: '#142235',
        border: '#1e3a52',
        border2: '#234060',
        accent: '#00e5ff',
        accent2: '#0090b8',
        accent3: '#00ffc8',
        danger: '#ff4d6a',
        orange: '#ff8c42',
        yellow: '#ffd166',
        purple: '#bd93f9',
        green: '#00ffc8',
        text1: '#e0eef8',
        text2: '#8ab0cc',
        text3: '#4a7a9b',
      },
      fontFamily: {
        sans: ['Syne', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        serif: ['Instrument Serif', 'serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'scan': 'scan 8s linear infinite',
        'fadeInUp': 'fadeInUp 0.5s ease forwards',
        'glow': 'glow 2s ease-in-out infinite',
      },
      keyframes: {
        scan: {
          '0%': { top: '-2px' },
          '100%': { top: '100%' },
        },
        fadeInUp: {
          from: { opacity: '0', transform: 'translateY(20px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        glow: {
          '0%, 100%': { boxShadow: '0 0 5px rgba(0,229,255,0.3)' },
          '50%': { boxShadow: '0 0 20px rgba(0,229,255,0.6)' },
        }
      },
      boxShadow: {
        'accent': '0 0 20px rgba(0,229,255,0.2)',
        'danger': '0 0 20px rgba(255,77,106,0.2)',
        'green': '0 0 20px rgba(0,255,200,0.2)',
      }
    },
  },
  plugins: [],
}
