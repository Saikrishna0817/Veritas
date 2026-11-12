import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { MotionConfig } from 'framer-motion'
import './index.css'
import App from './App.jsx'
import { AuthProvider } from './context/AuthContext.jsx'

// reducedMotion="user": framer-motion skips movement for visitors who ask their OS for less motion.
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <MotionConfig reducedMotion="user">
      <AuthProvider><App /></AuthProvider>
    </MotionConfig>
  </StrictMode>,
)
