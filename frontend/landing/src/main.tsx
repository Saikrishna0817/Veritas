import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/inter'
import '@fontsource-variable/archivo/wdth.css'
import './styles/tokens.css'
import './styles/global.css'
import App from './App'
import { look } from './looks'
import { content } from './content'

// the look's tokens override tokens.css, and the page takes its name
for (const [name, value] of Object.entries(look.tokens)) document.documentElement.style.setProperty(name, value)
document.documentElement.dataset.look = look.id
document.title = `${content.brand} — ${content.label.split(' — ')[0].toLowerCase().replace(/^./, (c) => c.toUpperCase())}`

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
