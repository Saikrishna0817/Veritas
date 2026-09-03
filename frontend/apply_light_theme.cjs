const fs = require('fs');
const path = require('path');

const pagesDir = path.join(__dirname, 'src', 'pages');

function processFile(filePath) {
    if (filePath.endsWith('UploadPage.jsx') || filePath.endsWith('Dashboard.jsx')) return; // Already updated manually

    let content = fs.readFileSync(filePath, 'utf8');

    // Text colors
    content = content.replace(/color:\s*['"]#f1f5f9['"]/g, "color: '#141414'"); // Main headers
    content = content.replace(/color:\s*['"]#94a3b8['"]/g, "color: '#334155'"); // Subtext
    content = content.replace(/color:\s*['"]#64748b['"]/g, "color: '#475569'"); // Subtext 2
    content = content.replace(/color:\s*['"]#a5b4fc['"]/g, "color: '#E8622C'"); // Accent text
    content = content.replace(/color:\s*['"]#fff['"]/g, "color: '#fff'"); // White stays white (usually for buttons)
    
    // Backgrounds
    content = content.replace(/background:\s*['"]rgba\(255,255,255,0\.02\)['"]/g, "background: 'rgba(255,255,255,0.4)'");
    content = content.replace(/background:\s*['"]rgba\(255,255,255,0\.03\)['"]/g, "background: 'rgba(255,255,255,0.5)'");
    content = content.replace(/background:\s*['"]rgba\(255,255,255,0\.04\)['"]/g, "background: 'rgba(0,0,0,0.03)'");
    content = content.replace(/background:\s*['"]rgba\(255,255,255,0\.05\)['"]/g, "background: 'rgba(0,0,0,0.05)'");
    content = content.replace(/background:\s*['"]rgba\(255,255,255,0\.06\)['"]/g, "background: 'rgba(0,0,0,0.06)'");
    content = content.replace(/background:\s*['"]rgba\(255,255,255,0\.08\)['"]/g, "background: 'rgba(0,0,0,0.08)'");
    
    // Borders
    content = content.replace(/border:\s*['"]1px solid rgba\(255,255,255,0\.08\)['"]/g, "border: '1px solid rgba(0,0,0,0.1)'");
    content = content.replace(/border:\s*['"]1px solid rgba\(255,255,255,0\.1\)['"]/g, "border: '1px solid rgba(0,0,0,0.15)'");
    content = content.replace(/border:\s*['"]2px dashed rgba\(255,255,255,0\.12\)['"]/g, "border: '3px dashed rgba(0,0,0,0.15)'");

    // Primary Buttons & Accents
    content = content.replace(/linear-gradient\(135deg, #6366f1, #8b5cf6\)/g, "#E8622C");
    content = content.replace(/linear-gradient\(90deg, #6366f1, #8b5cf6, #a855f7\)/g, "#E8622C");
    content = content.replace(/boxShadow:\s*loading \? 'none' : '0 4px 24px rgba\(99,102,241,0\.4\)'/g, "boxShadow: loading ? 'none' : '0 8px 24px rgba(232, 98, 44, 0.3)'");
    content = content.replace(/background:\s*['"]rgba\(99,102,241,0\.1\)['"]/g, "background: 'rgba(232,98,44,0.1)'");
    content = content.replace(/border:\s*['"]1px solid rgba\(99,102,241,0\.3\)['"]/g, "border: '2px solid rgba(232,98,44,0.4)'");
    
    // Hover states for the accent buttons
    content = content.replace(/rgba\(99,102,241,0\.08\)/g, "rgba(232,98,44,0.05)");
    content = content.replace(/rgba\(99,102,241,0\.12\)/g, "rgba(232,98,44,0.1)");
    content = content.replace(/rgba\(99,102,241,0\.2\)/g, "rgba(232,98,44,0.2)");
    
    // Layout Widths & Fonts (Main container)
    // Most start with: padding: '32px 40px', maxWidth: 1000, margin: '0 auto'
    // We want to make them fully wide, larger text.
    content = content.replace(/padding:\s*['"]32px 40px['"],\s*maxWidth:\s*\d+,\s*margin:\s*['"]0 auto['"]/g, "padding: '48px 64px', width: '100%', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '32px'");
    
    // Font sizing on headers
    content = content.replace(/fontSize:\s*26,\s*fontWeight:\s*800/g, "fontSize: 48, fontWeight: 900");
    content = content.replace(/fontSize:\s*28,\s*fontWeight:\s*800/g, "fontSize: 48, fontWeight: 900");
    content = content.replace(/marginTop:\s*8,\s*fontSize:\s*14/g, "marginTop: 12, fontSize: 18, maxWidth: '80%', lineHeight: 1.6");
    
    fs.writeFileSync(filePath, content, 'utf8');
}

fs.readdirSync(pagesDir).forEach(file => {
    if (file.endsWith('.jsx')) {
        processFile(path.join(pagesDir, file));
    }
});

console.log('Processed light theme styles.');
