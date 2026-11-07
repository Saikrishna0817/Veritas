import { defineConfig } from '@playwright/test';
// Run against a running stack:  BASE_URL=http://localhost:5173 E2E_USER=... E2E_PASS=... npx playwright test
export default defineConfig({
  testDir: '.', timeout: 60000, retries: 0, workers: 1,
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:5173', headless: true, screenshot: 'only-on-failure', trace: 'retain-on-failure',
    // PW_CHROME: path to an installed Chromium/Chrome when `playwright install` cannot fetch the headless shell
    launchOptions: process.env.PW_CHROME ? { executablePath: process.env.PW_CHROME } : {},
  },
});
