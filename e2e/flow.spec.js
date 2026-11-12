import { test, expect } from '@playwright/test';
import fs from 'fs';
import os from 'os';
import path from 'path';

const USER = process.env.E2E_USER, PASS = process.env.E2E_PASS;

function csv() {
  const rows = ['a,b,c,label'];
  let s = 7; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < 500; i++) {
    const a = rnd(), b = rnd(); let y = a + b > 1 ? 1 : 0;
    if (i >= 350 && i < 410) y = 1 - y;                       // 60 flipped labels in the incoming part
    rows.push(`${a.toFixed(4)},${b.toFixed(4)},${(a + b).toFixed(4)},${y}`);
  }
  const f = path.join(os.tmpdir(), 'veritas-e2e.csv');
  fs.writeFileSync(f, rows.join('\n'));
  return f;
}

test('sign in, audit a dataset, see the verdict, sign out; session is an httpOnly cookie', async ({ page, context }) => {
  await page.goto('/login');
  await page.getByPlaceholder(/username/i).fill(USER);
  await page.locator('input[type="password"]').fill(PASS);
  await page.locator('input[type="password"]').press('Enter');   // the submit button animates on load
  await page.waitForURL((u) => u.pathname === '/' || u.pathname.startsWith('/admin'));

  const cookies = await context.cookies();
  expect(cookies.find((c) => c.name === 'veritas_session')?.httpOnly).toBe(true);
  expect(await page.evaluate(() => JSON.stringify(localStorage))).not.toContain('token');   // nothing for a script to steal

  await page.goto('/upload');
  await page.setInputFiles('input[type="file"]', csv());
  await page.getByRole('button', { name: /run poisoning detection/i }).click();
  await expect(page.getByText(/CONFIRMED|SUSPICIOUS/i).first()).toBeVisible({ timeout: 45000 });
  await expect(page.getByText(/Label Check/i).first()).toBeVisible();
});

test('anonymous visitors are sent to sign-in and cannot call the API', async ({ page, request }) => {
  await page.goto('/upload');
  await expect(page).toHaveURL(/\/login/);
  expect((await request.get('/api/v1/history')).status()).toBe(401);
});

// The API allows 5 sign-ins per IP per minute, so tests that need a session sign in once each, through this helper.
async function signIn(page) {
  await page.goto('/login');
  await page.getByPlaceholder(/username/i).fill(USER);
  await page.locator('input[type="password"]').fill(PASS);
  await page.locator('input[type="password"]').press('Enter');
  await page.waitForURL((u) => u.pathname === '/');
}

// Over 10 MB, so the frontend sends it as a background job (?background=true, then GET /jobs/{id}).
function largeCsv() {
  const cols = 40, rows = 36000;
  let s = 11; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const lines = [Array.from({ length: cols }, (_, k) => `f${k}`).join(',') + ',label'];
  for (let i = 0; i < rows; i++) {
    const x = Array.from({ length: cols }, () => rnd() * 2 - 1);
    let y = x[0] + x[1] - x[2] > 0 ? 1 : 0;
    if (i >= 25200 && rnd() < 0.05) y = 1 - y;
    lines.push(x.map((v) => v.toFixed(5)).join(',') + ',' + y);
  }
  const f = path.join(os.tmpdir(), 'veritas-e2e-large.csv');
  fs.writeFileSync(f, lines.join('\n'));
  expect(fs.statSync(f).size).toBeGreaterThan(10 * 1024 * 1024);
  return f;
}

test('the landing page is public and leads to sign-in', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1, name: 'veritas' })).toBeVisible();
  await expect(page.getByText(/Measured on data/i)).toBeVisible();
  await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Sign in' }).click();
  await expect(page).toHaveURL(/\/login$/);
});

test('account page: create an API key, see it once, revoke it', async ({ page }) => {
  await signIn(page);
  await page.goto('/account');
  await expect(page.getByText(/Analyses today/i)).toBeVisible();
  const name = 'e2e-key-' + Date.now();
  await page.getByPlaceholder(/Key name/i).fill(name);
  await page.getByRole('button', { name: /create key/i }).click();
  await expect(page.getByText(/Copy this key now/i)).toBeVisible();
  await expect(page.locator('code', { hasText: /^vk_/ })).toBeVisible();
  await page.getByRole('button', { name: `Revoke key ${name}` }).click();
  await expect(page.getByText(new RegExp(`revoked keys?: .*${name}`))).toBeVisible();
  await expect(page.getByText(/Copy this key now/i)).toHaveCount(0);   // the banner goes with the key
});

test('a large CSV runs as a background job and opens the result', async ({ page }) => {
  test.setTimeout(240000);
  const file = largeCsv();
  await signIn(page);
  await page.goto('/upload');
  await page.setInputFiles('input[type="file"]', file);
  await page.getByRole('button', { name: /run poisoning detection/i }).click();
  await expect(page.getByText(/running in the background/i)).toBeVisible({ timeout: 30000 });
  await expect(page.getByRole('heading', { level: 1, name: 'Examination' })).toBeVisible({ timeout: 200000 });
  await expect(page.getByText('veritas-e2e-large.csv').first()).toBeVisible();
});

test('sign-up page validates and explains the next step', async ({ page }) => {
  await page.goto('/signup');
  await page.getByPlaceholder('Username').fill('e2e_' + Date.now());
  await page.getByPlaceholder('Email').fill(`e2e${Date.now()}@example.com`);
  await page.getByPlaceholder(/Password/).fill('a-long-enough-password');
  await page.getByRole('button', { name: /create account/i }).click();
  await expect(page.getByRole('status')).toContainText(/check your email|account created/i);
});
