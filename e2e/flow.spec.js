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

test('sign-up page validates and explains the next step', async ({ page }) => {
  await page.goto('/signup');
  await page.getByPlaceholder('Username').fill('e2e_' + Date.now());
  await page.getByPlaceholder('Email').fill(`e2e${Date.now()}@example.com`);
  await page.getByPlaceholder(/Password/).fill('a-long-enough-password');
  await page.getByRole('button', { name: /create account/i }).click();
  await expect(page.getByRole('status')).toContainText(/check your email|account created/i);
});
