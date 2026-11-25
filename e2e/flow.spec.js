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
  await page.setInputFiles('.in-drop input[type="file"]', csv());
  await expect(page.getByRole('radio', { name: /no trusted reference/i })).toBeChecked();   // the default
  await page.getByRole('radio', { name: /first 70%/i }).check();   // the flips sit in the last 30%: a verdict
  await page.getByRole('button', { name: 'Analyse' }).click();
  await expect(page.getByRole('heading', { name: /evidence of poisoning/i })).toBeVisible({ timeout: 60000 });
  await expect(page.getByText('Rows to review')).toBeVisible();
  await expect(page.locator('.in-table tbody tr').first()).toBeVisible();   // the flipped rows are listed

  // a user account has no admin console, and the admin API answers 403
  await page.goto('/admin');
  await expect(page.getByRole('heading', { name: 'This page is for administrators' })).toBeVisible();
  const csrf = (await context.cookies()).find((c) => c.name === 'veritas_csrf')?.value || '';
  for (const [method, path] of [['get', '/api/v1/users'], ['post', '/api/v1/users'], ['delete', '/api/v1/users/x'], ['delete', '/api/v1/history/x'], ['post', '/api/v1/defense/quarantine']]) {
    const r = await page.request[method](path, { headers: { 'X-CSRF-Token': csrf }, data: method === 'post' ? {} : undefined });
    expect(r.status(), `${method.toUpperCase()} ${path}`).toBe(403);
  }
});

test('a signed-out visitor at the front door is sent to the landing page', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveURL(/\/welcome\/$/);    // the landing page's content needs its own server (nginx, or :5180 locally)
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

test('a deep link survives signing in, and only addresses inside the app are followed', async ({ page }) => {
  await page.goto('/history?from=e2e');
  await expect(page).toHaveURL(/\/login\?next=%2Fhistory%3Ffrom%3De2e$/);
  await page.getByPlaceholder(/username/i).fill(USER);
  await page.locator('input[type="password"]').fill(PASS);
  await page.locator('input[type="password"]').press('Enter');
  await page.waitForURL((u) => u.pathname === '/history' && u.search === '?from=e2e');

  const home = new URL('/', page.url()).href;
  await page.goto('/login?next=' + encodeURIComponent('//example.com'));    // signed in: sent home, never off-site
  await expect(page).toHaveURL(home);
});

test('server validation messages appear beside the field they are about', async ({ page }) => {
  await page.goto('/signup');
  await page.getByPlaceholder('Username').fill('e2e_' + Date.now());
  await page.getByPlaceholder('Email').fill('someone@localhost');           // the browser accepts it; the server does not
  await page.getByPlaceholder(/Password/).fill('a-long-enough-password');
  await page.getByRole('button', { name: /create account/i }).click();
  await expect(page.getByPlaceholder('Email')).toHaveAttribute('aria-invalid', 'true');
  await expect(page.locator('.in-field-error')).toContainText(/email/i);
});

// ── Admin console ──────────────────────────────────────────────────────────────────────────────────────────────────
const ADMIN = process.env.E2E_ADMIN_USER, ADMIN_PASS = process.env.E2E_ADMIN_PASS;

async function signIn(page, path, user, pass) {
  await page.goto(path);
  await page.getByPlaceholder(/username/i).fill(user);
  await page.locator('input[type="password"]').fill(pass);
  await page.locator('input[type="password"]').press('Enter');
}

test('admin creates an account, changes its role, reviews the queue, deletes it, and the audit log records each step', async ({ page, browser }) => {
  test.skip(!ADMIN || !ADMIN_PASS, 'set E2E_ADMIN_USER and E2E_ADMIN_PASS');
  const name = 'e2e_u' + Date.now().toString(36), pass = 'e2e-temp-password-1';   // never signed in: logins are rate limited

  await signIn(page, '/admin/login', ADMIN, ADMIN_PASS);
  await page.waitForURL((u) => u.pathname === '/admin');

  // create
  await page.getByRole('button', { name: 'Create an account' }).click();
  await page.getByLabel('Username', { exact: true }).fill(name);
  await page.getByLabel('Password', { exact: true }).fill(pass);
  await page.locator('form').getByRole('button', { name: 'Create account' }).click();
  await expect(page.getByRole('status')).toContainText(`Account “${name}” created`);
  const row = page.locator('tbody tr', { hasText: name });
  await expect(row).toBeVisible();

  // change the role
  await row.getByLabel(`Role of ${name}`).selectOption('analyst');
  await row.getByRole('button', { name: 'Save role' }).click();
  await expect(page.getByRole('status')).toContainText(`“${name}” is now analyst`);
  await expect(row.getByLabel(`Role of ${name}`)).toHaveValue('analyst');

  // disable, then enable (your own row offers neither)
  await row.getByRole('button', { name: 'Disable' }).click();
  await page.getByRole('button', { name: `Disable ${name}` }).click();
  await expect(page.getByRole('status')).toContainText(`“${name}” is disabled`);
  await expect(row).toContainText('Disabled');
  await row.getByRole('button', { name: 'Enable' }).click();
  await page.getByRole('button', { name: `Enable ${name}` }).click();
  await expect(page.getByRole('status')).toContainText(`“${name}” can sign in again`);
  await expect(row).toContainText('Active');

  // review queue: decide the newest case when there is one, else the empty state
  await page.getByRole('tab', { name: /Review queue/ }).click();
  const decide = page.getByRole('button', { name: 'Close, no action' }).first();
  if (await decide.isVisible().catch(() => false)) {
    await decide.click();
    await page.getByRole('button', { name: 'Close case' }).click();
    await expect(page.getByRole('status')).toContainText(/closed with no action/);
  } else {
    await expect(page.getByText('No cases are waiting for review.')).toBeVisible();
  }

  // delete
  await page.getByRole('tab', { name: /Accounts/ }).click();
  await row.getByRole('button', { name: 'Delete' }).click();
  await page.getByRole('button', { name: `Delete ${name}` }).click();
  await expect(page.getByRole('status')).toContainText(`Account “${name}” deleted`);
  await expect(page.locator('tbody tr', { hasText: name })).toHaveCount(0);

  // the audit log has all three, as the admin, about this account
  await page.getByRole('tab', { name: /Audit log/ }).click();
  await expect(page.getByRole('heading', { name: 'Audit log' })).toBeVisible();
  const events = (await (await page.request.get('/api/v1/audit/events?limit=200')).json()).events
    .filter((e) => e.details?.username === name).map((e) => e.action).sort();
  expect(events).toEqual(['user.created', 'user.deleted', 'user.disabled', 'user.enabled', 'user.role_changed']);
  for (const action of ['user.created', 'user.role_changed', 'user.disabled', 'user.enabled', 'user.deleted']) {
    await expect(page.locator('td', { hasText: action }).first()).toBeVisible();
  }
});
