# Veritas frontend

The web app for Veritas. Read [docs/PRODUCT.md](../docs/PRODUCT.md) before changing anything visible: it sets the
tone, principles and anti-references every page follows.

## Stack

- React 19 and React Router 7, built with Vite 7.
- Plain CSS for the app's design system (`src/styles/instrument.css`). Tailwind 3 is still configured (`index.css`,
  `tailwind.config.js`). Only the admin console, `PageErrorBoundary` and `ProtectedRoute` still use its classes.
- lucide-react for icons.
- ESLint 9 (flat config in `eslint.config.js`).
- No chart library and no state library. All server calls go through `src/services/api.js`.

## Layout

```
src/
├── main.jsx                 # entry; imports index.css
├── App.jsx                  # routes, navigation, footer, signed-in shell
├── index.css, App.css       # older global styles (Tailwind base, dark frame)
├── styles/instrument.css    # the current design system: tokens and .in-* classes, scoped to .instrument
├── services/api.js          # the only place that calls the API (cookies + CSRF header)
├── context/AuthContext.jsx  # signed-in user; hooks/useAuth.js reads it
├── components/
│   ├── results.jsx          # data types (KINDS), verdict words, the shared result view for every data type
│   ├── Brand.jsx, ProtectedRoute.jsx, PageErrorBoundary.jsx
└── pages/                   # one file per route; see the route table in the root README
```

## Design tokens

Colours, type and easing are CSS variables at the top of `src/styles/instrument.css`, on `.instrument`. They are:

- paper, sheet and rule for surfaces;
- ink 1–3 for text;
- `--accent` (graphite indigo) for controls;
- `--signal` (magenta), used only where Veritas is signalling something;
- `--ev-0` to `--ev-3` for the evidence scale;
- `--font-ui` (Instrument Sans) and `--font-num` (IBM Plex Mono).

Use the variables and the `.in-*` classes. Don't hardcode colours.

## Run

The app needs the API on `localhost:8001`. See the root README for starting it. Vite proxies `/api` and `/ws` there;
set `VITE_API_PROXY` to point at another address.

```bash
npm install
npm run dev        # http://localhost:5173
```

## Check

```bash
npm run lint       # ESLint
npm run build      # production build into dist/
```

End-to-end tests live in `../e2e` (Playwright) and run against a running stack with a real account:

```bash
cd ../e2e
npm install
BASE_URL=http://localhost:5173 E2E_USER=<user> E2E_PASS=<password> npx playwright test
```

If Playwright cannot download its browser, set `PW_CHROME` to the path of an installed Chrome or Edge.
