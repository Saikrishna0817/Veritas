# Veritas — landing page

The public page for Veritas, built on the keycap sphere template by ubernatural (uberdesigner.ai), in its ARGILLA
look: matte ceramic keys glazed in bands, red sandstone, a low sun, warm light against a lilac sky. A sphere of 77
keycaps floats over a valley; scrolling breaks it apart ring by ring, and ten keys fly out to spell **TRUE OR LIES**,
then re-spell **OUTLIERS**. The whole page is one continuous 3D scene. It is a separate project from the app in
`../src` and shares nothing with it.

## Run

```bash
npm install
npm run dev
```

Production build: `npm run build`, then `npm run preview`. `dist/` uses relative paths, so it works from any folder.
"Sign in ↗" goes to `/login` on the same site; set `VITE_APP_URL` at build time when the app lives elsewhere.

- `?demo=1` plays a scripted take with no scrolling; `?at=<0…1>` shows a still of any scroll position with no loader
  (`?at=0.676` is TRUE OR LIES, `?at=1` is OUTLIERS).
- `npm run test:choreography` checks the choreography in Node: keycap overlaps, that scrolling back restores the
  sphere, that the phrase keys land in their slots and that no two keys touch while the word is re-spelled.
- `npm run solve:respell` finds new re-spelling paths after the words on the keys change.
- `npm run lint` (oxlint) and `tsc --noEmit` (part of `npm run build`).

## What's where

```
src/looks/veritas.ts        this look: copy, interface tokens, overrides of the scene's defaults
src/config.ts               the scene's defaults: sphere, choreography, camera, light, palette, post, demo
src/styles/tokens.css       the interface's default tokens and the sizes derived from the look's inputs
src/choreography/           pure logic, tested in Node: rings, phrase keys, each key's pose = f(scroll, time)
src/scroll/                 one pinned, scrubbed GSAP timeline for the scene and the UI; the demo take and stills
src/scene/                  camera, lights, post chain, environment (valley, massifs, sky, atmosphere), keys, cursor light
src/ui/                     Interface, Preloader, Cursor, the title and body-copy reveals
scripts/                    the choreography check, the re-spelling solver, their shared overlap test
public/models/key-3d.gltf   the keycap model
```

## Licences

Inter and Archivo: SIL Open Font License 1.1 (the licence files ship inside their `@fontsource-variable` packages).
Everything else, including the keycap model and the mountains, is part of the template.
