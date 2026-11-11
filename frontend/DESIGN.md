# Veritas design system

Primary visual reference: "Superconscious" by Outcrowd (dribbble.com/shots/27239697, live at superconscious-app.webflow.io). We follow its design language (palette, type, radii, gradient-edge cards, pill nav, slow expo motion) with our own content, logo, copy and 3D object. None of its assets are used.

## Colour (tokens in `src/index.css`, hex mirrors in `tailwind.config.js`)
| Role | Token | Value |
|---|---|---|
| Page | `--bg-surface` / `bgSurface` | #0C0B0C |
| Deep / rail | `--bg-void` | #080708 |
| Panel | `--bg-panel` | #141314 |
| Step card | `--bg-panel-raised` | #1A1A1A |
| Big card | `--bg-card-deep` | #0F0D10 |
| Edge | `--edge` | #4A494B, or gradient grey → #E73825 → #FF1900 → #4700D6 → grey |
| Accent | `--uv` (lavender) | #DDB5FC; hover #EDC5FC |
| Action / links | `--violet` | #B9A3FF; deep violet #4700D6 for light |
| Flagged rows, alarms | `--flag` | #FF6A4D |
| Verdicts | safe / info / warn / critical | #6EE7A0 / #9DB8FF / #FFB86B / #FF2327 |
| Text | white; secondary #CFCED0; muted #9A989C; serif accent #7C7A7E |

Legacy Tailwind names (`redPrimary`, `uv`, `ink-*`) are kept as aliases so older pages inherit the palette.

## Type
- **Wix Madefor Display** for everything, weight 500 for headings, tracking −0.05em (`.font-display`), −0.07em on the wordmark.
- **Tiro Kannada italic** for one grey word per headline (`.serif-accent`, or `PageHeader accent="…"`).
- **JetBrains Mono** for data only: evidence values, file names, hashes.
- Landing scale: wordmark `clamp(3.6rem,7vw,7rem)`, mega headline `clamp(3.6rem,13vw,13.2rem)`. App page titles `clamp(2.6rem,5.6vw,5rem)`.

## Shape
- 28px radius on cards (`.panel`, `.edge-card`, `.glass-card`), 16px on buttons, inputs and nav (`rounded-2xl`), 8px on tags.
- `.edge-card` > `.edge-inner`: 1.4px gradient edge for hero and result cards. `.glass-card`: lavender-to-dark edge for small step cards.
- `.icon-chip`: round #2C2C2C chip with a black ring.
- Primary button is solid white with dark text; secondary is a 6% white fill.
- Inputs: 6% white fill, no visible border, lavender focus.

## Backgrounds
`.uv-ground` (violet light from the top), `.stars` (CSS starfield), radial violet glows behind 3D objects, SVG wavy stripes behind the closing wordmark.

## 3D — `components/Specimen3D.jsx`
Point cloud of rows: lavender-white points, flagged rows in `--flag`, a sweep line that lifts rows by evidence. Real data comes from `layer_results.row_flags`; otherwise a labelled illustrative specimen. It stands where the reference puts its phone render.

## Motion
Easing `cubic-bezier(0.19, 1, 0.22, 1)`, 1.4–2s on landing reveals. Landing: hero parallax and fade, word-by-word paragraph fill, parallax step cards (desktop only), sticky stacking cards, bars that grow in view. App: 250ms route fade. Reduced motion collapses durations and freezes the specimen.

## Pages
- `/` signed out and `/welcome`: public landing page (`pages/LandingPage.jsx`).
- App shell: left rail with labels (the reference's dashboard mock shows an icon rail; labels are kept for nine destinations).

## Components — `src/components/ui.jsx`
Pages build from these; no page keeps its own copy of a control.
- **Layout:** `Page` (1360px container), `PageHeader` (title + grey serif `accent`), `Panel` (28px card), `StatCard` (#1A1A1A step card, display number, optional icon chip), `Reveal`.
- **Actions:** `Button` (`primary` white · `ghost` 6% fill · `danger` red outline; `size` md/sm; `loading` spinner + `aria-busy`), `IconButton` (label required), `ConfirmButton` (two-step destructive action, no modal).
- **Choices:** `Segmented` (toggle pills, `aria-pressed`, white sliding highlight) for filters; `Tabs` + `TabPanel` (same look, real tab semantics with arrow/Home/End keys) when a panel swaps; `RadioCards` for choices that need a sentence each.
- **Forms:** `Field` (label, `hint`, `error` with `aria-invalid`/`aria-describedby`; `as` input/select/textarea; `mono` for data values), styled by `.field-input`.
- **Data:** `Table` (`.data-table`: lavender header labels, hairline rows, `num` right-aligned tabular cells, selectable rows via `tabIndex` + `aria-selected`), `Readout`, `Badge` (tinted pill + dot by tone), `VerdictBadge` (backend verdict code → label and colour; unknown codes stay neutral, never "clean").
- **Feedback:** `Notice` (critical/safe/info, optional dismiss), `ErrorNote`, `Loading` (announced), `Empty` (says why and what to do), `PreviewTag`.
- **Dialog:** native `<dialog>` portalled to `<body>`: blurred backdrop, focus on the first field, Escape and backdrop click close. Only for irreversible actions that also need input (deleting the account).
- **States:** every control has hover, `:focus-visible` (2px lavender ring), active (1px press), disabled (45% opacity, no press) and, where it waits on the API, loading.

## Honesty rules
Heuristic features carry `PreviewTag`. Mock panels on the landing page are labelled "Example run". Benchmark numbers come only from `docs/model-card.md`.
