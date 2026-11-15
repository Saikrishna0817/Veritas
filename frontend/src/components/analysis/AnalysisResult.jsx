import { useMemo, useState, lazy, Suspense } from 'react';
import { motion } from 'framer-motion';
import { Download, RotateCcw, AlertTriangle } from 'lucide-react';
import {
  Button, Panel, Readout, Table, Badge, downloadJson, EASE, DURATION, verdictInfo, orDash,
} from '../ui';
import { DATA_KINDS, ATTACK_NAMES, PROFILE_FEATURES, CHECK_NAMES, checkName, kindOf } from './kinds';

const Specimen3D = lazy(() => import('../Specimen3D'));

const SCALE_MAX = 8;
const TICKS = [['0.05', 1.3], ['0.01', 2], ['1e-4', 4]];   // the verdict cut-offs as false-alarm levels
const SHOW_FIRST = 25;

function downloadCsv(rows, filename) {
  const esc = (x) => `"${String(x ?? '').replace(/"/g, '""')}"`;
  const url = URL.createObjectURL(new Blob([rows.map((r) => r.map(esc).join(',')).join('\n')], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

const baseName = (name = 'batch') => name.replace(/\.(csv|zip)$/i, '');

// ── Evidence ──────────────────────────────────────────────────────────────────────────────────────

function EvidenceBar({ check }) {
  const [name, what] = CHECK_NAMES[check.key] || [checkName(check.key), ''];
  const reported = check.evidence != null;
  const ev = reported ? check.evidence : 0;
  const width = Math.min(ev / SCALE_MAX, 1) * 100;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="min-w-0">
          <span className="font-semibold">{name}</span>
          {what && <span className="block text-xs text-textMuted">{what}</span>}
        </span>
        <span className="shrink-0 text-right font-mono text-sm text-textSecondary">
          {check.ran === false ? 'skipped (one label value)' : !reported ? 'not reported' : <>−log₁₀ p = <span className={check.alarm ? 'text-flag' : ''}>{ev >= SCALE_MAX ? `>${SCALE_MAX}` : ev.toFixed(2)}</span></>}
          {check.inRanking != null && <span className="block text-[11px] text-textMuted">{check.inRanking ? 'in the ranking' : 'not in the ranking'}</span>}
        </span>
      </div>
      <div className="relative mt-2 h-3 rounded-sm bg-ink-950">
        {TICKS.map(([, t]) => (
          <span key={t} className="absolute inset-y-[-3px] w-px bg-borderStrong" style={{ left: `${(t / SCALE_MAX) * 100}%` }} aria-hidden />
        ))}
        <motion.div
          className="h-full origin-left rounded-sm"
          style={{ width: `${width}%`, background: check.alarm ? 'var(--flag)' : 'var(--violet)' }}
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: DURATION.grow, ease: EASE }}
        />
      </div>
      <div className="relative mt-1 h-4 font-mono text-[10px] text-textMuted" aria-hidden>
        {TICKS.map(([l, t]) => (
          <span key={l} className="absolute -translate-x-1/2" style={{ left: `${(t / SCALE_MAX) * 100}%` }}>{l}</span>
        ))}
      </div>
    </div>
  );
}

function EvidencePanel({ checks, title = 'Evidence from each check', note }) {
  if (!checks.length) return null;
  return (
    <Panel title={title}>
      <div className="space-y-6">
        {checks.map((c) => <EvidenceBar key={c.key} check={c} />)}
      </div>
      <p className="mt-4 text-xs text-textMuted">
        Each bar is −log₁₀ of a p-value measured against your reference; ticks mark the verdict cut-offs. {note}
      </p>
    </Panel>
  );
}

// ── Lists to review ───────────────────────────────────────────────────────────────────────────────

function ShowMore({ total, open, onToggle }) {
  if (total <= SHOW_FIRST) return null;
  return (
    <button type="button" onClick={onToggle} className="inline-flex min-h-[44px] items-center text-sm text-violet hover:text-[var(--red-bright)]">
      {open ? `Show the first ${SHOW_FIRST}` : `Show all ${total.toLocaleString()}`}
    </button>
  );
}

/** Table rows from scores alone (results saved before suspicious cells existed, or a clean batch). */
function RankedRows({ scores, flagged, offset }) {
  const [showAll, setShowAll] = useState(false);
  const rows = useMemo(() => {
    const flagSet = new Set(flagged);
    return scores.map((s, i) => ({ i, s, f: flagSet.has(i) })).sort((a, b) => b.s - a.s);
  }, [scores, flagged]);
  const visible = rows.slice(0, showAll ? 200 : SHOW_FIRST);
  const max = rows[0]?.s || 1;
  return (
    <Panel
      title="Rows to review first"
      aside={<span className="font-mono text-xs text-textMuted">{flagged.length} flagged · {scores.length.toLocaleString()} ranked</span>}
    >
      <p className="-mt-2 mb-4 max-w-[70ch] text-sm text-textSecondary">
        Ranked by row evidence: −log₁₀ of the share of clean reference rows that score at least as high. 2.0 means the row is in the clean top 1%.
        Row numbers are positions in your file.
      </p>
      <Table minWidth={420} className="-mx-3">
        <thead>
          <tr><th>Rank</th><th>File row</th><th>Evidence</th><th className="w-[40%]"><span className="sr-only">Evidence bar</span></th></tr>
        </thead>
        <tbody className="font-mono">
          {visible.map((r, k) => (
            <tr key={r.i}>
              <td className="py-2 text-textMuted">{k + 1}</td>
              <td className="py-2">{(r.i + offset + 1).toLocaleString()}</td>
              <td className={`py-2 ${r.f ? 'text-flag' : 'text-textSecondary'}`}>{r.s.toFixed(2)}{r.f && <Badge tone="flag" dot={false} className="ml-2 font-sans">flagged</Badge>}</td>
              <td className="py-2">
                <div className="h-1.5 rounded-sm bg-ink-950"><div className="h-full rounded-sm" style={{ width: `${(r.s / max) * 100}%`, background: r.f ? 'var(--flag)' : '#4A494B' }} /></div>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
      {rows.length > SHOW_FIRST && (
        <button type="button" onClick={() => setShowAll((x) => !x)} className="mt-2 inline-flex min-h-[44px] items-center text-sm text-violet hover:text-[var(--red-bright)]">
          {showAll ? `Show top ${SHOW_FIRST}` : `Show top ${Math.min(rows.length, 200)}`}
        </button>
      )}
    </Panel>
  );
}

/** Table rows with their line in the user's file and the cells most to blame, as the user wrote them. */
function SuspiciousCells({ rows, title = 'Rows to review', lede, filename }) {
  const [open, setOpen] = useState(false);
  const shown = open ? rows : rows.slice(0, SHOW_FIRST);
  const exportCsv = () => downloadCsv(
    [['line', 'row_score', 'column', 'value', 'cell_score'], ...rows.flatMap((r) => r.cells.map((c) => [r.csv_line, r.row_score, c.column, c.value, c.score]))],
    `veritas_${baseName(filename)}_rows.csv`,
  );
  return (
    <Panel
      title={title}
      aside={<Button variant="ghost" size="sm" className="min-h-[44px]" onClick={exportCsv}><Download className="h-4 w-4" /> CSV</Button>}
    >
      <p className="-mt-2 mb-4 max-w-[70ch] text-sm text-textSecondary">
        {lede || 'Most suspicious first.'} Line numbers are lines in your file (line 1 is the header); values are as you wrote them.
      </p>
      <Table minWidth={520} className="-mx-3">
        <thead>
          <tr><th className="num">Line</th><th className="num">Row score</th><th>Cells most to blame (column, value, score)</th></tr>
        </thead>
        <tbody>
          {shown.map((r) => (
            <tr key={`${r.csv_line}-${r.row}`}>
              <td className="num py-2 font-mono">{r.csv_line?.toLocaleString() ?? '—'}</td>
              <td className="num py-2 font-mono text-flag">{r.row_score?.toFixed(2) ?? '—'}</td>
              <td className="py-2">
                <div className="flex flex-wrap gap-1.5">
                  {r.cells.map((c) => (
                    <span key={c.column} className="inline-flex items-baseline gap-1.5 rounded-lg border border-borderHairline bg-ink-950 px-2 py-0.5 text-xs">
                      <span className={c.is_label ? 'text-uv' : 'text-textSecondary'}>{c.column}</span>
                      <span className="max-w-[18ch] truncate font-mono text-textPrimary">{String(c.value)}</span>
                      <span className="font-mono text-[10px] text-textMuted">{c.score?.toFixed(1)}</span>
                    </span>
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-3 text-xs text-textMuted">
        <span>{rows.length.toLocaleString()} listed</span>
        <ShowMore total={rows.length} open={open} onToggle={() => setOpen((x) => !x)} />
      </div>
    </Panel>
  );
}

/** Image, audio or video files to review: path in the zip, class folder, score, and the check that flagged it. */
function FlaggedItems({ items, total, kind, title, lede, filename }) {
  const [open, setOpen] = useState(false);
  const shown = open ? items : items.slice(0, SHOW_FIRST);
  const exportCsv = () => downloadCsv(
    [['file', 'class', 'score', 'check'], ...items.map((r) => [r.file, r.class, r.score, r.check])],
    `veritas_${baseName(filename)}_${kind}.csv`,
  );
  return (
    <Panel
      title={title}
      aside={items.length > 0 && <Button variant="ghost" size="sm" className="min-h-[44px]" onClick={exportCsv}><Download className="h-4 w-4" /> CSV</Button>}
    >
      {lede && <p className="-mt-2 mb-4 max-w-[70ch] text-sm text-textSecondary">{lede}</p>}
      {items.length === 0 ? (
        <p className="text-sm text-textSecondary">
          Nothing is listed. Files are listed only when the batch reaches the 1-in-100 false-alarm level, so a weak result does not send you chasing noise.
        </p>
      ) : (
        <>
          <Table minWidth={520} className="-mx-3">
            <thead><tr><th>File</th><th>Class</th><th className="num">Score</th><th>Flagged by</th></tr></thead>
            <tbody>
              {shown.map((r) => (
                <tr key={r.file}>
                  <td className="max-w-[320px] truncate py-2 font-mono text-sm" title={r.file}>{r.file}</td>
                  <td className="nowrap py-2 text-textSecondary">{r.class ?? 'none'}</td>
                  <td className="num py-2 font-mono text-flag">{r.score?.toFixed(2)}</td>
                  <td className="nowrap py-2 text-textSecondary">{checkName(r.check)}</td>
                </tr>
              ))}
            </tbody>
          </Table>
          <div className="mt-2 flex flex-wrap items-center justify-between gap-3 text-xs text-textMuted">
            <span>{items.length.toLocaleString()} listed{total > items.length ? ` of ${total.toLocaleString()} flagged` : ''}, most suspicious first</span>
            <ShowMore total={items.length} open={open} onToggle={() => setOpen((x) => !x)} />
          </div>
        </>
      )}
    </Panel>
  );
}

// ── Attack profile (tables) ───────────────────────────────────────────────────────────────────────

function AttackProfile({ profile }) {
  const probs = Object.entries(profile.probabilities || {}).sort((a, b) => b[1] - a[1]);
  return (
    <Panel title="What the suspicious rows look like">
      <div className="font-display text-2xl font-medium">{ATTACK_NAMES[profile.likely] || profile.label || profile.likely}</div>
      <p className="mt-2 text-sm text-textSecondary">
        The closest of the six attacks Veritas was benchmarked on. On held-out data it names the exact attack 74% of the time and the right kind 86%.
        An attack of another kind still gets one of these names.
      </p>
      <ul className="mt-5 space-y-2.5">
        {probs.map(([k, p]) => (
          <li key={k} className="grid grid-cols-[minmax(0,1fr)_minmax(0,0.8fr)_3.2em] items-center gap-3 text-sm">
            <span className={k === profile.likely ? 'text-textPrimary' : 'text-textSecondary'}>{ATTACK_NAMES[k] || k}</span>
            <span className="h-1.5 rounded-sm bg-ink-950" aria-hidden>
              <motion.span
                className="block h-full origin-left rounded-sm"
                style={{ width: `${Math.max(p * 100, 1)}%`, background: k === profile.likely ? 'var(--uv)' : '#4A494B' }}
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: DURATION.grow, ease: EASE }}
              />
            </span>
            <span className="text-right font-mono tabular text-textSecondary">{Math.round(p * 100)}%</span>
          </li>
        ))}
      </ul>
      {profile.features && (
        <details className="mt-5 text-sm">
          <summary className="inline-flex min-h-[44px] cursor-pointer items-center text-violet">What the suspicious rows have in common</summary>
          <dl className="mt-2 grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1.5">
            {Object.entries(profile.features).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-textSecondary">{PROFILE_FEATURES[k] || k}</dt>
                <dd className="text-right font-mono">{v}</dd>
              </div>
            ))}
          </dl>
        </details>
      )}
    </Panel>
  );
}

// ── Shared pieces ─────────────────────────────────────────────────────────────────────────────────

function Hero({ eyebrow, title, color, blurb, readouts, actions, visual }) {
  return (
    <section className="edge-card"><div className={`edge-inner relative grid ${visual ? 'xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]' : ''}`}>
      <div aria-hidden className="pointer-events-none absolute inset-0" style={{ background: 'radial-gradient(55% 80% at 80% 60%, rgba(71,0,214,0.5), transparent 70%)' }} />
      <div className="relative z-10 p-8 md:p-12">
        <div className="rule-label break-all">{eyebrow}</div>
        <motion.div
          initial={{ opacity: 0.5, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: DURATION.slow, ease: EASE }}
          className="mt-3 font-display text-[clamp(2.4rem,5vw,4.4rem)] font-medium leading-[0.9]"
          style={{ color }}
        >
          {title}
        </motion.div>
        <p className="mt-4 max-w-[52ch] text-textSecondary">{blurb}</p>
        <div className={`mt-8 grid grid-cols-2 gap-6 ${visual ? '' : 'md:grid-cols-4'}`}>{readouts}</div>
        <div className="mt-8 flex flex-wrap gap-3">{actions}</div>
      </div>
      {visual}
    </div></section>
  );
}

function Actions({ result, onReset, resetLabel }) {
  const info = result.dataset_info || {};
  return (
    <>
      {onReset && <Button variant="ghost" onClick={onReset}><RotateCcw className="h-4 w-4" /> {resetLabel}</Button>}
      <Button variant="ghost" onClick={() => downloadJson(result, `veritas_${baseName(info.filename)}_${Date.now()}.json`)}>
        <Download className="h-4 w-4" /> Download JSON
      </Button>
    </>
  );
}

function Notes({ notes }) {
  if (!notes.length) return null;
  return (
    <div className="rounded-2xl border border-statusWarn/40 bg-statusWarn/[0.06] p-4 text-sm text-[#FFD9AE]">
      {notes.map((w) => <div key={w} className="flex gap-2"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />{w}</div>)}
    </div>
  );
}

function Limits({ kind }) {
  return (
    <Panel title="What this check cannot tell you">
      <ul className="space-y-2 text-sm text-textSecondary">
        {DATA_KINDS[kind].limits.map((l) => <li key={l} className="flex gap-2"><span className="text-uv" aria-hidden>·</span>{l}</li>)}
      </ul>
    </Panel>
  );
}

const seconds = (ms) => (ms ? (ms / 1000).toFixed(1) : '—');
const count = (n) => orDash(n, (x) => x.toLocaleString());

// ── Results ───────────────────────────────────────────────────────────────────────────────────────

function TableResult({ result, onReset, resetLabel }) {
  const v = verdictInfo(result.verdict);
  const lr = result.layer_results || {};
  const flags = lr.row_flags || {};
  const info = result.dataset_info || {};
  const cells = result.suspicious_cells || [];
  const profile = result.attack_profile || lr.attack_profile;
  const cls = result.attack_classification;
  const notes = [...new Set([...(info.warnings || []), ...(result.warnings || [])])];
  const checks = ['label_check', 'shift_check', 'resid_check'].filter((k) => lr[k]).map((k) => ({
    key: k, evidence: lr[k].evidence, alarm: lr[k].alarm, ran: lr[k].ran,
    inRanking: k === 'label_check' ? undefined : lr[k].in_row_ranking,
  }));

  return (
    <div className="space-y-8">
      <Hero
        eyebrow={info.filename}
        title={v.label}
        color={v.color}
        blurb={v.blurb}
        readouts={(
          <>
            <Readout label="Batch evidence" value={v.p} />
            <Readout label="Flagged rows" value={(flags.flagged_indices_incoming || []).length} color="var(--flag)" note={`of ${count(result.n_samples)} checked`} />
            <Readout label="Mode" value={result.detection_mode === 'supervised' || info.has_labels ? 'Labelled' : 'Unlabelled'} note={info.label_column ? `label: ${info.label_column}` : 'shift check only'} />
            <Readout label="Time" value={seconds(result.elapsed_ms)} unit="s" />
          </>
        )}
        actions={<Actions result={result} onReset={onReset} resetLabel={resetLabel} />}
        visual={(
          <div className="relative min-h-[360px] border-t border-borderHairline xl:border-l xl:border-t-0">
            <Suspense fallback={null}>
              <Specimen3D className="absolute inset-0" scores={flags.row_scores} flagged={flags.flagged_indices_incoming} />
            </Suspense>
            <span className="pointer-events-none absolute bottom-3 right-4 font-mono text-[11px] text-textMuted">your rows · height = evidence · move the pointer to tilt</span>
          </div>
        )}
      />

      <Notes notes={notes} />

      <div className="grid gap-8 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <div className="min-w-0 space-y-8">
          {cells.length > 0
            ? <SuspiciousCells rows={cells} filename={info.filename} lede="The flagged rows, most suspicious first, with the cells most to blame." />
            : flags.row_scores?.length
              ? <RankedRows scores={flags.row_scores} flagged={flags.flagged_indices_incoming || []} offset={info.reference_split || 0} />
              : null}
        </div>
        <div className="min-w-0 space-y-8">
          <EvidencePanel
            checks={checks}
            title={checks.length > 2 ? 'Three checks, one verdict' : 'Two checks, one verdict'}
            note="The verdict uses the strongest check. The shift and column checks join the row ranking only when their own evidence is strong."
          />
          {profile?.probabilities ? <AttackProfile profile={profile} /> : cls?.attack_type && cls.attack_type !== 'unknown' && (
            <Panel title="What the suspicious rows look like">
              <div className="font-display text-2xl font-medium">{ATTACK_NAMES[cls.attack_type] || cls.attack_type.replace(/_/g, ' ')}</div>
              {cls.description && <p className="mt-2 text-sm text-textSecondary">{cls.description}</p>}
            </Panel>
          )}
          <Panel title="Schema">
            <dl className="grid grid-cols-2 gap-4 text-sm">
              {[
                ['Rows in file', info.n_rows?.toLocaleString()],
                ['Features', info.n_features],
                ['Label column', info.label_column || 'none'],
                ['Reference rows', info.reference_split?.toLocaleString()],
              ].map(([k, val]) => (
                <div key={k}><dt className="rule-label">{k}</dt><dd className="mt-1 break-all font-mono">{val ?? '—'}</dd></div>
              ))}
            </dl>
            {info.feature_names?.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-1.5">
                {info.feature_names.slice(0, 40).map((f) => (
                  <span key={f} className="rounded-lg border border-borderHairline bg-ink-950 px-2 py-0.5 font-mono text-[11px] text-textSecondary">{f}</span>
                ))}
                {info.feature_names.length > 40 && <span className="px-2 py-0.5 font-mono text-[11px] text-textMuted">+{info.feature_names.length - 40}</span>}
              </div>
            )}
          </Panel>
          <Limits kind="csv" />
        </div>
      </div>
    </div>
  );
}

function MediaResult({ result, kind, onReset, resetLabel }) {
  const v = verdictInfo(result.verdict);
  const spec = DATA_KINDS[kind];
  const info = result.dataset_info || {};
  const items = result.flagged_images || result.flagged_clips || [];
  const checks = Object.entries(result.checks || {}).map(([key, c]) => ({ key, evidence: c.evidence, alarm: c.evidence >= 2, inRanking: c.in_row_ranking }));
  return (
    <div className="space-y-8">
      <Hero
        eyebrow={`${spec.label} · ${info.filename || ''}`}
        title={v.label}
        color={v.color}
        blurb={v.blurb}
        readouts={(
          <>
            <Readout label="Batch evidence" value={v.p} />
            <Readout label={`Flagged ${spec.noun}`} value={result.n_flagged ?? items.length} color="var(--flag)" note={`of ${count(info.n_incoming ?? result.n_samples)} checked`} />
            <Readout label="Reference" value={count(info.n_reference)} note={`trusted ${spec.noun}`} />
            <Readout label="Time" value={seconds(result.elapsed_ms)} unit="s" />
          </>
        )}
        actions={<Actions result={result} onReset={onReset} resetLabel={resetLabel} />}
      />

      <Notes notes={info.warnings || []} />

      <div className="grid gap-8 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <div className="min-w-0">
          <FlaggedItems
            items={items}
            total={result.n_flagged ?? items.length}
            kind={kind}
            filename={info.filename}
            title={`${spec.label} to review`}
            lede={result.leading_check ? `The strongest check was ${checkName(result.leading_check).toLowerCase()}. Paths are as they are in your zip.` : 'Paths are as they are in your zip.'}
          />
        </div>
        <div className="min-w-0 space-y-8">
          <EvidencePanel checks={checks} note="The verdict uses the strongest check; only checks marked as in the ranking order the list." />
          <Panel title="What was read">
            <dl className="grid grid-cols-2 gap-4 text-sm">
              {[
                ['Classes', info.class_names?.length],
                ['Labelled', info.has_labels ? 'yes, by folder' : 'no'],
                ['View', info.image_size || info.view],
                info.clip_seconds != null && ['Clip length', `${info.clip_seconds} s`],
              ].filter(Boolean).map(([k, val]) => (
                <div key={k}><dt className="rule-label">{k}</dt><dd className="mt-1 break-words font-mono">{val ?? '—'}</dd></div>
              ))}
            </dl>
            {info.class_names?.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-1.5">
                {info.class_names.slice(0, 40).map((c) => (
                  <span key={c} className="rounded-lg border border-borderHairline bg-ink-950 px-2 py-0.5 font-mono text-[11px] text-textSecondary">{c}</span>
                ))}
                {info.class_names.length > 40 && <span className="px-2 py-0.5 font-mono text-[11px] text-textMuted">+{info.class_names.length - 40}</span>}
              </div>
            )}
          </Panel>
          <Limits kind={kind} />
        </div>
      </div>
    </div>
  );
}

/** No trusted reference: one ranked list per check, deliberately without a verdict. */
function Screening({ result, kind, onReset, resetLabel }) {
  const v = verdictInfo('SCREENED');
  const spec = DATA_KINDS[kind];
  const info = result.dataset_info || {};
  const lists = Object.entries(result.lists || {});
  return (
    <div className="space-y-8">
      <Hero
        eyebrow={`${spec.label} · ${info.filename || ''}`}
        title={v.label}
        color={v.color}
        blurb={`Every one of the ${count(result.n_samples)} ${spec.noun} was checked against the others. This ranks what to look at first; without data you trust it cannot say whether the set is poisoned.`}
        readouts={(
          <>
            <Readout label={`${spec.noun.charAt(0).toUpperCase()}${spec.noun.slice(1)} checked`} value={count(result.n_samples)} />
            <Readout label="Ranked lists" value={lists.length} note="one per check" />
            <Readout label="Reference" value="none" note="self-check, 5 folds" />
            <Readout label="Time" value={seconds(result.elapsed_ms)} unit="s" />
          </>
        )}
        actions={<Actions result={result} onReset={onReset} resetLabel={resetLabel} />}
      />

      <Notes notes={info.warnings || []} />

      <Panel title="How to read a screening">
        <ul className="space-y-2 text-sm text-textSecondary">
          <li className="flex gap-2"><span className="text-uv" aria-hidden>·</span>Label flips and trigger patterns are found this way; changes that look like ordinary data (feature collisions, a shifted source) are not.</li>
          <li className="flex gap-2"><span className="text-uv" aria-hidden>·</span>The top of each list is worth a look even when nothing is wrong: compare a few items with ones you know.</li>
        </ul>
      </Panel>

      {lists.map(([key, items]) => {
        const title = result.checks?.[key]?.description || checkName(key);
        const lede = `The ${items.length.toLocaleString()} highest-ranked ${spec.noun} for this check, most unusual first.`;
        return kind === 'csv'
          ? <SuspiciousCells key={key} rows={items} title={title} lede={lede} filename={`${info.filename}_${key}`} />
          : <FlaggedItems key={key} items={items} total={items.length} kind={kind} title={title} lede={lede} filename={`${info.filename}_${key}`} />;
      })}
    </div>
  );
}

/**
 * Any analysis result: a table batch, an image/audio/video zip, or a screening without a reference.
 * onReset (optional) adds a button that starts over with `resetLabel`.
 */
export default function AnalysisResult({ result, onReset, resetLabel = 'Examine another batch' }) {
  const kind = kindOf(result);
  if (result.mode === 'self_check') return <Screening result={result} kind={kind} onReset={onReset} resetLabel={resetLabel} />;
  if (kind !== 'csv') return <MediaResult result={result} kind={kind} onReset={onReset} resetLabel={resetLabel} />;
  return <TableResult result={result} onReset={onReset} resetLabel={resetLabel} />;
}
