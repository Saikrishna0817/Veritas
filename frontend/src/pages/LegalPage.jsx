import { Fragment } from 'react';
import { Link } from 'react-router-dom';
import Brand from '../components/Brand';
import { PageTitle } from '../ui/Reveal';
import terms from '../content/legal/terms-of-service.md?raw';
import privacy from '../content/legal/privacy-policy.md?raw';

/**
 * The Terms of Service and the Privacy Policy, shown verbatim from the drafts in docs/legal (copied to
 * content/legal). They are drafts awaiting review: the page says so above the text, and the text says so itself.
 * A small renderer for the Markdown these two files use (headings, quote, paragraphs, tables, bold, code, links);
 * links between the two documents open the other page, links to other docs stay plain text.
 */
const DOCS = { terms: { md: terms, other: ['privacy', 'Privacy Policy'] }, privacy: { md: privacy, other: ['terms', 'Terms of Service'] } };
const ROUTE_OF = { 'privacy-policy.md': '/privacy', 'terms-of-service.md': '/terms' };

function inline(text) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0];
    if (t.startsWith('**')) out.push(<strong key={m.index}>{inline(t.slice(2, -2))}</strong>);
    else if (t.startsWith('`')) out.push(<code key={m.index} className="in-code">{t.slice(1, -1)}</code>);
    else {
      const [, label, href] = t.match(/\[([^\]]+)\]\(([^)]+)\)/);
      const to = ROUTE_OF[href.split('/').pop()];
      out.push(to ? <Link key={m.index} to={to} className="in-link-text">{label}</Link> : <Fragment key={m.index}>{label}</Fragment>);
    }
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function blocks(md) {
  const lines = md.replace(/\r\n/g, '\n').split('\n');
  const out = [];
  let title = '';
  for (let i = 0; i < lines.length; ) {
    const l = lines[i];
    if (!l.trim()) { i++; continue; }
    if (l.startsWith('# ')) { title = l.slice(2).trim(); i++; continue; }
    if (l.startsWith('## ')) { out.push(<h2 key={i} className="in-h2">{inline(l.slice(3))}</h2>); i++; continue; }
    if (l.startsWith('> ')) {
      const start = i;
      const q = [];
      while (i < lines.length && lines[i].startsWith('>')) q.push(lines[i++].replace(/^>\s?/, ''));
      out.push(<blockquote key={start} className="in-legal-quote">{inline(q.join(' '))}</blockquote>);
      continue;
    }
    if (l.startsWith('|')) {
      const start = i;
      const rows = [];
      while (i < lines.length && lines[i].startsWith('|')) rows.push(lines[i++]);
      const cells = (r) => r.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
      const [head, , ...body] = rows;
      out.push(
        <div key={start} className="in-table-wrap">
          <table className="in-table in-table-stack">
            <thead><tr>{cells(head).map((c) => <th key={c}>{inline(c)}</th>)}</tr></thead>
            <tbody>{body.map((r, ri) => <tr key={ri}>{cells(r).map((c, ci) => <td key={ci} data-label={cells(head)[ci]}>{inline(c)}</td>)}</tr>)}</tbody>
          </table>
        </div>,
      );
      continue;
    }
    const start = i;
    const p = [];
    while (i < lines.length && lines[i].trim() && !/^(#|>|\|)/.test(lines[i])) p.push(lines[i++]);
    out.push(<p key={start}>{inline(p.join(' '))}</p>);
  }
  return { title, body: out };
}

export default function LegalPage({ doc }) {
  const { md, other } = DOCS[doc];
  const { title, body } = blocks(md);
  return (
    <main className="instrument in-legal">
      <div className="in-page">
        <Brand />
        <p className="in-legal-note" role="note">Draft awaiting review. This text is not yet in force.</p>
        <PageTitle>{title}</PageTitle>
        <div className="in-prose in-legal-body">{body}</div>
        <div className="in-auth-alt">
          <Link to={`/${other[0]}`} className="in-link-text">{other[1]}</Link>
          <Link to="/" className="in-link-text">Back to Veritas</Link>
        </div>
      </div>
    </main>
  );
}
