import { Link, useLocation } from 'react-router-dom';
import { SearchX } from 'lucide-react';
import { Empty, Page, PageHeader } from '../components/ui';

/** An address inside the app that no page answers. Says so, instead of quietly showing the Overview. */
export default function NotFoundPage() {
  const { pathname } = useLocation();
  return (
    <Page>
      <PageHeader title="Nothing" accent="here" />
      <Empty
        icon={SearchX}
        title="No page at this address"
        action={<Link to="/" className="inline-flex min-h-[44px] items-center text-sm text-violet hover:underline">Go to the Overview →</Link>}
      >
        <span className="font-mono text-sm break-all">{pathname}</span> is not a Veritas page. The link may be old, or the address mistyped.
      </Empty>
    </Page>
  );
}
