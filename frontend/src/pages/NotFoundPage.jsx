import { Link, useLocation } from 'react-router-dom';
import { PageIntro, PageTitle } from '../ui/Reveal';

/** An address the app does not know. It used to redirect silently to the Overview; now it says so. */
export default function NotFoundPage() {
  const { pathname } = useLocation();
  return (
    <main className="in-main"><div className="in-page">
      <p className="in-h3">Not found</p>
      <PageTitle>This page does not exist.</PageTitle>
      <PageIntro>{`Nothing lives at ${pathname}. It may have moved, or the link may be mistyped.`}</PageIntro>
      <div className="in-actions">
        <Link to="/" className="in-btn" style={{ textDecoration: 'none' }}>Go to the Overview</Link>
        <Link to="/history" className="in-link-text">Open History</Link>
      </div>
    </div></main>
  );
}
