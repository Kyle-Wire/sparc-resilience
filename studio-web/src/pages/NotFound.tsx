import { Link, useRoute } from "../router";

/** The `*` route: the URL matches no registered page. */
export default function NotFound() {
  const { pathname } = useRoute();
  return (
    <section className="page-head" aria-labelledby="nf-title">
      <p className="eyebrow">404</p>
      <h1 id="nf-title">Page not found</h1>
      <p className="prose">
        Nothing in Studio lives at <code>{pathname}</code>. The link may be from an older version, or the run or project was deleted.
      </p>
      <p className="row">
        <Link to="/" className="btn primary">
          Go home
        </Link>
        <Link to="/jobs" className="btn">
          Activity
        </Link>
      </p>
    </section>
  );
}
