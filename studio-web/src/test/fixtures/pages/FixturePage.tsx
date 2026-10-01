import { useRoute } from "../../../router";

/** A page used by the registry fixtures: prints its route title and params. */
export default function FixturePage() {
  const r = useRoute();
  return (
    <p data-testid="fixture-page">
      {r.route?.title} {JSON.stringify(r.params)}
    </p>
  );
}
