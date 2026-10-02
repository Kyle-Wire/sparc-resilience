// Custom multiverse variants: each is a name plus dotted config overrides ("cv.block_m" = 2000),
// sent as custom_variants = {name: {"dotted.key": value}} (api.md §8). Values are read as JSON
// when they parse (numbers, true/false, null, lists), else as text; the JSON sent is shown.
import { Button } from "../../../components/ui/Button";
import { IconButton } from "../../../components/ui/IconButton";
import { customVariantsJson, type CustomVariantDraft } from "../model/multiverse";

export function CustomVariants({ value, onChange }: { value: CustomVariantDraft[]; onChange: (v: CustomVariantDraft[]) => void }) {
  const set = (i: number, d: CustomVariantDraft) => onChange(value.map((x, k) => (k === i ? d : x)));
  const json = customVariantsJson(value);
  return (
    <div className="stack" style={{ gap: 8 }} aria-label="Custom variants" role="group">
      {value.map((d, i) => (
        <div className="mv-variant" key={i} data-variant={i}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <input
              type="text"
              aria-label={`Custom variant ${i + 1} name`}
              placeholder="variant_name"
              value={d.name}
              onChange={(e) => set(i, { ...d, name: e.target.value })}
              style={{ maxWidth: "16em" }}
            />
            <IconButton icon="x" label={`Remove custom variant ${i + 1}`} onClick={() => onChange(value.filter((_, k) => k !== i))} />
          </div>
          {d.rows.map((r, j) => (
            <div className="mv-row" key={j}>
              <input
                type="text"
                aria-label={`Variant ${i + 1} override ${j + 1} key`}
                placeholder="dotted.key (e.g. cv.block_m)"
                value={r.key}
                onChange={(e) => set(i, { ...d, rows: d.rows.map((x, k) => (k === j ? { ...x, key: e.target.value } : x)) })}
              />
              <input
                type="text"
                aria-label={`Variant ${i + 1} override ${j + 1} value`}
                placeholder="value (2000, false, [1.0], text)"
                value={r.value}
                onChange={(e) => set(i, { ...d, rows: d.rows.map((x, k) => (k === j ? { ...x, value: e.target.value } : x)) })}
              />
              <IconButton
                icon="minus"
                label={`Remove override ${j + 1} of variant ${i + 1}`}
                onClick={() => set(i, { ...d, rows: d.rows.filter((_, k) => k !== j) })}
                disabled={d.rows.length <= 1}
              />
            </div>
          ))}
          <div>
            <Button size="small" icon="plus" onClick={() => set(i, { ...d, rows: [...d.rows, { key: "", value: "" }] })}>
              Add override
            </Button>
          </div>
        </div>
      ))}
      <div>
        <Button size="small" icon="plus" onClick={() => onChange([...value, { name: "", rows: [{ key: "", value: "" }] }])}>
          Add custom variant
        </Button>
      </div>
      {value.length ? (
        <details>
          <summary className="cap">JSON sent as custom_variants</summary>
          <pre className="mv-json" data-testid="custom-variants-json">
            {JSON.stringify(json.value, null, 2)}
          </pre>
        </details>
      ) : null}
    </div>
  );
}
