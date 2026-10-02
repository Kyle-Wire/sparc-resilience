// Contract check (SPEC §13.3, §14.4): the hand-written wire types of src/api/*.ts (and the two
// shell resources of src/layouts/resources.ts) against the types generated from the server's
// OpenAPI document (schema.gen.ts, `npm run gen:api`).
//
// Type-only: nothing here is imported by the app, bundled or run. `npm run typecheck` fails when
// a check below does not resolve to `true`; the error names the endpoint and field, e.g.
//   Type '"GET /api/jobs/{jid}.run_id: source may be null"' does not satisfy the constraint 'true'.
//
// How a pair is compared (`Fit<Source, Target>`, recursive, field by field):
// - Responses (`Out`): the source is the server's model as it goes on the wire (`Wire<…>`: FastAPI
//   serialises every field, defaults included) and the target is the client type. The client
//   type must declare exactly the server's fields (no client-only fields, none missing), and
//   every server value must fit the client field: a nullable server field needs a nullable (or
//   optional) client field, arrays stay arrays, primitive types match. The server may be less
//   precise than the client: `str` against a literal union the client takes from api.md, a free
//   `dict` against a structured client object, a typed `dict[str, X]` against fixed client keys.
//   When both sides are literal unions, every server value must be in the client's union.
// - Subset views (`View`): as `Out`, but the client type may leave server fields out (types that
//   are documented as partial views of a larger reply, e.g. `RunDetailLite`).
// - Requests (`In`): the source is the client's body type and the target the server's model.
//   Every client field must exist on the server model with a compatible type, and a client field
//   that can be null must be nullable on the server. Client omissions are not checked: the
//   generated types mark fields with a default as required (openapi-typescript's
//   `defaultNonNullable`), so a defaulted field cannot be told from a required one here.
// - Server replies the OpenAPI document leaves as free objects (run view sections, input views,
//   study views) only check their envelope; their bodies are covered by the backend view tests
//   and the e2e suite.
//
// Known differences are allowed per entry with `Gate<Result, "exact error">` and a reason; an
// allowance that no longer matches an error fails as stale, so fixing the drift forces its
// removal. Each one is reported to the owning work item.
import type { components, paths } from "./schema.gen";
import type * as A from "./analysis";
import type * as E from "./exports";
import type * as F from "./findings";
import type * as I from "./inputs";
import type * as L from "./lab";
import type * as P from "./projects";
import type * as R from "./runs";
import type * as St from "./studies";
import type * as T from "./tracking";
import type * as C from "./types";
import type { ProjectDetail as ShellProjectDetail, RunDetailHead } from "../layouts/resources";

type S = components["schemas"];

// ---------------------------------------------------------------- type utilities

type IsAny<X> = 0 extends 1 & X ? true : false;
type IsUnknown<X> = IsAny<X> extends true ? false : unknown extends X ? true : false;
type IsNever<X> = [X] extends [never] ? true : false;
type Same<X, Y> = [X] extends [Y] ? ([Y] extends [X] ? true : false) : false;
type Bare<X> = Exclude<X, null | undefined>;
type Prim = string | number | boolean | bigint;
type Widen<X> = X extends string ? string : X extends number ? number : X extends boolean ? boolean : X;
type Depth = unknown[];
type Mode = "out" | "view" | "in";

/** The object without its index signatures. */
type Known<X> = { [K in keyof X as string extends K ? never : number extends K ? never : symbol extends K ? never : K]: X[K] };
type KKeys<X> = keyof Known<X> & string;
type HasIndex<X> = string extends keyof X ? true : false;
type IndexValue<X> = X[string & keyof X];
/** A free object (`dict[str, Any]`, `Record<string, unknown>`): an index signature with unknown values and no known keys. */
type IsOpaque<X> = X extends readonly unknown[]
  ? false
  : X extends object
    ? HasIndex<X> extends true
      ? IsNever<KKeys<X>> extends true
        ? IsUnknown<IndexValue<X>> extends true ? true : IsAny<IndexValue<X>>
        : false
      : false
    : false;
/** A typed dict (`dict[str, X]`): an index signature with a known value type. */
type IsTypedDict<X> = HasIndex<X> extends true
  ? IsUnknown<IndexValue<X>> extends true ? false : IsAny<IndexValue<X>> extends true ? false : true
  : false;

/** A response model as serialised: every field present (FastAPI includes defaults). */
type Wire<X> = IsAny<X> extends true
  ? X
  : X extends Prim | null | undefined
    ? X
    : X extends readonly unknown[]
      ? { [K in keyof X]: Wire<X[K]> }
      : X extends object
        ? { [K in keyof X]-?: Wire<Exclude<X[K], undefined>> }
        : X;

type Errs<X> = Exclude<X[keyof X], true>;
/** Error unions are re-created member by member so compiler messages list them instead of an alias name. */
type Flat<Er> = Er extends string ? `${Er}` : Er;
type Collect<Er> = IsNever<Er> extends true ? true : Flat<Er>;

// ---------------------------------------------------------------- the comparison

/** Recursive client types compared by their own entry; nested occurrences are not walked again. */
type Separate<H> = Same<Bare<H>, C.SelectionSpec>;

type Lax<M extends Mode> = M extends "in" ? false : true;

type Fit<Src, Tgt, Pth extends string, M extends Mode, D extends Depth = []> =
  D["length"] extends 10 ? true
  : D["length"] extends 0 ? Fit1<Src, Tgt, Pth, M, D>
  : Separate<M extends "in" ? Src : Tgt> extends true ? Nullability<Src, Tgt, Pth, M>
  : Fit1<Src, Tgt, Pth, M, D>;

/** Only the null / absent rules (for types whose body is compared by their own entry). */
type Nullability<Src, Tgt, Pth extends string, M extends Mode> =
  null extends Src
    ? null extends Tgt ? true : Lax<M> extends true ? (undefined extends Tgt ? true : `${Pth}: source may be null`) : `${Pth}: source may be null`
  : undefined extends Src
    ? undefined extends Tgt ? true : Lax<M> extends true ? `${Pth}: source may omit it` : true
  : true;

type Fit1<Src, Tgt, Pth extends string, M extends Mode, D extends Depth> =
  IsAny<Tgt> extends true ? true
  : IsUnknown<Tgt> extends true ? true
  : IsAny<Src> extends true ? true
  : IsUnknown<Src> extends true ? (Lax<M> extends true ? true : `${Pth}: unknown on the client`)
  // plain assignability settles scalars only: for objects it would accept extra or missing optional fields
  : [Src] extends [Prim | null | undefined] ? ([Src] extends [Tgt] ? true : Fit2<Src, Tgt, Pth, M, D>)
  : Fit2<Src, Tgt, Pth, M, D>;

type Fit2<Src, Tgt, Pth extends string, M extends Mode, D extends Depth> =
  null extends Src
    ? null extends Tgt ? FitBare<Bare<Src>, Bare<Tgt>, Pth, M, D>
      // a server null reads as "absent" for an optional client field
      : Lax<M> extends true ? (undefined extends Tgt ? FitBare<Bare<Src>, Bare<Tgt>, Pth, M, D> : `${Pth}: source may be null`)
      : `${Pth}: source may be null`
  : undefined extends Src
    ? undefined extends Tgt ? FitBare<Bare<Src>, Bare<Tgt>, Pth, M, D>
      // requests: an optional client field may still go to a server field with a default
      : Lax<M> extends true ? `${Pth}: source may omit it` : FitBare<Bare<Src>, Bare<Tgt>, Pth, M, D>
  : FitBare<Bare<Src>, Bare<Tgt>, Pth, M, D>;

type FitBare<Sx, Tx, Pth extends string, M extends Mode, D extends Depth> =
  IsNever<Sx> extends true ? true
  : IsNever<Tx> extends true ? `${Pth}: not accepted by the target`
  : [Sx] extends [Prim] ? ([Sx] extends [Tx] ? true : FitPrim<Sx, Tx, Pth, M>)
  : [Sx] extends [readonly unknown[]] ? FitBare2<Sx, Tx, Pth, M, D>
  : [Sx] extends [object] ? FitBare2<Sx, Tx, Pth, M, D>
  // a union mixing scalars, arrays and objects: each kind against the target's members of that kind
  : Collect<Exclude<
      | FitBare<Extract<Sx, Prim>, Extract<Tx, Prim>, Pth, M, D>
      | FitBare<Extract<Sx, readonly unknown[]>, Extract<Tx, readonly unknown[]>, Pth, M, D>
      | FitBare<Exclude<Sx, Prim | readonly unknown[]>, Exclude<Tx, Prim | readonly unknown[]>, Pth, M, D>,
      true>>;

type FitPrim<Sx, Tx, Pth extends string, M extends Mode> =
  Lax<M> extends true
    ? [Widen<Sx>] extends [Sx]
      ? ([Sx] extends [Widen<Tx>] ? true : `${Pth}: primitive type differs`)
      : `${Pth}: values differ`
    : `${Pth}: primitive type differs`;

type FitBare2<Sx, Tx, Pth extends string, M extends Mode, D extends Depth> =
  IsOpaque<Sx> extends true
    ? Lax<M> extends true ? ([Tx] extends [object] ? true : `${Pth}: object vs non-object`)
      : IsOpaque<Tx> extends true ? true : `${Pth}: free object on the client`
  : [Sx] extends [readonly unknown[]]
    ? [Tx] extends [readonly unknown[]] ? Fit<Sx[number], Tx[number], `${Pth}[]`, M, [...D, 1]> : `${Pth}: array vs non-array`
  : [Tx] extends [readonly unknown[]] ? `${Pth}: non-array vs array`
  : [Sx] extends [object] ? ([Tx] extends [object] ? FitUnion<Sx, Tx, Pth, M, D> : `${Pth}: object vs non-object`)
  : `${Pth}: types differ`;

/** Every member of the source union must fit some member of the target union. */
type FitUnion<Sx, Tx, Pth extends string, M extends Mode, D extends Depth> = Collect<Exclude<FitEach<Sx, Tx, Pth, M, D>, true>>;
type FitEach<Sx, Tx, Pth extends string, M extends Mode, D extends Depth> =
  Sx extends unknown
    ? true extends (Tx extends unknown ? (FitObj<Sx, Tx, Pth, M, D> extends true ? true : false) : never)
      ? true
      : FitObj<Sx, Narrow<Sx, Tx>, Pth, M, D>
    : never;
/** The target members a source member is meant to match: same `kind` or `op` discriminant, else all. */
type Narrow<Sx, Tx> =
  Sx extends { kind: infer Kd } ? OrAll<Extract<Tx, { kind: Kd }>, Tx>
  : Sx extends { op: infer Op } ? OrAll<Extract<Tx, { op: Op }>, Tx>
  : Tx;
type OrAll<X, All> = IsNever<X> extends true ? All : X;

type FitObj<Sx, Tx, Pth extends string, M extends Mode, D extends Depth> = Collect<
  | Missing<Sx, Tx, Pth, M, D>
  | Extra<Sx, Tx, Pth, M>
  | Errs<{ [K in KKeys<Sx> & KKeys<Tx>]: Fit<Known<Sx>[K], Known<Tx>[K], `${Pth}.${K}`, M, [...D, 1]> }>
  | IndexFit<Sx, Tx, Pth, M, D>
>;

/** Target fields the source lacks (responses: every client field; requests: not checked, see the header). */
type Missing<Sx, Tx, Pth extends string, M extends Mode, D extends Depth> =
  M extends "in" ? never
  : IsTypedDict<Sx> extends true
    // the server's dict may hold the client's fixed keys: its value type must fit each of them
    ? Errs<{ [K in Exclude<KKeys<Tx>, KKeys<Sx>>]: Fit<IndexValue<Sx>, Known<Tx>[K], `${Pth}.${K}`, M, [...D, 1]> }>
  : Exclude<KKeys<Tx>, KKeys<Sx>> extends infer Ks extends string
    ? IsNever<Ks> extends true ? never : `${Pth}: source lacks ${Ks}`
    : never;

/** Source fields the target does not declare (not checked for subset views or open targets). */
type Extra<Sx, Tx, Pth extends string, M extends Mode> =
  M extends "view" ? never
  : HasIndex<Tx> extends true ? never
  : Exclude<KKeys<Sx>, KKeys<Tx>> extends infer Ks extends string
    ? IsNever<Ks> extends true ? never : `${Pth}: target lacks ${Ks}`
    : never;

/** A typed-dict target: the source's fields (and its own dict values) must fit the target's value type. */
type IndexFit<Sx, Tx, Pth extends string, M extends Mode, D extends Depth> =
  IsTypedDict<Tx> extends true
    ? | Errs<{ [K in Exclude<KKeys<Sx>, KKeys<Tx>>]: Fit<Known<Sx>[K], IndexValue<Tx>, `${Pth}.${K}`, M, [...D, 1]> }>
      | (IsTypedDict<Sx> extends true ? Exclude<Fit<IndexValue<Sx>, IndexValue<Tx>, `${Pth}[*]`, M, [...D, 1]>, true> : never)
    : never;

// ---------------------------------------------------------------- endpoints

type Method = "get" | "post" | "put" | "patch" | "delete";
type Json<X> = X extends { content: { "application/json": infer B } } ? B : never;
/** The JSON body of every 2xx reply of `M path` (several when the endpoint answers 200 or 202). */
type Res<Path extends keyof paths, M extends Method> = paths[Path][M] extends { responses: infer Rs }
  ? { [K in keyof Rs]: K extends 200 | 201 | 202 ? Json<Rs[K]> : never }[keyof Rs]
  : never;
/** The JSON request body of `M path` (an optional body is the model itself). */
type Req<Path extends keyof paths, M extends Method> = paths[Path][M] extends { requestBody?: infer B }
  ? Bare<Json<Bare<B>>>
  : never;

type Ret<Fn> = Fn extends (...args: never[]) => infer Rv ? Awaited<Rv> : never;
type Arg<Fn, N extends number> = Fn extends (...args: infer As) => unknown ? As[N] : never;

type Out<Hand, Gen, Pth extends string> = Fit<Wire<Gen>, Hand, Pth, "out">;
type View<Hand, Gen, Pth extends string> = Fit<Wire<Gen>, Hand, Pth, "view">;
type In<Hand, Gen, Pth extends string> = Fit<Hand, Gen, Pth, "in">;

/** `Rs` must be `true` apart from the listed known errors, and every known error must still occur. */
export type Gate<Rs, Kn extends string = never> =
  [Exclude<Rs, true | Kn>] extends [never]
    ? [Exclude<Kn, Rs>] extends [never] ? true : `stale allowance: ${Exclude<Kn, Rs>}`
    : Exclude<Rs, true | Kn>;
type Expect<X extends true> = X;

/** The generated selection union (api.md §1 `SelectionSpec`), as the server writes and reads it. */
type GenSelectionOut =
  | S["SelAll"] | S["SelZones"] | S["SelPolygon"] | S["SelCircle"] | S["SelRect"] | S["SelCells"] | S["SelBlob"]
  | S["SelHex"] | S["SelFilter"] | S["SelTop-Output"] | S["SelBuffer-Output"] | S["SelRegion"] | S["SelOp-Output"] | S["SelNot-Output"];
type GenSelectionIn =
  | S["SelAll"] | S["SelZones"] | S["SelPolygon"] | S["SelCircle"] | S["SelRect"] | S["SelCells"] | S["SelBlob"]
  | S["SelHex"] | S["SelFilter"] | S["SelTop-Input"] | S["SelBuffer-Input"] | S["SelRegion"] | S["SelOp-Input"] | S["SelNot-Input"];

// ---------------------------------------------------------------- shared types (api.md §0–1)

export interface SharedChecks {
  errorEnvelope: Expect<Out<C.ErrorEnvelope, S["ErrorEnvelope"], "ErrorEnvelope">>;
  job: Expect<Out<C.Job, S["Job"], "Job">>;
  planNode: Expect<Out<C.PlanNode, S["PlanNode"], "PlanNode">>;
  issue: Expect<Out<C.Issue, S["Issue"], "Issue">>;
  likely: Expect<Out<C.Likely, S["Likely"], "Likely">>;
  outputEntry: Expect<Out<C.OutputEntry, S["OutputEntry"], "OutputEntry">>;
  selectionOut: Expect<Out<C.SelectionSpec, GenSelectionOut, "SelectionSpec">>;
  selectionIn: Expect<In<C.SelectionSpec, GenSelectionIn, "SelectionSpec">>;
  sparseEdit: Expect<Out<C.SparseEdit, S["SparseEdit"], "SparseEdit">>;
}

// ---------------------------------------------------------------- system, jobs and tracking (api.md §2–3)

export interface TrackingChecks {
  health: Expect<Out<C.Health, Res<"/api/health", "get">, "GET /api/health">>;
  meta: Expect<Out<C.Meta, Res<"/api/meta", "get">, "GET /api/meta">>;
  settings: Expect<Out<T.Settings, Res<"/api/settings", "get">, "GET /api/settings">>;
  settingsPut: Expect<Out<Ret<typeof T.putSettings>, Res<"/api/settings", "put">, "PUT /api/settings">>;
  settingsPutBody: Expect<In<Arg<typeof T.putSettings, 0>, Req<"/api/settings", "put">, "PUT /api/settings body">>;
  projectSettings: Expect<View<P.StudioSettings, Res<"/api/settings", "get">, "GET /api/settings (projects)">>;
  threadsHeavy: Expect<View<{ threads_heavy?: number }, Res<"/api/settings", "get">, "GET /api/settings (studies)">>;
  system: Expect<Out<T.SystemInfo, Res<"/api/system", "get">, "GET /api/system">>;
  netcheck: Expect<Out<T.NetcheckResult, Res<"/api/system/netcheck", "post">, "POST /api/system/netcheck">>;
  netcheckInputs: Expect<Out<Ret<typeof I.netcheck>, Res<"/api/system/netcheck", "post">, "POST /api/system/netcheck (inputs)">>;
  netcheckBody: Expect<In<{ hosts?: string[] }, Req<"/api/system/netcheck", "post">, "POST /api/system/netcheck body">>;
  storage: Expect<Out<T.StorageInfo, Res<"/api/storage", "get">, "GET /api/storage">>;
  storageSummary: Expect<View<P.StorageSummary, Res<"/api/storage", "get">, "GET /api/storage (projects)">>;
  deleteCache: Expect<Out<Ret<typeof T.deleteCache>, Res<"/api/storage/cache/{name}", "delete">, "DELETE /api/storage/cache/{name}">>;

  jobs: Expect<Out<C.Page<C.Job>, Res<"/api/jobs", "get">, "GET /api/jobs">>;
  job: Expect<Out<C.Job, Res<"/api/jobs/{jid}", "get">, "GET /api/jobs/{jid}">>;
  jobPatch: Expect<Out<C.Job, Res<"/api/jobs/{jid}", "patch">, "PATCH /api/jobs/{jid}">>;
  jobPatchBody: Expect<In<{ priority: number }, Req<"/api/jobs/{jid}", "patch">, "PATCH /api/jobs/{jid} body">>;
  jobDelete: Expect<Out<Ret<typeof T.deleteJob>, Res<"/api/jobs/{jid}", "delete">, "DELETE /api/jobs/{jid}">>;
  cancel: Expect<Out<C.Job, Res<"/api/jobs/{jid}/cancel", "post">, "POST /api/jobs/{jid}/cancel">>;
  kill: Expect<Out<C.Job, Res<"/api/jobs/{jid}/kill", "post">, "POST /api/jobs/{jid}/kill">>;
  killBody: Expect<In<{ force_now?: boolean }, Req<"/api/jobs/{jid}/kill", "post">, "POST /api/jobs/{jid}/kill body">>;
  retry: Expect<Out<C.Job, Res<"/api/jobs/{jid}/retry", "post">, "POST /api/jobs/{jid}/retry">>;
  tracker: Expect<Out<T.TrackerSnapshot, Res<"/api/jobs/{jid}/tracker", "get">, "GET /api/jobs/{jid}/tracker">>;
  spans: Expect<Out<T.Span[], Res<"/api/jobs/{jid}/spans", "get">, "GET /api/jobs/{jid}/spans">>;
  events: Expect<Out<T.EventsPage, Res<"/api/jobs/{jid}/events", "get">, "GET /api/jobs/{jid}/events">>;
  logs: Expect<Out<T.LogsPage, Res<"/api/jobs/{jid}/logs", "get">, "GET /api/jobs/{jid}/logs">>;
  metrics: Expect<Out<Record<string, T.MetricPoint[]>, Res<"/api/jobs/{jid}/metrics", "get">, "GET /api/jobs/{jid}/metrics">>;
  warnings: Expect<Out<T.WarningRow[], Res<"/api/jobs/{jid}/warnings", "get">, "GET /api/jobs/{jid}/warnings">>;
  resources: Expect<Out<T.ResourceRow[], Res<"/api/jobs/{jid}/resources", "get">, "GET /api/jobs/{jid}/resources">>;
  queue: Expect<Out<T.QueueState, Res<"/api/queue", "get">, "GET /api/queue">>;
  queuePause: Expect<Out<T.QueueState, Res<"/api/queue/pause", "post">, "POST /api/queue/pause">>;
  queueResume: Expect<Out<T.QueueState, Res<"/api/queue/resume", "post">, "POST /api/queue/resume">>;
  timings: Expect<Out<T.Timings, Res<"/api/timings", "get">, "GET /api/timings">>;
}

// ---------------------------------------------------------------- projects, config, inputs (api.md §5)

export interface ProjectChecks {
  list: Expect<Out<C.Project[], Res<"/api/projects", "get">, "GET /api/projects">>;
  create: Expect<Out<P.CreateProjectResult, Res<"/api/projects", "post">, "POST /api/projects">>;
  createBody: Expect<In<P.CreateProjectBody, Req<"/api/projects", "post">, "POST /api/projects body">>;
  importProject: Expect<Out<P.ImportProjectResult, Res<"/api/projects/import", "post">, "POST /api/projects/import">>;
  importBody: Expect<In<P.ImportProjectBody, Req<"/api/projects/import", "post">, "POST /api/projects/import body">>;
  detail: Expect<Out<P.ProjectDetail, Res<"/api/projects/{pid}", "get">, "GET /api/projects/{pid}">>;
  detailShell: Expect<Out<ShellProjectDetail, Res<"/api/projects/{pid}", "get">, "GET /api/projects/{pid} (shell)">>;
  patch: Expect<Out<C.Project, Res<"/api/projects/{pid}", "patch">, "PATCH /api/projects/{pid}">>;
  patchBody: Expect<In<P.ProjectPatch, Req<"/api/projects/{pid}", "patch">, "PATCH /api/projects/{pid} body">>;
  remove: Expect<Out<Ret<typeof P.deleteProject>, Res<"/api/projects/{pid}", "delete">, "DELETE /api/projects/{pid}">>;

  files: Expect<Out<P.ProjectFile[], Res<"/api/projects/{pid}/files", "get">, "GET /api/projects/{pid}/files">>;
  upload: Expect<Out<P.UploadResult, Res<"/api/projects/{pid}/files/{kind}/{filename}", "put">, "PUT /api/projects/{pid}/files/{kind}/{filename}">>;
  inspect: Expect<Out<P.FileInspect, Res<"/api/projects/{pid}/files/inspect", "get">, "GET /api/projects/{pid}/files/inspect">>;
  deleteFile: Expect<Out<Ret<typeof P.deleteFile>, Res<"/api/projects/{pid}/files", "delete">, "DELETE /api/projects/{pid}/files">>;
  suggest: Expect<Out<P.ColumnSuggestion, Res<"/api/projects/{pid}/columns/suggest", "post">, "POST /api/projects/{pid}/columns/suggest">>;
  suggestBody: Expect<In<{ path: string }, Req<"/api/projects/{pid}/columns/suggest", "post">, "POST /api/projects/{pid}/columns/suggest body">>;
  dataCheck: Expect<Out<P.DataCheck, Res<"/api/projects/{pid}/data/check", "post">, "POST /api/projects/{pid}/data/check">>;
  dataCheckBody: Expect<In<{ config_patch?: Arg<typeof P.checkData, 1> }, Req<"/api/projects/{pid}/data/check", "post">, "POST /api/projects/{pid}/data/check body">>;

  config: Expect<Out<P.ConfigDoc, Res<"/api/projects/{pid}/config", "get">, "GET /api/projects/{pid}/config">>;
  configPut: Expect<Out<P.ConfigSaveResult, Res<"/api/projects/{pid}/config", "put">, "PUT /api/projects/{pid}/config">>;
  configPutBody: Expect<In<P.ConfigBody, Req<"/api/projects/{pid}/config", "put">, "PUT /api/projects/{pid}/config body">>;
  section: Expect<Out<P.ConfigSaveResult, Res<"/api/projects/{pid}/config/sections/{section}", "patch">, "PATCH /api/projects/{pid}/config/sections/{section}">>;
  sectionBody: Expect<In<{ value: unknown; note?: string }, Req<"/api/projects/{pid}/config/sections/{section}", "patch">, "PATCH /api/projects/{pid}/config/sections/{section} body">>;
  validate: Expect<Out<P.ValidateResult, Res<"/api/projects/{pid}/config/validate", "post">, "POST /api/projects/{pid}/config/validate">>;
  validateBody: Expect<In<Arg<typeof P.validateConfig, 1>, Req<"/api/projects/{pid}/config/validate", "post">, "POST /api/projects/{pid}/config/validate body">>;
  impact: Expect<Out<P.ImpactResult, Res<"/api/projects/{pid}/config/impact", "post">, "POST /api/projects/{pid}/config/impact">>;
  impactBody: Expect<In<Arg<typeof P.configImpact, 1>, Req<"/api/projects/{pid}/config/impact", "post">, "POST /api/projects/{pid}/config/impact body">>;
  history: Expect<Out<P.ConfigHistoryEntry[], Res<"/api/projects/{pid}/config/history", "get">, "GET /api/projects/{pid}/config/history">>;
  historyVersion: Expect<Out<Ret<typeof P.configAtVersion>, Res<"/api/projects/{pid}/config/history/{version}", "get">, "GET /api/projects/{pid}/config/history/{version}">>;
  link: Expect<Out<P.LinkResult, Res<"/api/projects/{pid}/link", "post">, "POST /api/projects/{pid}/link">>;
  linkBody: Expect<In<P.LinkBody, Req<"/api/projects/{pid}/link", "post">, "POST /api/projects/{pid}/link body">>;

  inputs: Expect<Out<I.InputsSummary, Res<"/api/projects/{pid}/inputs", "get">, "GET /api/projects/{pid}/inputs">>;
  inputView: Expect<Out<I.InputViews[I.InputViewKind], Res<"/api/projects/{pid}/inputs/{kind}/view", "get">, "GET /api/projects/{pid}/inputs/{kind}/view">>;
  forcing: Expect<Out<C.Job, Res<"/api/projects/{pid}/inputs/forcing", "post">, "POST /api/projects/{pid}/inputs/forcing">>;
  forcingBody: Expect<In<I.ForcingBody, Req<"/api/projects/{pid}/inputs/forcing", "post">, "POST /api/projects/{pid}/inputs/forcing body">>;
  layers: Expect<Out<C.Job, Res<"/api/projects/{pid}/inputs/layers", "post">, "POST /api/projects/{pid}/inputs/layers">>;
  layersBody: Expect<In<I.LayersBody, Req<"/api/projects/{pid}/inputs/layers", "post">, "POST /api/projects/{pid}/inputs/layers body">>;
  features: Expect<Out<C.Job, Res<"/api/projects/{pid}/inputs/features", "post">, "POST /api/projects/{pid}/inputs/features">>;
  featuresBody: Expect<In<I.FeaturesBody, Req<"/api/projects/{pid}/inputs/features", "post">, "POST /api/projects/{pid}/inputs/features body">>;
  cmip6: Expect<Out<C.Job, Res<"/api/projects/{pid}/inputs/cmip6", "post">, "POST /api/projects/{pid}/inputs/cmip6">>;
  cmip6Body: Expect<In<I.Cmip6Body, Req<"/api/projects/{pid}/inputs/cmip6", "post">, "POST /api/projects/{pid}/inputs/cmip6 body">>;
  ghcn: Expect<Out<C.Job, Res<"/api/projects/{pid}/inputs/ghcn", "post">, "POST /api/projects/{pid}/inputs/ghcn">>;
  ghcnBody: Expect<In<I.GhcnBody, Req<"/api/projects/{pid}/inputs/ghcn", "post">, "POST /api/projects/{pid}/inputs/ghcn body">>;
  stations: Expect<Out<C.Job, Res<"/api/projects/{pid}/inputs/stations", "post">, "POST /api/projects/{pid}/inputs/stations">>;
  stationList: Expect<Out<I.Station[], Res<"/api/projects/{pid}/forcing/stations", "get">, "GET /api/projects/{pid}/forcing/stations">>;
}

// ---------------------------------------------------------------- runs, outputs, analysis (api.md §6)

export interface RunChecks {
  runs: Expect<Out<C.Page<C.RunSummary>, Res<"/api/runs", "get">, "GET /api/runs">>;
  projectRuns: Expect<Out<C.Page<C.RunSummary>, Res<"/api/projects/{pid}/runs", "get">, "GET /api/projects/{pid}/runs">>;
  plan: Expect<Out<P.RunPlan, Res<"/api/projects/{pid}/runs/plan", "post">, "POST /api/projects/{pid}/runs/plan">>;
  planBody: Expect<In<P.RunPlanBody, Req<"/api/projects/{pid}/runs/plan", "post">, "POST /api/projects/{pid}/runs/plan body">>;
  launch: Expect<Out<P.LaunchResult, Res<"/api/projects/{pid}/runs", "post">, "POST /api/projects/{pid}/runs">>;
  launchBody: Expect<In<P.LaunchBody, Req<"/api/projects/{pid}/runs", "post">, "POST /api/projects/{pid}/runs body">>;
  importRun: Expect<Out<C.RunSummary, Res<"/api/runs/import", "post">, "POST /api/runs/import">>;
  importRunBody: Expect<In<P.ImportRunBody, Req<"/api/runs/import", "post">, "POST /api/runs/import body">>;
  trustBody: Expect<In<{ dir: string; project_id: string | null; config_path?: string; trust_pickles: boolean }, Req<"/api/runs/import", "post">, "POST /api/runs/import body (trust)">>;
  detail: Expect<Out<R.RunDetail, Res<"/api/runs/{rid}", "get">, "GET /api/runs/{rid}">>;
  detailShell: Expect<View<RunDetailHead, Res<"/api/runs/{rid}", "get">, "GET /api/runs/{rid} (shell)">>;
  detailLite: Expect<View<T.RunDetailLite, Res<"/api/runs/{rid}", "get">, "GET /api/runs/{rid} (tracking)">>;
  launchSource: Expect<View<P.RunLaunchSource, Res<"/api/runs/{rid}", "get">, "GET /api/runs/{rid} (launch)">>;
  remove: Expect<Out<Ret<typeof T.deleteRunData>, Res<"/api/runs/{rid}", "delete">, "DELETE /api/runs/{rid}">>;
  deleteCheckpoint: Expect<Out<Ret<typeof R.deleteCheckpoint>, Res<"/api/runs/{rid}/checkpoint", "delete">, "DELETE /api/runs/{rid}/checkpoint">>;
  resume: Expect<Out<C.Job, Res<"/api/runs/{rid}/resume", "post">, "POST /api/runs/{rid}/resume">>;
  resumeBody: Expect<In<Arg<typeof T.resumeRun, 1>, Req<"/api/runs/{rid}/resume", "post">, "POST /api/runs/{rid}/resume body">>;
  rerun: Expect<Out<Ret<typeof T.rerunRun>, Res<"/api/runs/{rid}/rerun", "post">, "POST /api/runs/{rid}/rerun">>;
  rerunBody: Expect<In<Arg<typeof T.rerunRun, 1>, Req<"/api/runs/{rid}/rerun", "post">, "POST /api/runs/{rid}/rerun body">>;
  statusBoard: Expect<Out<T.StatusBoard, Res<"/api/projects/{pid}/status-board", "get">, "GET /api/projects/{pid}/status-board">>;
  statusBoardStudies: Expect<Out<St.StatusBoard, Res<"/api/projects/{pid}/status-board", "get">, "GET /api/projects/{pid}/status-board (studies)">>;
  config: Expect<Out<R.RunConfig, Res<"/api/runs/{rid}/config", "get">, "GET /api/runs/{rid}/config">>;
  configTracking: Expect<Out<T.RunConfig, Res<"/api/runs/{rid}/config", "get">, "GET /api/runs/{rid}/config (tracking)">>;
  provenance: Expect<Out<T.RunProvenance, Res<"/api/runs/{rid}/provenance", "get">, "GET /api/runs/{rid}/provenance">>;
  environment: Expect<Out<R.RunEnvironment, Res<"/api/runs/{rid}/environment", "get">, "GET /api/runs/{rid}/environment">>;

  outputs: Expect<Out<C.RunOutputs, Res<"/api/runs/{rid}/outputs", "get">, "GET /api/runs/{rid}/outputs">>;
  view: Expect<Out<R.ViewModelOf<R.ViewName>, Res<"/api/runs/{rid}/views/{view}", "get">, "GET /api/runs/{rid}/views/{view}">>;
  docs: Expect<Out<R.DocEntry[], Res<"/api/runs/{rid}/docs", "get">, "GET /api/runs/{rid}/docs">>;
  doc: Expect<Out<R.DocContent, Res<"/api/runs/{rid}/docs/{doc}", "get">, "GET /api/runs/{rid}/docs/{doc}">>;
  files: Expect<Out<R.FileEntry[], Res<"/api/runs/{rid}/files", "get">, "GET /api/runs/{rid}/files">>;
  fileTable: Expect<Out<R.FileTable, Res<"/api/runs/{rid}/files/table", "get">, "GET /api/runs/{rid}/files/table">>;
  dictionary: Expect<Out<R.DictionaryRow[], Res<"/api/runs/{rid}/dictionary", "get">, "GET /api/runs/{rid}/dictionary">>;
  // the map kit's layer catalogue (src/map/data.ts); each LayerMeta is a free object on the server
  layers: Expect<Out<{ groups: C.LayerGroup[] }, Res<"/api/runs/{rid}/layers", "get">, "GET /api/runs/{rid}/layers">>;
  grid: Expect<Out<C.GridMeta, Res<"/api/runs/{rid}/grid", "get">, "GET /api/runs/{rid}/grid">>;
  cell: Expect<Out<R.CellInfo, Res<"/api/runs/{rid}/cells/{index}", "get">, "GET /api/runs/{rid}/cells/{index}">>;

  resolve: Expect<Out<A.ResolvedSelectionReply, Res<"/api/runs/{rid}/selection/resolve", "post">, "POST /api/runs/{rid}/selection/resolve">>;
  resolveLab: Expect<Out<L.SelectionReply, Res<"/api/runs/{rid}/selection/resolve", "post">, "POST /api/runs/{rid}/selection/resolve (lab)">>;
  resolveBody: Expect<In<{ selection: C.SelectionSpec }, Req<"/api/runs/{rid}/selection/resolve", "post">, "POST /api/runs/{rid}/selection/resolve body">>;
  blob: Expect<Out<{ blob_id: string; bytes: number }, Res<"/api/runs/{rid}/blobs", "put">, "PUT /api/runs/{rid}/blobs">>;
  regions: Expect<Out<A.Region[], Res<"/api/runs/{rid}/regions", "get">, "GET /api/runs/{rid}/regions">>;
  regionsLab: Expect<Out<L.Region[], Res<"/api/runs/{rid}/regions", "get">, "GET /api/runs/{rid}/regions (lab)">>;
  region: Expect<Out<A.Region, Res<"/api/runs/{rid}/regions", "post">, "POST /api/runs/{rid}/regions">>;
  regionBody: Expect<In<{ name: string; spec: C.SelectionSpec }, Req<"/api/runs/{rid}/regions", "post">, "POST /api/runs/{rid}/regions body">>;
  regionDelete: Expect<Out<Ret<typeof A.deleteRegion>, Res<"/api/runs/{rid}/regions/{rgid}", "delete">, "DELETE /api/runs/{rid}/regions/{rgid}">>;
  regionStats: Expect<Out<A.RegionStats, Res<"/api/runs/{rid}/stats/region", "post">, "POST /api/runs/{rid}/stats/region">>;
  regionStatsBody: Expect<In<A.RegionStatsRequest, Req<"/api/runs/{rid}/stats/region", "post">, "POST /api/runs/{rid}/stats/region body">>;
  breakdown: Expect<Out<A.Breakdown, Res<"/api/runs/{rid}/stats/breakdown", "post">, "POST /api/runs/{rid}/stats/breakdown">>;
  breakdownBody: Expect<In<A.BreakdownRequest, Req<"/api/runs/{rid}/stats/breakdown", "post">, "POST /api/runs/{rid}/stats/breakdown body">>;
  hexbin: Expect<Out<A.Hexbin, Res<"/api/runs/{rid}/stats/hexbin", "post">, "POST /api/runs/{rid}/stats/hexbin">>;
  hexbinBody: Expect<In<A.HexbinRequest, Req<"/api/runs/{rid}/stats/hexbin", "post">, "POST /api/runs/{rid}/stats/hexbin body">>;
  acf: Expect<Out<A.Acf, Res<"/api/runs/{rid}/stats/acf", "post">, "POST /api/runs/{rid}/stats/acf">>;
  acfBody: Expect<In<A.AcfRequest, Req<"/api/runs/{rid}/stats/acf", "post">, "POST /api/runs/{rid}/stats/acf body">>;

  compareRuns: Expect<Out<R.CompareRuns, Res<"/api/compare/runs", "get">, "GET /api/compare/runs">>;
  priority: Expect<Out<R.PriorityAgreement, Res<"/api/compare/priority", "post">, "POST /api/compare/priority">>;
  priorityBody: Expect<In<{ a: string; b: string; layer: string }, Req<"/api/compare/priority", "post">, "POST /api/compare/priority body">>;
}

// ---------------------------------------------------------------- Scenario Lab (api.md §7)

export interface LabChecks {
  engineHost: Expect<Out<L.EngineHostStatus, Res<"/api/engine", "get">, "GET /api/engine">>;
  engine: Expect<Out<L.RunEngineStatus, Res<"/api/runs/{rid}/engine", "get">, "GET /api/runs/{rid}/engine">>;
  engineOpen: Expect<Out<C.Job | L.RunEngineStatus, Res<"/api/runs/{rid}/engine/open", "post">, "POST /api/runs/{rid}/engine/open">>;
  engineClose: Expect<Out<L.RunEngineStatus, Res<"/api/runs/{rid}/engine", "delete">, "DELETE /api/runs/{rid}/engine">>;
  levers: Expect<Out<L.Lever[], Res<"/api/runs/{rid}/levers", "get">, "GET /api/runs/{rid}/levers">>;
  emulator: Expect<Out<L.EmulatorInfo, Res<"/api/runs/{rid}/emulator", "get">, "GET /api/runs/{rid}/emulator">>;
  previewBody: Expect<In<L.PreviewRequest, Req<"/api/runs/{rid}/preview", "post">, "POST /api/runs/{rid}/preview body">>;
  compile: Expect<Out<L.CompileResult, Res<"/api/runs/{rid}/compile", "post">, "POST /api/runs/{rid}/compile">>;
  compileBody: Expect<In<{ scenario: L.ScenarioDoc }, Req<"/api/runs/{rid}/compile", "post">, "POST /api/runs/{rid}/compile body">>;
  templates: Expect<Out<L.ScenarioTemplate[], Res<"/api/scenario-templates", "get">, "GET /api/scenario-templates">>;
  fromTemplate: Expect<Out<L.Scenario, Res<"/api/projects/{pid}/scenarios/from-template", "post">, "POST /api/projects/{pid}/scenarios/from-template">>;
  fromTemplateBody: Expect<In<{ template: string; params: Record<string, unknown>; run_id: string }, Req<"/api/projects/{pid}/scenarios/from-template", "post">, "POST /api/projects/{pid}/scenarios/from-template body">>;
  scenarios: Expect<Out<L.ScenarioSummary[], Res<"/api/projects/{pid}/scenarios", "get">, "GET /api/projects/{pid}/scenarios">>;
  scenarioCreate: Expect<Out<L.Scenario, Res<"/api/projects/{pid}/scenarios", "post">, "POST /api/projects/{pid}/scenarios">>;
  scenarioCreateBody: Expect<In<{ doc: L.ScenarioDoc }, Req<"/api/projects/{pid}/scenarios", "post">, "POST /api/projects/{pid}/scenarios body">>;
  scenarioClone: Expect<View<{ id: string }, Res<"/api/projects/{pid}/scenarios", "post">, "POST /api/projects/{pid}/scenarios (clone)">>;
  scenario: Expect<Out<L.Scenario, Res<"/api/scenarios/{sid}", "get">, "GET /api/scenarios/{sid}">>;
  scenarioPatch: Expect<Out<L.Scenario, Res<"/api/scenarios/{sid}", "patch">, "PATCH /api/scenarios/{sid}">>;
  scenarioPatchBody: Expect<In<L.ScenarioPatch, Req<"/api/scenarios/{sid}", "patch">, "PATCH /api/scenarios/{sid} body">>;
  scenarioDelete: Expect<Out<Ret<typeof L.deleteScenario>, Res<"/api/scenarios/{sid}", "delete">, "DELETE /api/scenarios/{sid}">>;
  fork: Expect<Out<L.Scenario, Res<"/api/scenarios/{sid}/fork", "post">, "POST /api/scenarios/{sid}/fork">>;
  forkBody: Expect<In<Arg<typeof L.forkScenario, 1>, Req<"/api/scenarios/{sid}/fork", "post">, "POST /api/scenarios/{sid}/fork body">>;
  runExact: Expect<Out<Ret<typeof L.runScenarioExact>, Res<"/api/scenarios/{sid}/run", "post">, "POST /api/scenarios/{sid}/run">>;
  runExactBody: Expect<In<{ run_id: string; force: boolean }, Req<"/api/scenarios/{sid}/run", "post">, "POST /api/scenarios/{sid}/run body">>;
  runBatch: Expect<Out<{ job: C.Job }, Res<"/api/scenarios/run-batch", "post">, "POST /api/scenarios/run-batch">>;
  ladder: Expect<Out<Ret<typeof L.makeLadder>, Res<"/api/scenarios/{sid}/ladder", "post">, "POST /api/scenarios/{sid}/ladder">>;
  ladderBody: Expect<In<{ edit_index: number; amounts: number[]; run_id?: string }, Req<"/api/scenarios/{sid}/ladder", "post">, "POST /api/scenarios/{sid}/ladder body">>;
  acrossEstimate: Expect<Out<L.AcrossRunsEstimate, Res<"/api/scenarios/{sid}/across-runs/estimate", "post">, "POST /api/scenarios/{sid}/across-runs/estimate">>;
  acrossBody: Expect<In<{ run_ids: string[] }, Req<"/api/scenarios/{sid}/across-runs", "post">, "POST /api/scenarios/{sid}/across-runs body">>;
  across: Expect<Out<C.Job, Res<"/api/scenarios/{sid}/across-runs", "post">, "POST /api/scenarios/{sid}/across-runs">>;
  promote: Expect<Out<L.PromoteResult, Res<"/api/scenarios/{sid}/promote", "post">, "POST /api/scenarios/{sid}/promote">>;
  promoteBody: Expect<In<{ apply: boolean }, Req<"/api/scenarios/{sid}/promote", "post">, "POST /api/scenarios/{sid}/promote body">>;
  designImport: Expect<Out<L.DesignImport, Res<"/api/runs/{rid}/designs/import", "post">, "POST /api/runs/{rid}/designs/import">>;
  runScenarios: Expect<Out<L.RunScenarios, Res<"/api/runs/{rid}/scenarios", "get">, "GET /api/runs/{rid}/scenarios">>;
  rerunConfigured: Expect<Out<C.Job, Res<"/api/runs/{rid}/configured/{slug}/rerun-exact", "post">, "POST /api/runs/{rid}/configured/{slug}/rerun-exact">>;

  result: Expect<Out<L.Result, Res<"/api/results/{res_id}", "get">, "GET /api/results/{res_id}">>;
  impacts: Expect<Out<L.Impacts, Res<"/api/results/{res_id}/impacts", "post">, "POST /api/results/{res_id}/impacts">>;
  impactsBody: Expect<In<L.ImpactsRequest, Req<"/api/results/{res_id}/impacts", "post">, "POST /api/results/{res_id}/impacts body">>;
  resultDelete: Expect<Out<Ret<typeof L.deleteResult>, Res<"/api/results/{res_id}", "delete">, "DELETE /api/results/{res_id}">>;
  compare: Expect<Out<L.Comparison, Res<"/api/runs/{rid}/compare", "post">, "POST /api/runs/{rid}/compare">>;
  compareBody: Expect<In<{ items: L.ItemRef[] } & Arg<typeof L.createComparison, 2>, Req<"/api/runs/{rid}/compare", "post">, "POST /api/runs/{rid}/compare body">>;
  comparison: Expect<Out<L.Comparison, Res<"/api/comparisons/{cid}", "get">, "GET /api/comparisons/{cid}">>;
  comparisons: Expect<Out<L.ComparisonListItem[], Res<"/api/runs/{rid}/comparisons", "get">, "GET /api/runs/{rid}/comparisons">>;
  comparisonDelete: Expect<Out<Ret<typeof L.deleteComparison>, Res<"/api/comparisons/{cid}", "delete">, "DELETE /api/comparisons/{cid}">>;
  climateFactors: Expect<Out<L.ClimateFactors, Res<"/api/runs/{rid}/climate/factors", "get">, "GET /api/runs/{rid}/climate/factors">>;
  climateExplore: Expect<Out<L.ClimateExplore, Res<"/api/runs/{rid}/climate/explore", "post">, "POST /api/runs/{rid}/climate/explore">>;
  climateExploreBody: Expect<In<L.ClimateExploreRequest, Req<"/api/runs/{rid}/climate/explore", "post">, "POST /api/runs/{rid}/climate/explore body">>;
  sweepCreate: Expect<Out<Ret<typeof L.createSweep>, Res<"/api/runs/{rid}/sweeps", "post">, "POST /api/runs/{rid}/sweeps">>;
  sweepCreateBody: Expect<In<Arg<typeof L.createSweep, 1>, Req<"/api/runs/{rid}/sweeps", "post">, "POST /api/runs/{rid}/sweeps body">>;
  sweeps: Expect<Out<L.SweepListItem[], Res<"/api/runs/{rid}/sweeps", "get">, "GET /api/runs/{rid}/sweeps">>;
  sweep: Expect<Out<L.Sweep, Res<"/api/sweeps/{swid}", "get">, "GET /api/sweeps/{swid}">>;
  sweepDelete: Expect<Out<Ret<typeof L.deleteSweep>, Res<"/api/sweeps/{swid}", "delete">, "DELETE /api/sweeps/{swid}">>;
  planPreview: Expect<Out<L.PlanPreview, Res<"/api/runs/{rid}/plans/preview", "post">, "POST /api/runs/{rid}/plans/preview">>;
  planPreviewBody: Expect<In<L.PlanParams, Req<"/api/runs/{rid}/plans/preview", "post">, "POST /api/runs/{rid}/plans/preview body">>;
  planCreate: Expect<Out<Ret<typeof L.createPlan>, Res<"/api/runs/{rid}/plans", "post">, "POST /api/runs/{rid}/plans">>;
  planCreateBody: Expect<In<{ params: L.PlanParams; name: string; verify: boolean }, Req<"/api/runs/{rid}/plans", "post">, "POST /api/runs/{rid}/plans body">>;
  plans: Expect<Out<L.Plan[], Res<"/api/runs/{rid}/plans", "get">, "GET /api/runs/{rid}/plans">>;
  plan: Expect<Out<L.Plan, Res<"/api/plans/{plid}", "get">, "GET /api/plans/{plid}">>;
  verify: Expect<Out<C.Job, Res<"/api/plans/{plid}/verify", "post">, "POST /api/plans/{plid}/verify">>;
  verifyBody: Expect<In<{ frontier: boolean }, Req<"/api/plans/{plid}/verify", "post">, "POST /api/plans/{plid}/verify body">>;
  toScenario: Expect<Out<L.Scenario, Res<"/api/plans/{plid}/to-scenario", "post">, "POST /api/plans/{plid}/to-scenario">>;
  fieldKit: Expect<Out<L.FieldKit, Res<"/api/plans/{plid}/field-kit", "post">, "POST /api/plans/{plid}/field-kit">>;
  fieldKitBody: Expect<In<Arg<typeof L.getFieldKit, 1>, Req<"/api/plans/{plid}/field-kit", "post">, "POST /api/plans/{plid}/field-kit body">>;
  planDelete: Expect<Out<Ret<typeof L.deletePlan>, Res<"/api/plans/{plid}", "delete">, "DELETE /api/plans/{plid}">>;
  pack: Expect<Out<Ret<typeof L.exportPack>, Res<"/api/exports", "post">, "POST /api/exports (packs)">>;
}

// ---------------------------------------------------------------- studies, exports, findings (api.md §9–11)

export interface StudyChecks {
  runStudies: Expect<Out<St.StudyStatusRow[], Res<"/api/runs/{rid}/studies", "get">, "GET /api/runs/{rid}/studies">>;
  action: Expect<Out<C.Job, Res<"/api/runs/{rid}/actions/{kind}", "post">, "POST /api/runs/{rid}/actions/{kind}">>;
  launch: Expect<Out<St.LaunchResult, Res<"/api/runs/{rid}/studies/{kind}", "post">, "POST /api/runs/{rid}/studies/{kind}">>;
  launchBody: Expect<In<St.StudyParams[St.LaunchableKind], Req<"/api/runs/{rid}/studies/{kind}", "post">, "POST /api/runs/{rid}/studies/{kind} body">>;
  benchmark: Expect<Out<St.LaunchResult, Res<"/api/projects/{pid}/studies/benchmark", "post">, "POST /api/projects/{pid}/studies/benchmark">>;
  reproduceBody: Expect<In<Arg<typeof R.reproduceRun, 1>, Req<"/api/runs/{rid}/studies/{kind}", "post">, "POST /api/runs/{rid}/studies/reproduce body">>;
  estimate: Expect<Out<St.StudyCost, Res<"/api/studies/estimate", "post">, "POST /api/studies/estimate">>;
  estimateBody: Expect<In<{ kind: string; run_id?: string; params: object }, Req<"/api/studies/estimate", "post">, "POST /api/studies/estimate body">>;
  projectStudies: Expect<Out<St.Study[], Res<"/api/projects/{pid}/studies", "get">, "GET /api/projects/{pid}/studies">>;
  study: Expect<Out<St.Study, Res<"/api/studies/{stid}", "get">, "GET /api/studies/{stid}">>;
  studyView: Expect<Out<St.StudyViews[St.ViewKind], Res<"/api/studies/{stid}/view", "get">, "GET /api/studies/{stid}/view">>;
  studyViewTracking: Expect<Out<T.StudyView, Res<"/api/studies/{stid}/view", "get">, "GET /api/studies/{stid}/view (tracking)">>;
  resume: Expect<Out<C.Job, Res<"/api/studies/{stid}/resume", "post">, "POST /api/studies/{stid}/resume">>;
  attach: Expect<Out<Ret<typeof St.attachStudy>, Res<"/api/studies/{stid}/attach", "post">, "POST /api/studies/{stid}/attach">>;
  detach: Expect<Out<Ret<typeof St.attachStudy>, Res<"/api/studies/{stid}/detach", "post">, "POST /api/studies/{stid}/detach">>;
  attachBody: Expect<In<{ run_id: string }, Req<"/api/studies/{stid}/attach", "post">, "POST /api/studies/{stid}/attach body">>;
  merge: Expect<Out<St.SimcheckMerge, Res<"/api/studies/simcheck/merge", "post">, "POST /api/studies/simcheck/merge">>;
  mergeBody: Expect<In<{ study_ids: string[] }, Req<"/api/studies/simcheck/merge", "post">, "POST /api/studies/simcheck/merge body">>;
  studyDelete: Expect<Out<Ret<typeof St.deleteStudy>, Res<"/api/studies/{stid}", "delete">, "DELETE /api/studies/{stid}">>;
  truth: Expect<Out<Ret<typeof St.getTruth>, Res<"/api/runs/{rid}/truth", "get">, "GET /api/runs/{rid}/truth">>;

  exportCreate: Expect<Out<Ret<typeof E.createExport>, Res<"/api/exports", "post">, "POST /api/exports">>;
  exportBody: Expect<In<{ kind: E.ExportKind; project_id: string; params: E.ExportParams[E.ExportKind] }, Req<"/api/exports", "post">, "POST /api/exports body">>;
  packBody: Expect<In<{ kind: L.PackKind; project_id: string; params: Record<string, unknown> }, Req<"/api/exports", "post">, "POST /api/exports body (packs)">>;
  exports: Expect<Out<E.Export[], Res<"/api/projects/{pid}/exports", "get">, "GET /api/projects/{pid}/exports">>;
  exportGet: Expect<Out<E.Export, Res<"/api/exports/{eid}", "get">, "GET /api/exports/{eid}">>;
  exportDelete: Expect<Out<Ret<typeof E.deleteExport>, Res<"/api/exports/{eid}", "delete">, "DELETE /api/exports/{eid}">>;
  reportPreview: Expect<Out<Ret<typeof E.previewReport>, Res<"/api/projects/{pid}/report/preview", "post">, "POST /api/projects/{pid}/report/preview">>;
  reportPreviewBody: Expect<In<E.ReportPreviewBody, Req<"/api/projects/{pid}/report/preview", "post">, "POST /api/projects/{pid}/report/preview body">>;

  findings: Expect<Out<F.Finding[], Res<"/api/findings", "get">, "GET /api/findings">>;
  findingCreate: Expect<Out<F.Finding, Res<"/api/findings", "post">, "POST /api/findings">>;
  findingCreateBody: Expect<In<F.FindingCreate, Req<"/api/findings", "post">, "POST /api/findings body">>;
  findingPatch: Expect<Out<F.Finding, Res<"/api/findings/{fid}", "patch">, "PATCH /api/findings/{fid}">>;
  findingPatchBody: Expect<In<F.FindingPatch, Req<"/api/findings/{fid}", "patch">, "PATCH /api/findings/{fid} body">>;
  findingDelete: Expect<Out<Ret<typeof F.deleteFinding>, Res<"/api/findings/{fid}", "delete">, "DELETE /api/findings/{fid}">>;
  findingImage: Expect<Out<Ret<typeof F.putFindingImage>, Res<"/api/findings/{fid}/image", "put">, "PUT /api/findings/{fid}/image">>;
}
