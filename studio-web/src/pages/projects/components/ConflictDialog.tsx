// The optimistic-concurrency conflict (api.md §0.1, §5.3): a save sent `If-Match: <version>`
// but the config was saved elsewhere since. The user chooses: load the saved version
// (discarding theirs), overwrite it with theirs, or go back and copy their text first.
import { Button } from "../../../components/ui/Button";
import { Dialog } from "../../../components/ui/Dialog";

export function ConflictDialog({
  open,
  mine,
  server,
  busy,
  onReload,
  onOverwrite,
  onClose,
}: {
  open: boolean;
  /** The version the edit was based on. */
  mine: number;
  /** The version now on the server (`detail.current_version`), when known. */
  server: number | null;
  busy?: boolean;
  onReload: () => void;
  onOverwrite: () => void;
  onClose: () => void;
}) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      busy={busy}
      title="The config changed while you were editing"
      className="conflict-dialog"
      footer={
        <>
          <Button onClick={onClose} disabled={busy}>
            Keep editing
          </Button>
          <Button onClick={onReload} disabled={busy}>
            Load the saved version
          </Button>
          <Button variant="danger" onClick={onOverwrite} busy={busy} data-autofocus>
            Overwrite with mine
          </Button>
        </>
      }
    >
      <div className="prose" role="alert">
        <p>
          Your changes are based on version <b className="num">{mine}</b>, but version <b className="num">{server ?? "newer"}</b> was saved since (another tab, the
          config editor, or an input job that linked a file into the config).
        </p>
        <p>
          <b>Load the saved version</b> discards your changes. <b>Overwrite with mine</b> replaces the saved version with yours; the other change stays in the
          version history.
        </p>
      </div>
    </Dialog>
  );
}
