// `/jobs/:jid` — Mission Control of one job (SPEC §5.10).
import { useRoute } from "../../router";
import { JobTracker } from "./mission/JobTracker";

export default function MissionControl() {
  const { params } = useRoute();
  return params.jid ? <JobTracker key={params.jid} jid={params.jid} /> : null;
}
