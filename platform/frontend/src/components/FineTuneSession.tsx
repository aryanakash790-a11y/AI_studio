import { useEffect, useState } from "react";
import { Button, Callout, Collapse, HTMLTable, Icon, Tag } from "@blueprintjs/core";
import { api } from "../api/client";
import { toast } from "./toaster";
import { Markdown } from "./Markdown";
import { FineTuneProgress } from "./FineTuneProgress";

// UI state the Fine-Tuning Assistant attaches to a reply (run.progress, kind === "ft_session"):
// an open question with clickable choices, a proposed plan with a Launch button, and the jobs
// touched in that turn (live progress). Rendered in Chat and in Domain AI › Fine-Tuning Studio.
export type FtSessionUi = {
  kind: "ft_session"; session_id: string; conversation_id?: string;
  question?: { text: string; options?: string[] } | null;
  plan?: FtPlan | null;
  jobs?: string[];
  runtime?: { name?: string; gpu_type?: string } | null;
  notebook?: { name?: string; path?: string; version?: number } | null;
};

export type FtPlan = {
  plan_id: string; status: string; summary?: string;
  runtime?: { name?: string; gpu_type?: string };
  notebook?: { name?: string; path?: string; version?: number | null; is_draft?: boolean };
  selection?: { parameters?: Record<string, unknown>; dataset?: { path: string } | null; dataset_parameter?: string };
  warnings?: string[]; job_id?: string;
};

const ACTIVE = ["queued", "running"];
const runStatusOf = (s?: string) => (s && ACTIVE.includes(s)) ? "running"
  : s === "completed" ? "succeeded" : s;

// Live progress of one job; polls while it is queued/running.
export function JobProgress({ jobId, compact = false }: { jobId: string; compact?: boolean }) {
  const [job, setJob] = useState<any>(null);
  const [showSummary, setShowSummary] = useState(false);
  useEffect(() => {
    let stop = false;
    let t: any;
    const tick = async () => {
      try {
        const j = await api.ftJob(jobId);
        if (stop) return;
        setJob(j);
        if (ACTIVE.includes(j.status)) t = setTimeout(tick, 3000);
      } catch { if (!stop) t = setTimeout(tick, 6000); }
    };
    tick();
    return () => { stop = true; clearTimeout(t); };
  }, [jobId]);
  if (!job) return <div className="muted" style={{ fontSize: 12 }}><span className="thinking-dots">loading job {jobId}</span></div>;
  const p = { ...(job.progress || {}), selection: { ...(job.progress?.selection || {}), notebook:
    `${job.selection?.notebook || ""}${job.selection?.version ? ` v${job.selection.version}` : ""}` } };
  const running = ACTIVE.includes(job.status);
  return (
    <div style={{ marginTop: 8 }}>
      <FineTuneProgress p={p} status={runStatusOf(job.status)} error={p.error}
        compact={compact && !running}
        onStop={running ? async () => {
          try { await api.ftCancelJob(jobId); toast.info("Stopping the job…"); } catch (e: any) { toast.error(e.message); }
        } : undefined} />
      {!running && p.summary && (
        <>
          <Button small minimal icon="document" text={showSummary ? "Hide job summary" : "Job summary"}
            onClick={() => setShowSummary(!showSummary)} />
          <Collapse isOpen={showSummary}>
            <div style={{ fontSize: 12, borderLeft: "2px solid var(--border)", paddingLeft: 10 }}><Markdown text={p.summary} /></div>
          </Collapse>
        </>
      )}
    </div>
  );
}

export function PlanCard({ plan, sessionId, canLaunch, onLaunched }: {
  plan: FtPlan; sessionId: string; canLaunch: boolean; onLaunched?: (jobId: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [jobId, setJobId] = useState<string | null>(plan.job_id || null);
  // the server knows whether this plan is still the current one, was launched, or went stale
  const [serverStatus, setServerStatus] = useState<string | null>(null);
  useEffect(() => {
    if (jobId) return;
    api.ftSession(sessionId).then((s: any) => {
      const cur = s.plan || {};
      if (cur.plan_id !== plan.plan_id) { setServerStatus("superseded"); return; }
      setServerStatus(cur.status);
      if (cur.job_id) setJobId(cur.job_id);
    }).catch(() => {});
    /* eslint-disable-next-line */
  }, [sessionId, plan.plan_id]);
  const launchable = serverStatus ? serverStatus === "proposed" : canLaunch;
  const why = serverStatus === "superseded" ? "A newer plan replaced this one."
    : serverStatus === "stale" ? "The notebook changed after this plan — ask the assistant to propose it again."
    : !launchable ? "This plan is no longer current." : "Nothing runs until you launch. You can also reply with changes.";
  const params = Object.entries(plan.selection?.parameters || {});
  const launch = async () => {
    setBusy(true);
    try {
      const j = await api.ftLaunch(sessionId, plan.plan_id);
      setJobId(j.job_id); onLaunched?.(j.job_id);
      toast.success(`Job ${j.job_id} launched`);
    } catch (e: any) { toast.error(e.message); }
    finally { setBusy(false); }
  };
  return (
    <div style={{ border: "1px solid var(--accent)", borderRadius: 10, padding: 12, marginTop: 8, background: "var(--panel)", fontSize: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 6 }}>
        <Icon icon="clipboard" color="var(--accent)" />
        <b style={{ fontSize: 13 }}>Fine-tuning plan</b>
        {plan.runtime?.name && <Tag minimal icon="lab-test">{plan.runtime.name}{plan.runtime.gpu_type ? ` · ${plan.runtime.gpu_type}` : ""}</Tag>}
        {plan.notebook?.name && <Tag minimal icon="document">{plan.notebook.name}{plan.notebook.version ? ` v${plan.notebook.version}` : ""}</Tag>}
        {plan.notebook?.is_draft && <Tag minimal intent="primary">edited draft</Tag>}
      </div>
      {plan.summary && <div style={{ marginBottom: 6 }}>{plan.summary}</div>}
      {params.length > 0 && (
        <HTMLTable compact striped style={{ width: "100%", fontSize: 12, marginBottom: 6 }}>
          <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
          <tbody>{params.map(([k, v]) => <tr key={k}><td><code>{k}</code></td><td><code>{JSON.stringify(v)}</code></td></tr>)}</tbody>
        </HTMLTable>
      )}
      {plan.selection?.dataset?.path && <div className="muted">Dataset: <code>{plan.selection.dataset.path}</code>
        {plan.selection.dataset_parameter ? <> → <code>{plan.selection.dataset_parameter}</code></> : null}</div>}
      {(plan.warnings?.length ?? 0) > 0 && (
        <Callout intent="warning" icon="warning-sign" style={{ margin: "6px 0", fontSize: 12 }}>
          {plan.warnings!.map((w, i) => <div key={i}>{w.replace(/`/g, "")}</div>)}
        </Callout>
      )}
      {jobId ? <JobProgress jobId={jobId} /> : (
        <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
          <Button intent="success" icon="play" text="Launch" loading={busy} disabled={!launchable} onClick={launch} />
          <span className="muted">{why}</span>
        </div>
      )}
    </div>
  );
}

export function FineTuneSession({ p, isLast, onChoose }: {
  p: FtSessionUi; isLast: boolean; onChoose?: (text: string) => void;
}) {
  const [launched, setLaunched] = useState<string[]>([]);
  const jobs = [...(p.jobs || []), ...launched.filter((j) => !(p.jobs || []).includes(j))];
  const planJob = p.plan?.job_id;
  return (
    <div>
      {p.question && (
        <div style={{ marginTop: 8 }}>
          {(p.question.options?.length ?? 0) > 0 && (
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {p.question.options!.map((o) => (
                <Button key={o} small outlined icon="key-enter" text={o} disabled={!isLast || !onChoose}
                  onClick={() => onChoose?.(o)} />
              ))}
            </div>
          )}
        </div>
      )}
      {p.plan && <PlanCard plan={p.plan} sessionId={p.session_id} canLaunch={isLast}
        onLaunched={(id) => setLaunched((l) => [...l, id])} />}
      {jobs.filter((j) => j !== planJob && !(p.plan && launched.includes(j))).map((j) => <JobProgress key={j} jobId={j} compact={!isLast} />)}
    </div>
  );
}
