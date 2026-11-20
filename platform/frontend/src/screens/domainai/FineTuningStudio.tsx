import { useCallback, useEffect, useState } from "react";
import { Button, Callout, Card, Collapse, HTMLSelect, Icon, Tag } from "@blueprintjs/core";
import { api } from "../../api/client";
import { Chat } from "../Chat";
import { JobProgress, PlanCard } from "../../components/FineTuneSession";

// Domain AI › Fine-Tuning Studio: the Fine-Tuning Assistant conversation (same workflow as in
// Chat) next to a workspace showing the session: runtime, working notebook with its changes,
// the plan to launch, and every job of the conversation with live progress.
const WORKFLOW = "Fine-Tuning Assistant";

export function FineTuningStudio() {
  const [cid, setCid] = useState<string | null>(null);
  const [ses, setSes] = useState<any>(null);
  const [tick, setTick] = useState(0);

  const load = useCallback(async () => {
    if (!cid) { setSes(null); return; }
    try { setSes(await api.ftSessionByConversation(cid)); } catch { setSes(null); }
  }, [cid]);

  useEffect(() => { load(); }, [load, tick]);
  // keep the workspace fresh while jobs run or the assistant works
  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 5000);
    return () => clearInterval(t);
  }, []);

  return (
    <div style={{ height: "100%", display: "grid", gap: 14, minHeight: 0,
      gridTemplateColumns: "minmax(0, 1.35fr) minmax(320px, 1fr)" }} className="ft-studio">
      <div style={{ minHeight: 0, display: "flex", flexDirection: "column" }}>
        <Chat lockWorkflowName={WORKFLOW} title="Fine-Tuning Studio" storageKey="tcs_active_ft_chat"
          onConversation={setCid} onRunSettled={() => setTick((n) => n + 1)} />
      </div>
      <div style={{ minHeight: 0, overflow: "auto", paddingRight: 4 }}>
        <Workspace ses={ses} hasConversation={!!cid} onChanged={() => setTick((n) => n + 1)} />
      </div>
    </div>
  );
}

function Section({ icon, title, children, right }: { icon: any; title: string; children: any; right?: any }) {
  return (
    <Card style={{ padding: 12, marginBottom: 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
        <Icon icon={icon} color="var(--accent)" size={14} />
        <b style={{ fontSize: 13 }}>{title}</b>
        <span style={{ marginLeft: "auto" }}>{right}</span>
      </div>
      {children}
    </Card>
  );
}

function Workspace({ ses, hasConversation, onChanged }: { ses: any; hasConversation: boolean; onChanged: () => void }) {
  const [runtimes, setRuntimes] = useState<any>(null);
  useEffect(() => { api.ftRuntimes().then(setRuntimes).catch(() => setRuntimes(null)); }, []);
  const bound = (runtimes?.runtimes || []).filter((r: any) => (r.bound_agents || []).includes(WORKFLOW));

  return (
    <div style={{ fontSize: 12 }}>
      <div className="section-head" style={{ marginBottom: 8 }}>
        <div><h3 style={{ margin: 0 }}>Workspace</h3>
          <span className="muted">Runtime, notebook, plan and jobs of this conversation</span></div>
      </div>

      <Section icon="lab-test" title="Runtime">
        {ses?.runtime ? (
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
            <Tag large minimal intent="success" icon="tick">{ses.runtime.name}</Tag>
            {ses.runtime.gpu_type && <Tag minimal>{ses.runtime.gpu_type}</Tag>}
            {ses.runtime_card?.status && <Tag minimal>{ses.runtime_card.status}</Tag>}
            {ses.runtime_card && !ses.runtime_card.usable && <Tag minimal intent="danger">{ses.runtime_card.problem}</Tag>}
          </div>
        ) : null}
        {ses?.runtime && ses.runtime_check && (
          ses.runtime_check.ok
            ? <details style={{ marginTop: 6 }}><summary className="muted" style={{ cursor: "pointer" }}>
                <Icon icon="tick-circle" size={11} color="var(--success)" /> Runtime responded (checked {fmtWhen(ses.runtime_check.at)})</summary>
                <pre style={{ margin: "4px 0 0", fontSize: 11, whiteSpace: "pre-wrap", fontFamily: "var(--mono)" }}>{ses.runtime_check.output}</pre></details>
            : <Callout intent="danger" icon="offline" title="Runtime did not respond" style={{ marginTop: 6 }}>
                <div style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>{ses.runtime_check.output}</div>
                <div className="muted" style={{ marginTop: 4 }}>Open the lab from Data Science Lab › Runtimes to see the connection steps, then ask the assistant to check again.</div>
              </Callout>
        )}
        {!ses?.runtime && <div className="muted">Not chosen yet — the assistant asks which runtime to use.</div>}
        {runtimes && (
          <div style={{ marginTop: 8 }}>
            {!runtimes.dsl_enabled && <Callout intent="warning" icon="warning-sign">Data Science Lab is disabled.</Callout>}
            {runtimes.dsl_enabled && bound.length === 0 && (
              <Callout intent="warning" icon="warning-sign" title="No runtime available to the assistant">
                In <b>Data Science Lab</b> open a runtime and click <b>Enable agent access</b>, then in
                <b> Agent Garden</b> edit <b>{WORKFLOW}</b> › Tools › MCP servers and tick <b>DSLab · &lt;runtime&gt;</b>.
              </Callout>
            )}
            {bound.length > 0 && <div className="muted">Available to the assistant: {bound.map((r: any) =>
              `${r.name}${r.gpu_type ? ` (${r.gpu_type})` : ""}`).join(", ")}</div>}
          </div>
        )}
      </Section>

      <NotebookSection ses={ses} />

      {ses?.plan?.status === "proposed" && (
        <Section icon="clipboard" title="Plan awaiting launch">
          <PlanCard plan={ses.plan} sessionId={ses.session_id} canLaunch onLaunched={onChanged} />
        </Section>
      )}

      <Section icon="predictive-analysis" title={`Jobs${ses?.jobs?.length ? ` (${ses.jobs.length})` : ""}`}>
        {!ses?.jobs?.length ? <div className="muted">{hasConversation ? "No jobs launched in this conversation yet."
          : "Start a conversation on the left."}</div>
          : ses.jobs.map((j: any, i: number) => <JobRow key={j.job_id} j={j} open={i === 0} />)}
      </Section>
    </div>
  );
}

function JobRow({ j, open }: { j: any; open: boolean }) {
  const [show, setShow] = useState(open || ["queued", "running"].includes(j.status));
  const intents: Record<string, any> = { completed: "success", failed: "danger", cancelled: "warning", running: "primary", queued: "primary" };
  const intent: any = intents[String(j.status)] || "none";
  return (
    <div style={{ borderTop: "1px solid var(--border)", padding: "6px 0" }}>
      <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", cursor: "pointer" }} onClick={() => setShow(!show)}>
        <Icon icon={show ? "chevron-down" : "chevron-right"} size={12} />
        <Tag minimal round intent={intent}>{j.status}</Tag>
        <code>{j.notebook}{j.version ? ` v${j.version}` : ""}</code>
        {typeof j.percent === "number" && ["queued", "running"].includes(j.status) && <span className="metric">{j.percent.toFixed(0)}%</span>}
        <span className="muted" style={{ marginLeft: "auto" }}>{j.runtime} · {fmtWhen(j.created_at)}</span>
      </div>
      <Collapse isOpen={show}><JobProgress jobId={j.job_id} compact /></Collapse>
    </div>
  );
}

function NotebookSection({ ses }: { ses: any }) {
  const nbRec = ses?.notebook;
  const [view, setView] = useState<"changes" | "cells">("changes");
  const [version, setVersion] = useState(0);
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState("");
  const key = `${ses?.session_id}|${nbRec?.path}|${nbRec?.version}|${version}`;

  useEffect(() => {
    if (!ses?.session_id || !nbRec?.path) { setData(null); return; }
    api.ftSessionNotebook(ses.session_id, nbRec.path, version)
      .then((d) => { setData(d); setErr(""); }).catch((e) => setErr(e.message));
    /* eslint-disable-next-line */
  }, [key]);

  if (!nbRec) {
    return <Section icon="document" title="Notebook"><div className="muted">No working notebook yet — the assistant
      lists the templates, creates a new notebook or edits one from your instructions.</div></Section>;
  }
  const versions = Array.from({ length: nbRec.version || 0 }, (_, i) => i + 1);
  return (
    <Section icon="document" title="Notebook" right={
      <div style={{ display: "flex", gap: 4 }}>
        <Button small minimal active={view === "changes"} text="Changes" onClick={() => setView("changes")} />
        <Button small minimal active={view === "cells"} text="Cells" onClick={() => setView("cells")} />
      </div>}>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center", marginBottom: 6 }}>
        <code>{nbRec.name}</code>
        {versions.length > 0 && (
          <HTMLSelect minimal value={version} onChange={(e) => setVersion(Number(e.target.value))}
            options={[{ label: `current (v${nbRec.version})`, value: 0 },
                      ...versions.slice().reverse().map((v) => ({ label: `v${v}`, value: v }))]} />
        )}
        {data?.validation && <Tag minimal intent={data.validation.ok ? "success" : "danger"}
          icon={data.validation.ok ? "tick" : "error"}>{data.validation.ok ? "valid" : `${data.validation.errors.length} error(s)`}</Tag>}
      </div>
      {nbRec.base && nbRec.base !== nbRec.path && <div className="muted" style={{ marginBottom: 6 }}>
        Based on <code>{nbRec.base}</code> (the template is never modified)</div>}
      {err && <Callout intent="danger">{err}</Callout>}
      {data && view === "changes" && (
        data.diff ? <DiffView text={data.diff} /> : <div className="muted">No changes against the base notebook.</div>
      )}
      {data && view === "cells" && (
        <div style={{ maxHeight: 420, overflow: "auto" }}>
          {(data.cells || []).map((c: any) => (
            <div key={c.index} style={{ marginBottom: 6 }}>
              <div className="muted" style={{ fontSize: 11 }}>
                [{c.index}] {c.type}{c.tags?.length ? ` · ${c.tags.join(", ")}` : ""}
              </div>
              <pre style={{ margin: 0, padding: 6, fontSize: 11, whiteSpace: "pre-wrap", fontFamily: "var(--mono)",
                background: c.type === "code" ? "var(--app-bg)" : "transparent", border: "1px solid var(--border)", borderRadius: 6 }}>
                {c.source || " "}
              </pre>
            </div>
          ))}
        </div>
      )}
      {data?.validation && !data.validation.ok && (
        <Callout intent="danger" style={{ marginTop: 6 }}>
          {data.validation.errors.map((e: any, i: number) => <div key={i}>cell {e.cell ?? "-"}: {e.error}</div>)}
        </Callout>
      )}
    </Section>
  );
}

function DiffView({ text }: { text: string }) {
  return (
    <pre style={{ margin: 0, maxHeight: 420, overflow: "auto", fontSize: 11, fontFamily: "var(--mono)",
      border: "1px solid var(--border)", borderRadius: 6, padding: 6, background: "var(--app-bg)" }}>
      {text.split("\n").map((l, i) => {
        const color = l.startsWith("+++") || l.startsWith("---") ? "var(--muted)"
          : l.startsWith("+") ? "var(--success)" : l.startsWith("-") ? "var(--danger)"
          : l.startsWith("@@") || l.includes("── cell") ? "var(--accent-2)" : undefined;
        return <div key={i} style={{ color, whiteSpace: "pre-wrap" }}>{l || " "}</div>;
      })}
    </pre>
  );
}

function fmtWhen(iso?: string) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

