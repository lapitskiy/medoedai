"use client";

import { useEffect, useMemo, useRef, useState } from "react";

class JsonRequestError extends Error {
  payload: unknown;

  constructor(message: string, payload: unknown) {
    super(message);
    this.name = "JsonRequestError";
    this.payload = payload;
  }
}

async function jsonFetch<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  const data = (await response.json()) as T;
  if (!response.ok) {
    const error = data && typeof data === "object" && "error" in data
      ? String((data as { error?: string }).error || `HTTP ${response.status}`)
      : `HTTP ${response.status}`;
    throw new JsonRequestError(error, data);
  }
  return data;
}

type ActiveTrainingTask = {
  task_id?: string;
  state?: string;
  grid_type?: string;
  symbol?: string;
  task_name?: string;
  done?: number;
  total?: number;
  elapsed_sec?: number;
  eta_sec?: number;
  avg_per_run_sec?: number;
  last_log?: string;
  logs_tail?: string[];
};

type ActiveTrainingResponse = {
  success?: boolean;
  active?: ActiveTrainingTask[];
  error?: string;
};

type GridStatusResponse = {
  success?: boolean;
  task_id?: string;
  state?: string;
  done?: number;
  total?: number;
  error?: string;
  last_log?: string;
};

type SymbolsResponse = { success?: boolean; symbols?: string[]; error?: string };
type RunsResponse = {
  success?: boolean;
  runs?: Array<{
    run_id?: string;
    direction?: string | null;
    task?: string | null;
    val_acc?: number | null;
  }>;
  error?: string;
};

type CreateVersionResponse = {
  success?: boolean;
  version?: string;
  path?: string;
  files?: string[];
  error?: string;
};

type TrainResponse = { success?: boolean; task_id?: string; error?: string; task?: string };

type GridPresetResponse = { success?: boolean; values?: Record<string, string>; error?: string };

function formatDuration(sec?: number): string {
  if (!sec || sec <= 0) return "—";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}ч ${m}м`;
  if (m > 0) return `${m}м ${s}с`;
  return `${s}с`;
}

function parseNumber(raw: unknown): number {
  const s = String(raw ?? "").trim().replace(",", ".");
  const x = Number.parseFloat(s);
  return Number.isFinite(x) ? x : Number.NaN;
}

function rangeValues(from: string, to: string, step: string): number[] {
  const f = parseNumber(from);
  const t = parseNumber(to);
  const s = parseNumber(step);
  if (Number.isNaN(f)) return [];
  if (Number.isNaN(t) || t <= f || Number.isNaN(s) || s <= 0) return [f];
  const arr: number[] = [];
  for (let v = f; v <= t + s * 0.001; v += s) {
    arr.push(Math.round(v * 1e8) / 1e8);
  }
  return arr;
}

type FieldState = Record<string, string>;

const GF_CACHE_KEY = "xgb_grid_cache";

const GF_IDS = [
  "gfSymbol",
  "gfTask",
  "gfDirection",
  "gfLimitCandles",
  "gfEarlyStopping",
  "gfKeepTopN",
  "gfRankByProxy",
  "gfDeleteRest",
  "gfHsFrom",
  "gfHsTo",
  "gfHsStep",
  "gfThrFrom",
  "gfThrTo",
  "gfThrStep",
  "gfMhFrom",
  "gfMhTo",
  "gfMhStep",
  "gfMpFrom",
  "gfMpTo",
  "gfMpStep",
  "gfFeeFrom",
  "gfFeeTo",
  "gfFeeStep",
  "gfPeFrom",
  "gfPeTo",
  "gfPeStep",
  "gfEtpFrom",
  "gfEtpTo",
  "gfEtpStep",
  "gfEslFrom",
  "gfEslTo",
  "gfEslStep",
  "gfEtrFrom",
  "gfEtrTo",
  "gfEtrStep",
  "gfUse1mMicrovol",
  "gfUse1mMomentum",
  "gfUse1mCandleStructure",
  "gfUse1mVolume",
  "gfUse1dRegime",
  "gfUseSrFeatures",
  "gfMdFrom",
  "gfMdTo",
  "gfMdStep",
  "gfLrFrom",
  "gfLrTo",
  "gfLrStep",
  "gfNeFrom",
  "gfNeTo",
  "gfNeStep",
  "gfSsFrom",
  "gfSsTo",
  "gfSsStep",
  "gfCbFrom",
  "gfCbTo",
  "gfCbStep",
  "gfRlFrom",
  "gfRlTo",
  "gfRlStep",
  "gfMcwFrom",
  "gfMcwTo",
  "gfMcwStep",
  "gfGmFrom",
  "gfGmTo",
  "gfGmStep",
  "gfSpwFrom",
  "gfSpwTo",
  "gfSpwStep",
] as const;

const GF_DEFAULTS: FieldState = {
  gfSymbol: "BTCUSDT",
  gfTask: "entry_long",
  gfDirection: "long",
  gfLimitCandles: "100000",
  gfEarlyStopping: "50",
  gfKeepTopN: "20",
  gfRankByProxy: "1",
  gfDeleteRest: "1",
  gfHsFrom: "12",
  gfHsTo: "48",
  gfHsStep: "12",
  gfThrFrom: "0.10",
  gfThrTo: "0.40",
  gfThrStep: "0.10",
  gfMhFrom: "48",
  gfMhTo: "96",
  gfMhStep: "48",
  gfMpFrom: "0.00",
  gfMpTo: "0.50",
  gfMpStep: "0.25",
  gfFeeFrom: "0.06",
  gfFeeTo: "0.06",
  gfFeeStep: "0.01",
  gfPeFrom: "0.60",
  gfPeTo: "0.80",
  gfPeStep: "0.10",
  gfEtpFrom: "1.2",
  gfEtpTo: "1.2",
  gfEtpStep: "0.1",
  gfEslFrom: "-0.7",
  gfEslTo: "-0.7",
  gfEslStep: "0.1",
  gfEtrFrom: "0.6",
  gfEtrTo: "0.6",
  gfEtrStep: "0.1",
  gfUse1mMicrovol: "0",
  gfUse1mMomentum: "0",
  gfUse1mCandleStructure: "0",
  gfUse1mVolume: "0",
  gfUse1dRegime: "0",
  gfUseSrFeatures: "1",
  gfMdFrom: "4",
  gfMdTo: "8",
  gfMdStep: "2",
  gfLrFrom: "0.01",
  gfLrTo: "0.10",
  gfLrStep: "0.03",
  gfNeFrom: "400",
  gfNeTo: "800",
  gfNeStep: "200",
  gfSsFrom: "0.7",
  gfSsTo: "0.9",
  gfSsStep: "0.1",
  gfCbFrom: "0.7",
  gfCbTo: "0.9",
  gfCbStep: "0.1",
  gfRlFrom: "1.0",
  gfRlTo: "1.0",
  gfRlStep: "2.0",
  gfMcwFrom: "1",
  gfMcwTo: "5",
  gfMcwStep: "2",
  gfGmFrom: "0.0",
  gfGmTo: "0.0",
  gfGmStep: "0.5",
  gfSpwFrom: "-1",
  gfSpwTo: "1",
  gfSpwStep: "2",
};

function fieldIsCheckbox(id: string): boolean {
  return id.startsWith("gfUse");
}

function calcTotalCombos(fields: FieldState): number {
  const task = fields.gfTask || "entry_long";
  const labelN = task === "directional"
    ? rangeValues(fields.gfHsFrom, fields.gfHsTo, fields.gfHsStep).length *
      rangeValues(fields.gfThrFrom, fields.gfThrTo, fields.gfThrStep).length
    : rangeValues(fields.gfMhFrom, fields.gfMhTo, fields.gfMhStep).length *
      rangeValues(fields.gfMpFrom, fields.gfMpTo, fields.gfMpStep).length *
      rangeValues(fields.gfFeeFrom, fields.gfFeeTo, fields.gfFeeStep).length;

  const modelN =
    rangeValues(fields.gfMdFrom, fields.gfMdTo, fields.gfMdStep).length *
    rangeValues(fields.gfLrFrom, fields.gfLrTo, fields.gfLrStep).length *
    rangeValues(fields.gfNeFrom, fields.gfNeTo, fields.gfNeStep).length *
    rangeValues(fields.gfSsFrom, fields.gfSsTo, fields.gfSsStep).length *
    rangeValues(fields.gfCbFrom, fields.gfCbTo, fields.gfCbStep).length *
    rangeValues(fields.gfRlFrom, fields.gfRlTo, fields.gfRlStep).length *
    rangeValues(fields.gfMcwFrom, fields.gfMcwTo, fields.gfMcwStep).length *
    rangeValues(fields.gfGmFrom, fields.gfGmTo, fields.gfGmStep).length *
    rangeValues(fields.gfSpwFrom, fields.gfSpwTo, fields.gfSpwStep).length;

  const policyN = task === "directional"
    ? 1
    : rangeValues(fields.gfPeFrom, fields.gfPeTo, fields.gfPeStep).length;

  const exitPolicyN = task.startsWith("entry")
    ? rangeValues(fields.gfEtpFrom, fields.gfEtpTo, fields.gfEtpStep).length *
      rangeValues(fields.gfEslFrom, fields.gfEslTo, fields.gfEslStep).length *
      rangeValues(fields.gfEtrFrom, fields.gfEtrTo, fields.gfEtrStep).length
    : 1;

  return labelN * modelN * policyN * exitPolicyN;
}

function FieldGroup({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
        {label}
      </div>
      {children}
    </div>
  );
}

function SectionCard({
  id,
  tone,
  title,
  description,
  meta,
  children,
}: {
  id?: string;
  tone: "blue" | "indigo" | "amber" | "emerald";
  title: string;
  description?: string;
  meta?: React.ReactNode;
  children: React.ReactNode;
}) {
  const toneStyles = tone === "blue"
    ? "border-t-blue-500"
    : tone === "indigo"
      ? "border-t-indigo-500"
      : tone === "amber"
        ? "border-t-amber-500"
        : "border-t-emerald-500";

  return (
    <section
      id={id}
      className={`rounded-[28px] border border-slate-200 bg-white p-6 shadow-sm lg:p-8 ${toneStyles} border-t-4`}
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-2">
          <h2 className="text-xl font-semibold text-slate-900">{title}</h2>
          {description ? (
            <p className="max-w-4xl text-sm leading-7 text-slate-600">
              {description}
            </p>
          ) : null}
        </div>
        {meta ? <div className="text-sm text-slate-600">{meta}</div> : null}
      </div>

      <div className="mt-6">{children}</div>
    </section>
  );
}

function Input({
  value,
  onChange,
  type = "text",
  min,
  step,
}: {
  value: string;
  onChange: (value: string) => void;
  type?: string;
  min?: number;
  step?: number;
}) {
  return (
    <input
      className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-400 focus:outline-none"
      type={type}
      value={value}
      min={min}
      step={step}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string }>;
}) {
  return (
    <select
      className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-400 focus:outline-none"
      value={value}
      onChange={(event) => onChange(event.target.value)}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

function StatusBox({ tone, text }: { tone: "info" | "ok" | "bad"; text: string }) {
  const styles = tone === "ok"
    ? "border-emerald-200 bg-emerald-50 text-emerald-900"
    : tone === "bad"
      ? "border-rose-200 bg-rose-50 text-rose-900"
      : "border-slate-200 bg-slate-50 text-slate-700";
  return (
    <div className={`rounded-2xl border px-4 py-3 text-sm leading-6 ${styles}`}>
      {text}
    </div>
  );
}

export function XgbTrainingView() {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [symbolsError, setSymbolsError] = useState<string | null>(null);

  const [active, setActive] = useState<ActiveTrainingTask[]>([]);
  const [activeError, setActiveError] = useState<string | null>(null);

  const [gridSymbol, setGridSymbol] = useState("BTCUSDT");
  const [gridDirection, setGridDirection] = useState("long");
  const [gridTask, setGridTask] = useState("directional");
  const [gridMaxHold, setGridMaxHold] = useState("48");
  const [gridFeePct, setGridFeePct] = useState("0.06");
  const [gridMinProfitPct, setGridMinProfitPct] = useState("0.00");
  const [gridLabelDeltaPct, setGridLabelDeltaPct] = useState("0.05");
  const [gridLimitCandles, setGridLimitCandles] = useState("100000");
  const [gridStatus, setGridStatus] = useState<{ tone: "info" | "ok" | "bad"; text: string } | null>(
    null,
  );

  const [gfFields, setGfFields] = useState<FieldState>(GF_DEFAULTS);
  const [isPresetLoaded, setIsPresetLoaded] = useState(false);
  const [gfStatus, setGfStatus] = useState<{ tone: "info" | "ok" | "bad"; text: string } | null>(
    null,
  );
  const [gfTaskId, setGfTaskId] = useState<string | null>(null);

  const [createSymbol, setCreateSymbol] = useState<string>("");
  const [createRuns, setCreateRuns] = useState<RunsResponse["runs"]>([]);
  const [createRunId, setCreateRunId] = useState<string>("");
  const [createEnsemble, setCreateEnsemble] = useState<string>("ensemble-a");
  const [createStatus, setCreateStatus] = useState<{ tone: "info" | "ok" | "bad"; text: string } | null>(
    null,
  );
  const [createBusy, setCreateBusy] = useState(false);

  const presetSaveTimerRef = useRef<number | null>(null);

  const gfTotalCombos = useMemo(() => calcTotalCombos(gfFields), [gfFields]);

  const gfShowDirectional = gfFields.gfTask === "directional";
  const gfShowExitPolicy = gfFields.gfTask.startsWith("entry");

  const lastGridPollRef = useRef<number>(0);
  const lastActivePollRef = useRef<number>(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await jsonFetch<SymbolsResponse>("/api/xgb_training/symbols", {
          method: "GET",
          cache: "no-store",
        });
        if (!data.success) throw new Error(data.error || "symbols unavailable");
        const list = Array.isArray(data.symbols) ? data.symbols : [];
        if (cancelled) return;
        setSymbols(list);
        setSymbolsError(null);
        if (list.length) {
          setGridSymbol(list[0]);
          setGfFields((current) => ({ ...current, gfSymbol: list[0] }));
          setCreateSymbol(list[0]);
        }
      } catch (error) {
        if (cancelled) return;
        setSymbolsError(error instanceof Error ? error.message : String(error));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      const now = Date.now();
      if (now - lastActivePollRef.current < 1200) return;
      lastActivePollRef.current = now;
      try {
        const data = await jsonFetch<ActiveTrainingResponse>("/api/xgb_training/active_training", {
          method: "GET",
          cache: "no-store",
        });
        if (!data.success) throw new Error(data.error || "active training unavailable");
        if (cancelled) return;
        setActive(Array.isArray(data.active) ? data.active : []);
        setActiveError(null);
      } catch (error) {
        if (cancelled) return;
        setActiveError(error instanceof Error ? error.message : String(error));
      }
    };

    poll();
    const timer = window.setInterval(poll, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const serverPreset = await jsonFetch<GridPresetResponse>("/api/xgb_training/grid_full_preset", {
          method: "GET",
          cache: "no-store",
        }).catch(() => null);

        const merged: FieldState = { ...GF_DEFAULTS };

        if (serverPreset && serverPreset.success && serverPreset.values && typeof serverPreset.values === "object") {
          for (const [k, v] of Object.entries(serverPreset.values)) {
            if (GF_IDS.includes(k as (typeof GF_IDS)[number])) merged[k] = String(v ?? "");
          }
          try {
            localStorage.setItem(GF_CACHE_KEY, JSON.stringify(serverPreset.values));
          } catch {}
        } else {
          try {
            const local = JSON.parse(localStorage.getItem(GF_CACHE_KEY) || "null") as unknown;
            if (local && typeof local === "object") {
              for (const [k, v] of Object.entries(local as Record<string, unknown>)) {
                if (GF_IDS.includes(k as (typeof GF_IDS)[number])) merged[k] = String(v ?? "");
              }
            }
          } catch {}
        }

        if (cancelled) return;
        setGfFields(merged);
        setIsPresetLoaded(true);
      } catch {
        // ignore preset load errors
        if (!cancelled) setIsPresetLoaded(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!isPresetLoaded) return;

    // keep in local storage + debounce server save
    const values: Record<string, string> = {};
    for (const id of GF_IDS) values[id] = gfFields[id] ?? "";
    try {
      localStorage.setItem(GF_CACHE_KEY, JSON.stringify(values));
    } catch {}

    if (presetSaveTimerRef.current) window.clearTimeout(presetSaveTimerRef.current);
    presetSaveTimerRef.current = window.setTimeout(() => {
      jsonFetch("/api/xgb_training/grid_full_preset", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ values }),
      }).catch(() => null);
    }, 350);
  }, [gfFields, isPresetLoaded]);

  const resetFullGridFields = () => {
    setGfFields(GF_DEFAULTS);
    setGfStatus({ tone: "info", text: "Full Grid: поля сброшены к дефолту." });
  };

  const shrinkRegGamma = () => {
    setGfFields((current) => ({
      ...current,
      gfRlTo: current.gfRlFrom,
      gfGmTo: current.gfGmFrom,
    }));
    setGfStatus({
      tone: "info",
      text: "Сетка упрощена: reg_lambda и gamma зафиксированы в 1 значение.",
    });
  };

  useEffect(() => {
    let cancelled = false;
    if (!createSymbol) {
      setCreateRuns([]);
      setCreateRunId("");
      return;
    }
    (async () => {
      try {
        const data = await jsonFetch<RunsResponse>(
          `/api/xgb_training/runs?symbol=${encodeURIComponent(createSymbol)}`,
          { method: "GET", cache: "no-store" },
        );
        if (!data.success) throw new Error(data.error || "runs unavailable");
        const list = Array.isArray(data.runs) ? data.runs : [];
        if (cancelled) return;
        setCreateRuns(list);
        setCreateRunId(list[0]?.run_id ? String(list[0].run_id) : "");
      } catch (error) {
        if (cancelled) return;
        setCreateRuns([]);
        setCreateRunId("");
        setCreateStatus({
          tone: "bad",
          text: `Ошибка загрузки runs: ${error instanceof Error ? error.message : String(error)}`,
        });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [createSymbol]);

  const startGridSelected = async () => {
    setGridStatus({
      tone: "info",
      text: `Запуск XGB grid: ${gridSymbol} task=${gridTask}...`,
    });
    try {
      const limitFinal = Number.parseInt(gridLimitCandles, 10) || 100000;
      const limitQuick = Math.min(50000, limitFinal || 100000);
      const feeBps = (parseNumber(gridFeePct) || 0) * 100.0;
      const minProfit = (parseNumber(gridMinProfitPct) || 0) / 100.0;
      const payload = {
        symbol: gridSymbol,
        task: gridTask,
        direction: gridDirection,
        limit_candles_final: limitFinal,
        limit_candles_quick: limitQuick,
        max_hold_steps: Number.parseInt(gridMaxHold, 10) || 48,
        fee_bps: feeBps,
        min_profit: minProfit,
      };
      const data = await jsonFetch<TrainResponse>("/api/xgb_training/train_grid_task", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(payload),
      });
      if (!data.success) throw new Error(data.error || "train start failed");
      setGridStatus({
        tone: "ok",
        text: `XGB grid запущен: task=${data.task || gridTask} (celery: ${data.task_id || "?"})`,
      });
    } catch (error) {
      setGridStatus({
        tone: "bad",
        text: `Ошибка запуска grid: ${error instanceof Error ? error.message : String(error)}`,
      });
    }
  };

  const pollGridStatus = async (taskId: string, expectedTotal: number) => {
    const now = Date.now();
    if (now - lastGridPollRef.current < 1000) return;
    lastGridPollRef.current = now;
    try {
      const data = await jsonFetch<GridStatusResponse>(
        `/api/xgb_training/grid_status?task_id=${encodeURIComponent(taskId)}`,
        { method: "GET", cache: "no-store" },
      );
      if (!data.success) throw new Error(data.error || "status unavailable");
      if (data.state === "PROGRESS") {
        const done = data.done || 0;
        const total = data.total || expectedTotal || 0;
        const pct = total > 0 ? Math.round((done / total) * 100) : 0;
        setGfStatus({
          tone: "info",
          text: `Grid: ${done}/${total} (${pct}%)${data.last_log ? `\n${data.last_log}` : ""}`,
        });
      } else if (data.state === "SUCCESS") {
        const cnt = data.done || data.total || expectedTotal || 0;
        setGfStatus({ tone: "ok", text: `Grid завершён: ${cnt} моделей обучено.` });
        setGfTaskId(null);
      } else if (data.state === "FAILURE") {
        setGfStatus({ tone: "bad", text: `Grid ошибка: ${data.error || "unknown"}` });
        setGfTaskId(null);
      } else {
        setGfStatus({ tone: "info", text: `Grid state=${data.state || "?"}...` });
      }
    } catch {
      // ignore single network errors
    }
  };

  useEffect(() => {
    if (!gfTaskId) return;
    const timer = window.setInterval(() => pollGridStatus(gfTaskId, gfTotalCombos), 3000);
    return () => window.clearInterval(timer);
  }, [gfTaskId, gfTotalCombos]);

  const startFullGrid = async (parallelMode = false) => {
    const task = gfFields.gfTask;
    const payload: Record<string, unknown> = {
      symbol: gfFields.gfSymbol,
      direction: gfFields.gfDirection,
      task,
      limit_candles: Number.parseInt(gfFields.gfLimitCandles, 10) || 100000,
      early_stopping_rounds: Number.parseInt(gfFields.gfEarlyStopping, 10) || 50,
      keep_top_n: Number.parseInt(gfFields.gfKeepTopN, 10) || 20,
      rank_by_proxy_pnl: gfFields.gfRankByProxy === "1",
      delete_rest: gfFields.gfDeleteRest === "1",
    };

    if (task === "directional") {
      payload.horizon_steps_list = rangeValues(gfFields.gfHsFrom, gfFields.gfHsTo, gfFields.gfHsStep).map((v) =>
        Math.round(v)
      );
      payload.threshold_list = rangeValues(gfFields.gfThrFrom, gfFields.gfThrTo, gfFields.gfThrStep).map((v) =>
        v / 100
      );
    } else {
      payload.max_hold_steps_list = rangeValues(gfFields.gfMhFrom, gfFields.gfMhTo, gfFields.gfMhStep).map((v) =>
        Math.round(v)
      );
      payload.min_profit_list = rangeValues(gfFields.gfMpFrom, gfFields.gfMpTo, gfFields.gfMpStep).map((v) =>
        v / 100
      );
      payload.fee_bps_list = rangeValues(gfFields.gfFeeFrom, gfFields.gfFeeTo, gfFields.gfFeeStep).map((v) =>
        v * 100
      );
      payload.p_enter_threshold_list = rangeValues(gfFields.gfPeFrom, gfFields.gfPeTo, gfFields.gfPeStep);
      if (task.startsWith("entry")) {
        payload.entry_tp_pct_list = rangeValues(gfFields.gfEtpFrom, gfFields.gfEtpTo, gfFields.gfEtpStep).map((v) =>
          v / 100
        );
        payload.entry_sl_pct_list = rangeValues(gfFields.gfEslFrom, gfFields.gfEslTo, gfFields.gfEslStep).map((v) =>
          v / 100
        );
        payload.entry_trail_pct_list = rangeValues(gfFields.gfEtrFrom, gfFields.gfEtrTo, gfFields.gfEtrStep).map((v) =>
          v / 100
        );
      }
    }

    payload.max_depth_list = rangeValues(gfFields.gfMdFrom, gfFields.gfMdTo, gfFields.gfMdStep).map((v) =>
      Math.round(v)
    );
    payload.learning_rate_list = rangeValues(gfFields.gfLrFrom, gfFields.gfLrTo, gfFields.gfLrStep);
    payload.n_estimators_list = rangeValues(gfFields.gfNeFrom, gfFields.gfNeTo, gfFields.gfNeStep).map((v) =>
      Math.round(v)
    );
    payload.subsample_list = rangeValues(gfFields.gfSsFrom, gfFields.gfSsTo, gfFields.gfSsStep);
    payload.colsample_bytree_list = rangeValues(gfFields.gfCbFrom, gfFields.gfCbTo, gfFields.gfCbStep);
    payload.reg_lambda_list = rangeValues(gfFields.gfRlFrom, gfFields.gfRlTo, gfFields.gfRlStep);
    payload.min_child_weight_list = rangeValues(gfFields.gfMcwFrom, gfFields.gfMcwTo, gfFields.gfMcwStep);
    payload.gamma_list = rangeValues(gfFields.gfGmFrom, gfFields.gfGmTo, gfFields.gfGmStep);
    payload.scale_pos_weight_list = rangeValues(gfFields.gfSpwFrom, gfFields.gfSpwTo, gfFields.gfSpwStep);

    payload.use_1m_microvol = gfFields.gfUse1mMicrovol === "1";
    payload.use_1m_momentum = gfFields.gfUse1mMomentum === "1";
    payload.use_1m_candle_structure = gfFields.gfUse1mCandleStructure === "1";
    payload.use_1m_volume = gfFields.gfUse1mVolume === "1";
    payload.use_1d_regime = gfFields.gfUse1dRegime === "1";
    payload.use_sr_features = gfFields.gfUseSrFeatures !== "0";
    payload.parallel_mode = parallelMode;
    if (parallelMode) {
      payload.parallel_workers = 4;
    }

    setGfStatus({
      tone: "info",
      text: `Запуск Full Grid${parallelMode ? " (parallel)" : ""}: ${gfTotalCombos} комбинаций…`,
    });
    try {
      const data = await jsonFetch<TrainResponse>("/api/xgb_training/train_grid_full", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(payload),
      });
      if (!data.success || !data.task_id) throw new Error(data.error || "grid start failed");
      setGfTaskId(data.task_id);
      setGfStatus({
        tone: "info",
        text: `Grid${parallelMode ? " (parallel)" : ""} запущен: 0/${gfTotalCombos} (celery: ${data.task_id})`,
      });
    } catch (error) {
      setGfTaskId(null);
      setGfStatus({
        tone: "bad",
        text: error instanceof Error ? error.message : String(error),
      });
    }
  };

  const createModelVersion = async () => {
    if (!createSymbol || !createRunId) {
      setCreateStatus({ tone: "bad", text: "Выберите символ и run" });
      return;
    }
    setCreateBusy(true);
    setCreateStatus({ tone: "info", text: "Создание новой версии..." });
    try {
      const data = await jsonFetch<CreateVersionResponse>("/api/xgb_training/create_model_version", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol: createSymbol,
          run_id: createRunId,
          ensemble: createEnsemble,
        }),
      });
      if (!data.success) throw new Error(data.error || "create failed");
      setCreateStatus({
        tone: "ok",
        text: `Версия создана: ${data.version}\nПуть: ${data.path}\nФайлы: ${(data.files || []).join(", ")}`,
      });
    } catch (error) {
      setCreateStatus({
        tone: "bad",
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setCreateBusy(false);
    }
  };

  const symbolOptions = symbols.length
    ? symbols.map((symbol) => ({ value: symbol, label: symbol }))
    : [{ value: "BTCUSDT", label: "BTCUSDT" }];

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      {symbolsError ? (
        <StatusBox tone="bad" text={`Ошибка загрузки символов: ${symbolsError}`} />
      ) : null}

      <SectionCard
        id="xgb-active-training"
        tone="blue"
        title="Статус обучения XGB"
        description="Показывает активные Celery-задачи обучения XGB и прогресс. Обновляется автоматически."
        meta={<span className="text-xs text-slate-500">auto-refresh: 5s</span>}
      >
        {activeError ? (
          <StatusBox tone="bad" text={`Ошибка проверки: ${activeError}`} />
        ) : active.length === 0 ? (
          <StatusBox tone="info" text="Активных обучений XGB нет" />
        ) : (
          <div className="space-y-3">
            {active.map((task) => {
              const done = task.done || 0;
              const total = task.total || 0;
              const pct = total > 0 ? Math.round((done / total) * 100) : 0;
              const stateLabel = task.state === "PROGRESS"
                ? "Обучение"
                : task.state === "PENDING"
                  ? "В очереди"
                  : task.state || "UNKNOWN";
              const title = `${stateLabel}: ${task.symbol || ""} / ${task.task_name || task.grid_type || ""}`.trim();
              const taskIdShort = (task.task_id || "").slice(0, 8);
              const logs = (task.logs_tail && task.logs_tail.length)
                ? task.logs_tail.join("\n")
                : task.last_log || "";
              return (
                <div key={task.task_id || `${task.symbol}-${task.grid_type}`} className="rounded-3xl border border-slate-200 bg-white p-5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="text-sm font-semibold text-slate-900">{title}</div>
                    <div className="text-xs text-slate-500">
                      task: {taskIdShort ? `${taskIdShort}…` : "—"}
                    </div>
                  </div>
                  {total > 0 ? (
                    <div className="mt-3">
                      <div className="h-5 overflow-hidden rounded-xl bg-slate-200">
                        <div
                          className="flex h-full items-center justify-center bg-blue-600 text-[11px] font-semibold text-white transition-[width]"
                          style={{ width: `${Math.max(2, pct)}%` }}
                        >
                          {pct}%
                        </div>
                      </div>
                      <div className="mt-2 flex flex-wrap gap-4 text-sm text-slate-700">
                        <span>
                          ✅ Готово: <b>{done}</b> / {total}
                        </span>
                        <span>
                          ⏱ Прошло: <b>{formatDuration(task.elapsed_sec)}</b>
                        </span>
                        <span>
                          ⏳ Осталось: <b>{formatDuration(task.eta_sec)}</b>
                        </span>
                        {task.avg_per_run_sec ? (
                          <span>⚡ {task.avg_per_run_sec.toFixed(1)}с/run</span>
                        ) : null}
                      </div>
                    </div>
                  ) : null}
                  {logs ? (
                    <pre className="mt-3 max-h-28 overflow-auto whitespace-pre-wrap rounded-2xl border border-slate-200 bg-slate-50 p-3 text-[11px] leading-5 text-slate-700">
                      {logs}
                    </pre>
                  ) : null}
                </div>
              );
            })}
          </div>
        )}
      </SectionCard>

      <SectionCard
        id="xgb-training"
        tone="indigo"
        title="Быстрый Grid (auto → финал)"
        description="Запускает базовый grid для выбранного symbol/task. Это короткий путь, когда не нужен полный перебор гиперпараметров."
      >
        <div className="grid gap-4 lg:grid-cols-3">
          <FieldGroup label="Символ">
            <Select value={gridSymbol} onChange={setGridSymbol} options={symbolOptions} />
          </FieldGroup>
          <FieldGroup label="Направление">
            <Select
              value={gridDirection}
              onChange={setGridDirection}
              options={[
                { value: "long", label: "Long" },
                { value: "short", label: "Short" },
              ]}
            />
          </FieldGroup>
          <FieldGroup label="Task">
            <Select
              value={gridTask}
              onChange={setGridTask}
              options={[
                { value: "directional", label: "directional (hold/buy/sell)" },
                { value: "entry_long", label: "entry_long (flat→enter)" },
                { value: "exit_long", label: "exit_long (long→exit)" },
                { value: "entry_short", label: "entry_short (flat→enter)" },
                { value: "exit_short", label: "exit_short (short→exit)" },
              ]}
            />
          </FieldGroup>
          <FieldGroup label="max_hold_steps (5m)">
            <Input value={gridMaxHold} onChange={setGridMaxHold} type="number" min={2} step={1} />
          </FieldGroup>
          <FieldGroup label="fee, % (round-trip)">
            <Input value={gridFeePct} onChange={setGridFeePct} type="number" min={0} step={0.01} />
          </FieldGroup>
          <FieldGroup label="min_profit, %">
            <Input value={gridMinProfitPct} onChange={setGridMinProfitPct} type="number" min={0} step={0.01} />
          </FieldGroup>
          <FieldGroup label="label_delta, % (legacy, сейчас не используется)">
            <Input value={gridLabelDeltaPct} onChange={setGridLabelDeltaPct} type="number" min={0} step={0.01} />
          </FieldGroup>
          <FieldGroup label="limit_candles (5m)">
            <Input value={gridLimitCandles} onChange={setGridLimitCandles} type="number" min={1000} step={1000} />
          </FieldGroup>
        </div>

        <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
          <button
            type="button"
            className="rounded-2xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-700"
            onClick={startGridSelected}
          >
            Grid (auto → финал)
          </button>
          <div className="text-xs text-slate-500">
            label_delta перенесён как в legacy, но backend его не принимает (как и раньше).
          </div>
        </div>
        {gridStatus ? <StatusBox tone={gridStatus.tone} text={gridStatus.text} /> : null}
      </SectionCard>

      <SectionCard
        id="xgb-grid-full"
        tone="amber"
        title="Full Grid Search"
        description="Полный перебор: labeling + hyper-params + (опционально) policy. Чтобы не было каши, параметры разбиты на раскрывающиеся блоки."
        meta={
          <span>
            Комбинаций:{" "}
            <span className="font-semibold text-amber-700">{gfTotalCombos}</span>
          </span>
        }
      >
        <div className="grid gap-4 lg:grid-cols-3">
          <FieldGroup label="Символ">
            <Select
              value={gfFields.gfSymbol}
              onChange={(value) => setGfFields((s) => ({ ...s, gfSymbol: value }))}
              options={symbolOptions}
            />
          </FieldGroup>
          <FieldGroup label="Task">
            <Select
              value={gfFields.gfTask}
              onChange={(value) => setGfFields((s) => ({ ...s, gfTask: value }))}
              options={[
                { value: "entry_long", label: "entry_long" },
                { value: "exit_long", label: "exit_long" },
                { value: "entry_short", label: "entry_short" },
                { value: "exit_short", label: "exit_short" },
                { value: "directional", label: "directional" },
              ]}
            />
          </FieldGroup>
          <FieldGroup label="Direction">
            <Select
              value={gfFields.gfDirection}
              onChange={(value) => setGfFields((s) => ({ ...s, gfDirection: value }))}
              options={[
                { value: "long", label: "Long" },
                { value: "short", label: "Short" },
              ]}
            />
          </FieldGroup>
          <FieldGroup label="limit_candles (5m)">
            <Input
              value={gfFields.gfLimitCandles}
              onChange={(value) => setGfFields((s) => ({ ...s, gfLimitCandles: value }))}
              type="number"
              min={5000}
              step={5000}
            />
          </FieldGroup>
          <FieldGroup label="early_stopping_rounds">
            <Input
              value={gfFields.gfEarlyStopping}
              onChange={(value) => setGfFields((s) => ({ ...s, gfEarlyStopping: value }))}
              type="number"
              min={0}
              step={10}
            />
          </FieldGroup>
          <FieldGroup label="keep_top_n">
            <Input
              value={gfFields.gfKeepTopN}
              onChange={(value) => setGfFields((s) => ({ ...s, gfKeepTopN: value }))}
              type="number"
              min={1}
              step={1}
            />
          </FieldGroup>
          <FieldGroup label="rank_by_proxy_pnl">
            <div className="flex h-full items-center">
              <input
                type="checkbox"
                className="h-5 w-5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                checked={gfFields.gfRankByProxy === "1"}
                onChange={(e) =>
                  setGfFields((s) => ({ ...s, gfRankByProxy: e.target.checked ? "1" : "0" }))
                }
              />
            </div>
          </FieldGroup>
          <FieldGroup label="delete_rest">
            <div className="flex h-full items-center">
              <input
                type="checkbox"
                className="h-5 w-5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                checked={gfFields.gfDeleteRest === "1"}
                onChange={(e) =>
                  setGfFields((s) => ({ ...s, gfDeleteRest: e.target.checked ? "1" : "0" }))
                }
              />
            </div>
          </FieldGroup>
        </div>

        <div className="mt-6 space-y-4">
          <details className="rounded-3xl border border-slate-200 bg-slate-50 p-5" open>
            <summary className="cursor-pointer select-none text-sm font-semibold text-slate-900">
              Labeling params (from / to / step)
              <span className="ml-2 text-xs font-normal text-slate-500">
                {gfShowDirectional ? "directional" : "entry/exit"}
              </span>
            </summary>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              directional использует <code>threshold</code>; entry_* использует <code>min_profit</code>. Значения в % будут
              сконвертированы как в legacy.
            </p>
            {gfShowDirectional ? (
            <div className="mt-4 grid gap-4 lg:grid-cols-2">
              <FieldGroup label="horizon_steps">
                <div className="grid grid-cols-3 gap-2">
                  <Input value={gfFields.gfHsFrom} onChange={(v) => setGfFields((s) => ({ ...s, gfHsFrom: v }))} type="number" step={1} />
                  <Input value={gfFields.gfHsTo} onChange={(v) => setGfFields((s) => ({ ...s, gfHsTo: v }))} type="number" step={1} />
                  <Input value={gfFields.gfHsStep} onChange={(v) => setGfFields((s) => ({ ...s, gfHsStep: v }))} type="number" step={1} />
                </div>
              </FieldGroup>
              <FieldGroup label="threshold, % (0.2 = 0.002)">
                <div className="grid grid-cols-3 gap-2">
                  <Input value={gfFields.gfThrFrom} onChange={(v) => setGfFields((s) => ({ ...s, gfThrFrom: v }))} type="number" step={0.01} />
                  <Input value={gfFields.gfThrTo} onChange={(v) => setGfFields((s) => ({ ...s, gfThrTo: v }))} type="number" step={0.01} />
                  <Input value={gfFields.gfThrStep} onChange={(v) => setGfFields((s) => ({ ...s, gfThrStep: v }))} type="number" step={0.01} />
                </div>
              </FieldGroup>
            </div>
          ) : (
            <div className="mt-4 grid gap-4 lg:grid-cols-3">
              <FieldGroup label="max_hold_steps">
                <div className="grid grid-cols-3 gap-2">
                  <Input value={gfFields.gfMhFrom} onChange={(v) => setGfFields((s) => ({ ...s, gfMhFrom: v }))} type="number" step={1} />
                  <Input value={gfFields.gfMhTo} onChange={(v) => setGfFields((s) => ({ ...s, gfMhTo: v }))} type="number" step={1} />
                  <Input value={gfFields.gfMhStep} onChange={(v) => setGfFields((s) => ({ ...s, gfMhStep: v }))} type="number" step={1} />
                </div>
              </FieldGroup>
              <FieldGroup label="min_profit, % (1 = 1%)">
                <div className="grid grid-cols-3 gap-2">
                  <Input value={gfFields.gfMpFrom} onChange={(v) => setGfFields((s) => ({ ...s, gfMpFrom: v }))} type="number" step={0.01} />
                  <Input value={gfFields.gfMpTo} onChange={(v) => setGfFields((s) => ({ ...s, gfMpTo: v }))} type="number" step={0.01} />
                  <Input value={gfFields.gfMpStep} onChange={(v) => setGfFields((s) => ({ ...s, gfMpStep: v }))} type="number" step={0.01} />
                </div>
              </FieldGroup>
              <FieldGroup label="fee, % (0.06 = 6bps)">
                <div className="grid grid-cols-3 gap-2">
                  <Input value={gfFields.gfFeeFrom} onChange={(v) => setGfFields((s) => ({ ...s, gfFeeFrom: v }))} type="number" step={0.01} />
                  <Input value={gfFields.gfFeeTo} onChange={(v) => setGfFields((s) => ({ ...s, gfFeeTo: v }))} type="number" step={0.01} />
                  <Input value={gfFields.gfFeeStep} onChange={(v) => setGfFields((s) => ({ ...s, gfFeeStep: v }))} type="number" step={0.01} />
                </div>
              </FieldGroup>
            </div>
          )}

          {!gfShowDirectional ? (
            <div className="mt-4 grid gap-4 lg:grid-cols-2">
              <FieldGroup label="p_enter_threshold (from / to / step)">
                <div className="grid grid-cols-3 gap-2">
                  <Input value={gfFields.gfPeFrom} onChange={(v) => setGfFields((s) => ({ ...s, gfPeFrom: v }))} type="number" step={0.05} />
                  <Input value={gfFields.gfPeTo} onChange={(v) => setGfFields((s) => ({ ...s, gfPeTo: v }))} type="number" step={0.05} />
                  <Input value={gfFields.gfPeStep} onChange={(v) => setGfFields((s) => ({ ...s, gfPeStep: v }))} type="number" step={0.05} />
                </div>
              </FieldGroup>
              {gfShowExitPolicy ? (
                <FieldGroup label="Exit-policy for entry_* (TP/SL/Trail)">
                  <div className="grid gap-2">
                    <div className="grid grid-cols-3 gap-2">
                      <Input value={gfFields.gfEtpFrom} onChange={(v) => setGfFields((s) => ({ ...s, gfEtpFrom: v }))} type="number" step={0.1} />
                      <Input value={gfFields.gfEtpTo} onChange={(v) => setGfFields((s) => ({ ...s, gfEtpTo: v }))} type="number" step={0.1} />
                      <Input value={gfFields.gfEtpStep} onChange={(v) => setGfFields((s) => ({ ...s, gfEtpStep: v }))} type="number" step={0.1} />
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <Input value={gfFields.gfEslFrom} onChange={(v) => setGfFields((s) => ({ ...s, gfEslFrom: v }))} type="number" step={0.1} />
                      <Input value={gfFields.gfEslTo} onChange={(v) => setGfFields((s) => ({ ...s, gfEslTo: v }))} type="number" step={0.1} />
                      <Input value={gfFields.gfEslStep} onChange={(v) => setGfFields((s) => ({ ...s, gfEslStep: v }))} type="number" step={0.1} />
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <Input value={gfFields.gfEtrFrom} onChange={(v) => setGfFields((s) => ({ ...s, gfEtrFrom: v }))} type="number" step={0.1} />
                      <Input value={gfFields.gfEtrTo} onChange={(v) => setGfFields((s) => ({ ...s, gfEtrTo: v }))} type="number" step={0.1} />
                      <Input value={gfFields.gfEtrStep} onChange={(v) => setGfFields((s) => ({ ...s, gfEtrStep: v }))} type="number" step={0.1} />
                    </div>
                  </div>
                </FieldGroup>
              ) : null}
            </div>
          ) : null}
          </details>

          <details className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
            <summary className="cursor-pointer select-none text-sm font-semibold text-slate-900">
              Features (1m/1d/SR)
            </summary>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              Если включено, train/live/OOS требуют реальные свечи Bybit соответствующих таймфреймов.
            </p>
            <div className="mt-4 grid gap-3 lg:grid-cols-3">
            <label className="flex items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={gfFields.gfUse1mMicrovol === "1"}
                onChange={(e) =>
                  setGfFields((s) => ({ ...s, gfUse1mMicrovol: e.target.checked ? "1" : "0" }))
                }
              />
              micro-volatility
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={gfFields.gfUse1mMomentum === "1"}
                onChange={(e) =>
                  setGfFields((s) => ({ ...s, gfUse1mMomentum: e.target.checked ? "1" : "0" }))
                }
              />
              momentum
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={gfFields.gfUse1mCandleStructure === "1"}
                onChange={(e) =>
                  setGfFields((s) => ({
                    ...s,
                    gfUse1mCandleStructure: e.target.checked ? "1" : "0",
                  }))
                }
              />
              candle structure
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={gfFields.gfUse1mVolume === "1"}
                onChange={(e) =>
                  setGfFields((s) => ({ ...s, gfUse1mVolume: e.target.checked ? "1" : "0" }))
                }
              />
              volume proxy
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={gfFields.gfUse1dRegime === "1"}
                onChange={(e) =>
                  setGfFields((s) => ({ ...s, gfUse1dRegime: e.target.checked ? "1" : "0" }))
                }
              />
              use 1d regime
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={gfFields.gfUseSrFeatures !== "0"}
                onChange={(e) =>
                  setGfFields((s) => ({ ...s, gfUseSrFeatures: e.target.checked ? "1" : "0" }))
                }
              />
              use support/resistance
            </label>
          </div>
          </details>

          <details className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
            <summary className="cursor-pointer select-none text-sm font-semibold text-slate-900">
              Model hyper-params (from / to / step)
            </summary>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              Большой блок, поэтому по умолчанию свернут. Можно открыть и настроить диапазоны.
            </p>
            <div className="mt-4 grid gap-4 lg:grid-cols-3">
            {[
              ["max_depth", "gfMdFrom", "gfMdTo", "gfMdStep", 1],
              ["learning_rate", "gfLrFrom", "gfLrTo", "gfLrStep", 0.001],
              ["n_estimators", "gfNeFrom", "gfNeTo", "gfNeStep", 100],
              ["subsample", "gfSsFrom", "gfSsTo", "gfSsStep", 0.05],
              ["colsample_bytree", "gfCbFrom", "gfCbTo", "gfCbStep", 0.05],
              ["reg_lambda", "gfRlFrom", "gfRlTo", "gfRlStep", 0.5],
              ["min_child_weight", "gfMcwFrom", "gfMcwTo", "gfMcwStep", 1],
              ["gamma", "gfGmFrom", "gfGmTo", "gfGmStep", 0.1],
              ["scale_pos_weight (-1=auto)", "gfSpwFrom", "gfSpwTo", "gfSpwStep", 0.5],
            ].map(([label, from, to, step, stepDefault]) => (
              <FieldGroup key={from} label={String(label)}>
                <div className="grid grid-cols-3 gap-2">
                  <Input
                    value={gfFields[from as string]}
                    onChange={(v) => setGfFields((s) => ({ ...s, [from as string]: v }))}
                    type="number"
                    step={Number(stepDefault)}
                  />
                  <Input
                    value={gfFields[to as string]}
                    onChange={(v) => setGfFields((s) => ({ ...s, [to as string]: v }))}
                    type="number"
                    step={Number(stepDefault)}
                  />
                  <Input
                    value={gfFields[step as string]}
                    onChange={(v) => setGfFields((s) => ({ ...s, [step as string]: v }))}
                    type="number"
                    step={Number(stepDefault)}
                  />
                </div>
              </FieldGroup>
            ))}
          </div>
          </details>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            className="rounded-2xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-700"
            onClick={() => startFullGrid(false)}
          >
            Запустить Full Grid
          </button>
          <button
            type="button"
            className="rounded-2xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700"
            onClick={() => startFullGrid(true)}
          >
            Запустить Full Grid (parallel)
          </button>
          <button
            type="button"
            className="rounded-2xl border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm hover:bg-slate-50"
            onClick={shrinkRegGamma}
          >
            Сжать reg_lambda + gamma
          </button>
          <button
            type="button"
            className="rounded-2xl border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm hover:bg-slate-50"
            onClick={resetFullGridFields}
          >
            Сбросить поля
          </button>
          {gfTaskId ? (
            <div className="text-xs text-slate-500">celery: {gfTaskId}</div>
          ) : null}
        </div>
        {gfStatus ? <StatusBox tone={gfStatus.tone} text={gfStatus.text} /> : null}
      </SectionCard>

      <SectionCard
        id="xgb-create-model"
        tone="emerald"
        title="Создать новую версию модели"
        description="Копирует выбранный run из result/xgb в models/xgb/<symbol>/<ensemble>/vN и обновляет current."
      >
        <div className="grid gap-4 lg:grid-cols-3">
          <FieldGroup label="Символ">
            <Select
              value={createSymbol}
              onChange={setCreateSymbol}
              options={[{ value: "", label: "Выберите..." }, ...symbolOptions]}
            />
          </FieldGroup>
          <FieldGroup label="Run из result/xgb">
            <Select
              value={createRunId}
              onChange={setCreateRunId}
              options={
                createRuns && createRuns.length
                  ? createRuns.map((r) => {
                      const runId = String(r.run_id || "");
                      const acc = typeof r.val_acc === "number" ? ` acc=${(r.val_acc * 100).toFixed(1)}%` : "";
                      const dir = r.direction ? ` [${r.direction}]` : "";
                      const task = r.task ? ` (${r.task})` : "";
                      return { value: runId, label: `${runId}${task}${dir}${acc}` };
                    })
                  : [{ value: "", label: createSymbol ? "Нет run-ов" : "Сначала выберите символ..." }]
              }
            />
          </FieldGroup>
          <FieldGroup label="Ансамбль назначения">
            <Select
              value={createEnsemble}
              onChange={setCreateEnsemble}
              options={[
                { value: "ensemble-a", label: "ensemble-a" },
                { value: "ensemble-b", label: "ensemble-b" },
                { value: "ensemble-c", label: "ensemble-c" },
              ]}
            />
          </FieldGroup>
        </div>

        <button
          type="button"
          className="rounded-2xl bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-emerald-700 disabled:opacity-60"
          onClick={createModelVersion}
          disabled={createBusy}
        >
          {createBusy ? "Создание..." : "Создать новую версию"}
        </button>
        {createStatus ? <StatusBox tone={createStatus.tone} text={createStatus.text} /> : null}
      </SectionCard>
    </div>
  );
}

