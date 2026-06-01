"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { XgbOosRun } from "@/lib/backend";

type XgbOosViewProps = {
  initialRuns: XgbOosRun[];
};

type CsvFile = {
  name: string;
  size: number;
  mtime: string;
};

type StatusTone = "info" | "success" | "error";

type StatusState = {
  tone: StatusTone;
  text: React.ReactNode;
} | null;

type OosMetrics = {
  val_acc?: number | null;
  y_counts_val?: unknown;
  y_non_hold_rate_val?: number | null;
  pred_non_hold_rate_val?: number | null;
  f1_buy_sell_val?: number | null;
  f1_val?: number[] | null;
  precision_val?: number[] | null;
  recall_val?: number[] | null;
};

type BacktestMetrics = {
  skip?: boolean;
  pnl_total?: number | null;
  roi_pct?: number | null;
  trades_count?: number | null;
  wins?: number | null;
  losses?: number | null;
  winrate?: number | null;
  profit_factor?: number | null;
  max_dd?: number | null;
  avg_bars_held?: number | null;
  start_capital?: number | null;
  equity_end?: number | null;
  reason_counts?: Record<string, unknown> | null;
  signal_exit?: {
    enabled?: boolean;
    active?: boolean;
    window?: number | null;
    start_step?: number | null;
    start_pct?: number | null;
    history_size?: number | null;
    last_signal?: number | null;
    last_threshold?: number | null;
    avg_signal?: number | null;
    avg_threshold?: number | null;
    ready?: boolean;
  } | null;
};

type OosResult = {
  success?: boolean;
  error?: string;
  symbol?: string | null;
  task?: string | null;
  direction?: string | null;
  days?: number | null;
  bars?: number | null;
  run_dir?: string | null;
  exit_mode?: string | null;
  atr_len?: number | null;
  atr_mult?: number | null;
  cfg_snapshot?: Record<string, unknown> | null;
  oos_metrics?: OosMetrics | null;
  backtest?: BacktestMetrics | null;
  signal_exit_enabled?: boolean;
  signal_exit_window?: number | null;
  signal_exit_start_pct?: number | null;
  signal_exit_start_step?: number | null;
  signal_exit_threshold?: number | null;
};

type TaskStatusResponse = {
  success?: boolean;
  state?: string;
  result?: OosResult;
  error?: string;
  traceback?: string;
};

type BatchTaskMeta = {
  result_dir: string;
  days: number;
  exit_mode: string;
  task_id: string;
  p_enter_threshold?: number | null;
  signal_exit_threshold?: number | null;
};

type BatchAsyncResponse = {
  success?: boolean;
  error?: string;
  batch_id?: string;
  tasks?: BatchTaskMeta[];
};

type BatchStatusResponse = {
  success?: boolean;
  error?: string;
  batch_id?: string;
  expected?: number;
  done?: number;
  success_count?: number;
  status?: string;
  csv_filename?: string | null;
  csv_rows?: number;
  elapsed_sec?: number;
  eta_sec?: number;
  avg_per_run_sec?: number;
};

type BatchResultsResponse = {
  success?: boolean;
  error?: string;
  batch_id?: string;
  offset?: number;
  limit?: number;
  total?: number;
  results?: OosResult[];
};

type SaveExperimentResponse = {
  success?: boolean;
  error?: string;
  path?: string;
  selected_count?: number;
};

type ForceFinalizeResponse = {
  success?: boolean;
  error?: string;
  batch_id?: string;
  filename?: string;
  rows?: number;
  forced?: boolean;
  expected?: number;
  done?: number;
  success_count?: number;
};

type SelectMode =
  | "all_safe"
  | "all"
  | "collapsed"
  | "pred_lt_005"
  | "pred_lt_01"
  | "weak_high"
  | "overtrade_03"
  | "proxy0"
  | "f1lt10"
  | "f1lt30"
  | "recall_lt_03"
  | "prec_lt_03"
  | "y_nonhold_lt_002"
  | "accdelta_le_neg001";

const SELECT_MODE_OPTIONS: Array<{ value: SelectMode; label: string }> = [
  { value: "all_safe", label: "ALL safe" },
  { value: "all", label: "ALL strict" },
  { value: "collapsed", label: "collapsed" },
  { value: "pred_lt_005", label: "pred_non_hold < 0.005" },
  { value: "pred_lt_01", label: "pred_non_hold < 0.01" },
  { value: "weak_high", label: "pred_non_hold > 0.2" },
  { value: "overtrade_03", label: "pred_non_hold > 0.30" },
  { value: "proxy0", label: "proxy_trades = 0" },
  { value: "f1lt10", label: "f1(1) < 0.10" },
  { value: "f1lt30", label: "f1(1) < 0.30" },
  { value: "recall_lt_03", label: "recall(1) < 0.30" },
  { value: "prec_lt_03", label: "prec(1) < 0.30" },
  { value: "y_nonhold_lt_002", label: "y_non_hold < 0.002" },
  { value: "accdelta_le_neg001", label: "acc_delta <= -0.01" },
];

const DAYS_GRID = ["30", "60", "90"];
const EXIT_GRID = ["policy", "hold_steps", "atr_trail"];
const ENSEMBLES = ["ensemble-a", "ensemble-b", "ensemble-c"];
const SIGNAL_EXIT_WINDOW_DEFAULT = "20";
const SIGNAL_EXIT_START_PCT_DEFAULT = "65";
const RUNS_PER_PAGE = 200;
const BATCH_POLL_CONCURRENCY = 8;
const POLL_INTERVAL_MS = 1500;
const POLL_NETWORK_RETRY_DELAY_MS = 1200;
const POLL_MAX_CONSECUTIVE_NETWORK_ERRORS = 20;
const OOS_BATCH_CACHE_KEY = "xgb_oos:last_batch_v2";
const OOS_OLD_BATCH_CACHE_KEYS = ["xgb_oos:last_batch_v1"];
const OOS_LIVE_SNAPSHOT_STALE_SEC = 120;
const BATCH_CSV_WAIT_MS = 30 * 60 * 1000;
const BATCH_CSV_POLL_MS = 2000;
const OOS_STALE_NOTICE =
  "Сохранённый snapshot прогресса (не live): страница была перезагружена или контейнер рестартован.\n" +
  "Для нового live нажмите Run selected OOS или Очистить кеш блока.";
const OOS_BATCH_HINT =
  "Подсказка: Celery часто не отдаёт STARTED (без task_track_started), поэтому задача может висеть как PENDING до SUCCESS. Для проверки worker: docker logs -f celery-oos.";

function formatNumber(value: number | null | undefined, digits = 4): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return new Intl.NumberFormat("ru-RU", {
    maximumFractionDigits: digits,
  }).format(value);
}

function formatSignedPercentFromRatio(value: number | null | undefined, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

function compactPath(value: string | null | undefined, parts = 5): string {
  if (!value) {
    return "—";
  }
  return value.replace(/\\/g, "/").split("/").filter(Boolean).slice(-parts).join("/");
}

function shortRunName(resultDir: string | null | undefined): string {
  if (!resultDir) {
    return "unknown-run";
  }
  const parts = resultDir.replace(/\\/g, "/").split("/").filter(Boolean);
  return parts.at(-1) ?? "unknown-run";
}

function createExperimentName() {
  const now = new Date();
  const stamp = [
    now.getUTCFullYear(),
    String(now.getUTCMonth() + 1).padStart(2, "0"),
    String(now.getUTCDate()).padStart(2, "0"),
    "_",
    String(now.getUTCHours()).padStart(2, "0"),
    String(now.getUTCMinutes()).padStart(2, "0"),
    String(now.getUTCSeconds()).padStart(2, "0"),
  ].join("");
  const uuid =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID().slice(0, 8)
      : Math.random().toString(36).slice(2, 10);
  return `preset_${uuid}_${stamp}`;
}

function parseLooseNumber(raw: string): number | null {
  const normalized = raw.trim().replace(",", ".");
  if (!normalized) {
    return null;
  }
  const value = Number.parseFloat(normalized);
  return Number.isFinite(value) ? value : null;
}

function toNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function toNumberArray(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is number => typeof item === "number" && Number.isFinite(item));
}

function metricValue(source: Record<string, unknown>, key: string): number | null {
  return toNumber(source[key]);
}

function arrayMetricItem(source: Record<string, unknown>, key: string, index: number): number | null {
  const values = toNumberArray(source[key]);
  return typeof values[index] === "number" ? values[index] : null;
}

function proxyTradesCount(source: Record<string, unknown>): number | null {
  const proxy = source.proxy_pnl_val;
  if (!proxy || typeof proxy !== "object") {
    return null;
  }
  return toNumber((proxy as Record<string, unknown>).trades);
}

function normalizeXgbOosRun(raw: Record<string, unknown>): XgbOosRun {
  const metrics =
    raw.metrics && typeof raw.metrics === "object"
      ? (raw.metrics as Record<string, unknown>)
      : {};
  const cfg = raw.cfg && typeof raw.cfg === "object" ? (raw.cfg as Record<string, unknown>) : {};

  return {
    symbol: String(raw.symbol ?? ""),
    runName: String(raw.run_name ?? raw.runName ?? ""),
    direction: raw.direction ? String(raw.direction) : null,
    task: raw.task ? String(raw.task) : null,
    source: raw.source ? String(raw.source) : null,
    gridId: raw.grid_id ? String(raw.grid_id) : raw.gridId ? String(raw.gridId) : null,
    resultDir: String(raw.result_dir ?? raw.resultDir ?? ""),
    modelPath: raw.model_path ? String(raw.model_path) : raw.modelPath ? String(raw.modelPath) : null,
    mtime: toNumber(raw.mtime),
    metrics,
    cfg,
  };
}

function normalizeXgbOosRuns(rawRuns: unknown): XgbOosRun[] {
  if (!Array.isArray(rawRuns)) {
    return [];
  }
  return rawRuns
    .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
    .map((item) => normalizeXgbOosRun(item));
}

function renderPrefetchError(payload: { error?: string; prefetch?: Record<string, unknown> | null }) {
  const prefetch = payload.prefetch ?? {};
  const symbol = prefetch.symbol ? String(prefetch.symbol) : "—";
  const age = prefetch.age_sec != null ? String(prefetch.age_sec) : "—";
  const lastTs = prefetch.db_last_ts != null ? String(prefetch.db_last_ts) : "—";
  const fetchError = prefetch.fetch_error ? String(prefetch.fetch_error) : "";
  const started = Boolean(prefetch.prefetch_task_started);
  const taskId = prefetch.prefetch_task_id ? String(prefetch.prefetch_task_id) : "—";
  const wait =
    prefetch.prefetch_wait && typeof prefetch.prefetch_wait === "object"
      ? (prefetch.prefetch_wait as Record<string, unknown>)
      : null;
  const waitState = wait?.state ? String(wait.state) : "";
  const waitReady = wait?.ready === true;

  const lines = [`${payload.error || "Prefetch failed"}`];
  if (started) {
    lines.push(`Prefetch task: ${taskId}`);
  }
  if (wait) {
    lines.push(
      waitReady
        ? `Prefetch finished: state=${waitState || "UNKNOWN"}`
        : `Prefetch still running or timed out: state=${waitState || "UNKNOWN"}`,
    );
  }
  lines.push("OOS не запущен: после prefetch в БД всё ещё не хватает нужных свечей.");
  lines.push(`sym=${symbol} age_sec=${age} db_last_ts=${lastTs}`);
  if (fetchError) {
    lines.push(`fetch_error=${fetchError}`);
  }
  return lines.join("\n");
}

function buildPEnterGrid(fromRaw: string, toRaw: string, stepRaw: string): number[] {
  const from = parseLooseNumber(fromRaw);
  const to = parseLooseNumber(toRaw);
  const step = Math.abs(parseLooseNumber(stepRaw) ?? Number.NaN);
  if (from === null || to === null || !Number.isFinite(step) || step <= 0) {
    throw new Error("bad p_enter grid values");
  }
  if (from <= 0 || from >= 1 || to <= 0 || to >= 1) {
    throw new Error("p_enter grid values must be between 0 and 1");
  }

  const direction = from >= to ? -1 : 1;
  const values: number[] = [];
  for (let value = from, guard = 0; guard < 200; value += direction * step, guard += 1) {
    if ((direction < 0 && value < to - 1e-9) || (direction > 0 && value > to + 1e-9)) {
      break;
    }
    values.push(Number(value.toFixed(6)));
  }
  return [...new Set(values)];
}

function formatDuration(sec?: number): string {
  if (!sec || sec <= 0) return "—";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}ч ${m}м`;
  if (m > 0) return `${m}м ${s}с`;
  return `${s}с`;
}

function statusClasses(tone: StatusTone): string {
  if (tone === "success") {
    return "border-emerald-200 bg-emerald-50 text-emerald-700";
  }
  if (tone === "error") {
    return "border-rose-200 bg-rose-50 text-rose-700";
  }
  return "border-slate-200 bg-slate-50 text-slate-700";
}

function selectionModeMatches(run: XgbOosRun, mode: SelectMode) {
  const metrics = run.metrics;
  const predNonHold = metricValue(metrics, "pred_non_hold_rate_val");
  const f1One = arrayMetricItem(metrics, "f1_val", 1);
  const precisionOne = arrayMetricItem(metrics, "precision_val", 1);
  const recallOne = arrayMetricItem(metrics, "recall_val", 1);
  const valAcc = metricValue(metrics, "val_acc");
  const yNonHold = metricValue(metrics, "y_non_hold_rate_val");
  const proxyTrades = proxyTradesCount(metrics);

  const hasPred = predNonHold !== null;
  const isCollapsed = hasPred && predNonHold < 0.001;
  const isPredLt005 = hasPred && predNonHold > 0 && predNonHold < 0.005;
  const isPredLt01 = hasPred && predNonHold > 0 && predNonHold < 0.01;
  const isWeakHigh = hasPred && predNonHold > 0.2;
  const isOvertrade = hasPred && predNonHold > 0.3;
  const isProxy0 = proxyTrades === 0;
  const isF1Lt10 = f1One !== null && f1One < 0.1;
  const isF1Lt30 = f1One !== null && f1One < 0.3;
  const isRecallLt03 = recallOne !== null && recallOne < 0.3;
  const isPrecLt03 = precisionOne !== null && precisionOne < 0.3;
  const isYNonHoldLt002 = yNonHold !== null && yNonHold < 0.002;
  const isAccDeltaBad =
    valAcc !== null && yNonHold !== null ? valAcc - (1 - yNonHold) <= -0.01 : false;

  if (mode === "all") {
    return (
      isCollapsed || isWeakHigh || isOvertrade || isProxy0 || isF1Lt10 || isAccDeltaBad
    );
  }
  if (mode === "all_safe") {
    return isCollapsed || isWeakHigh || isOvertrade || isProxy0 || isF1Lt10;
  }
  if (mode === "pred_lt_005") {
    return isPredLt005;
  }
  if (mode === "pred_lt_01") {
    return isPredLt01;
  }
  if (mode === "weak_high") {
    return isWeakHigh;
  }
  if (mode === "overtrade_03") {
    return isOvertrade;
  }
  if (mode === "proxy0") {
    return isProxy0;
  }
  if (mode === "f1lt10") {
    return isF1Lt10;
  }
  if (mode === "f1lt30") {
    return isF1Lt30;
  }
  if (mode === "recall_lt_03") {
    return isRecallLt03;
  }
  if (mode === "prec_lt_03") {
    return isPrecLt03;
  }
  if (mode === "y_nonhold_lt_002") {
    return isYNonHoldLt002;
  }
  return isAccDeltaBad;
}

function batchResultLine(taskMeta: BatchTaskMeta, result?: OosResult, failedByState = false) {
  const resultThreshold = toNumber(result?.cfg_snapshot?.p_enter_threshold);
  const peThreshold = taskMeta.p_enter_threshold ?? resultThreshold;
  const pePart = peThreshold != null ? ` pe=${peThreshold.toFixed(2)}` : "";
  const prefix = `[${shortRunName(taskMeta.result_dir)}] d=${taskMeta.days} exit=${taskMeta.exit_mode}${pePart}`;
  if (failedByState) {
    return `FAIL ${prefix} :: ${result?.error || "task crashed"}`;
  }
  if (!result || result.success !== true) {
    return `FAIL ${prefix} :: ${result?.error || "unknown error"}`;
  }
  const metrics = result.oos_metrics ?? {};
  const backtest = result.backtest ?? {};
  const valAcc = formatNumber(metrics.val_acc ?? null, 4);
  const roi = formatNumber(backtest.roi_pct ?? null, 2);
  const trades = formatNumber(backtest.trades_count ?? null, 0);
  return `OK ${prefix} :: val_acc=${valAcc} roi=${roi}% trades=${trades}`;
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

class JsonRequestError extends Error {
  payload: unknown;

  constructor(message: string, payload: unknown) {
    super(message);
    this.name = "JsonRequestError";
    this.payload = payload;
  }
}

async function waitForBatchCsv(batchId: string): Promise<{ filename: string; rows?: number }> {
  const started = Date.now();
  while (Date.now() - started < BATCH_CSV_WAIT_MS) {
    const status = await jsonFetch<BatchStatusResponse>(
      `/api/oos_xgb/batch_status?batch_id=${encodeURIComponent(batchId)}`,
      { method: "GET", cache: "no-store" },
    );
    if (!status.success) {
      throw new Error(status.error || "batch status failed");
    }
    if (["done", "partial_timeout", "cancelled"].includes(status.status || "") && status.csv_filename) {
      return { filename: status.csv_filename, rows: status.csv_rows };
    }
    if (status.status === "failed") {
      throw new Error(status.error || "batch CSV failed on server");
    }
    await new Promise((resolve) => setTimeout(resolve, BATCH_CSV_POLL_MS));
  }
  throw new Error("timeout waiting for server CSV");
}

async function pollTask(
  taskId: string,
  onState?: (state: string) => void,
): Promise<OosResult> {
  let consecutiveNetworkErrors = 0;
  while (true) {
    let response: TaskStatusResponse;
    try {
      response = await jsonFetch<TaskStatusResponse>(
        `/api/oos_xgb/test_status?task_id=${encodeURIComponent(taskId)}`,
        {
          method: "GET",
          cache: "no-store",
        },
      );
      consecutiveNetworkErrors = 0;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const isNetworkError = message.toLowerCase().includes("failed to fetch");
      if (!isNetworkError) {
        throw error;
      }
      consecutiveNetworkErrors += 1;
      if (consecutiveNetworkErrors >= POLL_MAX_CONSECUTIVE_NETWORK_ERRORS) {
        throw new Error(`Polling status unavailable: ${message}`);
      }
      await new Promise((resolve) => setTimeout(resolve, POLL_NETWORK_RETRY_DELAY_MS));
      continue;
    }

    if (typeof response.state === "string") {
      onState?.(response.state);
    }

    if (response.state === "SUCCESS") {
      return response.result ?? { success: false, error: "Empty result" };
    }
    if (response.state === "FAILURE") {
      const details = [response.error, response.result?.error].filter(Boolean).join(" | ");
      throw new Error(details || "FAILURE");
    }

    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
}

function StatusBanner({ status }: { status: StatusState }) {
  if (!status) {
    return null;
  }

  return (
    <div
      className={`rounded-2xl border px-4 py-3 text-sm whitespace-pre-wrap ${statusClasses(status.tone)}`}
    >
      {status.text}
    </div>
  );
}

export function XgbOosView({ initialRuns }: XgbOosViewProps) {
  const [runs, setRuns] = useState<XgbOosRun[]>(initialRuns);
  const [csvFiles, setCsvFiles] = useState<CsvFile[]>([]);
  const [isLoadingRuns, setIsLoadingRuns] = useState(initialRuns.length === 0);
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
  const [runsPage, setRunsPage] = useState(1);

  const [signalExitEnabled, setSignalExitEnabled] = useState(false);
  const [signalExitWindow, setSignalExitWindow] = useState(SIGNAL_EXIT_WINDOW_DEFAULT);
  const [signalExitStartPct, setSignalExitStartPct] = useState(SIGNAL_EXIT_START_PCT_DEFAULT);
  const [signalExitThrGridEnabled, setSignalExitThrGridEnabled] = useState(true);
  const [signalExitThrFrom, setSignalExitThrFrom] = useState("0.2");
  const [signalExitThrTo, setSignalExitThrTo] = useState("0.3");
  const [signalExitThrStep, setSignalExitThrStep] = useState("0.01");

  const [prodEnsemble, setProdEnsemble] = useState("ensemble-a");
  const [selectMode, setSelectMode] = useState<SelectMode>("all_safe");
  const [pruneKeepPct, setPruneKeepPct] = useState("10");
  const [daysGrid, setDaysGrid] = useState<Set<string>>(new Set(["90"]));
  const [exitGrid, setExitGrid] = useState<Set<string>>(new Set(["policy"]));
  const [pEnterGridEnabled, setPEnterGridEnabled] = useState(false);
  const [pEnterFrom, setPEnterFrom] = useState("0.86");
  const [pEnterTo, setPEnterTo] = useState("0.68");
  const [pEnterStep, setPEnterStep] = useState("0.02");

  const [batchStatus, setBatchStatus] = useState<StatusState>(null);
  const [managementStatus, setManagementStatus] = useState<StatusState>(null);
  const [lastBatchResults, setLastBatchResults] = useState<OosResult[]>([]);
  const [lastBatchCsv, setLastBatchCsv] = useState<string | null>(null);
  const [lastBatchUpdatedAt, setLastBatchUpdatedAt] = useState<string | null>(null);
  const [batchLiveLines, setBatchLiveLines] = useState<string[]>([]);
  const [showBatchLiveLines, setShowBatchLiveLines] = useState(false);
  const [activeBatchTasks, setActiveBatchTasks] = useState<BatchTaskMeta[]>([]);
  const [activeBatchId, setActiveBatchId] = useState<string | null>(null);
  const resumeBatchRef = useRef(false);
  const restoredFromCacheRef = useRef(false);
  const batchResultsOffsetRef = useRef(0);
  const [experimentCsv, setExperimentCsv] = useState("");
  const [experimentName, setExperimentName] = useState(() => createExperimentName());
  const totalRunPages = Math.max(1, Math.ceil(runs.length / RUNS_PER_PAGE));
  const safeRunsPage = Math.min(runsPage, totalRunPages);
  const pageStartIndex = (safeRunsPage - 1) * RUNS_PER_PAGE;
  const pageRuns = useMemo(
    () => runs.slice(pageStartIndex, pageStartIndex + RUNS_PER_PAGE),
    [pageStartIndex, runs],
  );
  const pageEndIndex = pageStartIndex + pageRuns.length;
  const pageAllSelected =
    pageRuns.length > 0 && pageRuns.every((run) => selectedPaths.has(run.resultDir));
  const lastBatchStats = useMemo(() => {
    const total = lastBatchResults.length;
    const success = lastBatchResults.filter((item) => item.success === true).length;
    const failed = total - success;
    return { total, success, failed };
  }, [lastBatchResults]);
  const lastBatchPreviewLines = useMemo(() => {
    return lastBatchResults.slice(-30).map((result) => {
      const cfg = result.cfg_snapshot && typeof result.cfg_snapshot === "object"
        ? (result.cfg_snapshot as Record<string, unknown>)
        : {};
      const pe = toNumber(cfg.p_enter_threshold);
      const pePart = pe !== null ? ` pe=${pe.toFixed(2)}` : "";
      const runPart = shortRunName(result.run_dir);
      const prefix = `[${runPart}] d=${result.days ?? "?"} exit=${result.exit_mode ?? "?"}${pePart}`;
      if (result.success !== true) {
        return `FAIL ${prefix} :: ${result.error || "unknown error"}`;
      }
      const backtest = result.backtest ?? {};
      const valAcc = formatNumber(result.oos_metrics?.val_acc ?? null, 4);
      const roi = formatNumber(backtest.roi_pct ?? null, 2);
      const trades = formatNumber(backtest.trades_count ?? null, 0);
      return `OK ${prefix} :: val_acc=${valAcc} roi=${roi}% trades=${trades}`;
    });
  }, [lastBatchResults]);

  useEffect(() => {
    if (restoredFromCacheRef.current && activeBatchId && !lastBatchCsv) {
      setBatchLiveLines(lastBatchPreviewLines);
    }
  }, [activeBatchId, lastBatchCsv, lastBatchPreviewLines]);

  useEffect(() => {
    async function loadRecent() {
      try {
        const data = await jsonFetch<{ success?: boolean; batches?: BatchStatusResponse[] }>(
          "/api/oos_xgb/recent_batches",
          { method: "GET", cache: "no-store" },
        );
        if (data.success && Array.isArray(data.batches)) {
          const running = data.batches.find((b) => b.status === "running" || b.status === "writing");
          const done = data.batches.find((b) => ["done", "partial_timeout", "cancelled"].includes(b.status || ""));
          
          if (running && running.batch_id) {
             setActiveBatchId(running.batch_id);
             setLastBatchCsv(null);
             setLastBatchResults([]);
             batchResultsOffsetRef.current = 0;
             restoredFromCacheRef.current = true;
          } else if (done && !activeBatchId) {
             setBatchStatus({
               tone: "success",
               text: `Последний завершённый batch: ${done.csv_filename || "нет CSV"} (${done.done}/${done.expected} успешно)`
             });
             setLastBatchCsv(done.csv_filename ?? null);
             if (done.finished_at) {
               setLastBatchUpdatedAt(done.finished_at);
             }
          }
        }
      } catch (e) {
        console.error("Failed to load recent batches", e);
      } finally {
        restoredFromCacheRef.current = true;
      }
    }
    void loadRecent();
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadRuns() {
      setIsLoadingRuns(true);
      try {
        const data = await jsonFetch<{ success?: boolean; runs?: unknown[] }>(
          "/api/oos_xgb/runs",
          {
            method: "GET",
            cache: "no-store",
          },
        );
        if (!cancelled) {
          setRuns(normalizeXgbOosRuns(data.runs));
        }
      } catch {
        if (!cancelled) {
          setRuns(initialRuns);
        }
      } finally {
        if (!cancelled) {
          setIsLoadingRuns(false);
        }
      }
    }

    if (initialRuns.length === 0) {
      void loadRuns();
    }

    return () => {
      cancelled = true;
    };
  }, [initialRuns]);

  useEffect(() => {
    if (runsPage > totalRunPages) {
      setRunsPage(totalRunPages);
    }
  }, [runsPage, totalRunPages]);

  useEffect(() => {
    let cancelled = false;

    async function loadCsv() {
      try {
        const data = await jsonFetch<{ success?: boolean; files?: CsvFile[] }>(
          "/api/oos_xgb/csv_list",
          {
            method: "GET",
            cache: "no-store",
          },
        );
        if (!cancelled) {
          setCsvFiles(Array.isArray(data.files) ? data.files : []);
        }
      } catch {
        if (!cancelled) {
          setCsvFiles([]);
        }
      }
    }

    void loadCsv();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!csvFiles.length) {
      setExperimentCsv("");
      return;
    }
    if (!experimentCsv || !csvFiles.some((file) => file.name === experimentCsv)) {
      setExperimentCsv(csvFiles[0]?.name ?? "");
    }
  }, [csvFiles, experimentCsv]);

  async function refreshRuns() {
    try {
      setIsLoadingRuns(true);
      const data = await jsonFetch<{ success?: boolean; runs?: unknown[] }>("/api/oos_xgb/runs", {
        method: "GET",
        cache: "no-store",
      });
      const nextRuns = normalizeXgbOosRuns(data.runs);
      setRuns(nextRuns);
      setSelectedPaths((current) => {
        const allowed = new Set(nextRuns.map((run) => run.resultDir));
        return new Set([...current].filter((path) => allowed.has(path)));
      });
    } finally {
      setIsLoadingRuns(false);
    }
  }

  async function refreshCsv() {
    const data = await jsonFetch<{ success?: boolean; files?: CsvFile[] }>("/api/oos_xgb/csv_list", {
      method: "GET",
      cache: "no-store",
    });
    setCsvFiles(Array.isArray(data.files) ? data.files : []);
  }

  useEffect(() => {
    if (!restoredFromCacheRef.current) {
      return;
    }
    if (!activeBatchId || lastBatchCsv) {
      return;
    }

    let cancelled = false;
    const batchId = activeBatchId;
    resumeBatchRef.current = false;

    const renderServerStatus = (status: BatchStatusResponse) => {
      const expected = Number(status.expected ?? 0);
      const done = Number(status.done ?? 0);
      const successCount = Number(status.success_count ?? 0);
      const failureCount = Math.max(0, done - successCount);
      const state = String(status.status ?? "running");
      
      const pct = expected > 0 ? Math.round((done / expected) * 100) : 0;
      
      setBatchStatus({
        tone: "info",
        text: (
          <div className="flex flex-col gap-2">
            <div className="font-semibold text-slate-800">
              Batch {batchId.slice(0, 8)}... (выполняется на сервере)
            </div>
            <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-slate-200">
              <div
                className="h-full bg-indigo-500 transition-all duration-300"
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="mt-2 flex flex-wrap gap-4 text-sm text-slate-700">
              <span>
                ✅ Готово: <b>{done}</b> / {expected || "?"}
              </span>
              <span>Успешно: {successCount} | Ошибки: {failureCount}</span>
              {status.elapsed_sec ? (
                <span>
                  ⏱ Прошло: <b>{formatDuration(status.elapsed_sec)}</b>
                </span>
              ) : null}
              {status.eta_sec ? (
                <span>
                  ⏳ Осталось: <b>{formatDuration(status.eta_sec)}</b>
                </span>
              ) : null}
              {status.avg_per_run_sec ? (
                <span>⚡ {status.avg_per_run_sec.toFixed(1)}с/run</span>
              ) : null}
            </div>
          </div>
        ),
      });
    };

    async function tick() {
      if (cancelled || resumeBatchRef.current) {
        return;
      }
      resumeBatchRef.current = true;
      try {
        const status = await jsonFetch<BatchStatusResponse>(
          `/api/oos_xgb/batch_status?batch_id=${encodeURIComponent(batchId)}`,
          { method: "GET", cache: "no-store" },
        );
        if (!status.success) {
          throw new Error(status.error || "batch status failed");
        }
        renderServerStatus(status);

        // Pull successful results from Redis in chunks (fast restore after refresh).
        let pages = 0;
        while (!cancelled && pages < 3) {
          const offset = batchResultsOffsetRef.current;
          const chunk = await jsonFetch<BatchResultsResponse>(
            `/api/oos_xgb/batch_results?batch_id=${encodeURIComponent(batchId)}&offset=${offset}&limit=500`,
            { method: "GET", cache: "no-store" },
          );
          if (!chunk.success) {
            break;
          }
          const items = Array.isArray(chunk.results) ? chunk.results : [];
          if (items.length === 0) {
            break;
          }
          batchResultsOffsetRef.current = offset + items.length;
          setLastBatchResults((current) => [...current, ...items]);
          pages += 1;
          if (items.length < 500) {
            break;
          }
        }

        if (status.status === "failed") {
          setBatchStatus({ tone: "error", text: status.error || "Batch failed on server" });
          setActiveBatchTasks([]);
          setActiveBatchId(null);
          restoredFromCacheRef.current = false;
          return;
        }
        if (["done", "partial_timeout", "cancelled"].includes(status.status || "") && status.csv_filename) {
          await refreshCsv();
          setLastBatchCsv(status.csv_filename);
          setLastBatchUpdatedAt(new Date().toISOString());
          setBatchStatus({
            tone: "success",
            text: `Batch завершён на сервере. CSV: ${status.csv_filename} (${status.csv_rows ?? status.success_count ?? "?"} строк)`,
          });
          setActiveBatchTasks([]);
          setActiveBatchId(null);
          setShowBatchLiveLines(false);
          restoredFromCacheRef.current = false;
        }
      } catch (error) {
        if (!cancelled) {
          setBatchStatus({
            tone: "error",
            text: error instanceof Error ? error.message : "Ошибка restore batch OOS",
          });
        }
      } finally {
        resumeBatchRef.current = false;
      }
    }

    void tick();
    const timer = window.setInterval(() => void tick(), BATCH_CSV_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeBatchId, lastBatchCsv]);

  function toggleRunSelection(path: string) {
    setSelectedPaths((current) => {
      const next = new Set(current);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }

  function toggleCurrentPageRuns(checked: boolean) {
    setSelectedPaths((current) => {
      const next = new Set(current);
      for (const run of pageRuns) {
        if (checked) {
          next.add(run.resultDir);
        } else {
          next.delete(run.resultDir);
        }
      }
      return next;
    });
  }

  function toggleDaysGrid(value: string) {
    setDaysGrid((current) => {
      const next = new Set(current);
      if (next.has(value)) {
        next.delete(value);
      } else {
        next.add(value);
      }
      return next;
    });
  }

  function toggleExitGrid(value: string) {
    setExitGrid((current) => {
      const next = new Set(current);
      if (next.has(value)) {
        next.delete(value);
      } else {
        next.add(value);
      }
      return next;
    });
  }

  function selectByMode() {
    const matched = runs
      .filter((run) => selectionModeMatches(run, selectMode))
      .map((run) => run.resultDir);

    if (matched.length === 0) {
      window.alert("Под выбранный режим ничего не найдено.");
      return;
    }

    setSelectedPaths(new Set(matched));
  }

  async function startBatchOos() {
    const resultDirs = [...selectedPaths];
    if (resultDirs.length === 0) {
      setBatchStatus({ tone: "error", text: "Выберите модели галочками." });
      return;
    }

    const selectedDays = [...daysGrid].map((item) => Number.parseInt(item, 10)).filter(Boolean);
    if (selectedDays.length === 0) {
      setBatchStatus({ tone: "error", text: "Выберите days grid." });
      return;
    }

    const selectedExitModes = [...exitGrid];
    if (selectedExitModes.length === 0) {
      setBatchStatus({ tone: "error", text: "Выберите exit grid." });
      return;
    }

    let pEnterThresholds: number[] = [];
    if (pEnterGridEnabled) {
      try {
        pEnterThresholds = buildPEnterGrid(pEnterFrom, pEnterTo, pEnterStep);
      } catch (error) {
        setBatchStatus({
          tone: "error",
          text: error instanceof Error ? error.message : "bad p_enter grid values",
        });
        return;
      }
      if (pEnterThresholds.length === 0) {
        setBatchStatus({ tone: "error", text: "p_enter grid is empty" });
        return;
      }
    }

    try {
      setLastBatchResults([]);
      batchResultsOffsetRef.current = 0;
      setLastBatchCsv(null);
      setActiveBatchId(null);
      restoredFromCacheRef.current = false;
      setBatchStatus({
        tone: "info",
        text: `Шаг 1/2: запускаю batch для ${resultDirs.length} моделей...`,
      });

      const body: Record<string, unknown> = {
        result_dirs: resultDirs,
        days_grid: selectedDays,
        exit_modes: selectedExitModes,
        atr_len: 14,
        atr_mult: 2,
      };

      if (pEnterGridEnabled) {
        body.p_enter_grid_enabled = true;
        body.p_enter_thresholds = pEnterThresholds;
      }

      let signalExitThresholds: number[] = [];
      if (signalExitEnabled) {
        if (signalExitThrGridEnabled) {
          try {
            signalExitThresholds = buildPEnterGrid(signalExitThrFrom, signalExitThrTo, signalExitThrStep);
          } catch (error) {
            setBatchStatus({
              tone: "error",
              text: error instanceof Error ? error.message : "bad signal_exit threshold grid values",
            });
            return;
          }
          if (signalExitThresholds.length === 0) {
            setBatchStatus({ tone: "error", text: "signal_exit threshold grid is empty" });
            return;
          }
        }

        body.signal_exit_enabled = true;
        body.signal_exit_window = Math.max(1, Number.parseInt(signalExitWindow, 10) || 20);
        body.signal_exit_start_pct = parseLooseNumber(signalExitStartPct) ?? 65;
        if (signalExitThrGridEnabled) {
          body.signal_exit_threshold_grid_enabled = true;
          body.signal_exit_thresholds = signalExitThresholds;
        }
      }

      const response = await jsonFetch<BatchAsyncResponse>("/api/oos_xgb/batch_async", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });

      if (!response.success || !Array.isArray(response.tasks) || response.tasks.length === 0) {
        throw new Error(response.error || "Batch OOS не запустился");
      }
      if (!response.batch_id) {
        throw new Error("batch_id missing in server response");
      }
      restoredFromCacheRef.current = true;
      setActiveBatchId(response.batch_id);
      setActiveBatchTasks(response.tasks);

    } catch (error) {
      const message = error instanceof Error ? error.message : "Ошибка batch OOS";
      if (message.includes("Prefetch stale")) {
        const payload =
          error instanceof JsonRequestError && error.payload && typeof error.payload === "object"
            ? (error.payload as { error?: string; prefetch?: Record<string, unknown> | null })
            : { error: message };
        setBatchStatus({ tone: "error", text: renderPrefetchError(payload) });
      } else {
        setBatchStatus({ tone: "error", text: message });
      }
      setBatchLiveLines([]);
    }
  }

  async function saveSelectedExperiment() {
    if (lastBatchResults.length === 0) {
      setManagementStatus({
        tone: "error",
        text: "Сначала запустите batch OOS, чтобы появились результаты для эксперимента.",
      });
      return;
    }

    const selectedResultDirs = new Set(selectedPaths);
    const resultsToSave = lastBatchResults.filter((result) =>
      selectedResultDirs.has(String(result.run_dir || "")),
    );
    const finalResults = resultsToSave.length > 0 ? resultsToSave : lastBatchResults;
    const symbol = finalResults.find((result) => result.symbol)?.symbol ?? runs[0]?.symbol ?? "XGB";
    const name = experimentName.trim() || "oos_experiment";

    if (!window.confirm(`Сохранить эксперимент "${name}" для ${finalResults.length} OOS строк?`)) {
      return;
    }

    try {
      setManagementStatus({ tone: "info", text: "Сохраняю эксперимент..." });
      const response = await jsonFetch<SaveExperimentResponse>("/api/oos_xgb/save_experiment", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          symbol,
          experiment_name: name,
          oos_csv: lastBatchCsv,
          selection_metric: "manual",
          results: finalResults,
        }),
      });

      if (!response.success) {
        throw new Error(response.error || "Save experiment failed");
      }

      setManagementStatus({
        tone: "success",
        text: `Эксперимент сохранён: ${response.path ?? "path unavailable"}\nМоделей: ${response.selected_count ?? finalResults.length}`,
      });
    } catch (error) {
      setManagementStatus({
        tone: "error",
        text: error instanceof Error ? error.message : "Save experiment failed",
      });
    }
  }

  async function saveTopCsvExperiment() {
    if (!experimentCsv) {
      setManagementStatus({ tone: "error", text: "Выберите CSV для top-3 эксперимента." });
      return;
    }

    const name = experimentName.trim() || createExperimentName();
    if (!window.confirm(`Сохранить top-3 из CSV ${experimentCsv} как "${name}"?`)) {
      return;
    }

    try {
      setManagementStatus({ tone: "info", text: "Сохраняю top-3 из CSV..." });
      const response = await jsonFetch<SaveExperimentResponse>(
        "/api/oos_xgb/save_top_csv_experiment",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            filename: experimentCsv,
            experiment_name: name,
            top_n: 3,
          }),
        },
      );

      if (!response.success) {
        throw new Error(response.error || "Save top-3 experiment failed");
      }

      setExperimentName(createExperimentName());
      setManagementStatus({
        tone: "success",
        text: `Top-3 эксперимент сохранён: ${response.path ?? "path unavailable"}\nМоделей: ${response.selected_count ?? 3}`,
      });
    } catch (error) {
      setManagementStatus({
        tone: "error",
        text: error instanceof Error ? error.message : "Save top-3 experiment failed",
      });
    }
  }

  async function copySelectedRunsToProd() {
    const paths = [...selectedPaths];
    if (paths.length === 0) {
      window.alert("Выберите модели галочками.");
      return;
    }
    if (!window.confirm(`Скопировать ${paths.length} моделей в ${prodEnsemble}?`)) {
      return;
    }

    try {
      setManagementStatus({ tone: "info", text: "Копирую модели в прод..." });
      const response = await jsonFetch<{
        success?: boolean;
        copied_count?: number;
        copied?: Array<{ symbol_dir?: string; ensemble?: string; version?: string; run_id?: string }>;
        errors?: Array<{ path?: string; error?: string }>;
        error?: string;
      }>("/api/oos_xgb/copy_to_prod", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ paths, ensemble: prodEnsemble }),
      });

      if (!response.success) {
        throw new Error(response.error || "Copy to prod failed");
      }

      const preview = (response.copied ?? [])
        .slice(0, 8)
        .map((item) => `${item.symbol_dir}/${item.ensemble}/${item.version} <= ${item.run_id}`)
        .join("\n");

      setManagementStatus({
        tone: "success",
        text: preview
          ? `Скопировано: ${response.copied_count ?? 0}\n${preview}`
          : `Скопировано: ${response.copied_count ?? 0}`,
      });
    } catch (error) {
      setManagementStatus({
        tone: "error",
        text: error instanceof Error ? error.message : "Copy to prod failed",
      });
    }
  }

  async function copySelectedRunsToWf() {
    const paths = [...selectedPaths];
    if (paths.length === 0) {
      window.alert("Выберите модели галочками.");
      return;
    }
    if (!window.confirm(`Скопировать ${paths.length} моделей в WF archive?`)) {
      return;
    }

    try {
      setManagementStatus({ tone: "info", text: "Копирую модели в WF..." });
      const response = await jsonFetch<{
        success?: boolean;
        copied_count?: number;
        copied?: Array<{ symbol_dir?: string; run_id?: string; path?: string }>;
        errors?: Array<{ path?: string; error?: string }>;
        error?: string;
      }>("/api/oos_xgb/copy_to_wf", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ paths }),
      });

      if (!response.success) {
        throw new Error(response.error || "Copy to WF failed");
      }

      const preview = (response.copied ?? [])
        .slice(0, 8)
        .map((item) => `${item.symbol_dir}/${item.run_id}`)
        .join("\n");

      setManagementStatus({
        tone: "success",
        text: preview
          ? `Скопировано в WF: ${response.copied_count ?? 0}\n${preview}`
          : `Скопировано в WF: ${response.copied_count ?? 0}`,
      });
    } catch (error) {
      setManagementStatus({
        tone: "error",
        text: error instanceof Error ? error.message : "Copy to WF failed",
      });
    }
  }

  async function deleteSelectedRuns() {
    const paths = [...selectedPaths];
    if (paths.length === 0) {
      window.alert("Ничего не выбрано.");
      return;
    }
    if (!window.confirm(`Удалить ${paths.length} run?`)) {
      return;
    }

    try {
      setManagementStatus({ tone: "info", text: "Удаляю выбранные run..." });
      const response = await jsonFetch<{
        success?: boolean;
        deleted_count?: number;
        error?: string;
      }>("/api/oos_xgb/delete_runs", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ paths }),
      });

      if (!response.success) {
        throw new Error(response.error || "Delete failed");
      }

      setSelectedPaths(new Set());
      await refreshRuns();
      setManagementStatus({
        tone: "success",
        text: `Удалено: ${response.deleted_count ?? 0}`,
      });
    } catch (error) {
      setManagementStatus({
        tone: "error",
        text: error instanceof Error ? error.message : "Delete failed",
      });
    }
  }

  async function pruneSelectedRuns() {
    const paths = [...selectedPaths];
    if (paths.length === 0) {
      window.alert("Ничего не выбрано.");
      return;
    }

    const keepPct = Math.max(1, Math.min(Number.parseFloat(pruneKeepPct) || 10, 100));
    const keepCount = Math.max(1, Math.ceil(paths.length * (keepPct / 100)));
    if (!window.confirm(`Оставить top ${keepPct}% (~${keepCount}) и удалить остальное?`)) {
      return;
    }

    try {
      setManagementStatus({ tone: "info", text: "Prune selected..." });
      const response = await jsonFetch<{
        success?: boolean;
        kept_count?: number;
        deleted_count?: number;
        cutoff_score?: number | null;
        error?: string;
      }>("/api/oos_xgb/prune_runs", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          paths,
          keep_pct: keepPct,
        }),
      });

      if (!response.success) {
        throw new Error(response.error || "Prune failed");
      }

      setSelectedPaths(new Set());
      await refreshRuns();
      setManagementStatus({
        tone: "success",
        text: `Prune done. kept=${response.kept_count ?? 0} deleted=${response.deleted_count ?? 0} cutoff=${formatNumber(response.cutoff_score ?? null, 4)}`,
      });
    } catch (error) {
      setManagementStatus({
        tone: "error",
        text: error instanceof Error ? error.message : "Prune failed",
      });
    }
  }

  async function pruneDuplicateConfigs() {
    if (!window.confirm("Оставить top-1 по каждой конфигурации и удалить дубликаты?")) {
      return;
    }

    try {
      setManagementStatus({ tone: "info", text: "Prune duplicates..." });
      const response = await jsonFetch<{
        success?: boolean;
        kept_count?: number;
        deleted_count?: number;
        error?: string;
      }>("/api/oos_xgb/prune_duplicates", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      });

      if (!response.success) {
        throw new Error(response.error || "Prune duplicates failed");
      }

      setSelectedPaths(new Set());
      await refreshRuns();
      setManagementStatus({
        tone: "success",
        text: `Duplicates pruned. kept=${response.kept_count ?? 0} deleted=${response.deleted_count ?? 0}`,
      });
    } catch (error) {
      setManagementStatus({
        tone: "error",
        text: error instanceof Error ? error.message : "Prune duplicates failed",
      });
    }
  }

  async function deleteCsvFile(name: string) {
    if (!window.confirm(`Удалить CSV ${name}?`)) {
      return;
    }

    try {
      await jsonFetch<{ success?: boolean; error?: string }>("/api/oos_xgb/csv_delete", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ filename: name }),
      });
      await refreshCsv();
    } catch (error) {
      window.alert(error instanceof Error ? error.message : "Не удалось удалить CSV");
    }
  }

  function clearLastBatchCache() {
    restoredFromCacheRef.current = false;
    batchResultsOffsetRef.current = 0;
    setBatchStatus(null);
    setLastBatchResults([]);
    setLastBatchCsv(null);
    setLastBatchUpdatedAt(null);
    setActiveBatchTasks([]);
    setActiveBatchId(null);
    setBatchLiveLines([]);
    setShowBatchLiveLines(false);
  }

  async function cancelBatchOos() {
    if (!activeBatchId) {
      return;
    }
    const confirmed = window.confirm("Отменить OOS batch? Задачи будут остановлены, а текущие результаты сохранены в snapshot.");
    if (!confirmed) {
      return;
    }
    try {
      setBatchStatus({ tone: "info", text: "Отменяю batch..." });
      const response = await jsonFetch<{ success: boolean; error?: string; csv_filename?: string; csv_rows?: number }>("/api/oos_xgb/batch_cancel", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ batch_id: activeBatchId }),
      });
      if (!response.success) {
        throw new Error(response.error || "cancel failed");
      }
      await refreshCsv();
      if (response.csv_filename) {
        setLastBatchCsv(response.csv_filename);
        setLastBatchUpdatedAt(new Date().toISOString());
        setBatchStatus({
          tone: "success",
          text: `Batch отменён. Сохранён CSV (snapshot): ${response.csv_filename} (${response.csv_rows ?? "?"} строк)`,
        });
      } else {
        setBatchStatus({
          tone: "info",
          text: `Batch отменён. Нет успешных результатов для CSV.`,
        });
      }
      setActiveBatchTasks([]);
      setActiveBatchId(null);
      restoredFromCacheRef.current = false;
    } catch (error) {
      setBatchStatus({
        tone: "error",
        text: error instanceof Error ? error.message : "Не удалось отменить batch",
      });
    }
  }

  async function forceFinalizeBatchCsv() {
    if (!activeBatchId) {
      setBatchStatus({ tone: "error", text: "Нет active batch_id для сохранения CSV." });
      return;
    }
    try {
      setBatchStatus({ tone: "info", text: "Сохраняю CSV сейчас (partial)..." });
      const response = await jsonFetch<ForceFinalizeResponse>("/api/oos_xgb/batch_force_finalize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          batch_id: activeBatchId,
          confirm: "FORCE_PARTIAL",
          reason: "ui_force_partial",
        }),
      });
      if (!response.success || !response.filename) {
        throw new Error(response.error || "force finalize failed");
      }
      await refreshCsv();
      setLastBatchCsv(response.filename);
      setLastBatchUpdatedAt(new Date().toISOString());
      setBatchStatus({
        tone: "success",
        text: `CSV сохранён (partial): ${response.filename} (${response.rows ?? "?"} строк)`,
      });
      setActiveBatchTasks([]);
      setActiveBatchId(null);
      restoredFromCacheRef.current = false;
    } catch (error) {
      setBatchStatus({
        tone: "error",
        text: error instanceof Error ? error.message : "Не удалось сохранить CSV (partial)",
      });
    }
  }

  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-200 bg-white p-5">
        <div className="border-b border-slate-200 pb-4">
          <h3 className="text-lg font-semibold text-slate-900">Последний batch OOS</h3>
          <p className="mt-2 text-sm leading-7 text-slate-600">
            Прогресс и результат последнего запуска сохраняются в браузере и доступны после refresh.
          </p>
        </div>
        <div className="mt-4 space-y-4">
          <div className="flex justify-end">
            <div className="flex flex-wrap gap-2">
              {activeBatchId && !lastBatchCsv ? (
                <>
                  <button
                    type="button"
                    onClick={() => void forceFinalizeBatchCsv()}
                    className="rounded-xl border border-amber-300 bg-amber-50 px-3 py-2 text-xs font-semibold text-amber-900 transition hover:bg-amber-100"
                    title="Сохраняет CSV из уже собранных success результатов (snapshot), batch продолжит работу."
                  >
                    Сохранить CSV сейчас (partial)
                  </button>
                  <button
                    type="button"
                    onClick={() => void cancelBatchOos()}
                    className="rounded-xl border border-red-300 bg-red-50 px-3 py-2 text-xs font-semibold text-red-900 transition hover:bg-red-100"
                    title="Отменить выполнение batch: отзовет задачи из Celery и сохранит частичные результаты."
                  >
                    Отменить OOS
                  </button>
                </>
              ) : null}
              <button
                type="button"
                onClick={() => setShowBatchLiveLines((value) => !value)}
                className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-100"
              >
                {showBatchLiveLines ? "Скрыть детали" : "Показать детали"}
              </button>
              <button
                type="button"
                onClick={clearLastBatchCache}
                className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-100"
              >
                Очистить кеш блока
              </button>
            </div>
          </div>
          <StatusBanner status={batchStatus} />
          {showBatchLiveLines ? (
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Детали (последние строки)</p>
              <pre className="mt-3 max-h-72 overflow-auto rounded-xl border border-slate-200 bg-white p-3 text-xs leading-6 text-slate-700">
                {(batchLiveLines.length > 0 ? batchLiveLines : ["—"]).join("\n")}
              </pre>
              <p className="mt-3 text-xs leading-6 text-slate-500">{OOS_BATCH_HINT}</p>
            </div>
          ) : null}
          {lastBatchUpdatedAt ? (
            <p className="text-xs text-slate-500">
              Обновлено: {new Date(lastBatchUpdatedAt).toLocaleString("ru-RU")}
            </p>
          ) : null}
          {lastBatchStats.total > 0 ? (
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <p className="text-sm font-semibold text-slate-900">
                Итог: {lastBatchStats.success}/{lastBatchStats.total} успешно, ошибок: {lastBatchStats.failed}
                {lastBatchCsv ? `, CSV: ${lastBatchCsv}` : ""}
              </p>
            </div>
          ) : (
            <p className="text-sm text-slate-500">Пока нет сохранённых результатов batch OOS.</p>
          )}
        </div>
      </section>

      <section className="rounded-3xl border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-4">
          <div>
            <h3 className="text-lg font-semibold text-slate-900">Сохранённые CSV</h3>
            <p className="mt-2 text-sm leading-7 text-slate-600">
              Список сохранённых batch CSV, как в legacy `OOS`.
            </p>
          </div>
          <button
            type="button"
            onClick={() => void refreshCsv()}
            className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-100"
          >
            Обновить
          </button>
        </div>

        {csvFiles.length === 0 ? (
          <div className="mt-4 rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-4 py-5 text-sm text-slate-500">
            Нет сохранённых CSV.
          </div>
        ) : (
          <div className="mt-4 overflow-hidden rounded-2xl border border-slate-200">
            <div className="overflow-auto">
              <table className="min-w-full border-collapse text-left text-sm">
                <thead className="bg-slate-50 text-slate-600">
                  <tr>
                    <th className="border-b border-slate-200 px-3 py-3">Файл</th>
                    <th className="border-b border-slate-200 px-3 py-3">Размер</th>
                    <th className="border-b border-slate-200 px-3 py-3">Дата</th>
                    <th className="border-b border-slate-200 px-3 py-3">Действия</th>
                  </tr>
                </thead>
                <tbody className="bg-white">
                  {csvFiles.map((file) => (
                    <tr key={file.name} className="hover:bg-slate-50">
                      <td className="border-b border-slate-100 px-3 py-3 font-mono text-xs text-slate-900">
                        {file.name}
                      </td>
                      <td className="border-b border-slate-100 px-3 py-3">
                        {(file.size / 1024).toFixed(1)} KB
                      </td>
                      <td className="border-b border-slate-100 px-3 py-3">{file.mtime}</td>
                      <td className="border-b border-slate-100 px-3 py-3">
                        <div className="flex gap-3">
                          <a
                            href={`/api/oos_xgb/csv_download/${encodeURIComponent(file.name)}`}
                            className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-100"
                          >
                            Скачать
                          </a>
                          <button
                            type="button"
                            onClick={() => void deleteCsvFile(file.name)}
                            className="rounded-xl border border-rose-200 bg-white px-3 py-2 text-sm font-medium text-rose-700 transition hover:bg-rose-50"
                          >
                            Удалить
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      <div>
        <section className="space-y-4 rounded-3xl border border-slate-200 bg-white p-5">
          <div className="border-b border-slate-200 pb-4">
            <h3 className="text-lg font-semibold text-slate-900">Batch и управление</h3>
            <p className="mt-2 text-sm leading-7 text-slate-600">
              Массовый запуск OOS, copy to prod, delete и prune.
            </p>
          </div>

          <StatusBanner status={managementStatus} />

          <div className="grid gap-4 lg:grid-cols-2">
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="rounded-2xl border border-slate-200 bg-white p-4">
                <label className="mb-2 block text-sm font-semibold text-slate-900">
                  Prod ensemble
                </label>
                <select
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300"
                  value={prodEnsemble}
                  onChange={(event) => setProdEnsemble(event.target.value)}
                >
                  {ENSEMBLES.map((ensemble) => (
                    <option key={ensemble} value={ensemble}>
                      {ensemble}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => void copySelectedRunsToProd()}
                  className="mt-4 w-full rounded-xl border border-slate-300 bg-slate-900 px-4 py-3 text-sm font-semibold text-white transition hover:bg-slate-700"
                >
                  Copy to prod
                </button>
              </div>

              <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-4">
                <label className="mb-2 block text-sm font-semibold text-slate-900">
                  Select for delete
                </label>
                <select
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300"
                  value={selectMode}
                  onChange={(event) => setSelectMode(event.target.value as SelectMode)}
                >
                  {SELECT_MODE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>

                <button
                  type="button"
                  onClick={selectByMode}
                  className="mt-4 w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-100"
                >
                  Выделить по режиму
                </button>

                <label className="mt-5 mb-2 block text-sm font-semibold text-slate-900">
                  Keep %
                </label>
                <input
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300"
                  value={pruneKeepPct}
                  onChange={(event) => setPruneKeepPct(event.target.value)}
                />

                <div className="mt-4 grid gap-3">
                  <button
                    type="button"
                    onClick={() => void pruneSelectedRuns()}
                    className="rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-100"
                  >
                    Prune selected
                  </button>
                  <button
                    type="button"
                    onClick={() => void pruneDuplicateConfigs()}
                    className="rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-100"
                  >
                    Prune duplicates
                  </button>
                  <button
                    type="button"
                    onClick={() => void deleteSelectedRuns()}
                    className="rounded-xl bg-rose-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-rose-500"
                  >
                    Delete selected
                  </button>
                </div>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <p className="text-sm font-semibold text-slate-900">Days grid</p>
              <div className="mt-3 flex flex-wrap gap-3">
                {DAYS_GRID.map((item) => (
                  <label
                    key={item}
                    className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm text-slate-700"
                  >
                    <input
                      type="checkbox"
                      checked={daysGrid.has(item)}
                      onChange={() => toggleDaysGrid(item)}
                      className="h-4 w-4 rounded border-slate-300"
                    />
                    {item}
                  </label>
                ))}
              </div>

              <p className="mt-5 text-sm font-semibold text-slate-900">Exit grid</p>
              <div className="mt-3 flex flex-wrap gap-3">
                {EXIT_GRID.map((item) => (
                  <label
                    key={item}
                    className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm text-slate-700"
                  >
                    <input
                      type="checkbox"
                      checked={exitGrid.has(item)}
                      onChange={() => toggleExitGrid(item)}
                      className="h-4 w-4 rounded border-slate-300"
                    />
                    {item}
                  </label>
                ))}
              </div>

              <div className="mt-5 rounded-2xl border border-slate-200 bg-white p-4">
                <label className="flex items-center gap-3 text-sm font-semibold text-slate-900">
                  <input
                    type="checkbox"
                    checked={pEnterGridEnabled}
                    onChange={(event) => setPEnterGridEnabled(event.target.checked)}
                    className="h-4 w-4 rounded border-slate-300"
                  />
                  p_enter grid
                </label>
                <div className="mt-4 grid gap-4 sm:grid-cols-3">
                  <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    from
                    <input
                      className="mt-2 w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300 disabled:bg-slate-100"
                      value={pEnterFrom}
                      inputMode="decimal"
                      disabled={!pEnterGridEnabled}
                      onChange={(event) => setPEnterFrom(event.target.value)}
                    />
                  </label>
                  <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    to
                    <input
                      className="mt-2 w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300 disabled:bg-slate-100"
                      value={pEnterTo}
                      inputMode="decimal"
                      disabled={!pEnterGridEnabled}
                      onChange={(event) => setPEnterTo(event.target.value)}
                    />
                  </label>
                  <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    step
                    <input
                      className="mt-2 w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300 disabled:bg-slate-100"
                      value={pEnterStep}
                      inputMode="decimal"
                      disabled={!pEnterGridEnabled}
                      onChange={(event) => setPEnterStep(event.target.value)}
                    />
                  </label>
                </div>
              </div>

              <div className="mt-5 rounded-2xl border border-slate-200 bg-white p-4">
                <label className="flex items-center gap-3 text-sm font-semibold text-slate-900">
                  <input
                    type="checkbox"
                    checked={signalExitEnabled}
                    onChange={(event) => setSignalExitEnabled(event.target.checked)}
                    className="h-4 w-4 rounded border-slate-300"
                  />
                  Early signal exit
                </label>
                <p className="mt-2 text-xs leading-6 text-slate-500">
                  Выход, если avg(signal) за window ниже порога. Без grid порог = p_enter_thr.
                </p>
                <label className="mt-4 flex items-center gap-3 text-xs font-medium text-slate-700">
                  <input
                    type="checkbox"
                    checked={signalExitThrGridEnabled}
                    disabled={!signalExitEnabled}
                    onChange={(event) => setSignalExitThrGridEnabled(event.target.checked)}
                    className="h-4 w-4 rounded border-slate-300"
                  />
                  Grid avg-signal threshold
                </label>
                <div className="mt-3 grid gap-3 sm:grid-cols-3">
                  <label className="block text-xs text-slate-600">
                    From
                    <input
                      className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm disabled:bg-slate-100"
                      value={signalExitThrFrom}
                      inputMode="decimal"
                      disabled={!signalExitEnabled || !signalExitThrGridEnabled}
                      onChange={(event) => setSignalExitThrFrom(event.target.value)}
                    />
                  </label>
                  <label className="block text-xs text-slate-600">
                    To
                    <input
                      className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm disabled:bg-slate-100"
                      value={signalExitThrTo}
                      inputMode="decimal"
                      disabled={!signalExitEnabled || !signalExitThrGridEnabled}
                      onChange={(event) => setSignalExitThrTo(event.target.value)}
                    />
                  </label>
                  <label className="block text-xs text-slate-600">
                    Step
                    <input
                      className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm disabled:bg-slate-100"
                      value={signalExitThrStep}
                      inputMode="decimal"
                      disabled={!signalExitEnabled || !signalExitThrGridEnabled}
                      onChange={(event) => setSignalExitThrStep(event.target.value)}
                    />
                  </label>
                </div>
                <div className="mt-4 grid gap-4 sm:grid-cols-2">
                  <input
                    className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300 disabled:bg-slate-100"
                    value={signalExitWindow}
                    placeholder="Window"
                    disabled={!signalExitEnabled}
                    onChange={(event) => setSignalExitWindow(event.target.value)}
                  />
                  <input
                    className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300 disabled:bg-slate-100"
                    value={signalExitStartPct}
                    placeholder="Start % of hold"
                    disabled={!signalExitEnabled}
                    onChange={(event) => setSignalExitStartPct(event.target.value)}
                  />
                </div>
              </div>

            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => void startBatchOos()}
              className="rounded-xl bg-slate-900 px-4 py-3 text-sm font-semibold text-white transition hover:bg-slate-700"
            >
              Run selected OOS
            </button>
            <button
              type="button"
              onClick={() => void copySelectedRunsToWf()}
              className="rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-100"
            >
              Copy to WF
            </button>
          </div>
        </section>

      </div>

      <section className="rounded-3xl border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-4">
          <div>
            <h3 className="text-lg font-semibold text-slate-900">Все XGB run</h3>
            <p className="mt-2 text-sm leading-7 text-slate-600">
              Показывается по {RUNS_PER_PAGE} run на страницу, выбор сохраняется между страницами.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="rounded-full border border-slate-200 bg-slate-50 px-4 py-2 text-sm font-semibold text-slate-700">
              Выбрано: {selectedPaths.size}/{runs.length}
            </span>
            <button
              type="button"
              onClick={() => void refreshRuns()}
              className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-100"
            >
              Обновить список
            </button>
          </div>
        </div>

        {isLoadingRuns ? (
          <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-5 text-sm text-slate-500">
            Загружаю список run...
          </div>
        ) : runs.length === 0 ? (
          <div className="mt-4 rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-4 py-5 text-sm text-slate-500">
            XGB run пока не найдено.
          </div>
        ) : (
          <div className="mt-4 overflow-hidden rounded-2xl border border-slate-200">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-slate-50 px-3 py-3 text-sm text-slate-700">
              <span>
                Показано: {pageStartIndex + 1}-{pageEndIndex} из {runs.length}
              </span>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => setRunsPage(1)}
                  disabled={safeRunsPage === 1}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Первая
                </button>
                <button
                  type="button"
                  onClick={() => setRunsPage((page) => Math.max(1, page - 1))}
                  disabled={safeRunsPage === 1}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Назад
                </button>
                <span className="px-2 font-semibold">
                  {safeRunsPage}/{totalRunPages}
                </span>
                <button
                  type="button"
                  onClick={() => setRunsPage((page) => Math.min(totalRunPages, page + 1))}
                  disabled={safeRunsPage === totalRunPages}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Вперёд
                </button>
                <button
                  type="button"
                  onClick={() => setRunsPage(totalRunPages)}
                  disabled={safeRunsPage === totalRunPages}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Последняя
                </button>
              </div>
            </div>
            <div className="max-h-[70vh] overflow-auto">
              <table className="min-w-full border-collapse text-left text-sm">
                <thead className="sticky top-0 bg-slate-50 text-slate-600">
                  <tr>
                    <th className="border-b border-slate-200 px-3 py-3">
                      <input
                        type="checkbox"
                        checked={pageAllSelected}
                        onChange={(event) => toggleCurrentPageRuns(event.target.checked)}
                        className="h-4 w-4 rounded border-slate-300"
                      />
                    </th>
                    <th className="border-b border-slate-200 px-3 py-3">Symbol</th>
                    <th className="border-b border-slate-200 px-3 py-3">Run</th>
                    <th className="border-b border-slate-200 px-3 py-3">Task</th>
                    <th className="border-b border-slate-200 px-3 py-3">Dir</th>
                    <th className="border-b border-slate-200 px-3 py-3">Source</th>
                    <th className="border-b border-slate-200 px-3 py-3">H</th>
                    <th className="border-b border-slate-200 px-3 py-3">Thr</th>
                    <th className="border-b border-slate-200 px-3 py-3">max_hold</th>
                    <th className="border-b border-slate-200 px-3 py-3">min_profit%</th>
                    <th className="border-b border-slate-200 px-3 py-3">val_acc</th>
                    <th className="border-b border-slate-200 px-3 py-3">f1_buy_sell</th>
                    <th className="border-b border-slate-200 px-3 py-3">f1(1)</th>
                    <th className="border-b border-slate-200 px-3 py-3">prec(1)</th>
                    <th className="border-b border-slate-200 px-3 py-3">recall(1)</th>
                    <th className="border-b border-slate-200 px-3 py-3">y_non_hold</th>
                    <th className="border-b border-slate-200 px-3 py-3">pred_non_hold</th>
                    <th className="border-b border-slate-200 px-3 py-3">Path</th>
                  </tr>
                </thead>
                <tbody className="bg-white">
                  {pageRuns.map((run) => {
                    const cfg = run.cfg;
                    const metrics = run.metrics;
                    const task = run.task?.toLowerCase() ?? "";
                    const isBinaryTask = task.startsWith("entry") || task.startsWith("exit");

                    return (
                      <tr
                        key={run.resultDir}
                        className={selectedPaths.has(run.resultDir) ? "bg-slate-50" : "hover:bg-slate-50"}
                      >
                        <td className="border-b border-slate-100 px-3 py-3 align-top">
                          <input
                            type="checkbox"
                            checked={selectedPaths.has(run.resultDir)}
                            onChange={() => toggleRunSelection(run.resultDir)}
                            className="h-4 w-4 rounded border-slate-300"
                          />
                        </td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">{run.symbol}</td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">{run.runName}</td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">{run.task || "—"}</td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">{run.direction || "—"}</td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">{run.source || "—"}</td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">
                          {isBinaryTask ? "—" : formatNumber(toNumber(cfg.horizon_steps), 0)}
                        </td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">
                          {formatNumber(toNumber(cfg.threshold), 4)}
                        </td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">
                          {formatNumber(toNumber(cfg.max_hold_steps), 0)}
                        </td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">
                          {formatSignedPercentFromRatio(toNumber(cfg.min_profit), 2)}
                        </td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">
                          {formatNumber(metricValue(metrics, "val_acc"), 5)}
                        </td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">
                          {formatNumber(metricValue(metrics, "f1_buy_sell_val"), 4)}
                        </td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">
                          {formatNumber(arrayMetricItem(metrics, "f1_val", 1), 4)}
                        </td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">
                          {formatNumber(arrayMetricItem(metrics, "precision_val", 1), 4)}
                        </td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">
                          {formatNumber(arrayMetricItem(metrics, "recall_val", 1), 4)}
                        </td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">
                          {formatNumber(metricValue(metrics, "y_non_hold_rate_val"), 4)}
                        </td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top">
                          {formatNumber(metricValue(metrics, "pred_non_hold_rate_val"), 4)}
                        </td>
                        <td className="border-b border-slate-100 px-3 py-3 align-top font-mono text-xs text-slate-500">
                          {compactPath(run.resultDir, 6)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      <section className="rounded-3xl border border-slate-200 bg-white p-5">
        <div className="border-b border-slate-200 pb-4">
          <h3 className="text-lg font-semibold text-slate-900">Experiment name</h3>
          <p className="mt-2 text-sm leading-7 text-slate-600">
            Сохраняет последний успешный batch OOS: `preset.json` + `summary.json`
            в `predict_test/xgb_hypo/experiments/&lt;symbol&gt;/...`.
          </p>
        </div>

        <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_260px]">
          <div>
            <label className="mb-2 block text-sm font-semibold text-slate-900">
              Название эксперимента
            </label>
            <input
              className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300"
              value={experimentName}
              placeholder="preset_uuid_timestamp"
              onChange={(event) => setExperimentName(event.target.value)}
            />
            <button
              type="button"
              onClick={() => setExperimentName(createExperimentName())}
              className="mt-3 rounded-xl border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-100"
            >
              Сгенерировать имя
            </button>
            {lastBatchCsv ? (
              <p className="mt-2 font-mono text-xs text-slate-500">
                Последний CSV: {lastBatchCsv}
              </p>
            ) : (
              <p className="mt-2 text-xs leading-6 text-slate-500">
                Кнопка станет активной после успешного `Run selected OOS`, когда появятся
                результаты batch и CSV.
              </p>
            )}

            <label className="mt-4 mb-2 block text-sm font-semibold text-slate-900">
              CSV для top-3
            </label>
            <select
              className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300"
              value={experimentCsv}
              onChange={(event) => setExperimentCsv(event.target.value)}
            >
              {csvFiles.length === 0 ? (
                <option value="">CSV не найден</option>
              ) : (
                csvFiles.map((file) => (
                  <option key={file.name} value={file.name}>
                    {file.name}
                  </option>
                ))
              )}
            </select>
            <p className="mt-2 text-xs leading-6 text-slate-500">
              Эта кнопка не требует нового OOS: backend прочитает выбранный CSV,
              отсортирует по `roi_pct` и сохранит top-3.
            </p>
          </div>

          <div className="flex flex-col justify-end gap-3">
            <button
              type="button"
              onClick={() => void saveSelectedExperiment()}
              disabled={lastBatchResults.length === 0}
              className="w-full rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-semibold text-emerald-700 transition hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Сохранить эксперимент
            </button>
            <button
              type="button"
              onClick={() => void saveTopCsvExperiment()}
              disabled={!experimentCsv}
              className="w-full rounded-xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm font-semibold text-sky-700 transition hover:bg-sky-100 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Сохранить top-3 из CSV
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
