"use client";

import { useEffect, useMemo, useState } from "react";

type RawPrediction = {
  id?: number | null;
  session_id?: string | null;
  timestamp?: string | null;
  created_at?: string | null;
  symbol?: string | null;
  action?: string | null;
  confidence?: number | null;
  q_values?: number[] | null;
  current_price?: number | null;
  position_status?: string | null;
  model_path?: string | null;
  market_conditions?: Record<string, unknown> | null;
};

type PredictionsResponse = {
  success?: boolean;
  predictions?: RawPrediction[];
  total_predictions?: number;
  error?: string;
};

type XgbPrediction = {
  id: string;
  sessionId: string | null;
  timestamp: string | null;
  symbol: string;
  action: string | null;
  confidence: number | null;
  currentPrice: number | null;
  positionStatus: string | null;
  modelPath: string | null;
  direction: string | null;
  tradeMode: string | null;
  errorHuman: string | null;
  qValues: number[];
  signalScore: number | null;
  entryThreshold: number | null;
  qgateMaxQ: number | null;
  qgateGapQ: number | null;
  qgateT1: number | null;
  qgateT2: number | null;
  qgateFiltered: boolean;
  holdReason: string | null;
};

type XgbPredictionsViewProps = {
  symbols: string[];
  initialSymbol: string;
};

type RawActiveAgent = {
  session_id?: string | null;
  symbol?: string | null;
  model_path?: string | null;
  model_paths?: string[] | null;
  bybit_account_label?: string | null;
  bybit_account_id?: string | null;
  is_xgb?: boolean | null;
};

type ActiveAgentsResponse = {
  success?: boolean;
  active_agents?: RawActiveAgent[];
  error?: string;
};

type ActiveModelOption = {
  value: string;
  kind: "session" | "model";
  sessionId: string;
  symbol: string;
  modelPath: string | null;
  accountLabel: string | null;
};

const ACTIVE_MODELS_REFRESH_INTERVAL_MS = 5 * 60 * 1000;

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isXgbModelPath(modelPath: string | null | undefined): boolean {
  return Boolean(modelPath && modelPath.replace(/\\/g, "/").includes("/models/xgb/"));
}

function normalizeSymbol(value: string | null | undefined): string {
  const normalized = String(value ?? "").trim().toUpperCase();
  if (!normalized) {
    return "";
  }
  return normalized.endsWith("USDT") ? normalized.slice(0, -4) : normalized;
}

function formatNumber(value: number | null | undefined, digits = 4): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return new Intl.NumberFormat("ru-RU", {
    maximumFractionDigits: digits,
  }).format(value);
}

function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "—";
  }
  const normalizedValue =
    /(?:Z|[+-]\d{2}:\d{2})$/.test(value) ? value : `${value}Z`;
  const date = new Date(normalizedValue);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("ru-RU", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZone: "Europe/Moscow",
  }).format(date);
}

function compactModelPath(modelPath: string | null): string {
  if (!modelPath) {
    return "—";
  }
  return modelPath.replace(/\\/g, "/").split("/").filter(Boolean).slice(-5).join("/");
}

function formatModelLabel(modelPath: string | null, direction: string | null): string {
  if (!modelPath) {
    return "—";
  }
  const parts = modelPath.replace(/\\/g, "/").split("/").filter(Boolean);
  const xgbIndex = parts.lastIndexOf("xgb");
  if (xgbIndex < 0 || parts.length <= xgbIndex + 3) {
    return compactModelPath(modelPath);
  }
  const symbol = parts[xgbIndex + 1] || "—";
  const version = parts[xgbIndex + 3] || "—";
  const normalizedDirection = direction ? direction.trim().toUpperCase() : "";
  if (!normalizedDirection) {
    return `${symbol} ${version}`;
  }
  return `${normalizedDirection} ${symbol} ${version}`;
}

function formatModelOptionLabel(option: ActiveModelOption): string {
  const model = compactModelPath(option.modelPath);
  const account = option.accountLabel || "account";
  const symbol = normalizeSymbol(option.symbol) || option.symbol || "—";
  if (option.kind === "model") {
    return `${symbol} | ${model} | ${account} | конкретная модель`;
  }
  return `${symbol} | ${model} | ${account} | вся session`;
}

function actionTone(action: string | null): string {
  if (action === "buy") {
    return "border-emerald-200 bg-emerald-50 text-emerald-700";
  }
  if (action === "sell") {
    return "border-rose-200 bg-rose-50 text-rose-700";
  }
  return "border-slate-200 bg-slate-100 text-slate-700";
}

function toNumberOrNull(value: unknown): number | null {
  return isFiniteNumber(value) ? value : null;
}

function deriveSignalScore(
  qValues: number[],
  task: string | null,
  confidence: number | null,
): number | null {
  if (qValues.length >= 3) {
    if (task === "entry_short" || task === "exit_long") {
      return isFiniteNumber(qValues[2]) ? qValues[2] : null;
    }
    if (
      task === "entry_long" ||
      task === "exit_short" ||
      task === "directional" ||
      task === null
    ) {
      return isFiniteNumber(qValues[1]) ? qValues[1] : null;
    }
  }
  return confidence;
}

function deriveQgateStats(
  qValues: number[],
  qgateMaxQ: number | null,
  qgateGapQ: number | null,
): { maxQ: number | null; gapQ: number | null } {
  if (qgateMaxQ !== null && qgateGapQ !== null) {
    return { maxQ: qgateMaxQ, gapQ: qgateGapQ };
  }
  if (qValues.length < 2) {
    return { maxQ: qgateMaxQ, gapQ: qgateGapQ };
  }
  const sorted = [...qValues].filter((item) => Number.isFinite(item)).sort((a, b) => b - a);
  if (sorted.length < 2) {
    return { maxQ: qgateMaxQ, gapQ: qgateGapQ };
  }
  return {
    maxQ: qgateMaxQ ?? sorted[0],
    gapQ: qgateGapQ ?? sorted[0] - sorted[1],
  };
}

function deriveHoldReason(params: {
  action: string | null;
  signalScore: number | null;
  entryThreshold: number | null;
  qgateFiltered: boolean;
  qgateMaxQ: number | null;
  qgateGapQ: number | null;
  qgateT1: number | null;
  qgateT2: number | null;
  inPositionBlock: boolean;
}): string | null {
  if (params.action !== "hold") {
    return "Вход разрешен";
  }
  if (params.inPositionBlock) {
    return "Уже в позиции, ждем выход";
  }
  if (
    params.entryThreshold !== null &&
    params.signalScore !== null &&
    params.signalScore < params.entryThreshold
  ) {
    return `Ниже entry threshold: ${formatNumber(params.signalScore, 4)} < ${formatNumber(params.entryThreshold, 4)}`;
  }
  if (params.qgateFiltered) {
    return "Q-gate заблокировал вход";
  }
  if (
    params.qgateT1 !== null &&
    params.qgateMaxQ !== null &&
    params.qgateMaxQ < params.qgateT1
  ) {
    return `MaxQ ниже порога: ${formatNumber(params.qgateMaxQ, 4)} < ${formatNumber(params.qgateT1, 4)}`;
  }
  if (
    params.qgateT2 !== null &&
    params.qgateGapQ !== null &&
    params.qgateGapQ < params.qgateT2
  ) {
    return `GapQ ниже порога: ${formatNumber(params.qgateGapQ, 4)} < ${formatNumber(params.qgateT2, 4)}`;
  }
  return "Модель дала hold";
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-slate-100 py-2 last:border-b-0">
      <span className="text-sm text-slate-500">{label}</span>
      <span className="text-right text-sm font-medium text-slate-900">{value}</span>
    </div>
  );
}

export function XgbPredictionsView({
  symbols,
  initialSymbol,
}: XgbPredictionsViewProps) {
  const [selectedSymbol, setSelectedSymbol] = useState(initialSymbol);
  const [selectedLaunchValue, setSelectedLaunchValue] = useState("");
  const [predictions, setPredictions] = useState<XgbPrediction[]>([]);
  const [activeModels, setActiveModels] = useState<ActiveModelOption[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadActiveModels() {
      try {
        const response = await fetch("/api/trading/status_all", {
          method: "GET",
          cache: "no-store",
        });
        if (!response.ok) {
          throw new Error(`Backend returned ${response.status}`);
        }
        const data = (await response.json()) as ActiveAgentsResponse;
        if (!data.success) {
          throw new Error(data.error || "Failed to load active sessions");
        }
        const mapped: ActiveModelOption[] = [];
        const seen = new Set<string>();
        for (const item of data.active_agents ?? []) {
          if (!item.is_xgb || !item.session_id) {
            continue;
          }
          const sessionId = String(item.session_id ?? "");
          const symbol = String(item.symbol ?? "");
          const accountLabel = item.bybit_account_label
            ? String(item.bybit_account_label)
            : item.bybit_account_id
              ? `Account ${String(item.bybit_account_id)}`
              : null;
          const primaryModelPath = item.model_path ? String(item.model_path) : null;
          const sessionOption: ActiveModelOption = {
            value: `session:${sessionId}`,
            kind: "session",
            sessionId,
            symbol,
            modelPath: primaryModelPath,
            accountLabel,
          };
          if (!seen.has(sessionOption.value)) {
            seen.add(sessionOption.value);
            mapped.push(sessionOption);
          }
          for (const modelPath of item.model_paths ?? []) {
            const modelPathString = String(modelPath ?? "").trim();
            if (!modelPathString) {
              continue;
            }
            const modelOption: ActiveModelOption = {
              value: `model:${sessionId}:${modelPathString}`,
              kind: "model",
              sessionId,
              symbol,
              modelPath: modelPathString,
              accountLabel,
            };
            if (!seen.has(modelOption.value)) {
              seen.add(modelOption.value);
              mapped.push(modelOption);
            }
          }
        }
        if (!cancelled) {
          setActiveModels(mapped);
        }
      } catch {
        if (!cancelled) {
          setActiveModels([]);
        }
      }
    }

    void loadActiveModels();
    const timer = setInterval(() => {
      void loadActiveModels();
    }, ACTIVE_MODELS_REFRESH_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const filteredModelOptions = useMemo(() => {
    const selectedNormalized = normalizeSymbol(selectedSymbol);
    return activeModels.filter(
      (item) => !selectedNormalized || normalizeSymbol(item.symbol) === selectedNormalized,
    );
  }, [activeModels, selectedSymbol]);

  const selectedLaunchOption = useMemo(
    () => filteredModelOptions.find((item) => item.value === selectedLaunchValue) ?? null,
    [filteredModelOptions, selectedLaunchValue],
  );

  useEffect(() => {
    if (!selectedLaunchValue) {
      return;
    }
    const exists = filteredModelOptions.some((item) => item.value === selectedLaunchValue);
    if (!exists) {
      setSelectedLaunchValue("");
    }
  }, [filteredModelOptions, selectedLaunchValue]);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function load() {
      try {
        setError(null);
        const response = await fetch(
          `/api/predictions/recent?limit=200${
            selectedSymbol ? `&symbol=${encodeURIComponent(selectedSymbol)}` : ""
          }${
            selectedLaunchOption?.sessionId
              ? `&session_id=${encodeURIComponent(selectedLaunchOption.sessionId)}`
              : ""
          }${
            selectedLaunchOption?.kind === "model" && selectedLaunchOption.modelPath
              ? `&model_path=${encodeURIComponent(selectedLaunchOption.modelPath)}`
              : ""
          }`,
          {
            method: "GET",
            cache: "no-store",
          },
        );
        if (!response.ok) {
          throw new Error(`Backend returned ${response.status}`);
        }

        const data = (await response.json()) as PredictionsResponse;
        if (!data.success) {
          throw new Error(data.error || "Failed to load predictions");
        }

        const mapped = (data.predictions ?? [])
          .filter((item) => isXgbModelPath(item.model_path ?? null))
          .map<XgbPrediction>((item, index) => {
            const marketConditions =
              item.market_conditions && typeof item.market_conditions === "object"
                ? item.market_conditions
                : {};
            const qValues =
              Array.isArray(item.q_values)
                ? item.q_values.filter((value): value is number => isFiniteNumber(value))
                : [];
            const task =
              typeof marketConditions.xgb_task === "string"
                ? marketConditions.xgb_task
                : null;
            const entryThreshold =
              toNumberOrNull(marketConditions.xgb_runtime_threshold) ??
              toNumberOrNull(marketConditions.xgb_p_enter_threshold);
            const qgateT1 = toNumberOrNull(marketConditions.qgate_T1);
            const qgateT2 = toNumberOrNull(marketConditions.qgate_T2);
            const qgateStats = deriveQgateStats(
              qValues,
              toNumberOrNull(marketConditions.qgate_maxQ),
              toNumberOrNull(marketConditions.qgate_gapQ),
            );
            const signalScore = deriveSignalScore(
              qValues,
              task,
              isFiniteNumber(item.confidence) ? item.confidence : null,
            );
            const qgateFiltered = Boolean(marketConditions.qgate_filtered);
            const inPositionBlock = Boolean(marketConditions.xgb_in_position_block);
            const errorHuman =
              typeof marketConditions.error_human === "string"
                ? marketConditions.error_human
                : typeof marketConditions.error === "string"
                  ? marketConditions.error
                  : null;

            return {
              id: String(item.id ?? `${item.timestamp ?? "ts"}-${index}`),
              sessionId: item.session_id ? String(item.session_id) : null,
              timestamp: item.created_at ? String(item.created_at) : item.timestamp ? String(item.timestamp) : null,
              symbol: String(item.symbol ?? selectedSymbol ?? "—"),
              action: item.action ? String(item.action).toLowerCase() : null,
              confidence: isFiniteNumber(item.confidence) ? item.confidence : null,
              currentPrice: isFiniteNumber(item.current_price) ? item.current_price : null,
              positionStatus: item.position_status ? String(item.position_status) : null,
              modelPath: item.model_path ? String(item.model_path) : null,
              direction:
                typeof marketConditions.model_role === "string"
                  ? marketConditions.model_role
                  : typeof marketConditions.xgb_direction === "string"
                    ? marketConditions.xgb_direction
                    : typeof marketConditions.trade_direction === "string"
                      ? marketConditions.trade_direction
                      : typeof marketConditions.active_role === "string"
                        ? marketConditions.active_role
                        : null,
              tradeMode:
                typeof marketConditions.trade_mode === "string"
                  ? marketConditions.trade_mode
                  : null,
              errorHuman,
              qValues,
              signalScore,
              entryThreshold,
              qgateMaxQ: qgateStats.maxQ,
              qgateGapQ: qgateStats.gapQ,
              qgateT1,
              qgateT2,
              qgateFiltered,
              holdReason: deriveHoldReason({
                action: item.action ? String(item.action).toLowerCase() : null,
                signalScore,
                entryThreshold,
                qgateFiltered,
                qgateMaxQ: qgateStats.maxQ,
                qgateGapQ: qgateStats.gapQ,
                qgateT1,
                qgateT2,
                inPositionBlock,
              }),
            };
          });

        if (!cancelled) {
          setPredictions(mapped);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(
            loadError instanceof Error ? loadError.message : "Failed to load XGB predictions",
          );
          setPredictions([]);
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
          timer = setTimeout(() => {
            void load();
          }, 5000);
        }
      }
    }

    setIsLoading(true);
    void load();

    return () => {
      cancelled = true;
      if (timer) {
        clearTimeout(timer);
      }
    };
  }, [selectedLaunchOption, selectedSymbol]);

  const summary = useMemo(() => {
    const buyCount = predictions.filter((item) => item.action === "buy").length;
    const sellCount = predictions.filter((item) => item.action === "sell").length;
    const holdCount = predictions.filter((item) => item.action === "hold").length;

    return {
      total: predictions.length,
      buyCount,
      sellCount,
      holdCount,
    };
  }, [predictions]);

  return (
    <div className="space-y-6">
      <div className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
              XGB Predictions
            </p>
            <h2 className="mt-2 text-2xl font-semibold text-slate-900">
              Последние предсказания
            </h2>
            <p className="mt-2 text-sm text-slate-600">
              Здесь показываются только предсказания моделей XGB по выбранному символу и выбранной запущенной модели.
            </p>
          </div>

          <div className="flex flex-col gap-4 sm:flex-row">
            <label className="block">
              <span className="mb-2 block text-sm font-medium text-slate-700">Символ</span>
              <select
                className="min-w-[220px] rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-400"
                value={selectedSymbol}
                onChange={(event) => setSelectedSymbol(event.target.value)}
              >
                <option value="">Все XGB</option>
                {symbols.map((symbol) => (
                  <option key={symbol} value={symbol}>
                    {symbol}
                  </option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="mb-2 block text-sm font-medium text-slate-700">
                Запущенная модель
              </span>
              <select
                className="min-w-[320px] rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-400"
                value={selectedLaunchValue}
                onChange={(event) => setSelectedLaunchValue(event.target.value)}
              >
                <option value="">Все запущенные модели</option>
                {filteredModelOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {formatModelOptionLabel(option)}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Всего</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">{summary.total}</p>
        </div>
        <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-emerald-700">Buy</p>
          <p className="mt-2 text-2xl font-semibold text-emerald-800">{summary.buyCount}</p>
        </div>
        <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-rose-700">Sell</p>
          <p className="mt-2 text-2xl font-semibold text-rose-800">{summary.sellCount}</p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-slate-100 p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-600">Hold</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">{summary.holdCount}</p>
        </div>
      </div>

      {isLoading ? (
        <div className="rounded-3xl border border-slate-200 bg-slate-50 p-5 text-sm text-slate-500">
          Загружаю XGB предсказания...
        </div>
      ) : null}

      {!isLoading && error ? (
        <div className="rounded-3xl border border-rose-200 bg-rose-50 p-5 text-sm text-rose-700">
          Ошибка загрузки: {error}
        </div>
      ) : null}

      {!isLoading && !error && predictions.length === 0 ? (
        <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6 text-sm text-slate-500">
          {selectedSymbol
            ? `Для ${selectedSymbol}${selectedLaunchValue ? " и выбранной модели" : ""} XGB предсказаний пока нет.`
            : "XGB предсказаний пока нет."}
        </div>
      ) : null}

      {!isLoading && !error && predictions.length > 0 ? (
        <div className="grid gap-4">
          {predictions.map((prediction) => (
            <div
              key={prediction.id}
              className="rounded-3xl border border-slate-200 bg-white p-5"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
                      {prediction.symbol}
                    </p>
                    <span
                      className="inline-flex h-5 w-5 cursor-help items-center justify-center rounded-full border border-slate-300 text-xs font-semibold text-slate-600"
                      title={`${prediction.errorHuman ? `Причина: ${prediction.errorHuman}\n` : ""}Session: ${prediction.sessionId || "—"}\nQ-gate max / T1: ${
                        prediction.qgateMaxQ !== null || prediction.qgateT1 !== null
                          ? `${formatNumber(prediction.qgateMaxQ, 4)} / ${formatNumber(prediction.qgateT1, 4)}`
                          : "—"
                      }\nQ-gate gap / T2: ${
                        prediction.qgateGapQ !== null || prediction.qgateT2 !== null
                          ? `${formatNumber(prediction.qgateGapQ, 4)} / ${formatNumber(prediction.qgateT2, 4)}`
                          : "—"
                      }`}
                      aria-label="Prediction details"
                    >
                      ?
                    </span>
                  </div>
                </div>
                <span
                  className={`rounded-full border px-3 py-1 text-sm font-semibold ${actionTone(
                    prediction.action,
                  )}`}
                >
                  {prediction.action || "—"}
                </span>
              </div>

              <div className="mt-4">
                <MetaRow label="Время" value={formatDateTime(prediction.timestamp)} />
                {prediction.errorHuman ? (
                  <MetaRow label="Причина" value={prediction.errorHuman} />
                ) : null}
                <MetaRow
                  label="Model"
                  value={formatModelLabel(prediction.modelPath, prediction.direction)}
                />
                <MetaRow label="Trade mode" value={prediction.tradeMode || "—"} />
                <MetaRow
                  label="Signal / Threshold"
                  value={
                    prediction.signalScore !== null || prediction.entryThreshold !== null
                      ? `${formatNumber(prediction.signalScore, 4)} / ${formatNumber(prediction.entryThreshold, 4)}`
                      : "—"
                  }
                />
                <MetaRow
                  label="Confidence"
                  value={
                    prediction.confidence !== null
                      ? formatNumber(prediction.confidence, 4)
                      : "—"
                  }
                />
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
