"use client";

import { useEffect, useMemo, useState } from "react";

import {
  type BybitAccount,
  type XgbEnsembleSummary,
  type XgbModelSummary,
} from "@/lib/backend";

type XgbLaunchFormProps = {
  accounts: BybitAccount[];
  selectedAccountId: string | null;
  symbols: string[];
};

const ENSEMBLES = ["ensemble-a", "ensemble-b", "ensemble-c"];
const XGB_SIGNAL_EXIT_WINDOW = 20;
const XGB_SIGNAL_EXIT_THRESHOLD_DEFAULT = 0.5;

function formatAccountLabel(account: BybitAccount): string {
  return account.apiKeyMasked
    ? `${account.label} (${account.apiKeyMasked})`
    : account.label;
}

function formatNumber(
  value: number | null | undefined,
  options?: Intl.NumberFormatOptions,
): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return new Intl.NumberFormat("ru-RU", {
    maximumFractionDigits: 4,
    ...options,
  }).format(value);
}

function formatPercent(value: number | null | undefined, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return `${value.toFixed(digits)}%`;
}

function formatRatioPercent(value: number | null | undefined, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

function metricTone(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "text-slate-900";
  }
  if (value > 0) {
    return "text-emerald-600";
  }
  if (value < 0) {
    return "text-rose-600";
  }
  return "text-slate-900";
}

function summarizeDirection(model: XgbModelSummary | null | undefined): string {
  if (!model) {
    return "—";
  }
  return model.direction ?? model.trainedAs ?? "—";
}

function modelDirectionGroup(model: XgbModelSummary): "long" | "short" | "unknown" {
  const task = (model.task ?? "").toLowerCase();
  if (task.includes("short")) {
    return "short";
  }
  if (task.includes("long")) {
    return "long";
  }
  const direction = (model.direction ?? model.trainedAs ?? "").toLowerCase();
  if (direction === "short" || direction === "long") {
    return direction;
  }
  return "unknown";
}

function describeXgbTask(task: string | null): string {
  switch ((task ?? "").toLowerCase()) {
    case "entry_long":
    case "enter_long":
      return "entry_long: вход в long";
    case "entry_short":
    case "enter_short":
      return "entry_short: вход в short";
    case "exit_long":
      return "exit_long: выход из long";
    case "exit_short":
      return "exit_short: выход из short";
    case "directional":
      return "directional: hold / buy / sell";
    default:
      return task ? `${task}: описание не задано` : "task не задан";
  }
}

function strongestModelScore(model: XgbModelSummary): number {
  const pnl = model.bestOos.pnlTotal ?? -1_000_000;
  const profitFactor = model.bestOos.profitFactor ?? 0;
  const drawdown = model.bestOos.maxDd ?? 1;
  const roi = model.bestOos.roiPct ?? -10_000;
  const trades = model.bestOos.tradesCount ?? 0;
  const valAcc = model.trainMetrics.valAcc ?? 0;

  return (
    pnl * 1.0 +
    profitFactor * 2_000 +
    roi * 120 +
    trades * 2 +
    valAcc * 500 -
    drawdown * 4_000
  );
}

function modelPathForTrading(model: XgbModelSummary | null): string {
  if (!model?.path) {
    return "";
  }
  const normalizedPath = model.path.startsWith("/")
    ? model.path
    : `/workspace/${model.path.replace(/^\/+/, "")}`;
  return `${normalizedPath}/model.json`;
}

function defaultSignalExitStartStep(maxHoldSteps: number | null | undefined): number {
  if (typeof maxHoldSteps !== "number" || !Number.isFinite(maxHoldSteps) || maxHoldSteps <= 0) {
    return 90;
  }
  const rounded = Math.max(XGB_SIGNAL_EXIT_WINDOW, Math.round(maxHoldSteps));
  const proposed = Math.round(rounded * 0.65);
  return Math.max(
    XGB_SIGNAL_EXIT_WINDOW,
    Math.min(rounded, proposed),
  );
}

function mapEnsemblePayload(payload: unknown): Record<string, XgbEnsembleSummary> {
  if (!payload || typeof payload !== "object") {
    return {};
  }

  const source = payload as Record<string, unknown>;

  function toNumberOrNull(value: unknown): number | null {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  function toNumberArrayOrNull(value: unknown): number[] | null {
    if (!Array.isArray(value)) {
      return null;
    }
    const values = value.filter(
      (item): item is number => typeof item === "number" && Number.isFinite(item),
    );
    return values.length > 0 ? values : null;
  }

  function mapModel(raw: unknown): XgbModelSummary | null {
    if (!raw || typeof raw !== "object") {
      return null;
    }

    const item = raw as Record<string, unknown>;
    const trainMetrics =
      item.train_metrics && typeof item.train_metrics === "object"
        ? (item.train_metrics as Record<string, unknown>)
        : {};
    const cfg = item.cfg && typeof item.cfg === "object" ? (item.cfg as Record<string, unknown>) : {};
    const bestOos =
      item.best_oos && typeof item.best_oos === "object"
        ? (item.best_oos as Record<string, unknown>)
        : {};

    return {
      version: String(item.version ?? ""),
      path: String(item.path ?? ""),
      isCurrent: Boolean(item.is_current),
      modelId: String(item.model_id ?? ""),
      runId: String(item.run_id ?? ""),
      modelUuid: String(item.model_uuid ?? ""),
      direction: item.direction ? String(item.direction) : null,
      task: item.task ? String(item.task) : null,
      trainedAs: item.trained_as ? String(item.trained_as) : null,
      createdAt: item.created_at ? String(item.created_at) : null,
      sourceRunPath: item.source_run_path ? String(item.source_run_path) : null,
      trainMetrics: {
        valAcc: toNumberOrNull(trainMetrics.val_acc),
        f1BuySellVal: toNumberOrNull(trainMetrics.f1_buy_sell_val),
        f1Val: toNumberArrayOrNull(trainMetrics.f1_val),
        precisionVal: toNumberArrayOrNull(trainMetrics.precision_val),
        recallVal: toNumberArrayOrNull(trainMetrics.recall_val),
        proxyPnlVal:
          trainMetrics.proxy_pnl_val && typeof trainMetrics.proxy_pnl_val === "object"
            ? (trainMetrics.proxy_pnl_val as XgbModelSummary["trainMetrics"]["proxyPnlVal"])
            : null,
      },
      cfg: {
        horizonSteps: toNumberOrNull(cfg.horizon_steps),
        threshold: toNumberOrNull(cfg.threshold),
        maxHoldSteps: toNumberOrNull(cfg.max_hold_steps),
        minProfit: toNumberOrNull(cfg.min_profit),
        feeBps: toNumberOrNull(cfg.fee_bps),
        pEnterThreshold: toNumberOrNull(cfg.p_enter_threshold),
        entryTpPct: toNumberOrNull(cfg.entry_tp_pct),
        entrySlPct: toNumberOrNull(cfg.entry_sl_pct),
        entryTrailPct: toNumberOrNull(cfg.entry_trail_pct),
        nEstimators: toNumberOrNull(cfg.n_estimators),
        maxDepth: toNumberOrNull(cfg.max_depth),
        learningRate: toNumberOrNull(cfg.learning_rate),
        subsample: toNumberOrNull(cfg.subsample),
        colsampleBytree: toNumberOrNull(cfg.colsample_bytree),
        regLambda: toNumberOrNull(cfg.reg_lambda),
        minChildWeight: toNumberOrNull(cfg.min_child_weight),
        gamma: toNumberOrNull(cfg.gamma),
      },
      bestOos: {
        days: toNumberOrNull(bestOos.days),
        ts: bestOos.ts ? String(bestOos.ts) : null,
        pnlTotal: toNumberOrNull(bestOos.pnl_total),
        roiPct: toNumberOrNull(bestOos.roi_pct),
        profitFactor: toNumberOrNull(bestOos.profit_factor),
        maxDd: toNumberOrNull(bestOos.max_dd),
        tradesCount: toNumberOrNull(bestOos.trades_count),
        winrate: toNumberOrNull(bestOos.winrate),
        avgTradePnl: toNumberOrNull(bestOos.avg_trade_pnl),
        avgBarsHeld: toNumberOrNull(bestOos.avg_bars_held),
        equityEnd: toNumberOrNull(bestOos.equity_end),
      },
    };
  }

  return Object.fromEntries(
    Object.entries(source).map(([ensembleName, ensembleValue]) => {
      const ensemble =
        ensembleValue && typeof ensembleValue === "object"
          ? (ensembleValue as Record<string, unknown>)
          : {};
      const versionsRaw = Array.isArray(ensemble.versions) ? ensemble.versions : [];

      return [
        ensembleName,
        {
          currentVersion: ensemble.current_version
            ? String(ensemble.current_version)
            : null,
          currentModel: mapModel(ensemble.current_model),
          versions: versionsRaw
            .map((item) => mapModel(item))
            .filter((item): item is XgbModelSummary => item !== null),
        },
      ];
    }),
  );
}

function StatCard({
  label,
  value,
  tone = "text-slate-900",
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
      <p className="text-xs uppercase tracking-[0.2em] text-slate-500">{label}</p>
      <p className={`mt-2 text-xl font-semibold ${tone}`}>{value}</p>
    </div>
  );
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-slate-100 py-3 last:border-b-0">
      <span className="text-sm text-slate-500">{label}</span>
      <span className="text-right text-sm font-medium text-slate-900">{value}</span>
    </div>
  );
}

export function XgbLaunchForm({
  accounts,
  selectedAccountId,
  symbols,
}: XgbLaunchFormProps) {
  const defaultAccountId = useMemo(() => {
    if (selectedAccountId && accounts.some((item) => item.id === selectedAccountId)) {
      return selectedAccountId;
    }
    return accounts[0]?.id ?? "";
  }, [accounts, selectedAccountId]);

  const [accountId, setAccountId] = useState(defaultAccountId);
  const [symbol, setSymbol] = useState(symbols[0] ?? "BTCUSDT");
  const [ensemble, setEnsemble] = useState("ensemble-a");
  const [executionMode, setExecutionMode] = useState<"market" | "limit_post_only">("market");
  const [accountPct, setAccountPct] = useState(100);
  const [longSignalExitEnabled, setLongSignalExitEnabled] = useState(false);
  const [longSignalExitStartStep, setLongSignalExitStartStep] = useState(90);
  const [longSignalExitThreshold, setLongSignalExitThreshold] = useState(
    XGB_SIGNAL_EXIT_THRESHOLD_DEFAULT,
  );
  const [shortSignalExitEnabled, setShortSignalExitEnabled] = useState(false);
  const [shortSignalExitStartStep, setShortSignalExitStartStep] = useState(90);
  const [shortSignalExitThreshold, setShortSignalExitThreshold] = useState(
    XGB_SIGNAL_EXIT_THRESHOLD_DEFAULT,
  );
  const [postExitGuardEnabled, setPostExitGuardEnabled] = useState(true);
  const [postExitCooldownCandles, setPostExitCooldownCandles] = useState(5);
  const [postExitThresholdBoostPct, setPostExitThresholdBoostPct] = useState(10);
  const [postExitBoostCandles, setPostExitBoostCandles] = useState(20);
  const [takeProfitEnabled, setTakeProfitEnabled] = useState(true);
  const [stopLossEnabled, setStopLossEnabled] = useState(true);
  const [trailingEnabled, setTrailingEnabled] = useState(false);
  const [takeProfitPct, setTakeProfitPct] = useState(3);
  const [stopLossPct, setStopLossPct] = useState(1);
  const [selectedVersion, setSelectedVersion] = useState<string | null>(null);
  const [selectedLongVersion, setSelectedLongVersion] = useState<string | null>(null);
  const [selectedShortVersion, setSelectedShortVersion] = useState<string | null>(null);
  const [ensemblesMap, setEnsemblesMap] = useState<Record<string, XgbEnsembleSummary>>({});
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [launchMessage, setLaunchMessage] = useState<string | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [isLaunching, setIsLaunching] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadEnsembles() {
      setIsLoadingModels(true);
      setModelsError(null);

      try {
        const response = await fetch(`/api/xgb/ensembles?symbol=${encodeURIComponent(symbol)}`, {
          method: "GET",
        });
        if (!response.ok) {
          throw new Error(`Backend returned ${response.status}`);
        }
        const raw = (await response.json()) as {
          success?: boolean;
          error?: string;
          ensembles?: unknown;
        };
        if (!raw.success) {
          throw new Error(raw.error || "Failed to load ensembles");
        }
        const data = mapEnsemblePayload(raw.ensembles);
        if (!cancelled) {
          setEnsemblesMap(data);
        }
      } catch (error) {
        if (!cancelled) {
          setEnsemblesMap({});
          setModelsError(error instanceof Error ? error.message : "Failed to load XGB models");
        }
      } finally {
        if (!cancelled) {
          setIsLoadingModels(false);
        }
      }
    }

    void loadEnsembles();

    return () => {
      cancelled = true;
    };
  }, [symbol]);

  const selectedEnsemble = ensemblesMap[ensemble] ?? null;
  const selectedModel =
    selectedEnsemble?.versions.find((item) => item.version === selectedVersion) ??
    selectedEnsemble?.currentModel ??
    null;
  const versionRanks = useMemo(() => {
    if (!selectedEnsemble?.versions.length) {
      return new Map<string, number>();
    }
    const sorted = [...selectedEnsemble.versions].sort(
      (a, b) => strongestModelScore(b) - strongestModelScore(a),
    );
    return new Map(sorted.map((item, index) => [item.version, index + 1]));
  }, [selectedEnsemble]);
  const versionGroups = useMemo(() => {
    const versions = selectedEnsemble?.versions ?? [];
    return {
      long: versions.filter((item) => modelDirectionGroup(item) === "long"),
      short: versions.filter((item) => modelDirectionGroup(item) === "short"),
      unknown: versions.filter((item) => modelDirectionGroup(item) === "unknown"),
    };
  }, [selectedEnsemble]);
  const selectedLongModel =
    versionGroups.long.find((item) => item.version === selectedLongVersion) ??
    versionGroups.long.find((item) => item.isCurrent) ??
    versionGroups.long[0] ??
    null;
  const selectedShortModel =
    versionGroups.short.find((item) => item.version === selectedShortVersion) ??
    versionGroups.short.find((item) => item.isCurrent) ??
    versionGroups.short[0] ??
    null;
  const selectedLaunchModels = [selectedLongModel, selectedShortModel].filter(
    (item): item is XgbModelSummary => item !== null,
  );
  const launchMaxHoldSteps =
    selectedLaunchModels.length > 0
      ? Math.max(...selectedLaunchModels.map((item) => item.cfg.maxHoldSteps ?? 0))
      : null;
  const signalExitStartMin = XGB_SIGNAL_EXIT_WINDOW;
  const longSignalExitStartMax = Math.max(
    signalExitStartMin,
    Math.round(selectedLongModel?.cfg.maxHoldSteps ?? signalExitStartMin),
  );
  const shortSignalExitStartMax = Math.max(
    signalExitStartMin,
    Math.round(selectedShortModel?.cfg.maxHoldSteps ?? signalExitStartMin),
  );

  useEffect(() => {
    if (!selectedEnsemble) {
      setSelectedVersion(null);
      return;
    }

    const nextVersion =
      selectedEnsemble.versions.find((item) => item.version === selectedVersion)?.version ??
      selectedEnsemble.currentVersion ??
      selectedEnsemble.versions[0]?.version ??
      null;

    setSelectedVersion(nextVersion);
  }, [selectedEnsemble, selectedVersion]);

  useEffect(() => {
    setSelectedLongVersion(
      versionGroups.long.find((item) => item.version === selectedLongVersion)?.version ??
        versionGroups.long.find((item) => item.isCurrent)?.version ??
        versionGroups.long[0]?.version ??
        null,
    );
    setSelectedShortVersion(
      versionGroups.short.find((item) => item.version === selectedShortVersion)?.version ??
        versionGroups.short.find((item) => item.isCurrent)?.version ??
        versionGroups.short[0]?.version ??
        null,
    );
  }, [selectedLongVersion, selectedShortVersion, versionGroups]);

  useEffect(() => {
    const stopLossSource = selectedLongModel ?? selectedModel;
    if (!stopLossSource) {
      return;
    }
    const nextStopLoss =
      typeof stopLossSource.cfg.entrySlPct === "number" &&
      Number.isFinite(stopLossSource.cfg.entrySlPct)
        ? Math.abs(stopLossSource.cfg.entrySlPct) * 100
        : 1;
    setStopLossPct(Number(nextStopLoss.toFixed(2)));
    setTakeProfitPct(3);
    setStopLossEnabled(true);
    setTakeProfitEnabled(true);
    setTrailingEnabled(false);
    setPostExitGuardEnabled(true);
    setPostExitCooldownCandles(5);
    setPostExitThresholdBoostPct(10);
    setPostExitBoostCandles(20);
  }, [selectedLongModel, selectedModel]);

  useEffect(() => {
    if (!selectedLongModel) {
      return;
    }
    setLongSignalExitEnabled(false);
    setLongSignalExitStartStep(defaultSignalExitStartStep(selectedLongModel.cfg.maxHoldSteps));
    setLongSignalExitThreshold(XGB_SIGNAL_EXIT_THRESHOLD_DEFAULT);
  }, [selectedLongModel]);

  useEffect(() => {
    if (!selectedShortModel) {
      return;
    }
    setShortSignalExitEnabled(false);
    setShortSignalExitStartStep(defaultSignalExitStartStep(selectedShortModel.cfg.maxHoldSteps));
    setShortSignalExitThreshold(XGB_SIGNAL_EXIT_THRESHOLD_DEFAULT);
  }, [selectedShortModel]);

  async function handleDeleteModel(symbolCode: string, ensembleName: string, versionId: string) {
    try {
      const response = await fetch(`/api/xgb/prod/models`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol: symbolCode,
          ensemble: ensembleName,
          version: versionId,
        }),
      });
      const data = await response.json() as { success?: boolean; error?: string };
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Не удалось удалить модель");
      }
      
      const res = await fetch(`/api/xgb/ensembles?symbol=${encodeURIComponent(symbol)}`);
      if (res.ok) {
        const raw = await res.json() as { success?: boolean; ensembles?: unknown };
        if (raw.success) {
          setEnsemblesMap(mapEnsemblePayload(raw.ensembles));
        }
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : "Ошибка удаления");
    }
  }

  async function handleLaunch() {
    setLaunchMessage(null);
    setLaunchError(null);

    if (!selectedLongModel || !selectedShortModel) {
      setLaunchError("Для запуска выберите одну long-модель и одну short-модель.");
      return;
    }

    if (!accountId) {
      setLaunchError("Выберите аккаунт Bybit.");
      return;
    }

    if (
      typeof launchMaxHoldSteps !== "number" ||
      !Number.isFinite(launchMaxHoldSteps) ||
      launchMaxHoldSteps <= 0
    ) {
      setLaunchError("У выбранных моделей не задан корректный hold steps.");
      return;
    }

    setIsLaunching(true);

    try {
      const longModelPath = modelPathForTrading(selectedLongModel);
      const shortModelPath = modelPathForTrading(selectedShortModel);
      const anySignalExitEnabled = longSignalExitEnabled || shortSignalExitEnabled;
      const response = await fetch(`/api/trading/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          symbols: [symbol],
          account_id: accountId,
          model_path: longModelPath,
          model_paths: [longModelPath, shortModelPath],
          model_roles: {
            [longModelPath]: "long",
            [shortModelPath]: "short",
          },
          direction: "long",
          trade_mode: "long-short",
          execution_mode: executionMode,
          account_pct: accountPct,
          exit_mode: "hold_steps",
          max_hold_steps: Math.round(launchMaxHoldSteps),
          xgb_signal_exit_enabled: anySignalExitEnabled,
          xgb_signal_exit_start_step: longSignalExitEnabled
            ? Math.round(longSignalExitStartStep)
            : undefined,
          xgb_signal_exit_window: anySignalExitEnabled ? XGB_SIGNAL_EXIT_WINDOW : undefined,
          xgb_signal_exit_threshold: longSignalExitEnabled
            ? longSignalExitThreshold
            : undefined,
          xgb_signal_exit_long_enabled: longSignalExitEnabled,
          xgb_signal_exit_long_start_step: longSignalExitEnabled
            ? Math.round(longSignalExitStartStep)
            : undefined,
          xgb_signal_exit_long_threshold: longSignalExitEnabled
            ? longSignalExitThreshold
            : undefined,
          xgb_signal_exit_short_enabled: shortSignalExitEnabled,
          xgb_signal_exit_short_start_step: shortSignalExitEnabled
            ? Math.round(shortSignalExitStartStep)
            : undefined,
          xgb_signal_exit_short_threshold: shortSignalExitEnabled
            ? shortSignalExitThreshold
            : undefined,
          xgb_postexit_guard_enabled: postExitGuardEnabled,
          xgb_postexit_cooldown_candles: postExitGuardEnabled
            ? Math.round(postExitCooldownCandles)
            : undefined,
          xgb_postexit_threshold_boost_pct: postExitGuardEnabled
            ? postExitThresholdBoostPct
            : undefined,
          xgb_postexit_boost_candles: postExitGuardEnabled
            ? Math.round(postExitBoostCandles)
            : undefined,
          risk_management_type:
            takeProfitEnabled || stopLossEnabled ? "exchange_orders" : "none",
          take_profit_pct: takeProfitEnabled ? takeProfitPct : undefined,
          stop_loss_pct: stopLossEnabled ? stopLossPct : undefined,
          trailing_enabled: trailingEnabled,
          risk_stop_mode: trailingEnabled ? "atr_trailing" : "fixed_pct",
          trailing_mode: trailingEnabled ? "atr_trailing" : "",
          atr_trail_mult: 2.0,
        }),
      });

      const data = (await response.json()) as {
        success?: boolean;
        message?: string;
        error?: string;
        task_id?: string;
      };

      if (!response.ok || !data.success) {
        throw new Error(data.error || `Backend returned ${response.status}`);
      }

      setLaunchMessage(
        data.task_id
          ? `Запуск поставлен в очередь, task_id: ${data.task_id}`
          : data.message || "Запуск отправлен.",
      );
    } catch (error) {
      setLaunchError(error instanceof Error ? error.message : "Не удалось запустить XGB");
    } finally {
      setIsLaunching(false);
    }
  }

  function renderSignalExitBlock({
    title,
    enabled,
    startStep,
    threshold,
    startMax,
    onEnabledChange,
    onStartStepChange,
    onThresholdChange,
  }: {
    title: string;
    enabled: boolean;
    startStep: number;
    threshold: number;
    startMax: number;
    onEnabledChange: (value: boolean) => void;
    onStartStepChange: (value: number) => void;
    onThresholdChange: (value: number) => void;
  }) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
        <div className="flex items-center justify-between gap-3">
          <label className="flex items-center gap-3 text-sm font-semibold text-slate-900">
            <input
              type="checkbox"
              checked={enabled}
              onChange={(event) => onEnabledChange(event.target.checked)}
              className="h-4 w-4 rounded border-slate-300"
            />
            Early exit по среднему signal ({title})
          </label>
          <span className={`text-sm font-semibold ${enabled ? "text-emerald-700" : "text-slate-500"}`}>
            {enabled ? `c ${startStep} шага` : "выключен"}
          </span>
        </div>
        <div className="mt-4">
          <div className="mb-2 flex justify-between text-xs font-semibold text-slate-600">
            <span>Порог weak signal</span>
            <span>{threshold.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0.1}
            max={0.95}
            step={0.01}
            value={threshold}
            disabled={!enabled}
            onChange={(event) => onThresholdChange(Number(event.target.value))}
            className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
          />
        </div>
        <input
          type="range"
          min={signalExitStartMin}
          max={startMax}
          step={1}
          value={Math.min(startStep, startMax)}
          disabled={!enabled}
          onChange={(event) => onStartStepChange(Number(event.target.value))}
          className="mt-4 h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
        />
        <div className="mt-2 flex justify-between text-xs text-slate-500">
          <span>{signalExitStartMin}</span>
          <span>{startMax}</span>
        </div>
        <p className="mt-2 text-xs text-slate-500">
          После выбранного шага бот смотрит средний signal последних {XGB_SIGNAL_EXIT_WINDOW} свечей.
          Если средний signal ниже порога weak signal, позиция закрывается раньше hold steps.
        </p>
      </div>
    );
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[420px_minmax(0,1fr)]">
      <section className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
        <h2 className="text-2xl font-semibold text-slate-900">
          Управление торговлей
        </h2>
        <p className="mt-2 text-sm leading-7 text-slate-600">
          Запуск и остановка торгового бота
        </p>

        <div className="mt-6 space-y-5">
          <div>
            <label className="mb-2 block text-sm font-semibold text-slate-900">
              Аккаунт Bybit
            </label>
            <select
              className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300"
              value={accountId}
              onChange={(event) => setAccountId(event.target.value)}
            >
              {accounts.length > 0 ? (
                accounts.map((account) => (
                  <option key={account.id} value={account.id}>
                    {formatAccountLabel(account)}
                  </option>
                ))
              ) : (
                <option value="">Нет аккаунтов Bybit</option>
              )}
            </select>
          </div>

          <div>
            <label className="mb-2 block text-sm font-semibold text-slate-900">
              Ансамбль и версии
            </label>

            <div className="space-y-4 rounded-2xl border border-slate-200 bg-white p-4">
              <div>
                <label className="mb-2 block text-sm font-semibold text-slate-900">
                  Символ
                </label>
                <select
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300"
                  value={symbol}
                  onChange={(event) => setSymbol(event.target.value)}
                >
                  {symbols.length > 0 ? (
                    symbols.map((item) => (
                      <option key={item} value={item}>
                        {item}
                      </option>
                    ))
                  ) : (
                    <option value="BTCUSDT">BTCUSDT</option>
                  )}
                </select>
              </div>

              <div>
                <label className="mb-2 block text-sm font-semibold text-slate-900">
                  Ансамбль
                </label>
                <select
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300"
                  value={ensemble}
                  onChange={(event) => setEnsemble(event.target.value)}
                >
                  {ENSEMBLES.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="mb-2 block text-sm font-semibold text-slate-900">
                  Стратегия исполнения
                </label>
                <select
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300"
                  value={executionMode}
                  onChange={(event) =>
                    setExecutionMode(
                      event.target.value === "limit_post_only"
                        ? "limit_post_only"
                        : "market",
                    )
                  }
                >
                  <option value="market">Market</option>
                  <option value="limit_post_only">Limit</option>
                </select>
              </div>

              <div className="rounded-2xl border border-blue-100 bg-blue-50 p-4 text-sm text-slate-700">
                <p className="font-semibold text-slate-900">Модели для запуска long-short</p>
                <p className="mt-2">
                  Long: {selectedLongModel ? `${selectedLongModel.version} / ${describeXgbTask(selectedLongModel.task)}` : "не выбрана"}
                </p>
                <p>
                  Short: {selectedShortModel ? `${selectedShortModel.version} / ${describeXgbTask(selectedShortModel.task)}` : "не выбрана"}
                </p>
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between gap-3">
                  <label className="block text-sm font-semibold text-slate-900">
                    Доля счёта для сделки (%)
                  </label>
                  <span className="text-sm font-semibold text-slate-700">
                    {accountPct}%
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={100}
                  step={1}
                  value={accountPct}
                  onChange={(event) => setAccountPct(Number(event.target.value))}
                  className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-200"
                />
                <div className="mt-2 flex justify-between text-xs text-slate-500">
                  <span>1%</span>
                  <span>100%</span>
                </div>
              </div>

              <div className="space-y-3 rounded-2xl border border-slate-200 bg-white p-4">
                <p className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">
                  Long настройки
                </p>
                {renderSignalExitBlock({
                  title: "Long",
                  enabled: longSignalExitEnabled,
                  startStep: longSignalExitStartStep,
                  threshold: longSignalExitThreshold,
                  startMax: longSignalExitStartMax,
                  onEnabledChange: setLongSignalExitEnabled,
                  onStartStepChange: setLongSignalExitStartStep,
                  onThresholdChange: setLongSignalExitThreshold,
                })}
              </div>

              <div className="space-y-3 rounded-2xl border border-slate-200 bg-white p-4">
                <p className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">
                  Short настройки
                </p>
                {renderSignalExitBlock({
                  title: "Short",
                  enabled: shortSignalExitEnabled,
                  startStep: shortSignalExitStartStep,
                  threshold: shortSignalExitThreshold,
                  startMax: shortSignalExitStartMax,
                  onEnabledChange: setShortSignalExitEnabled,
                  onStartStepChange: setShortSignalExitStartStep,
                  onThresholdChange: setShortSignalExitThreshold,
                })}
              </div>

              <div className="space-y-4 rounded-2xl border border-slate-200 bg-white p-4">
                <p className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">
                  Общие настройки
                </p>

              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <div className="flex items-center justify-between gap-3">
                  <label className="flex items-center gap-3 text-sm font-semibold text-slate-900">
                    <input
                      type="checkbox"
                      checked={postExitGuardEnabled}
                      onChange={(event) => setPostExitGuardEnabled(event.target.checked)}
                      className="h-4 w-4 rounded border-slate-300"
                    />
                    Post-exit защита входа
                  </label>
                  <span className={`text-sm font-semibold ${postExitGuardEnabled ? "text-emerald-700" : "text-slate-500"}`}>
                    {postExitGuardEnabled ? "активна" : "выключена"}
                  </span>
                </div>

                <div className="mt-4 space-y-4">
                  <div>
                    <div className="mb-2 flex justify-between text-xs font-semibold text-slate-600">
                      <span>Cooldown после выхода</span>
                      <span>{postExitCooldownCandles} свечей</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={48}
                      step={1}
                      value={postExitCooldownCandles}
                      disabled={!postExitGuardEnabled}
                      onChange={(event) => setPostExitCooldownCandles(Number(event.target.value))}
                      className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
                    />
                  </div>

                  <div>
                    <div className="mb-2 flex justify-between text-xs font-semibold text-slate-600">
                      <span>Повышение threshold</span>
                      <span>+{postExitThresholdBoostPct}%</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={50}
                      step={1}
                      value={postExitThresholdBoostPct}
                      disabled={!postExitGuardEnabled}
                      onChange={(event) => setPostExitThresholdBoostPct(Number(event.target.value))}
                      className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
                    />
                  </div>

                  <div>
                    <div className="mb-2 flex justify-between text-xs font-semibold text-slate-600">
                      <span>Окно повышенного threshold</span>
                      <span>{postExitBoostCandles} свечей</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={96}
                      step={1}
                      value={postExitBoostCandles}
                      disabled={!postExitGuardEnabled}
                      onChange={(event) => setPostExitBoostCandles(Number(event.target.value))}
                      className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
                    />
                  </div>
                </div>

                <p className="mt-3 text-xs text-slate-500">
                  Если после TP/SL/trailing модель снова даёт вход, сначала будет hold, затем вход только при signal выше threshold на заданный процент.
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <div className="flex items-center justify-between gap-3">
                  <label className="flex items-center gap-3 text-sm font-semibold text-slate-900">
                    <input
                      type="checkbox"
                      checked={takeProfitEnabled}
                      onChange={(event) => setTakeProfitEnabled(event.target.checked)}
                      className="h-4 w-4 rounded border-slate-300"
                    />
                    Включить TP
                  </label>
                  <span className="text-sm font-semibold text-slate-700">
                    {takeProfitPct.toFixed(1)}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0.5}
                  max={10}
                  step={0.1}
                  value={takeProfitPct}
                  disabled={!takeProfitEnabled}
                  onChange={(event) => setTakeProfitPct(Number(event.target.value))}
                  className="mt-4 h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
                />
                <p className="mt-2 text-xs text-slate-500">
                  По умолчанию `3%` для резких пиков.
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <div className="flex items-center justify-between gap-3">
                  <label className="flex items-center gap-3 text-sm font-semibold text-slate-900">
                    <input
                      type="checkbox"
                      checked={stopLossEnabled}
                      onChange={(event) => setStopLossEnabled(event.target.checked)}
                      className="h-4 w-4 rounded border-slate-300"
                    />
                    Включить SL
                  </label>
                  <span className="text-sm font-semibold text-slate-700">
                    {stopLossPct.toFixed(1)}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0.3}
                  max={5}
                  step={0.1}
                  value={stopLossPct}
                  disabled={!stopLossEnabled}
                  onChange={(event) => setStopLossPct(Number(event.target.value))}
                  className="mt-4 h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
                />
                <p className="mt-2 text-xs text-slate-500">
                  По умолчанию берется из `entry_sl_pct` модели.
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <div className="flex items-center justify-between gap-3">
                  <label className="flex items-center gap-3 text-sm font-semibold text-slate-900">
                    <input
                      type="checkbox"
                      checked={trailingEnabled}
                      onChange={(event) => setTrailingEnabled(event.target.checked)}
                      className="h-4 w-4 rounded border-slate-300"
                    />
                    Включить trailing stop
                  </label>
                  <span
                    className={`text-sm font-semibold ${
                      trailingEnabled ? "text-emerald-700" : "text-slate-500"
                    }`}
                  >
                    {trailingEnabled ? "ATR trailing" : "выключен"}
                  </span>
                </div>
                <p className="mt-2 text-xs text-slate-500">
                  Если галочка снята, запуск принудительно отключает старый trailing из Redis.
                </p>
              </div>
              </div>

              <button
                type="button"
                className="w-full rounded-xl bg-slate-900 px-4 py-3 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
                disabled={!accountId || !selectedLongModel || !selectedShortModel || isLaunching}
                onClick={() => void handleLaunch()}
              >
                {isLaunching ? "Запуск..." : "Запустить"}
              </button>

              {launchMessage ? (
                <p className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
                  {launchMessage}
                </p>
              ) : null}

              {launchError ? (
                <p className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                  {launchError}
                </p>
              ) : null}
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-3xl border border-slate-200 bg-white p-5">
        <div className="mb-4 border-b border-slate-200 pb-4">
          <h3 className="text-lg font-semibold text-slate-900">Текущая модель</h3>
          <p className="mt-2 text-sm leading-7 text-slate-600">
            Справа показана текущая модель выбранного ансамбля и ее ключевые метрики.
          </p>
        </div>

        {isLoadingModels ? (
          <div className="flex min-h-[520px] items-center justify-center rounded-3xl border border-dashed border-slate-300 bg-slate-50 text-sm text-slate-500">
            Загружаю данные моделей...
          </div>
        ) : modelsError ? (
          <div className="flex min-h-[520px] items-center justify-center rounded-3xl border border-rose-200 bg-rose-50 px-6 text-center text-sm text-rose-700">
            {modelsError}
          </div>
        ) : !selectedModel ? (
          <div className="flex min-h-[520px] items-center justify-center rounded-3xl border border-dashed border-slate-300 bg-slate-50 px-6 text-center text-sm text-slate-500">
            Для {ensemble} по символу {symbol} пока нет текущей XGB модели.
          </div>
        ) : (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
                    {symbol} / {ensemble}
                  </p>
                  <h4 className="mt-2 text-2xl font-semibold text-slate-900">
                    {selectedModel.modelId || selectedModel.version}
                  </h4>
                </div>
                <div className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700">
                  {summarizeDirection(selectedModel)}
                </div>
              </div>

              <div className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <StatCard
                  label="Net PnL"
                  value={formatNumber(selectedModel.bestOos.pnlTotal, {
                    maximumFractionDigits: 2,
                  })}
                  tone={metricTone(selectedModel.bestOos.pnlTotal)}
                />
                <StatCard
                  label="Profit Factor"
                  value={formatNumber(selectedModel.bestOos.profitFactor, {
                    maximumFractionDigits: 3,
                  })}
                />
                <StatCard
                  label="Max Drawdown"
                  value={formatRatioPercent(selectedModel.bestOos.maxDd)}
                  tone={
                    typeof selectedModel.bestOos.maxDd === "number" &&
                    selectedModel.bestOos.maxDd > 0.2
                      ? "text-rose-600"
                      : "text-slate-900"
                  }
                />
                <StatCard
                  label="Hold Steps"
                  value={formatNumber(selectedModel.cfg.maxHoldSteps, {
                    maximumFractionDigits: 0,
                  })}
                />
              </div>

              <p className="mt-4 text-sm text-slate-500">
                Лучший OOS взят из сохраненной истории модели:
                {" "}
                {selectedModel.bestOos.days
                  ? `${selectedModel.bestOos.days}d`
                  : "days неизвестны"}
                {selectedModel.bestOos.ts ? `, ${selectedModel.bestOos.ts}` : ""}.
                {selectedModel.modelUuid ? ` UUID: ${selectedModel.modelUuid}.` : ""}
              </p>
            </div>

            <div className="rounded-3xl border border-slate-200 p-5">
              <div className="flex items-center justify-between gap-3">
                <h5 className="text-base font-semibold text-slate-900">Версии в ансамбле</h5>
                <span className="text-sm text-slate-500">
                  {selectedEnsemble?.versions.length ?? 0} шт.
                </span>
              </div>

              <div className="mt-4 space-y-6">
                {[
                  ["long", "Long", versionGroups.long],
                  ["short", "Short", versionGroups.short],
                  ["unknown", "Не определено", versionGroups.unknown],
                ].map(([groupKey, title, versions]) => {
                  const groupVersions = versions as XgbModelSummary[];
                  if (groupVersions.length === 0) {
                    return null;
                  }
                  return (
                    <div key={String(groupKey)}>
                      <div className="mb-3 flex items-center justify-between gap-3">
                        <h6 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">
                          {String(title)}
                        </h6>
                        <span className="text-xs text-slate-400">{groupVersions.length} шт.</span>
                      </div>
                      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                        {groupVersions.map((version) => {
                          const isSelected = version.version === selectedModel.version;
                          const isLaunchSelected =
                            (groupKey === "long" && version.version === selectedLongModel?.version) ||
                            (groupKey === "short" && version.version === selectedShortModel?.version);
                          const rank = versionRanks.get(version.version) ?? null;
                          const isStrongest = rank === 1;
                          return (
                            <button
                              key={version.version}
                              type="button"
                              onClick={() => {
                                setSelectedVersion(version.version);
                                if (groupKey === "long") {
                                  setSelectedLongVersion(version.version);
                                } else if (groupKey === "short") {
                                  setSelectedShortVersion(version.version);
                                }
                              }}
                              className={`rounded-2xl border p-4 text-left transition ${
                                isSelected
                                  ? "border-slate-900 bg-slate-900 text-white"
                                  : isStrongest
                                    ? "border-amber-300 bg-amber-50 text-slate-900 hover:border-amber-400"
                                    : isLaunchSelected
                                      ? "border-blue-300 bg-blue-50 text-slate-900 hover:border-blue-400"
                                      : "border-slate-200 bg-white text-slate-900 hover:border-slate-400"
                              }`}
                            >
                              <div className="flex items-start justify-between gap-3">
                                <div>
                                  <p className={`text-sm font-semibold ${isSelected ? "text-white" : "text-slate-900"}`}>
                                    {version.version}
                                  </p>
                                  <p className={`mt-1 text-xs ${isSelected ? "text-slate-200" : "text-slate-500"}`}>
                                    {version.modelId || "без id"}
                                  </p>
                                  <p className={`mt-1 text-xs ${isSelected ? "text-slate-300" : "text-slate-500"}`}>
                                    {describeXgbTask(version.task)}
                                  </p>
                                  {version.modelUuid ? (
                                    <p className={`mt-1 break-all text-xs ${isSelected ? "text-slate-300" : "text-slate-500"}`}>
                                      UUID: {version.modelUuid}
                                    </p>
                                  ) : null}
                                </div>
                                <div className="flex flex-col items-end gap-2">
                                  <div className="flex gap-2">
                                    {version.isCurrent ? (
                                      <span className={`rounded-full px-2 py-1 text-xs font-semibold ${isSelected ? "bg-white/15 text-white" : "bg-emerald-50 text-emerald-700"}`}>
                                        current
                                      </span>
                                    ) : null}
                                    {isLaunchSelected ? (
                                      <span className={`rounded-full px-2 py-1 text-xs font-semibold ${isSelected ? "bg-white/15 text-white" : "bg-blue-100 text-blue-700"}`}>
                                        для запуска
                                      </span>
                                    ) : null}
                                  </div>
                                  <button
                                    type="button"
                                    onClick={async (e) => {
                                      e.stopPropagation();
                                      if (window.confirm(`Точно удалить модель ${version.version}?`)) {
                                        await handleDeleteModel(symbol, ensemble, version.version);
                                      }
                                    }}
                                    className={`p-1 transition-colors ${
                                      isSelected
                                        ? "text-slate-400 hover:text-white"
                                        : "text-slate-400 hover:text-rose-600"
                                    }`}
                                    title="Удалить модель"
                                  >
                                    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                  </button>
                                </div>
                              </div>

                              {rank !== null ? (
                                <div className="mt-3">
                                  <span className={`rounded-full px-2 py-1 text-xs font-semibold ${isSelected ? (isStrongest ? "bg-amber-300/20 text-amber-100" : "bg-white/15 text-white") : isStrongest ? "bg-amber-100 text-amber-800" : "bg-slate-100 text-slate-700"}`}>
                                    rank: {rank}
                                  </span>
                                </div>
                              ) : null}

                              <div className={`mt-4 space-y-1 text-sm ${isSelected ? "text-slate-100" : "text-slate-600"}`}>
                                <p>Net PnL: {formatNumber(version.bestOos.pnlTotal, { maximumFractionDigits: 2 })}</p>
                                <p>Hold steps: {formatNumber(version.cfg.maxHoldSteps, { maximumFractionDigits: 0 })}</p>
                                <p>Profit factor: {formatNumber(version.bestOos.profitFactor, { maximumFractionDigits: 3 })}</p>
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="grid gap-6 xl:grid-cols-2">
              <div className="rounded-3xl border border-slate-200 p-5">
                <h5 className="text-base font-semibold text-slate-900">Параметры модели</h5>
                <div className="mt-4">
                  <MetaRow label="ID модели" value={selectedModel.modelId || "—"} />
                  <MetaRow label="UUID модели" value={selectedModel.modelUuid || "—"} />
                  <MetaRow label="Версия" value={selectedModel.version || "—"} />
                  <MetaRow label="Task" value={selectedModel.task || "—"} />
                  <MetaRow label="Long / Short" value={summarizeDirection(selectedModel)} />
                  <MetaRow
                    label="Horizon steps"
                    value={formatNumber(selectedModel.cfg.horizonSteps, {
                      maximumFractionDigits: 0,
                    })}
                  />
                  <MetaRow
                    label="Hold steps"
                    value={formatNumber(selectedModel.cfg.maxHoldSteps, {
                      maximumFractionDigits: 0,
                    })}
                  />
                  <MetaRow
                    label="Threshold"
                    value={formatNumber(selectedModel.cfg.threshold, {
                      maximumFractionDigits: 4,
                    })}
                  />
                  <MetaRow
                    label="p_enter"
                    value={formatNumber(selectedModel.cfg.pEnterThreshold, {
                      maximumFractionDigits: 4,
                    })}
                  />
                  <MetaRow
                    label="Min profit"
                    value={formatPercent(selectedModel.cfg.minProfit)}
                  />
                  <MetaRow
                    label="Fee bps"
                    value={formatNumber(selectedModel.cfg.feeBps, {
                      maximumFractionDigits: 2,
                    })}
                  />
                </div>
              </div>

              <div className="rounded-3xl border border-slate-200 p-5">
                <h5 className="text-base font-semibold text-slate-900">Обучение и OOS</h5>
                <div className="mt-4">
                  <MetaRow
                    label="Val accuracy"
                    value={formatRatioPercent(selectedModel.trainMetrics.valAcc)}
                  />
                  <MetaRow
                    label="F1 class=1"
                    value={formatNumber(selectedModel.trainMetrics.f1Val?.[1] ?? null, {
                      maximumFractionDigits: 4,
                    })}
                  />
                  <MetaRow
                    label="Precision class=1"
                    value={formatNumber(
                      selectedModel.trainMetrics.precisionVal?.[1] ?? null,
                      {
                        maximumFractionDigits: 4,
                      },
                    )}
                  />
                  <MetaRow
                    label="Recall class=1"
                    value={formatNumber(selectedModel.trainMetrics.recallVal?.[1] ?? null, {
                      maximumFractionDigits: 4,
                    })}
                  />
                  <MetaRow
                    label="Trades"
                    value={formatNumber(selectedModel.bestOos.tradesCount, {
                      maximumFractionDigits: 0,
                    })}
                  />
                  <MetaRow
                    label="Winrate"
                    value={formatRatioPercent(selectedModel.bestOos.winrate)}
                  />
                  <MetaRow
                    label="ROI"
                    value={formatPercent(selectedModel.bestOos.roiPct)}
                  />
                  <MetaRow
                    label="Avg trade PnL"
                    value={formatNumber(selectedModel.bestOos.avgTradePnl, {
                      maximumFractionDigits: 2,
                    })}
                  />
                  <MetaRow
                    label="Avg bars held"
                    value={formatNumber(selectedModel.bestOos.avgBarsHeld, {
                      maximumFractionDigits: 1,
                    })}
                  />
                  <MetaRow
                    label="Equity end"
                    value={formatNumber(selectedModel.bestOos.equityEnd, {
                      maximumFractionDigits: 2,
                    })}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
