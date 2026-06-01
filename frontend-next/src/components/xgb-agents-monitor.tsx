"use client";

import { useEffect, useMemo, useState } from "react";

type RawAgent = {
  session_id?: string | null;
  symbol?: string | null;
  is_xgb?: boolean | null;
  status?: string | null;
  current_price?: number | null;
  trades_count?: number | null;
  last_prediction?: string | null;
  model_path?: string | null;
  model_paths?: string[] | null;
  model_roles?: Record<string, string> | null;
  xgb_side_models?: RawSideModel[] | null;
  xgb_model_uuid?: string | null;
  execution_mode?: string | null;
  direction?: string | null;
  trade_mode?: string | null;
  ignore_trend_filter?: boolean | null;
  exit_mode?: string | null;
  risk_management_type?: string | null;
  account_pct?: number | null;
  leverage?: number | null;
  max_hold_steps?: number | null;
  ttl_seconds?: number | null;
  xgb_entry_threshold?: number | null;
  xgb_entry_threshold_default?: number | null;
  xgb_threshold_override_active?: boolean | null;
  xgb_stop_loss_pct?: number | null;
  xgb_stop_loss_pct_default?: number | null;
  xgb_stop_loss_override_active?: boolean | null;
  xgb_take_profit_pct?: number | null;
  xgb_take_profit_pct_default?: number | null;
  xgb_take_profit_override_active?: boolean | null;
  xgb_task?: string | null;
  bybit_account_id?: string | null;
  bybit_account_label?: string | null;
  bybit_api_key_hint?: string | null;
  position?: { type?: string | null } | null;
  amount_usdt?: number | null;
  position_entry_ts_ms?: number | null;
  hold_seconds_remaining?: number | null;
  xgb_signal_exit_enabled?: boolean | null;
  xgb_signal_exit_start_step?: number | null;
  xgb_signal_exit_window?: number | null;
  xgb_signal_exit_history_size?: number | null;
  xgb_signal_exit_avg_signal?: number | null;
  xgb_signal_exit_avg_threshold?: number | null;
  xgb_signal_exit_last_signal?: number | null;
  xgb_signal_exit_last_threshold?: number | null;
  xgb_signal_exit_ready?: boolean | null;
  xgb_signal_exit_passes_threshold?: boolean | null;
  xgb_postexit_guard_enabled?: boolean | null;
  xgb_postexit_cooldown_candles?: number | null;
  xgb_postexit_boost_candles?: number | null;
  xgb_postexit_threshold_boost_pct?: number | null;
  xgb_postexit_phase?: string | null;
  xgb_postexit_candles_since_exit?: number | null;
  xgb_postexit_candles_left?: number | null;
  xgb_postexit_hours_left?: number | null;
  xgb_last_exit_ts_ms?: number | null;
  xgb_last_exit_reason?: string | null;
  is_client_master?: boolean | null;
};

type RawSideModel = {
  role?: string | null;
  model_path?: string | null;
  model_uuid?: string | null;
  task?: string | null;
  max_hold_steps?: number | null;
  entry_threshold?: number | null;
  entry_threshold_default?: number | null;
  entry_threshold_override_active?: boolean | null;
  stop_loss_pct?: number | null;
  stop_loss_pct_default?: number | null;
  stop_loss_override_active?: boolean | null;
  take_profit_pct?: number | null;
  take_profit_pct_default?: number | null;
  take_profit_override_active?: boolean | null;
  signal_exit_enabled?: boolean | null;
  signal_exit_start_step?: number | null;
  signal_exit_threshold?: number | null;
  signal_exit_window?: number | null;
  opposite_role?: string | null;
  opposite_signal_max?: number | null;
  last_prediction_signal?: number | null;
  last_prediction_threshold?: number | null;
  last_prediction_action?: string | null;
  last_prediction_at?: string | null;
};

type StatusResponse = {
  success?: boolean;
  active_agents?: RawAgent[];
  xgb_entry_attempts_history?: XgbEntryAttempt[];
  error?: string;
};

type XgbEntryAttempt = {
  ts_ms?: number | null;
  timestamp?: string | null;
  session_id?: string | null;
  symbol?: string | null;
  long_signal?: number | null;
  short_signal?: number | null;
  state?: string | null;
  bybit_account_id?: string | null;
  bybit_account_label?: string | null;
  bybit_api_key_hint?: string | null;
};

type XgbAgent = {
  sessionId: string;
  symbol: string;
  status: string;
  currentPrice: number | null;
  tradesCount: number | null;
  lastPrediction: string | null;
  modelPath: string | null;
  modelPaths: string[];
  modelRoles: Record<string, string>;
  sideModels: XgbSideModel[];
  modelUuid: string | null;
  executionMode: string | null;
  direction: string | null;
  tradeMode: string | null;
  ignoreTrendFilter: boolean | null;
  exitMode: string | null;
  riskManagementType: string | null;
  accountPct: number | null;
  leverage: number;
  maxHoldSteps: number | null;
  ttlSeconds: number | null;
  xgbEntryThreshold: number | null;
  xgbEntryThresholdDefault: number | null;
  xgbThresholdOverrideActive: boolean;
  xgbStopLossPct: number | null;
  xgbStopLossPctDefault: number | null;
  xgbStopLossOverrideActive: boolean;
  xgbTakeProfitPct: number | null;
  xgbTakeProfitPctDefault: number | null;
  xgbTakeProfitOverrideActive: boolean;
  xgbTask: string | null;
  bybitAccountId: string | null;
  bybitAccountLabel: string | null;
  bybitApiKeyHint: string | null;
  positionType: string | null;
  amountUsdt: number | null;
  positionEntryTsMs: number | null;
  holdSecondsRemaining: number | null;
  xgbSignalExitEnabled: boolean;
  xgbSignalExitStartStep: number | null;
  xgbSignalExitWindow: number | null;
  xgbSignalExitHistorySize: number | null;
  xgbSignalExitAvgSignal: number | null;
  xgbSignalExitAvgThreshold: number | null;
  xgbSignalExitLastSignal: number | null;
  xgbSignalExitLastThreshold: number | null;
  xgbSignalExitReady: boolean;
  xgbSignalExitPassesThreshold: boolean | null;
  xgbPostExitGuardEnabled: boolean;
  xgbPostExitCooldownCandles: number | null;
  xgbPostExitBoostCandles: number | null;
  xgbPostExitThresholdBoostPct: number | null;
  xgbPostExitPhase: string | null;
  xgbPostExitCandlesSinceExit: number | null;
  xgbPostExitCandlesLeft: number | null;
  xgbPostExitHoursLeft: number | null;
  xgbLastExitTsMs: number | null;
  xgbLastExitReason: string | null;
  isClientMaster: boolean;
};

type XgbSideModel = {
  role: "long" | "short";
  modelPath: string | null;
  modelUuid: string | null;
  task: string | null;
  maxHoldSteps: number | null;
  entryThreshold: number | null;
  entryThresholdDefault: number | null;
  entryThresholdOverrideActive: boolean;
  stopLossPct: number | null;
  stopLossPctDefault: number | null;
  stopLossOverrideActive: boolean;
  takeProfitPct: number | null;
  takeProfitPctDefault: number | null;
  takeProfitOverrideActive: boolean;
  signalExitEnabled: boolean | null;
  signalExitStartStep: number | null;
  signalExitThreshold: number | null;
  signalExitWindow: number | null;
  oppositeRole: "long" | "short";
  oppositeSignalMax: number | null;
  lastPredictionSignal: number | null;
  lastPredictionThreshold: number | null;
  lastPredictionAction: string | null;
  lastPredictionAt: string | null;
};

type ExitExecutionMode = "market" | "limit";

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return new Intl.NumberFormat("ru-RU", {
    maximumFractionDigits: digits,
  }).format(value);
}

function formatDuration(seconds: number | null | undefined): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) {
    return "—";
  }
  if (seconds === 0) {
    return "0с";
  }
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  if (hours > 0) {
    return `${hours}ч ${minutes}м`;
  }
  if (minutes > 0) {
    return `${minutes}м ${secs}с`;
  }
  return `${secs}с`;
}

function compactModelName(modelPath: string | null): string {
  if (!modelPath) {
    return "—";
  }
  const parts = modelPath.replace(/\\/g, "/").split("/").filter(Boolean);
  const ensemble = parts[parts.length - 3];
  const version = parts[parts.length - 2];
  return ensemble && version ? `${ensemble} ${version}` : parts.slice(-2).join(" ");
}

function mapRawSideModel(item: RawSideModel): XgbSideModel | null {
  const role = String(item.role ?? "").toLowerCase();
  if (role !== "long" && role !== "short") {
    return null;
  }
  return {
    role,
    modelPath: item.model_path ? String(item.model_path) : null,
    modelUuid: item.model_uuid ? String(item.model_uuid) : null,
    task: item.task ? String(item.task) : null,
    maxHoldSteps: typeof item.max_hold_steps === "number" ? item.max_hold_steps : null,
    entryThreshold: typeof item.entry_threshold === "number" ? item.entry_threshold : null,
    entryThresholdDefault:
      typeof item.entry_threshold_default === "number" ? item.entry_threshold_default : null,
    entryThresholdOverrideActive: Boolean(item.entry_threshold_override_active),
    stopLossPct: typeof item.stop_loss_pct === "number" ? item.stop_loss_pct : null,
    stopLossPctDefault:
      typeof item.stop_loss_pct_default === "number" ? item.stop_loss_pct_default : null,
    stopLossOverrideActive: Boolean(item.stop_loss_override_active),
    takeProfitPct: typeof item.take_profit_pct === "number" ? item.take_profit_pct : null,
    takeProfitPctDefault:
      typeof item.take_profit_pct_default === "number" ? item.take_profit_pct_default : null,
    takeProfitOverrideActive: Boolean(item.take_profit_override_active),
    signalExitEnabled:
      typeof item.signal_exit_enabled === "boolean" ? item.signal_exit_enabled : null,
    signalExitStartStep:
      typeof item.signal_exit_start_step === "number" ? item.signal_exit_start_step : null,
    signalExitThreshold:
      typeof item.signal_exit_threshold === "number" ? item.signal_exit_threshold : null,
    signalExitWindow:
      typeof item.signal_exit_window === "number" ? item.signal_exit_window : null,
    oppositeRole: String(item.opposite_role ?? (role === "long" ? "short" : "long")) === "long"
      ? "long"
      : "short",
    oppositeSignalMax:
      typeof item.opposite_signal_max === "number" ? item.opposite_signal_max : null,
    lastPredictionSignal:
      typeof item.last_prediction_signal === "number" ? item.last_prediction_signal : null,
    lastPredictionThreshold:
      typeof item.last_prediction_threshold === "number" ? item.last_prediction_threshold : null,
    lastPredictionAction: item.last_prediction_action ? String(item.last_prediction_action) : null,
    lastPredictionAt: item.last_prediction_at ? String(item.last_prediction_at) : null,
  };
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-slate-100 py-2 last:border-b-0">
      <span className="text-sm text-slate-500">{label}</span>
      <span className="text-right text-sm font-medium text-slate-900">{value}</span>
    </div>
  );
}

function formatMoscowDateTimeFromMs(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return "—";
  }
  return new Intl.DateTimeFormat("ru-RU", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZone: "Europe/Moscow",
  }).format(new Date(value));
}

function isRiskPlacementEnabled(riskManagementType: string | null): boolean {
  return riskManagementType === "exchange_orders" || riskManagementType === "both";
}

function sideDraftKey(sessionId: string, role: "long" | "short"): string {
  return `${sessionId}:${role}`;
}

function mapRawAgent(item: RawAgent): XgbAgent {
  return {
    sessionId: String(item.session_id ?? ""),
    symbol: String(item.symbol ?? ""),
    status: String(item.status ?? "—"),
    currentPrice: typeof item.current_price === "number" ? item.current_price : null,
    tradesCount: typeof item.trades_count === "number" ? item.trades_count : null,
    lastPrediction: item.last_prediction ? String(item.last_prediction) : null,
    modelPath: item.model_path ? String(item.model_path) : null,
    modelPaths: Array.isArray(item.model_paths)
      ? item.model_paths.map((value) => String(value)).filter(Boolean)
      : [],
    modelRoles:
      item.model_roles && typeof item.model_roles === "object"
        ? Object.fromEntries(
            Object.entries(item.model_roles).map(([path, role]) => [String(path), String(role)]),
          )
        : {},
    sideModels: Array.isArray(item.xgb_side_models)
      ? item.xgb_side_models
          .map(mapRawSideModel)
          .filter((value): value is XgbSideModel => value !== null)
      : [],
    modelUuid: item.xgb_model_uuid ? String(item.xgb_model_uuid) : null,
    executionMode: item.execution_mode ? String(item.execution_mode) : null,
    direction: item.direction ? String(item.direction) : null,
    tradeMode: item.trade_mode ? String(item.trade_mode) : null,
    ignoreTrendFilter: typeof item.ignore_trend_filter === "boolean" ? item.ignore_trend_filter : null,
    exitMode: item.exit_mode ? String(item.exit_mode) : null,
    riskManagementType: item.risk_management_type ? String(item.risk_management_type) : null,
    accountPct: typeof item.account_pct === "number" ? item.account_pct : null,
    leverage: typeof item.leverage === "number" ? item.leverage : 1,
    maxHoldSteps: typeof item.max_hold_steps === "number" ? item.max_hold_steps : null,
    ttlSeconds: typeof item.ttl_seconds === "number" ? item.ttl_seconds : null,
    xgbEntryThreshold:
      typeof item.xgb_entry_threshold === "number" ? item.xgb_entry_threshold : null,
    xgbEntryThresholdDefault:
      typeof item.xgb_entry_threshold_default === "number"
        ? item.xgb_entry_threshold_default
        : null,
    xgbThresholdOverrideActive: Boolean(item.xgb_threshold_override_active),
    xgbStopLossPct:
      typeof item.xgb_stop_loss_pct === "number" ? item.xgb_stop_loss_pct : null,
    xgbStopLossPctDefault:
      typeof item.xgb_stop_loss_pct_default === "number"
        ? item.xgb_stop_loss_pct_default
        : null,
    xgbStopLossOverrideActive: Boolean(item.xgb_stop_loss_override_active),
    xgbTakeProfitPct:
      typeof item.xgb_take_profit_pct === "number" ? item.xgb_take_profit_pct : null,
    xgbTakeProfitPctDefault:
      typeof item.xgb_take_profit_pct_default === "number"
        ? item.xgb_take_profit_pct_default
        : null,
    xgbTakeProfitOverrideActive: Boolean(item.xgb_take_profit_override_active),
    xgbTask: item.xgb_task ? String(item.xgb_task) : null,
    bybitAccountId: item.bybit_account_id ? String(item.bybit_account_id) : null,
    bybitAccountLabel: item.bybit_account_label ? String(item.bybit_account_label) : null,
    bybitApiKeyHint: item.bybit_api_key_hint ? String(item.bybit_api_key_hint) : null,
    positionType:
      item.position && typeof item.position === "object" && item.position.type
        ? String(item.position.type)
        : null,
    amountUsdt: typeof item.amount_usdt === "number" ? item.amount_usdt : null,
    positionEntryTsMs:
      typeof item.position_entry_ts_ms === "number" ? item.position_entry_ts_ms : null,
    holdSecondsRemaining:
      typeof item.hold_seconds_remaining === "number" ? item.hold_seconds_remaining : null,
    xgbSignalExitEnabled: Boolean(item.xgb_signal_exit_enabled),
    xgbSignalExitStartStep:
      typeof item.xgb_signal_exit_start_step === "number"
        ? item.xgb_signal_exit_start_step
        : null,
    xgbSignalExitWindow:
      typeof item.xgb_signal_exit_window === "number" ? item.xgb_signal_exit_window : null,
    xgbSignalExitHistorySize:
      typeof item.xgb_signal_exit_history_size === "number"
        ? item.xgb_signal_exit_history_size
        : null,
    xgbSignalExitAvgSignal:
      typeof item.xgb_signal_exit_avg_signal === "number"
        ? item.xgb_signal_exit_avg_signal
        : null,
    xgbSignalExitAvgThreshold:
      typeof item.xgb_signal_exit_avg_threshold === "number"
        ? item.xgb_signal_exit_avg_threshold
        : null,
    xgbSignalExitLastSignal:
      typeof item.xgb_signal_exit_last_signal === "number"
        ? item.xgb_signal_exit_last_signal
        : null,
    xgbSignalExitLastThreshold:
      typeof item.xgb_signal_exit_last_threshold === "number"
        ? item.xgb_signal_exit_last_threshold
        : null,
    xgbSignalExitReady: Boolean(item.xgb_signal_exit_ready),
    xgbSignalExitPassesThreshold:
      typeof item.xgb_signal_exit_passes_threshold === "boolean"
        ? item.xgb_signal_exit_passes_threshold
        : null,
    xgbPostExitGuardEnabled: Boolean(item.xgb_postexit_guard_enabled),
    xgbPostExitCooldownCandles:
      typeof item.xgb_postexit_cooldown_candles === "number"
        ? item.xgb_postexit_cooldown_candles
        : null,
    xgbPostExitBoostCandles:
      typeof item.xgb_postexit_boost_candles === "number" ? item.xgb_postexit_boost_candles : null,
    xgbPostExitThresholdBoostPct:
      typeof item.xgb_postexit_threshold_boost_pct === "number"
        ? item.xgb_postexit_threshold_boost_pct
        : null,
    xgbPostExitPhase: item.xgb_postexit_phase ? String(item.xgb_postexit_phase) : null,
    xgbPostExitCandlesSinceExit:
      typeof item.xgb_postexit_candles_since_exit === "number"
        ? item.xgb_postexit_candles_since_exit
        : null,
    xgbPostExitCandlesLeft:
      typeof item.xgb_postexit_candles_left === "number" ? item.xgb_postexit_candles_left : null,
    xgbPostExitHoursLeft:
      typeof item.xgb_postexit_hours_left === "number" ? item.xgb_postexit_hours_left : null,
    xgbLastExitTsMs: typeof item.xgb_last_exit_ts_ms === "number" ? item.xgb_last_exit_ts_ms : null,
    xgbLastExitReason: item.xgb_last_exit_reason ? String(item.xgb_last_exit_reason) : null,
    isClientMaster: Boolean(item.is_client_master),
  };
}

function formatSignalExitState(agent: XgbAgent): string {
  if (!agent.xgbSignalExitEnabled) {
    return "выключен";
  }
  if (!agent.xgbSignalExitReady) {
    const size = agent.xgbSignalExitHistorySize ?? 0;
    const window = agent.xgbSignalExitWindow ?? 20;
    return `ожидание окна ${size}/${window}`;
  }
  if (agent.xgbSignalExitPassesThreshold === true) {
    return "держим, окно выше threshold";
  }
  if (agent.xgbSignalExitPassesThreshold === false) {
    return "ниже threshold, готов к early exit";
  }
  return "—";
}

function formatPostExitGuardState(agent: XgbAgent): string {
  if (!agent.xgbPostExitGuardEnabled) {
    return "выключена";
  }
  const phase = (agent.xgbPostExitPhase || "").toLowerCase();
  if (phase === "cooldown") {
    const left = agent.xgbPostExitCandlesLeft ?? null;
    const hrs = agent.xgbPostExitHoursLeft ?? null;
    const reason = agent.xgbLastExitReason ? ` (${agent.xgbLastExitReason})` : "";
    if (typeof left === "number" && Number.isFinite(left)) {
      return `cooldown: осталось ${left} свечей (~${formatNumber(hrs, 2)}ч)${reason}`;
    }
    return `cooldown${reason}`;
  }
  if (phase === "boost") {
    const pct = agent.xgbPostExitThresholdBoostPct;
    const reason = agent.xgbLastExitReason ? ` (${agent.xgbLastExitReason})` : "";
    return `boost: +${formatNumber(pct, 2)}% к threshold${reason}`;
  }
  if (phase === "expired") {
    return "активна (окно прошло)";
  }
  return "активна";
}

function getSideModel(agent: XgbAgent, role: "long" | "short"): XgbSideModel {
  const fromApi = agent.sideModels.find((item) => item.role === role);
  if (fromApi) {
    return fromApi;
  }
  const modelPath =
    agent.modelPaths.find((path) => agent.modelRoles[path] === role) ??
    (agent.direction === role ? agent.modelPath : null);
  return {
    role,
    modelPath,
    modelUuid: role === agent.direction ? agent.modelUuid : null,
    task: role === agent.direction ? agent.xgbTask : null,
    maxHoldSteps: null,
    entryThreshold: role === agent.direction ? agent.xgbEntryThreshold : null,
    entryThresholdDefault: role === agent.direction ? agent.xgbEntryThresholdDefault : null,
    entryThresholdOverrideActive: role === agent.direction ? agent.xgbThresholdOverrideActive : false,
    stopLossPct: role === agent.direction ? agent.xgbStopLossPct : null,
    stopLossPctDefault: role === agent.direction ? agent.xgbStopLossPctDefault : null,
    stopLossOverrideActive: role === agent.direction ? agent.xgbStopLossOverrideActive : false,
    takeProfitPct: role === agent.direction ? agent.xgbTakeProfitPct : null,
    takeProfitPctDefault: role === agent.direction ? agent.xgbTakeProfitPctDefault : null,
    takeProfitOverrideActive: role === agent.direction ? agent.xgbTakeProfitOverrideActive : false,
    signalExitEnabled: role === agent.direction ? agent.xgbSignalExitEnabled : null,
    signalExitStartStep: role === agent.direction ? agent.xgbSignalExitStartStep : null,
    signalExitThreshold: role === agent.direction ? agent.xgbSignalExitAvgThreshold : null,
    signalExitWindow: role === agent.direction ? agent.xgbSignalExitWindow : null,
    oppositeRole: role === "long" ? "short" : "long",
    oppositeSignalMax: null,
    lastPredictionSignal: null,
    lastPredictionThreshold: null,
    lastPredictionAction: null,
    lastPredictionAt: null,
  };
}

function SideModelBlock({ agent, role }: { agent: XgbAgent; role: "long" | "short" }) {
  const side = getSideModel(agent, role);
  const title = role === "long" ? "Long модель" : "Short модель";
  const tone =
    role === "long"
      ? "border-emerald-200 bg-emerald-50"
      : "border-rose-200 bg-rose-50";
  return (
    <div className={`rounded-2xl border p-4 ${tone}`}>
      <div className="mb-3 flex items-center justify-between gap-3">
        <p className="text-sm font-semibold text-slate-900">{title}</p>
        <span className="rounded-full bg-white/70 px-2 py-1 text-xs font-semibold uppercase text-slate-600">
          {role}
        </span>
      </div>
      <MetaRow label="Model" value={compactModelName(side.modelPath)} />
      <MetaRow label="UUID" value={side.modelUuid || "—"} />
      <MetaRow label="Task" value={side.task || "—"} />
      <MetaRow
        label="Hold steps"
        value={
          agent.positionType?.toLowerCase() === role && agent.holdSecondsRemaining != null
            ? `${formatNumber(side.maxHoldSteps, 0)} (осталось: ${formatDuration(agent.holdSecondsRemaining)})`
            : formatNumber(side.maxHoldSteps, 0)
        }
      />
      <MetaRow
        label="Last signal / runtime threshold"
        value={`${formatNumber(side.lastPredictionSignal, 4)} / ${formatNumber(side.entryThreshold, 4)}`}
      />
      <MetaRow
        label="Last action / pred threshold"
        value={`${side.lastPredictionAction || "—"} / ${formatNumber(side.lastPredictionThreshold, 4)}`}
      />
      <MetaRow
        label="Weak signal"
        value={
          side.signalExitEnabled === null
            ? "—"
            : side.signalExitEnabled
              ? "включен"
              : "выключен"
        }
      />
      <MetaRow
        label="Start / Window"
        value={
          side.signalExitEnabled
            ? `${formatNumber(side.signalExitStartStep, 0)} / ${formatNumber(side.signalExitWindow, 0)}`
            : "—"
        }
      />
      <MetaRow
        label="Weak threshold"
        value={side.signalExitEnabled ? formatNumber(side.signalExitThreshold, 4) : "—"}
      />
      <MetaRow
        label={`${side.oppositeRole} signal max`}
        value={formatNumber(side.oppositeSignalMax, 4)}
      />
    </div>
  );
}

export function XgbAgentsMonitor() {
  const [agents, setAgents] = useState<XgbAgent[]>([]);
  const [entryAttempts, setEntryAttempts] = useState<XgbEntryAttempt[]>([]);
  const [selectedEntryApi, setSelectedEntryApi] = useState<string>("__all__");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [draftThresholds, setDraftThresholds] = useState<Record<string, number>>({});
  const [draftSignalExitThresholds, setDraftSignalExitThresholds] = useState<Record<string, number>>({});
  const [draftOppositeSignalThresholds, setDraftOppositeSignalThresholds] = useState<Record<string, number>>({});
  const [draftStopLosses, setDraftStopLosses] = useState<Record<string, number>>({});
  const [draftTakeProfits, setDraftTakeProfits] = useState<Record<string, number>>({});
  const [draftLeverages, setDraftLeverages] = useState<Record<string, number>>({});
  const [draftHoldExtensions, setDraftHoldExtensions] = useState<Record<string, number>>({});
  const [draftExitModes, setDraftExitModes] = useState<Record<string, ExitExecutionMode>>({});
  const [savingSession, setSavingSession] = useState<string | null>(null);
  const [exitingSession, setExitingSession] = useState<string | null>(null);
  const [stoppingSession, setStoppingSession] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function load() {
      try {
        const response = await fetch("/api/trading/status_all", {
          method: "GET",
          cache: "no-store",
        });
        if (!response.ok) {
          throw new Error(`Backend returned ${response.status}`);
        }

        const data = (await response.json()) as StatusResponse;
        if (!data.success) {
          throw new Error(data.error || "Failed to load active agents");
        }

        const mapped = (data.active_agents ?? [])
          .filter((item) => Boolean(item.is_xgb))
          .map(mapRawAgent);

        if (!cancelled) {
          setAgents(mapped);
          setEntryAttempts((data.xgb_entry_attempts_history ?? []).slice(0, 300));
          setDraftThresholds((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              for (const role of ["long", "short"] as const) {
                const side = getSideModel(agent, role);
                const key = sideDraftKey(agent.sessionId, role);
                if (typeof next[key] !== "number") {
                  next[key] = side.entryThreshold ?? side.entryThresholdDefault ?? 0.5;
                }
              }
            }
            return next;
          });
          setDraftSignalExitThresholds((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              if (typeof next[agent.sessionId] !== "number") {
                next[agent.sessionId] = agent.xgbSignalExitAvgThreshold ?? 0.5;
              }
            }
            return next;
          });
          setDraftOppositeSignalThresholds((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              for (const role of ["long", "short"] as const) {
                const side = getSideModel(agent, role);
                const key = sideDraftKey(agent.sessionId, role);
                if (typeof next[key] !== "number") {
                  next[key] = side.oppositeSignalMax ?? 0.4;
                }
              }
            }
            return next;
          });
          setDraftStopLosses((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              for (const role of ["long", "short"] as const) {
                const side = getSideModel(agent, role);
                const key = sideDraftKey(agent.sessionId, role);
                if (typeof next[key] !== "number") {
                  next[key] = side.stopLossPct ?? side.stopLossPctDefault ?? 1.0;
                }
              }
            }
            return next;
          });
          setDraftTakeProfits((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              for (const role of ["long", "short"] as const) {
                const side = getSideModel(agent, role);
                const key = sideDraftKey(agent.sessionId, role);
                if (typeof next[key] !== "number") {
                  next[key] = side.takeProfitPct ?? side.takeProfitPctDefault ?? 3.0;
                }
              }
            }
            return next;
          });
          setDraftLeverages((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              if (typeof next[agent.sessionId] !== "number") {
                next[agent.sessionId] = agent.leverage ?? 1;
              }
            }
            return next;
          });
          setDraftHoldExtensions((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              if (typeof next[agent.sessionId] !== "number") {
                next[agent.sessionId] = 5;
              }
            }
            return next;
          });
          setDraftExitModes((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              if (!next[agent.sessionId]) {
                next[agent.sessionId] =
                  agent.executionMode === "limit_post_only" ? "limit" : "market";
              }
            }
            return next;
          });
          setError(null);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(
            loadError instanceof Error ? loadError.message : "Failed to load XGB agents",
          );
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

    void load();

    return () => {
      cancelled = true;
      if (timer) {
        clearTimeout(timer);
      }
    };
  }, []);

  const agentsBySession = useMemo(() => {
    const map = new Map<string, XgbAgent>();
    for (const agent of agents) {
      map.set(agent.sessionId, agent);
    }
    return map;
  }, [agents]);

  const entryApiOptions = useMemo(() => {
    const seen = new Map<string, string>();
    for (const agent of agents) {
      if (!agent.bybitAccountId) continue;
      const label = `${agent.bybitAccountLabel || "Account"} (id: ${agent.bybitAccountId}${
        agent.bybitApiKeyHint ? `, key: ${agent.bybitApiKeyHint}` : ""
      })`;
      seen.set(agent.bybitAccountId, label);
    }
    for (const row of entryAttempts) {
      const idRaw = row.bybit_account_id ? String(row.bybit_account_id).trim() : "";
      if (!idRaw || seen.has(idRaw)) continue;
      seen.set(idRaw, `Account (id: ${idRaw})`);
    }
    return Array.from(seen.entries())
      .map(([id, label]) => ({ id, label }))
      .sort((a, b) => a.label.localeCompare(b.label));
  }, [agents, entryAttempts]);

  const visibleEntryAttempts = useMemo(() => {
    if (selectedEntryApi === "__all__") return entryAttempts;
    return entryAttempts.filter((row) => {
      const direct = row.bybit_account_id ? String(row.bybit_account_id).trim() : "";
      if (direct) return direct === selectedEntryApi;
      const sid = row.session_id ? String(row.session_id) : "";
      const agent = sid ? agentsBySession.get(sid) : undefined;
      return agent?.bybitAccountId === selectedEntryApi;
    });
  }, [agentsBySession, entryAttempts, selectedEntryApi]);

  const summary = useMemo(
    () => ({
      total: agents.length,
      longCount: agents.filter((item) => item.direction === "long").length,
      shortCount: agents.filter((item) => item.direction === "short").length,
    }),
    [agents],
  );

  if (isLoading) {
    return (
      <div className="rounded-3xl border border-slate-200 bg-slate-50 p-5 text-sm text-slate-500">
        Загружаю активных XGB агентов...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-3xl border border-rose-200 bg-rose-50 p-5 text-sm text-rose-700">
        Ошибка мониторинга: {error}
      </div>
    );
  }

  async function updateThreshold(agent: XgbAgent, role: "long" | "short", reset = false) {
    try {
      const sessionId = agent.sessionId;
      const key = sideDraftKey(sessionId, role);
      setSavingSession(sessionId);
      setActionMessage(null);
      const response = await fetch("/api/trading/xgb_threshold", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(
          reset
            ? { session_id: sessionId, side: role, reset: true }
            : { session_id: sessionId, side: role, threshold: draftThresholds[key] },
        ),
      });
      const data = (await response.json()) as {
        success?: boolean;
        error?: string;
        xgb_entry_threshold?: number | null;
      };
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Failed to update XGB threshold");
      }
      if (typeof data.xgb_entry_threshold === "number") {
        setDraftThresholds((current) => ({
          ...current,
          [key]: data.xgb_entry_threshold as number,
        }));
      }
      setActionMessage(
        reset
          ? `Threshold ${role} для ${agent.symbol} сброшен к значению модели.`
          : `Threshold ${role} для ${agent.symbol} обновлен.`,
      );
      setTimeout(() => {
        void (async () => {
          const responseReload = await fetch("/api/trading/status_all", {
            method: "GET",
            cache: "no-store",
          });
          if (!responseReload.ok) {
            return;
          }
          const dataReload = (await responseReload.json()) as StatusResponse;
          if (!dataReload.success) {
            return;
          }
          const mapped = (dataReload.active_agents ?? [])
            .filter((item) => Boolean(item.is_xgb))
            .map(mapRawAgent);
          setAgents(mapped);
          setDraftStopLosses((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              if (typeof next[agent.sessionId] !== "number") {
                next[agent.sessionId] = agent.xgbStopLossPct ?? agent.xgbStopLossPctDefault ?? 1.0;
              }
            }
            return next;
          });
          setDraftTakeProfits((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              if (typeof next[agent.sessionId] !== "number") {
                next[agent.sessionId] = agent.xgbTakeProfitPct ?? agent.xgbTakeProfitPctDefault ?? 3.0;
              }
            }
            return next;
          });
          setDraftExitModes((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              if (!next[agent.sessionId]) {
                next[agent.sessionId] =
                  agent.executionMode === "limit_post_only" ? "limit" : "market";
              }
            }
            return next;
          });
        })();
      }, 250);
    } catch (saveError) {
      setActionMessage(
        saveError instanceof Error ? saveError.message : "Failed to update XGB threshold",
      );
    } finally {
      setSavingSession(null);
    }
  }

  async function updateStopLoss(agent: XgbAgent, role: "long" | "short", reset = false) {
    try {
      const sessionId = agent.sessionId;
      const key = sideDraftKey(sessionId, role);
      setSavingSession(sessionId);
      setActionMessage(null);
      const response = await fetch("/api/trading/xgb_stop_loss", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(
          reset
            ? { session_id: sessionId, side: role, reset: true }
            : { session_id: sessionId, side: role, stop_loss_pct: draftStopLosses[key] },
        ),
      });
      const data = (await response.json()) as {
        success?: boolean;
        error?: string;
        xgb_stop_loss_pct?: number | null;
      };
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Failed to update XGB stop-loss");
      }
      if (typeof data.xgb_stop_loss_pct === "number") {
        setDraftStopLosses((current) => ({
          ...current,
          [key]: data.xgb_stop_loss_pct as number,
        }));
      }
      setActionMessage(
        reset
          ? `Stop-loss ${role} для ${agent.symbol} сброшен к значению модели.`
          : `Stop-loss ${role} для ${agent.symbol} обновлен.`,
      );
      setTimeout(() => {
        void (async () => {
          const responseReload = await fetch("/api/trading/status_all", {
            method: "GET",
            cache: "no-store",
          });
          if (!responseReload.ok) {
            return;
          }
          const dataReload = (await responseReload.json()) as StatusResponse;
          if (!dataReload.success) {
            return;
          }
          const mapped = (dataReload.active_agents ?? [])
            .filter((item) => Boolean(item.is_xgb))
            .map(mapRawAgent);
          setAgents(mapped);
          setDraftStopLosses((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              if (typeof next[agent.sessionId] !== "number") {
                next[agent.sessionId] = agent.xgbStopLossPct ?? agent.xgbStopLossPctDefault ?? 1.0;
              }
            }
            return next;
          });
          setDraftTakeProfits((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              if (typeof next[agent.sessionId] !== "number") {
                next[agent.sessionId] = agent.xgbTakeProfitPct ?? agent.xgbTakeProfitPctDefault ?? 3.0;
              }
            }
            return next;
          });
          setDraftExitModes((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              if (!next[agent.sessionId]) {
                next[agent.sessionId] =
                  agent.executionMode === "limit_post_only" ? "limit" : "market";
              }
            }
            return next;
          });
        })();
      }, 250);
    } catch (saveError) {
      setActionMessage(
        saveError instanceof Error ? saveError.message : "Failed to update XGB stop-loss",
      );
    } finally {
      setSavingSession(null);
    }
  }

  async function updateTakeProfit(agent: XgbAgent, role: "long" | "short", reset = false) {
    try {
      const sessionId = agent.sessionId;
      const key = sideDraftKey(sessionId, role);
      setSavingSession(sessionId);
      setActionMessage(null);
      const response = await fetch("/api/trading/xgb_take_profit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(
          reset
            ? { session_id: sessionId, side: role, reset: true }
            : { session_id: sessionId, side: role, take_profit_pct: draftTakeProfits[key] },
        ),
      });
      const data = (await response.json()) as {
        success?: boolean;
        error?: string;
        xgb_take_profit_pct?: number | null;
      };
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Failed to update XGB take-profit");
      }
      if (typeof data.xgb_take_profit_pct === "number") {
        setDraftTakeProfits((current) => ({
          ...current,
          [key]: data.xgb_take_profit_pct as number,
        }));
      }
      setActionMessage(
        reset
          ? `Take-profit ${role} для ${agent.symbol} сброшен к значению модели.`
          : `Take-profit ${role} для ${agent.symbol} обновлен.`,
      );
      setTimeout(() => {
        void (async () => {
          const responseReload = await fetch("/api/trading/status_all", {
            method: "GET",
            cache: "no-store",
          });
          if (!responseReload.ok) {
            return;
          }
          const dataReload = (await responseReload.json()) as StatusResponse;
          if (!dataReload.success) {
            return;
          }
          const mapped = (dataReload.active_agents ?? [])
            .filter((item) => Boolean(item.is_xgb))
            .map(mapRawAgent);
          setAgents(mapped);
          setDraftTakeProfits((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              if (typeof next[agent.sessionId] !== "number") {
                next[agent.sessionId] = agent.xgbTakeProfitPct ?? agent.xgbTakeProfitPctDefault ?? 3.0;
              }
            }
            return next;
          });
          setDraftStopLosses((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              if (typeof next[agent.sessionId] !== "number") {
                next[agent.sessionId] = agent.xgbStopLossPct ?? agent.xgbStopLossPctDefault ?? 1.0;
              }
            }
            return next;
          });
          setDraftExitModes((current) => {
            const next = { ...current };
            for (const agent of mapped) {
              if (!next[agent.sessionId]) {
                next[agent.sessionId] =
                  agent.executionMode === "limit_post_only" ? "limit" : "market";
              }
            }
            return next;
          });
        })();
      }, 250);
    } catch (saveError) {
      setActionMessage(
        saveError instanceof Error ? saveError.message : "Failed to update XGB take-profit",
      );
    } finally {
      setSavingSession(null);
    }
  }

  async function toggleTrendFilter(agent: XgbAgent) {
    if (!confirm(`Изменить режим использования тренда для сессии ${agent.sessionId}?`)) return;

    setSavingSession(agent.sessionId);
    try {
      const newStatus = !agent.ignoreTrendFilter;
      const response = await fetch("/api/trading/toggle_trend_filter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: agent.sessionId,
          ignore_trend_filter: newStatus,
        }),
      });
      const data = (await response.json()) as {
        success?: boolean;
        ignore_trend_filter?: boolean;
        error?: string;
      };
      if (!data.success) {
        alert("Ошибка: " + (data.error || "Неизвестная ошибка"));
      } else {
        setAgents((prev) =>
          prev.map((item) => {
            if (item.sessionId === agent.sessionId) {
              return { ...item, ignoreTrendFilter: Boolean(data.ignore_trend_filter) };
            }
            return item;
          }),
        );
      }
    } catch (e: any) {
      alert("Ошибка сети: " + String(e.message || e));
    } finally {
      setSavingSession(null);
    }
  }

  async function toggleClientMaster(agent: XgbAgent) {
    try {
      const sessionId = agent.sessionId;
      const newStatus = !agent.isClientMaster;
      setSavingSession(sessionId);
      setActionMessage(null);
      const response = await fetch("/api/trading/toggle_client_master", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ session_id: sessionId, enable: newStatus }),
      });
      const data = (await response.json()) as {
        success?: boolean;
        error?: string;
        is_client_master?: boolean;
      };
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Failed to toggle client master");
      }
      setAgents((current) =>
        current.map((item) => {
          if (item.sessionId === sessionId) {
            return { ...item, isClientMaster: Boolean(data.is_client_master) };
          }
          // Если мы включили эту модель, то выключаем все остальные
          if (Boolean(data.is_client_master)) {
            return { ...item, isClientMaster: false };
          }
          return item;
        })
      );
      setActionMessage(
        newStatus
          ? `Сигналы модели ${agent.symbol} теперь отправляются клиентам.`
          : `Рассылка сигналов ${agent.symbol} клиентам отключена.`,
      );
    } catch (saveError) {
      setActionMessage(
        saveError instanceof Error ? saveError.message : "Failed to toggle client master",
      );
    } finally {
      setSavingSession(null);
    }
  }

  async function updateLeverage(agent: XgbAgent) {
    try {
      const sessionId = agent.sessionId;
      const leverage = Math.max(1, Math.min(5, Math.round(draftLeverages[sessionId] ?? 1)));
      setSavingSession(sessionId);
      setActionMessage(null);
      const response = await fetch("/api/trading/xgb_leverage", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ session_id: sessionId, leverage }),
      });
      const data = (await response.json()) as {
        success?: boolean;
        error?: string;
        xgb_leverage?: number | null;
      };
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Failed to update XGB leverage");
      }
      const savedLeverage =
        typeof data.xgb_leverage === "number" ? data.xgb_leverage : leverage;
      setDraftLeverages((current) => ({
        ...current,
        [sessionId]: savedLeverage,
      }));
      setAgents((current) =>
        current.map((item) =>
          item.sessionId === sessionId ? { ...item, leverage: savedLeverage } : item,
        ),
      );
      setActionMessage(`Плечо для ${agent.symbol} обновлено: x${savedLeverage}.`);
    } catch (saveError) {
      setActionMessage(
        saveError instanceof Error ? saveError.message : "Failed to update XGB leverage",
      );
    } finally {
      setSavingSession(null);
    }
  }

  async function updateSignalExitThreshold(agent: XgbAgent) {
    try {
      const sessionId = agent.sessionId;
      const threshold = draftSignalExitThresholds[sessionId] ?? 0.5;
      if (!Number.isFinite(threshold) || threshold <= 0 || threshold >= 1) {
        throw new Error("Signal exit threshold должен быть между 0 и 1");
      }
      setSavingSession(sessionId);
      setActionMessage(null);
      const response = await fetch("/api/trading/xgb_signal_exit_threshold", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ session_id: sessionId, threshold }),
      });
      const data = (await response.json()) as {
        success?: boolean;
        error?: string;
        xgb_signal_exit_threshold?: number | null;
      };
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Failed to update XGB signal exit threshold");
      }
      const savedThreshold =
        typeof data.xgb_signal_exit_threshold === "number"
          ? data.xgb_signal_exit_threshold
          : threshold;
      setDraftSignalExitThresholds((current) => ({
        ...current,
        [sessionId]: savedThreshold,
      }));
      setAgents((current) =>
        current.map((item) =>
          item.sessionId === sessionId
            ? {
                ...item,
                xgbSignalExitAvgThreshold: savedThreshold,
                xgbSignalExitLastThreshold: savedThreshold,
              }
            : item,
        ),
      );
      setActionMessage(`Avg signal threshold для ${agent.symbol} обновлен: ${formatNumber(savedThreshold, 4)}.`);
    } catch (saveError) {
      setActionMessage(
        saveError instanceof Error
          ? saveError.message
          : "Failed to update XGB signal exit threshold",
      );
    } finally {
      setSavingSession(null);
    }
  }

  async function updateOppositeSignalThreshold(agent: XgbAgent, role: "long" | "short") {
    try {
      const sessionId = agent.sessionId;
      const key = sideDraftKey(sessionId, role);
      const threshold = draftOppositeSignalThresholds[key] ?? 0.4;
      if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
        throw new Error("Opposite signal threshold должен быть между 0 и 1");
      }
      setSavingSession(sessionId);
      setActionMessage(null);
      const response = await fetch("/api/trading/xgb_opposite_signal_threshold", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ session_id: sessionId, side: role, threshold }),
      });
      const data = (await response.json()) as {
        success?: boolean;
        error?: string;
        opposite_signal_max?: number | null;
      };
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Failed to update opposite signal threshold");
      }
      const savedThreshold =
        typeof data.opposite_signal_max === "number"
          ? data.opposite_signal_max
          : threshold;
      setDraftOppositeSignalThresholds((current) => ({
        ...current,
        [key]: savedThreshold,
      }));
      setAgents((current) =>
        current.map((item) =>
          item.sessionId === sessionId
            ? {
                ...item,
                sideModels: item.sideModels.map((side) =>
                  side.role === role
                    ? { ...side, oppositeSignalMax: savedThreshold }
                    : side,
                ),
              }
            : item,
        ),
      );
      const opposite = role === "long" ? "short" : "long";
      setActionMessage(
        `${role}: ${opposite} signal должен быть <= ${formatNumber(savedThreshold, 4)}.`,
      );
    } catch (saveError) {
      setActionMessage(
        saveError instanceof Error
          ? saveError.message
          : "Failed to update opposite signal threshold",
      );
    } finally {
      setSavingSession(null);
    }
  }

  async function extendHoldSteps(agent: XgbAgent) {
    try {
      const sessionId = agent.sessionId;
      const increment = Math.max(
        5,
        Math.min(200, Math.round(draftHoldExtensions[sessionId] ?? 5)),
      );
      setSavingSession(sessionId);
      setActionMessage(null);
      const response = await fetch("/api/trading/xgb_hold_steps_extend", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ session_id: sessionId, increment_steps: increment }),
      });
      const data = (await response.json()) as {
        success?: boolean;
        error?: string;
        previous_max_hold_steps?: number | null;
        increment_steps?: number | null;
        max_hold_steps?: number | null;
      };
      if (!response.ok || !data.success || typeof data.max_hold_steps !== "number") {
        throw new Error(data.error || "Failed to extend XGB hold_steps");
      }
      setAgents((current) =>
        current.map((item) =>
          item.sessionId === sessionId
            ? {
                ...item,
                maxHoldSteps: data.max_hold_steps as number,
                holdSecondsRemaining:
                  typeof item.holdSecondsRemaining === "number"
                    ? item.holdSecondsRemaining + increment * 300
                    : item.holdSecondsRemaining,
              }
            : item,
        ),
      );
      setActionMessage(
        `Hold steps для ${agent.symbol}: ${data.previous_max_hold_steps} + ${increment} = ${data.max_hold_steps}.`,
      );
    } catch (saveError) {
      setActionMessage(
        saveError instanceof Error ? saveError.message : "Failed to extend XGB hold_steps",
      );
    } finally {
      setSavingSession(null);
    }
  }

  async function forceExit(agent: XgbAgent) {
    try {
      const sessionId = agent.sessionId;
      setExitingSession(sessionId);
      setActionMessage(null);
      const response = await fetch("/api/trading/manual_exit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          execution_mode: draftExitModes[sessionId] ?? "market",
        }),
      });
      const data = (await response.json()) as {
        success?: boolean;
        error?: string;
        execution_mode?: string | null;
        price?: number | null;
      };
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Failed to force exit position");
      }
      if (data.execution_mode === "limit") {
        setActionMessage(
          `Для ${agent.symbol} выставлен лимитный reduceOnly выход${
            typeof data.price === "number" ? ` по ${formatNumber(data.price, 4)}` : ""
          }.`,
        );
      } else {
        setActionMessage(`Для ${agent.symbol} отправлен принудительный market выход.`);
      }
      setTimeout(() => {
        void (async () => {
          const responseReload = await fetch("/api/trading/status_all", {
            method: "GET",
            cache: "no-store",
          });
          if (!responseReload.ok) {
            return;
          }
          const dataReload = (await responseReload.json()) as StatusResponse;
          if (!dataReload.success) {
            return;
          }
          const mapped = (dataReload.active_agents ?? [])
            .filter((item) => Boolean(item.is_xgb))
            .map(mapRawAgent);
          setAgents(mapped);
        })();
      }, 400);
    } catch (exitError) {
      setActionMessage(
        exitError instanceof Error ? exitError.message : "Failed to force exit position",
      );
    } finally {
      setExitingSession(null);
    }
  }

  async function stopSession(agent: XgbAgent) {
    const shouldStop = window.confirm(
      agent.positionType
        ? `Остановить модель ${agent.symbol}? Открытая позиция останется на Bybit.`
        : `Остановить модель ${agent.symbol}?`,
    );
    if (!shouldStop) {
      return;
    }

    try {
      const sessionId = agent.sessionId;
      setStoppingSession(sessionId);
      setActionMessage(null);
      const response = await fetch("/api/trading/stop_session", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ session_id: sessionId }),
      });
      const data = (await response.json()) as {
        success?: boolean;
        error?: string;
        symbol?: string;
      };
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Failed to stop trading session");
      }

      setAgents((current) => current.filter((item) => item.sessionId !== sessionId));
      setDraftThresholds((current) => {
        const next = { ...current };
        delete next[sessionId];
        return next;
      });
      setDraftStopLosses((current) => {
        const next = { ...current };
        delete next[sessionId];
        return next;
      });
      setDraftTakeProfits((current) => {
        const next = { ...current };
        delete next[sessionId];
        return next;
      });
      setDraftLeverages((current) => {
        const next = { ...current };
        delete next[sessionId];
        return next;
      });
      setDraftHoldExtensions((current) => {
        const next = { ...current };
        delete next[sessionId];
        return next;
      });
      setDraftExitModes((current) => {
        const next = { ...current };
        delete next[sessionId];
        return next;
      });
      setActionMessage(`Модель ${data.symbol || agent.symbol} остановлена.`);
    } catch (stopError) {
      setActionMessage(
        stopError instanceof Error ? stopError.message : "Failed to stop trading session",
      );
    } finally {
      setStoppingSession(null);
    }
  }

  function renderSideRiskControls(agent: XgbAgent, role: "long" | "short") {
    const side = getSideModel(agent, role);
    if (!side.modelPath) {
      return null;
    }
    const key = sideDraftKey(agent.sessionId, role);
    const title = role === "long" ? "Long настройки" : "Short настройки";
    const tone = role === "long" ? "border-emerald-200 bg-emerald-50" : "border-rose-200 bg-rose-50";
    const riskText = isRiskPlacementEnabled(agent.riskManagementType) ? "включена" : "выключена";
    const riskClass = isRiskPlacementEnabled(agent.riskManagementType) ? "text-emerald-700" : "text-rose-600";

    return (
      <div className={`rounded-2xl border p-4 ${tone}`}>
        <p className="text-sm font-semibold text-slate-900">{title}</p>
        <p className="mt-1 text-xs text-slate-600">{compactModelName(side.modelPath)}</p>

        <div className="mt-4 space-y-4">
          <div>
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm font-semibold text-slate-900">Порог входа XGB</p>
              <span className="text-sm font-semibold text-slate-900">
                {formatNumber(draftThresholds[key] ?? side.entryThreshold ?? side.entryThresholdDefault, 4)}
              </span>
            </div>
            <input
              type="range"
              min={0.3}
              max={0.95}
              step={0.01}
              value={draftThresholds[key] ?? side.entryThreshold ?? side.entryThresholdDefault ?? 0.5}
              onChange={(event) =>
                setDraftThresholds((current) => ({ ...current, [key]: Number(event.target.value) }))
              }
              className="mt-3 h-2 w-full cursor-pointer appearance-none rounded-lg bg-white/70"
            />
            <div className="mt-3 flex flex-wrap gap-2">
              <button type="button" disabled={savingSession === agent.sessionId} onClick={() => void updateThreshold(agent, role, false)} className="rounded-xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400">
                Обновить
              </button>
              <button type="button" disabled={savingSession === agent.sessionId} onClick={() => void updateThreshold(agent, role, true)} className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-400">
                Сбросить к модели
              </button>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm font-semibold text-slate-900">
                {role === "long" ? "Short signal должен быть ниже" : "Long signal должен быть ниже"}
              </p>
              <span className="text-sm font-semibold text-slate-900">
                {formatNumber(
                  draftOppositeSignalThresholds[key] ?? side.oppositeSignalMax ?? 0.4,
                  4,
                )}
              </span>
            </div>
            <p className="mt-1 text-xs text-slate-600">
              Вход {role} будет разрешён только если противоположная модель явно слабая.
            </p>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={draftOppositeSignalThresholds[key] ?? side.oppositeSignalMax ?? 0.4}
              onChange={(event) =>
                setDraftOppositeSignalThresholds((current) => ({
                  ...current,
                  [key]: Number(event.target.value),
                }))
              }
              className="mt-3 h-2 w-full cursor-pointer appearance-none rounded-lg bg-white/70"
            />
            <div className="mt-3 flex flex-wrap gap-2">
              <button type="button" disabled={savingSession === agent.sessionId} onClick={() => void updateOppositeSignalThreshold(agent, role)} className="rounded-xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400">
                Обновить
              </button>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm font-semibold text-slate-900">Stop-loss XGB</p>
              <span className="text-sm font-semibold text-slate-900">
                {formatNumber(draftStopLosses[key] ?? side.stopLossPct ?? side.stopLossPctDefault, 2)}%
              </span>
            </div>
            <p className={`mt-1 text-xs font-medium ${riskClass}`}>Простановка при входе: {riskText}</p>
            <input
              type="range"
              min={0.3}
              max={5}
              step={0.1}
              value={draftStopLosses[key] ?? side.stopLossPct ?? side.stopLossPctDefault ?? 1}
              onChange={(event) =>
                setDraftStopLosses((current) => ({ ...current, [key]: Number(event.target.value) }))
              }
              className="mt-3 h-2 w-full cursor-pointer appearance-none rounded-lg bg-white/70"
            />
            <div className="mt-3 flex flex-wrap gap-2">
              <button type="button" disabled={savingSession === agent.sessionId} onClick={() => void updateStopLoss(agent, role, false)} className="rounded-xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400">
                Обновить
              </button>
              <button type="button" disabled={savingSession === agent.sessionId} onClick={() => void updateStopLoss(agent, role, true)} className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-400">
                Сбросить к модели
              </button>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm font-semibold text-slate-900">Take-profit XGB</p>
              <span className="text-sm font-semibold text-slate-900">
                {formatNumber(draftTakeProfits[key] ?? side.takeProfitPct ?? side.takeProfitPctDefault, 2)}%
              </span>
            </div>
            <p className={`mt-1 text-xs font-medium ${riskClass}`}>Простановка при входе: {riskText}</p>
            <input
              type="range"
              min={0.5}
              max={10}
              step={0.1}
              value={draftTakeProfits[key] ?? side.takeProfitPct ?? side.takeProfitPctDefault ?? 3}
              onChange={(event) =>
                setDraftTakeProfits((current) => ({ ...current, [key]: Number(event.target.value) }))
              }
              className="mt-3 h-2 w-full cursor-pointer appearance-none rounded-lg bg-white/70"
            />
            <div className="mt-3 flex flex-wrap gap-2">
              <button type="button" disabled={savingSession === agent.sessionId} onClick={() => void updateTakeProfit(agent, role, false)} className="rounded-xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400">
                Обновить
              </button>
              <button type="button" disabled={savingSession === agent.sessionId} onClick={() => void updateTakeProfit(agent, role, true)} className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-400">
                Сбросить к модели
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6" id="xgb-active-training">
      {actionMessage ? (
        <div className="rounded-3xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
          {actionMessage}
        </div>
      ) : null}
      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Активных XGB</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">{summary.total}</p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Long</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">{summary.longCount}</p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Short</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">{summary.shortCount}</p>
        </div>
      </div>

      {agents.length === 0 ? (
        <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6 text-sm text-slate-500">
          Активных XGB агентов сейчас нет.
        </div>
      ) : (
        <div className="grid gap-4 xl:grid-cols-2">
          {agents.map((agent) => (
            <div
              key={agent.sessionId}
              className="rounded-3xl border border-slate-200 bg-white p-5"
            >
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
                    {agent.symbol}
                  </p>
                  <h3 className="mt-2 text-xl font-semibold text-slate-900">
                    {agent.direction || "—"}
                  </h3>
                  <p className="mt-1 text-sm text-slate-500">
                    API:{" "}
                    {agent.bybitAccountId
                      ? `${agent.bybitAccountLabel || "Account"} (id: ${agent.bybitAccountId}${
                          agent.bybitApiKeyHint ? `, ${agent.bybitApiKeyHint}` : ""
                        })`
                      : "—"}
                  </p>
                </div>
                <div className="flex flex-col items-end gap-2">
                  <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-sm font-semibold text-emerald-700">
                    active
                  </span>
                  <button
                    onClick={() => void toggleClientMaster(agent)}
                    disabled={savingSession === agent.sessionId}
                    className={`rounded-full border px-3 py-1 text-sm font-semibold transition ${
                      agent.isClientMaster
                        ? "border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100"
                        : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                    }`}
                  >
                    {agent.isClientMaster ? "✅ Использовать для клиентов" : "Отключено для клиентов"}
                  </button>
                  <button
                    onClick={() => void toggleTrendFilter(agent)}
                    disabled={savingSession === agent.sessionId}
                    className={`rounded-full border px-3 py-1 text-sm font-semibold transition ${
                      agent.ignoreTrendFilter
                        ? "border-amber-200 bg-amber-50 text-amber-700 hover:bg-amber-100"
                        : "border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100"
                    }`}
                  >
                    {agent.ignoreTrendFilter ? "Вход по сигналу (тренд игнорируется)" : "Вход по тренду (long-short)"}
                  </button>
                </div>
              </div>

              <div className="mt-5 space-y-4">
                <SideModelBlock agent={agent} role="long" />
                <SideModelBlock agent={agent} role="short" />
              </div>

              <div className="mt-4">
                <MetaRow label="Exit mode" value={agent.exitMode || "—"} />
                <MetaRow
                  label="Account %"
                  value={agent.accountPct !== null ? `${agent.accountPct}%` : "—"}
                />
                <MetaRow label="Leverage" value={`x${agent.leverage || 1}`} />
                <MetaRow
                  label="Trades count"
                  value={formatNumber(agent.tradesCount, 0)}
                />
                <MetaRow
                  label="Entry threshold"
                  value={
                    agent.xgbEntryThreshold !== null
                      ? `${formatNumber(agent.xgbEntryThreshold, 4)}${
                          agent.xgbThresholdOverrideActive ? " (override)" : ""
                        }`
                      : "—"
                  }
                />
                <MetaRow
                  label="Stop-loss"
                  value={
                    agent.xgbStopLossPct !== null
                      ? `${formatNumber(agent.xgbStopLossPct, 2)}%${
                          agent.xgbStopLossOverrideActive ? " (override)" : ""
                        }`
                      : "—"
                  }
                />
                <MetaRow
                  label="Take-profit"
                  value={
                    agent.xgbTakeProfitPct !== null
                      ? `${formatNumber(agent.xgbTakeProfitPct, 2)}%${
                          agent.xgbTakeProfitOverrideActive ? " (override)" : ""
                        }`
                      : "—"
                  }
                />
                <MetaRow
                  label="Lock TTL"
                  value={formatDuration(agent.ttlSeconds)}
                />
                <MetaRow
                  label="Bybit account"
                  value={
                    agent.bybitAccountId
                      ? `${agent.bybitAccountLabel || "Account"} (id: ${agent.bybitAccountId})`
                      : "—"
                  }
                />
                <MetaRow
                  label="Bybit API key"
                  value={agent.bybitApiKeyHint || "—"}
                />
              </div>

              {agent.exitMode === "hold_steps" ? (
                <div className="mt-5 rounded-2xl border border-blue-200 bg-blue-50 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold text-slate-900">
                        Продлить hold_steps
                      </p>
                      <p className="mt-1 text-xs text-slate-600">
                        Прибавляет шаги к runtime max_hold_steps, который закрывает позицию по timeout.
                      </p>
                    </div>
                    <span className="text-sm font-semibold text-slate-900">
                      +{draftHoldExtensions[agent.sessionId] ?? 5}
                    </span>
                  </div>

                  <input
                    type="range"
                    min={5}
                    max={200}
                    step={1}
                    value={draftHoldExtensions[agent.sessionId] ?? 5}
                    onChange={(event) =>
                      setDraftHoldExtensions((current) => ({
                        ...current,
                        [agent.sessionId]: Number(event.target.value),
                      }))
                    }
                    className="mt-4 h-2 w-full cursor-pointer appearance-none rounded-lg bg-blue-200"
                  />

                  <div className="mt-4 flex flex-wrap items-center gap-3">
                    <button
                      type="button"
                      disabled={savingSession === agent.sessionId}
                      onClick={() => void extendHoldSteps(agent)}
                      className="rounded-xl bg-blue-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-600 disabled:cursor-not-allowed disabled:bg-blue-300"
                    >
                      {savingSession === agent.sessionId ? "Обновляю..." : "Обновить"}
                    </button>
                    <span className="text-xs text-slate-600">
                      Сейчас: {agent.maxHoldSteps ?? "—"}, осталось:{" "}
                      {agent.positionType ? formatDuration(agent.holdSecondsRemaining) : "—"}
                    </span>
                  </div>
                </div>
              ) : null}

              <div className="mt-5 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold text-slate-900">Плечо XGB</p>
                    <p className="mt-1 text-xs text-slate-500">
                      x1 по умолчанию. Сохранённое значение применится на следующем входе этой модели.
                    </p>
                  </div>
                  <span className="text-sm font-semibold text-slate-900">
                    x{draftLeverages[agent.sessionId] ?? agent.leverage ?? 1}
                  </span>
                </div>

                <input
                  type="range"
                  min={1}
                  max={5}
                  step={1}
                  value={draftLeverages[agent.sessionId] ?? agent.leverage ?? 1}
                  onChange={(event) =>
                    setDraftLeverages((current) => ({
                      ...current,
                      [agent.sessionId]: Number(event.target.value),
                    }))
                  }
                  className="mt-4 h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-200"
                />

                <div className="mt-4 flex flex-wrap gap-3">
                  <button
                    type="button"
                    disabled={savingSession === agent.sessionId}
                    onClick={() => void updateLeverage(agent)}
                    className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
                  >
                    {savingSession === agent.sessionId ? "Сохраняю..." : "Обновить"}
                  </button>
                </div>
              </div>

              <div className="mt-5 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold text-slate-900">Статус позиции</p>
                    <p className="mt-1 text-xs text-slate-500">
                      Текущее состояние открытой сделки по этому XGB-агенту.
                    </p>
                  </div>
                  <span
                    className={`rounded-full border px-3 py-1 text-sm font-semibold ${
                      agent.positionType
                        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                        : "border-slate-200 bg-white text-slate-600"
                    }`}
                  >
                    {agent.positionType ? "в позиции" : "вне позиции"}
                  </span>
                </div>

                <MetaRow
                  label="Размер позиции"
                  value={
                    agent.positionType && agent.amountUsdt !== null
                      ? `${formatNumber(agent.amountUsdt, 2)} USDT`
                      : "—"
                  }
                />
                <MetaRow
                  label="Время входа"
                  value={
                    agent.positionType
                      ? formatMoscowDateTimeFromMs(agent.positionEntryTsMs)
                      : "—"
                  }
                />
                <MetaRow
                  label="До выхода hold_steps"
                  value={
                    agent.positionType && agent.exitMode === "hold_steps"
                      ? (agent.holdSecondsRemaining === 0
                          ? "0с (таймаут достигнут)"
                          : formatDuration(agent.holdSecondsRemaining))
                      : "—"
                  }
                />
                <MetaRow
                  label="Early exit signal"
                  value={formatSignalExitState(agent)}
                />
                <MetaRow
                  label="Post-exit guard"
                  value={formatPostExitGuardState(agent)}
                />
                <MetaRow
                  label="Start / Window"
                  value={
                    agent.xgbSignalExitEnabled
                      ? `${formatNumber(agent.xgbSignalExitStartStep, 0)} / ${formatNumber(agent.xgbSignalExitWindow, 0)}`
                      : "—"
                  }
                />
                <MetaRow
                  label="History ready"
                  value={
                    agent.xgbSignalExitEnabled
                      ? `${formatNumber(agent.xgbSignalExitHistorySize, 0)} / ${formatNumber(agent.xgbSignalExitWindow, 0)}`
                      : "—"
                  }
                />
                <MetaRow
                  label="Last signal / threshold"
                  value={
                    agent.xgbSignalExitEnabled
                      ? `${formatNumber(agent.xgbSignalExitLastSignal, 4)} / ${formatNumber(agent.xgbSignalExitLastThreshold, 4)}`
                      : "—"
                  }
                />
                <MetaRow
                  label="Avg signal / threshold"
                  value={
                    agent.xgbSignalExitEnabled
                      ? `${formatNumber(agent.xgbSignalExitAvgSignal, 4)} / ${formatNumber(agent.xgbSignalExitAvgThreshold, 4)}`
                      : "—"
                  }
                />
                {agent.exitMode === "hold_steps" ? (
                  <div className="mt-4 rounded-2xl border border-violet-200 bg-violet-50 p-4">
                    <div className="flex flex-wrap items-end gap-3">
                      <label className="block">
                        <span className="text-xs font-semibold uppercase tracking-wide text-violet-700">
                          Новый Avg threshold
                        </span>
                        <input
                          type="number"
                          min={0.01}
                          max={0.99}
                          step={0.01}
                          value={
                            draftSignalExitThresholds[agent.sessionId] ??
                            agent.xgbSignalExitAvgThreshold ??
                            0.5
                          }
                          onChange={(event) =>
                            setDraftSignalExitThresholds((current) => ({
                              ...current,
                              [agent.sessionId]: Number(event.target.value),
                            }))
                          }
                          className="mt-1 w-40 rounded-xl border border-violet-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900"
                        />
                      </label>
                      <button
                        type="button"
                        disabled={savingSession === agent.sessionId}
                        onClick={() => void updateSignalExitThreshold(agent)}
                        className="rounded-xl bg-violet-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-violet-600 disabled:cursor-not-allowed disabled:bg-violet-300"
                      >
                        {savingSession === agent.sessionId ? "Обновляю..." : "Обновить Avg threshold"}
                      </button>
                    </div>
                    <p className="mt-2 text-xs text-violet-700">
                      {agent.xgbSignalExitEnabled
                        ? "Применится в runtime на следующем 5m цикле без перезапуска модели."
                        : "Guard сейчас выключен, но threshold сохранится в runtime для этой сессии."}
                    </p>
                  </div>
                ) : null}
                <MetaRow
                  label="Последнее решение"
                  value={agent.lastPrediction || "—"}
                />
                <div className="mt-4 flex flex-wrap items-center gap-3">
                  <select
                    value={draftExitModes[agent.sessionId] ?? "market"}
                    disabled={!agent.positionType || exitingSession === agent.sessionId}
                    onChange={(event) =>
                      setDraftExitModes((current) => ({
                        ...current,
                        [agent.sessionId]: event.target.value as ExitExecutionMode,
                      }))
                    }
                    className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
                  >
                    <option value="market">Выход по market</option>
                    <option value="limit">Выход по limit</option>
                  </select>
                  <button
                    type="button"
                    disabled={!agent.positionType || exitingSession === agent.sessionId}
                    onClick={() => void forceExit(agent)}
                    className="rounded-xl bg-rose-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-500 disabled:cursor-not-allowed disabled:bg-rose-300"
                  >
                    {exitingSession === agent.sessionId
                      ? "Выходим..."
                      : "Принудительный выход"}
                  </button>
                  <button
                    type="button"
                    disabled={stoppingSession === agent.sessionId}
                    onClick={() => void stopSession(agent)}
                    className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
                  >
                    {stoppingSession === agent.sessionId
                      ? "Останавливаем..."
                      : "Остановить модель"}
                  </button>
                </div>
              </div>

              <div className="mt-5 grid gap-4 lg:grid-cols-2">
                {renderSideRiskControls(agent, "long")}
                {renderSideRiskControls(agent, "short")}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="rounded-3xl border border-slate-200 bg-white p-5">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-slate-900">История попыток входа (Redis)</p>
            <p className="mt-1 text-xs text-slate-500">
              Последние 300 записей: ожидание, вход long, вход short.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <select
              value={selectedEntryApi}
              onChange={(event) => setSelectedEntryApi(event.target.value)}
              className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm"
            >
              <option value="__all__">Все API</option>
              {entryApiOptions.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
            <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-semibold text-slate-700">
              {visibleEntryAttempts.length}/{entryAttempts.length}
            </span>
          </div>
        </div>
        {visibleEntryAttempts.length === 0 ? (
          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-500">
            История пока пустая.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                  <th className="px-2 py-2">Время</th>
                  <th className="px-2 py-2">API</th>
                  <th className="px-2 py-2">Символ</th>
                  <th className="px-2 py-2">Long signal</th>
                  <th className="px-2 py-2">Short signal</th>
                  <th className="px-2 py-2">Статус</th>
                </tr>
              </thead>
              <tbody>
                {visibleEntryAttempts.map((row, index) => {
                  const tsMs = typeof row.ts_ms === "number"
                    ? row.ts_ms
                    : (row.timestamp ? Date.parse(String(row.timestamp)) : NaN);
                  const sessionAgent =
                    row.session_id ? agentsBySession.get(String(row.session_id)) : undefined;
                  const showId = row.bybit_account_id
                    ? String(row.bybit_account_id)
                    : (sessionAgent?.bybitAccountId ?? null);
                  const showLabel = row.bybit_account_label
                    ? String(row.bybit_account_label)
                    : (sessionAgent?.bybitAccountLabel ?? "Account");
                  const showHint = row.bybit_api_key_hint
                    ? String(row.bybit_api_key_hint)
                    : (sessionAgent?.bybitApiKeyHint ?? null);
                  return (
                    <tr key={`${row.session_id ?? "sid"}:${row.symbol ?? "sym"}:${row.ts_ms ?? index}`} className="border-b border-slate-100 text-slate-700">
                      <td className="px-2 py-2">{formatMoscowDateTimeFromMs(Number.isFinite(tsMs) ? tsMs : null)}</td>
                      <td className="px-2 py-2">
                        {showId ? `${showLabel} (${showId}${showHint ? `, ${showHint}` : ""})` : "—"}
                      </td>
                      <td className="px-2 py-2">{row.symbol || "—"}</td>
                      <td className="px-2 py-2">{formatNumber(row.long_signal, 4)}</td>
                      <td className="px-2 py-2">{formatNumber(row.short_signal, 4)}</td>
                      <td className="px-2 py-2">{row.state || "ожидание"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
