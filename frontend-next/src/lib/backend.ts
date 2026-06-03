const INTERNAL_BACKEND_URL =
  process.env.MEDOED_BACKEND_URL ?? "http://medoedai:5050";

const LEGACY_UI_BASE_URL =
  process.env.NEXT_PUBLIC_LEGACY_UI_BASE_URL ?? "http://localhost:5050";

type TradingAgent = {
  symbol?: string | null;
};

type TradingStatusResponse = {
  active_agents?: TradingAgent[];
};

type BybitAccountResponse = {
  id: string | number;
  label?: string | null;
  api_key_masked?: string | null;
};

type BybitAccountsApiResponse = {
  accounts?: BybitAccountResponse[];
};

type BybitSelectedApiResponse = {
  selected?: string | number | null;
};

type XgbSymbolsApiResponse = {
  symbols?: string[];
};

type XgbOosRunsApiResponse = {
  runs?: Record<string, unknown>[];
};

type XgbWfRunsApiResponse = {
  runs?: Record<string, unknown>[];
};

type XgbProdModelsApiResponse = {
  models?: Record<string, unknown>[];
};

type XgbExperimentsApiResponse = {
  experiments?: Record<string, unknown>[];
};

type TelegramBotAccountsApiResponse = {
  users?: Record<string, unknown>[];
};

type MaxBotAccountsApiResponse = {
  users?: Record<string, unknown>[];
};

export type PromoCode = {
  id: number;
  code: string;
  durationDays: number;
  maxUses: number;
  usedCount: number;
  isActive: boolean;
  note: string | null;
  validUntil: string | null;
  createdAt: string;
};

type PromoCodesApiResponse = {
  promo_codes?: Record<string, unknown>[];
};

export type XgbModelSummary = {
  version: string;
  path: string;
  isCurrent: boolean;
  modelId: string;
  runId: string;
  modelUuid: string;
  direction: string | null;
  task: string | null;
  trainedAs: string | null;
  createdAt: string | null;
  sourceRunPath: string | null;
  trainMetrics: {
    valAcc: number | null;
    f1BuySellVal: number | null;
    f1Val: number[] | null;
    precisionVal: number[] | null;
    recallVal: number[] | null;
    proxyPnlVal: {
      trades?: number | null;
      pnl_sum?: number | null;
      max_hold_steps?: number | null;
      [key: string]: unknown;
    } | null;
  };
  cfg: {
    horizonSteps: number | null;
    threshold: number | null;
    maxHoldSteps: number | null;
    minProfit: number | null;
    feeBps: number | null;
    pEnterThreshold: number | null;
    entryTpPct: number | null;
    entrySlPct: number | null;
    entryTrailPct: number | null;
    nEstimators: number | null;
    maxDepth: number | null;
    learningRate: number | null;
    subsample: number | null;
    colsampleBytree: number | null;
    regLambda: number | null;
    minChildWeight: number | null;
    gamma: number | null;
  };
  bestOos: {
    days: number | null;
    ts: string | null;
    pnlTotal: number | null;
    roiPct: number | null;
    profitFactor: number | null;
    maxDd: number | null;
    tradesCount: number | null;
    winrate: number | null;
    avgTradePnl: number | null;
    avgBarsHeld: number | null;
    equityEnd: number | null;
  };
};

export type XgbEnsembleSummary = {
  currentVersion: string | null;
  currentModel: XgbModelSummary | null;
  versions: XgbModelSummary[];
};

export type XgbOosRun = {
  symbol: string;
  runName: string;
  direction: string | null;
  task: string | null;
  source: string | null;
  gridId: string | null;
  resultDir: string;
  modelPath: string | null;
  mtime: number | null;
  metrics: Record<string, unknown>;
  cfg: Record<string, unknown>;
};

export type XgbExperimentSummary = {
  symbol: string;
  experimentName: string;
  createdAt: string | null;
  path: string;
  summaryPath: string;
  presetPath: string;
  oosCsv: string | null;
  selectionMetric: string | null;
  selectedCount: number;
  topUuid: string | null;
  topRoiPct: number | null;
  topPnlTotal: number | null;
  topProfitFactor: number | null;
  topMaxDd: number | null;
  topTradesCount: number | null;
  top3Uuids: string[];
  avgRoiTop: number | null;
  avgProfitFactorTop: number | null;
  worstMaxDdTop: number | null;
  tradesSum: number | null;
};

export type XgbWfRun = {
  symbol: string;
  runName: string;
  resultDir: string;
  sourceRunPath: string | null;
  copiedAt: string | null;
  direction: string | null;
  task: string | null;
  metrics: Record<string, unknown>;
  mtime: number | null;
};

export type XgbProdModel = XgbModelSummary & {
  symbol: string;
  ensemble: string;
};

type XgbEnsemblesApiResponse = {
  ensembles?: Record<
    string,
    {
      current_version?: string | null;
      current_model?: Record<string, unknown> | null;
      versions?: Record<string, unknown>[];
    }
  >;
};

export type SidebarData = {
  backendUrl: string;
  symbols: string[];
  symbolsError: string | null;
};

export type BybitAccount = {
  id: string;
  label: string;
  apiKeyMasked: string | null;
};

export type TelegramBotAccount = {
  userId: string;
  identityId: string;
  telegramId: string;
  username: string | null;
  displayName: string | null;
  languageCode: string | null;
  status: string;
  role: string;
  registeredAt: string | null;
  lastSeenAt: string | null;
  hasActiveKeys: boolean;
  leverage: number;
  paidUntil: string | null;
  activePromos: string[];
  unreadSupportCount: number;
};

export type MaxBotAccount = {
  userId: string;
  identityId: string;
  maxUserId: string;
  username: string | null;
  displayName: string | null;
  languageCode: string | null;
  status: string;
  role: string;
  registeredAt: string | null;
  lastSeenAt: string | null;
};

function toNumberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function toNumberArrayOrNull(value: unknown): number[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const values = value
    .map((item) => (typeof item === "number" && Number.isFinite(item) ? item : null))
    .filter((item): item is number => item !== null);
  return values.length > 0 ? values : null;
}

function mapXgbModelSummary(raw: Record<string, unknown>): XgbModelSummary {
  const trainMetrics =
    raw.train_metrics && typeof raw.train_metrics === "object"
      ? (raw.train_metrics as Record<string, unknown>)
      : {};
  const cfg = raw.cfg && typeof raw.cfg === "object" ? (raw.cfg as Record<string, unknown>) : {};
  const bestOos =
    raw.best_oos && typeof raw.best_oos === "object"
      ? (raw.best_oos as Record<string, unknown>)
      : {};

  return {
    version: String(raw.version ?? ""),
    path: String(raw.path ?? ""),
    isCurrent: Boolean(raw.is_current),
    modelId: String(raw.model_id ?? ""),
    runId: String(raw.run_id ?? ""),
    modelUuid: String(raw.model_uuid ?? ""),
    direction: raw.direction ? String(raw.direction) : null,
    task: raw.task ? String(raw.task) : null,
    trainedAs: raw.trained_as ? String(raw.trained_as) : null,
    createdAt: raw.created_at ? String(raw.created_at) : null,
    sourceRunPath: raw.source_run_path ? String(raw.source_run_path) : null,
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

function mapXgbProdModel(raw: Record<string, unknown>): XgbProdModel {
  return {
    ...mapXgbModelSummary(raw),
    symbol: String(raw.symbol ?? ""),
    ensemble: String(raw.ensemble ?? ""),
  };
}

function mapXgbExperimentSummary(raw: Record<string, unknown>): XgbExperimentSummary {
  return {
    symbol: String(raw.symbol ?? ""),
    experimentName: String(raw.experiment_name ?? ""),
    createdAt: raw.created_at ? String(raw.created_at) : null,
    path: String(raw.path ?? ""),
    summaryPath: String(raw.summary_path ?? ""),
    presetPath: String(raw.preset_path ?? ""),
    oosCsv: raw.oos_csv ? String(raw.oos_csv) : null,
    selectionMetric: raw.selection_metric ? String(raw.selection_metric) : null,
    selectedCount: toNumberOrNull(raw.selected_count) ?? 0,
    topUuid: raw.top_uuid ? String(raw.top_uuid) : null,
    topRoiPct: toNumberOrNull(raw.top_roi_pct),
    topPnlTotal: toNumberOrNull(raw.top_pnl_total),
    topProfitFactor: toNumberOrNull(raw.top_profit_factor),
    topMaxDd: toNumberOrNull(raw.top_max_dd),
    topTradesCount: toNumberOrNull(raw.top_trades_count),
    top3Uuids: Array.isArray(raw.top3_uuids)
      ? raw.top3_uuids.map((item) => String(item)).filter(Boolean)
      : [],
    avgRoiTop: toNumberOrNull(raw.avg_roi_top),
    avgProfitFactorTop: toNumberOrNull(raw.avg_profit_factor_top),
    worstMaxDdTop: toNumberOrNull(raw.worst_max_dd_top),
    tradesSum: toNumberOrNull(raw.trades_sum),
  };
}

function mapXgbOosRun(raw: Record<string, unknown>): XgbOosRun {
  const metrics =
    raw.metrics && typeof raw.metrics === "object"
      ? (raw.metrics as Record<string, unknown>)
      : {};
  const cfg = raw.cfg && typeof raw.cfg === "object" ? (raw.cfg as Record<string, unknown>) : {};

  return {
    symbol: String(raw.symbol ?? ""),
    runName: String(raw.run_name ?? ""),
    direction: raw.direction ? String(raw.direction) : null,
    task: raw.task ? String(raw.task) : null,
    source: raw.source ? String(raw.source) : null,
    gridId: raw.grid_id ? String(raw.grid_id) : null,
    resultDir: String(raw.result_dir ?? ""),
    modelPath: raw.model_path ? String(raw.model_path) : null,
    mtime: toNumberOrNull(raw.mtime),
    metrics,
    cfg,
  };
}

function mapXgbWfRun(raw: Record<string, unknown>): XgbWfRun {
  const metrics =
    raw.metrics && typeof raw.metrics === "object"
      ? (raw.metrics as Record<string, unknown>)
      : {};
  return {
    symbol: String(raw.symbol ?? ""),
    runName: String(raw.run_name ?? ""),
    resultDir: String(raw.result_dir ?? ""),
    sourceRunPath: raw.source_run_path ? String(raw.source_run_path) : null,
    copiedAt: raw.copied_at ? String(raw.copied_at) : null,
    direction: raw.direction ? String(raw.direction) : null,
    task: raw.task ? String(raw.task) : null,
    metrics,
    mtime: toNumberOrNull(raw.mtime),
  };
}

export async function getSidebarData(): Promise<SidebarData> {
  try {
    const response = await fetch(
      `${INTERNAL_BACKEND_URL}/api/trading/status_all`,
      {
        cache: "no-store",
      },
    );

    if (!response.ok) {
      return {
        backendUrl: LEGACY_UI_BASE_URL,
        symbols: [],
        symbolsError: `Backend returned ${response.status}`,
      };
    }

    const data = (await response.json()) as TradingStatusResponse;
    const uniqueSymbols = Array.from(
      new Set(
        (data.active_agents ?? [])
          .map((agent) => agent.symbol?.trim())
          .filter((symbol): symbol is string => Boolean(symbol)),
      ),
    );

    return {
      backendUrl: LEGACY_UI_BASE_URL,
      symbols: uniqueSymbols,
      symbolsError: null,
    };
  } catch (error) {
    return {
      backendUrl: LEGACY_UI_BASE_URL,
      symbols: [],
      symbolsError:
        error instanceof Error ? error.message : "Backend is unavailable",
    };
  }
}

export async function getBybitAccounts(): Promise<BybitAccount[]> {
  try {
    const response = await fetch(`${INTERNAL_BACKEND_URL}/api/bybit/accounts`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return [];
    }

    const data = (await response.json()) as BybitAccountsApiResponse;
    const accounts = Array.isArray(data.accounts) ? data.accounts : [];

    return accounts.map((account) => ({
      id: String(account.id ?? ""),
      label: String(account.label ?? `Account ${account.id ?? ""}`),
      apiKeyMasked: account.api_key_masked
        ? String(account.api_key_masked)
        : null,
    }));
  } catch {
    return [];
  }
}

export async function getSelectedBybitAccountId(): Promise<string | null> {
  try {
    const response = await fetch(`${INTERNAL_BACKEND_URL}/api/bybit/selected`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return null;
    }

    const data = (await response.json()) as BybitSelectedApiResponse;
    return data.selected ? String(data.selected) : null;
  } catch {
    return null;
  }
}

export async function getTelegramBotAccounts(): Promise<TelegramBotAccount[]> {
  try {
    const response = await fetch(`${INTERNAL_BACKEND_URL}/api/bot/accounts/telegram`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return [];
    }

    const data = (await response.json()) as TelegramBotAccountsApiResponse;
    const users = Array.isArray(data.users) ? data.users : [];
    return users.map((user) => ({
      userId: String(user.user_id ?? ""),
      identityId: String(user.identity_id ?? ""),
      telegramId: String(user.platform_user_id ?? ""),
      username: user.username ? String(user.username) : null,
      displayName: user.display_name ? String(user.display_name) : null,
      languageCode: user.language_code ? String(user.language_code) : null,
      status: String(user.status ?? ""),
      role: String(user.role ?? ""),
      registeredAt: user.registered_at ? String(user.registered_at) : null,
      lastSeenAt: user.last_seen_at ? String(user.last_seen_at) : null,
      hasActiveKeys: Boolean(user.has_active_keys),
      leverage: toNumberOrNull(user.bybit_leverage) ?? 1,
      paidUntil: user.paid_until ? String(user.paid_until) : null,
      activePromos: Array.isArray(user.active_promos) ? user.active_promos.map(String) : [],
      unreadSupportCount: toNumberOrNull(user.unread_support_count) ?? 0,
    }));
  } catch {
    return [];
  }
}

export async function getMaxBotAccounts(): Promise<MaxBotAccount[]> {
  try {
    const response = await fetch(`${INTERNAL_BACKEND_URL}/api/bot/accounts/max`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return [];
    }

    const data = (await response.json()) as MaxBotAccountsApiResponse;
    const users = Array.isArray(data.users) ? data.users : [];
    return users.map((user) => ({
      userId: String(user.user_id ?? ""),
      identityId: String(user.identity_id ?? ""),
      maxUserId: String(user.platform_user_id ?? ""),
      username: user.username ? String(user.username) : null,
      displayName: user.display_name ? String(user.display_name) : null,
      languageCode: user.language_code ? String(user.language_code) : null,
      status: String(user.status ?? ""),
      role: String(user.role ?? ""),
      registeredAt: user.registered_at ? String(user.registered_at) : null,
      lastSeenAt: user.last_seen_at ? String(user.last_seen_at) : null,
    }));
  } catch {
    return [];
  }
}

export async function getPromoCodes(): Promise<PromoCode[]> {
  try {
    const response = await fetch(`${INTERNAL_BACKEND_URL}/api/bot/promo-codes`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return [];
    }

    const data = (await response.json()) as PromoCodesApiResponse;
    const codes = Array.isArray(data.promo_codes) ? data.promo_codes : [];
    return codes.map((code) => ({
      id: toNumberOrNull(code.id) ?? 0,
      code: String(code.code ?? ""),
      durationDays: toNumberOrNull(code.duration_days) ?? 0,
      maxUses: toNumberOrNull(code.max_uses) ?? 0,
      usedCount: toNumberOrNull(code.used_count) ?? 0,
      isActive: Boolean(code.is_active),
      note: code.note ? String(code.note) : null,
      validUntil: code.valid_until ? String(code.valid_until) : null,
      createdAt: String(code.created_at ?? ""),
    }));
  } catch {
    return [];
  }
}

export async function getXgbSymbols(): Promise<string[]> {
  try {
    const response = await fetch(`${INTERNAL_BACKEND_URL}/api/xgb/symbols`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return [];
    }

    const data = (await response.json()) as XgbSymbolsApiResponse;
    return Array.isArray(data.symbols) ? data.symbols : [];
  } catch {
    return [];
  }
}

export async function getXgbEnsembles(
  symbol: string,
): Promise<Record<string, XgbEnsembleSummary>> {
  try {
    const response = await fetch(
      `${INTERNAL_BACKEND_URL}/api/xgb/ensembles?symbol=${encodeURIComponent(symbol)}`,
      {
        cache: "no-store",
      },
    );

    if (!response.ok) {
      return {};
    }

    const data = (await response.json()) as XgbEnsemblesApiResponse;
    const ensembles = data.ensembles ?? {};

    return Object.fromEntries(
      Object.entries(ensembles).map(([ensembleName, ensembleValue]) => {
        const versions = Array.isArray(ensembleValue.versions)
          ? ensembleValue.versions.map((item) => mapXgbModelSummary(item))
          : [];
        const currentModel =
          ensembleValue.current_model && typeof ensembleValue.current_model === "object"
            ? mapXgbModelSummary(ensembleValue.current_model)
            : null;

        return [
          ensembleName,
          {
            currentVersion: ensembleValue.current_version
              ? String(ensembleValue.current_version)
              : null,
            currentModel,
            versions,
          },
        ];
      }),
    );
  } catch {
    return {};
  }
}

export async function getXgbOosRuns(): Promise<XgbOosRun[]> {
  try {
    const response = await fetch(`${INTERNAL_BACKEND_URL}/api/xgb/oos/runs`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return [];
    }

    const data = (await response.json()) as XgbOosRunsApiResponse;
    const runs = Array.isArray(data.runs) ? data.runs : [];
    return runs
      .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
      .map((item) => mapXgbOosRun(item));
  } catch {
    return [];
  }
}

export async function getXgbWfRuns(): Promise<XgbWfRun[]> {
  try {
    const response = await fetch(`${INTERNAL_BACKEND_URL}/api/xgb/wf/runs`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return [];
    }

    const data = (await response.json()) as XgbWfRunsApiResponse;
    const runs = Array.isArray(data.runs) ? data.runs : [];
    return runs
      .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
      .map((item) => mapXgbWfRun(item));
  } catch {
    return [];
  }
}

export async function getXgbProdModels(): Promise<XgbProdModel[]> {
  try {
    const response = await fetch(`${INTERNAL_BACKEND_URL}/api/xgb/prod/models`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return [];
    }

    const data = (await response.json()) as XgbProdModelsApiResponse;
    const models = Array.isArray(data.models) ? data.models : [];
    return models
      .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
      .map((item) => mapXgbProdModel(item));
  } catch {
    return [];
  }
}

export async function getXgbExperiments(): Promise<XgbExperimentSummary[]> {
  try {
    const response = await fetch(`${INTERNAL_BACKEND_URL}/api/xgb/oos/experiments`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return [];
    }

    const data = (await response.json()) as XgbExperimentsApiResponse;
    const experiments = Array.isArray(data.experiments) ? data.experiments : [];
    return experiments
      .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
      .map((item) => mapXgbExperimentSummary(item));
  } catch {
    return [];
  }
}

export type XgbShortAnchorData = {
  schema_version?: number;
  direction?: string;
  symbol?: string;
  task?: string;
  description?: string;
  updated_at?: string;
  compare_keys?: string[];
  models?: Record<string, any>[];
};

type XgbShortAnchorsApiResponse = {
  success?: boolean;
  data?: XgbShortAnchorData;
  error?: string;
};

export async function getXgbShortAnchors(): Promise<XgbShortAnchorData | null> {
  try {
    const response = await fetch(`${INTERNAL_BACKEND_URL}/api/xgb/hypo/short_anchors`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return null;
    }

    const resData = (await response.json()) as XgbShortAnchorsApiResponse;
    if (resData.success && resData.data) {
      return resData.data;
    }
    return null;
  } catch {
    return null;
  }
}
