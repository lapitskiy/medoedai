import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AdaptiveNormalizer:
    """
    Адаптивная нормализация для разных криптовалют
    Учитывает волатильность, объем и рыночные условия
    """
    
    def __init__(self):
        # Ручные профили криптовалют убраны: расчёт торговых параметров идёт динамически
        # через get_trading_params() -> adapt_parameters() -> _full_auto_base_profile().
        # get_crypto_profile() оставлен для обратной совместимости (возвращает дефолтный профиль).
        self.crypto_profiles = {}
        
        # Динамические параметры
        self.dynamic_params = {}

        # --- Full-auto calibration cache (small JSON files) ---
        # Храним только компактные параметры, без больших массивов regime_precomputed.
        self._cache_dir = os.environ.get("ADAPT_CACHE_DIR", "result/adaptive_cache")
        try:
            self._cache_ttl_sec = int(os.environ.get("ADAPT_CACHE_TTL_SEC", "3600"))
        except Exception:
            self._cache_ttl_sec = 3600
        self._cache_mem: dict[tuple, dict] = {}

    def _df_signature(self, df: pd.DataFrame) -> dict:
        """Компактная сигнатура данных для кеша (быстро, без хеширования всего массива)."""
        try:
            n = int(len(df))
        except Exception:
            n = 0
        last_ts = None
        try:
            if n > 0 and "timestamp" in df.columns:
                last_ts = int(df["timestamp"].iloc[-1])
        except Exception:
            last_ts = None
        try:
            last_close = float(df["close"].iloc[-1]) if (n > 0 and "close" in df.columns) else None
        except Exception:
            last_close = None
        return {"n": n, "last_ts": last_ts, "last_close": last_close}

    def _cache_key(self, symbol: str, timeframe: str, train_split_point: int | None, sig: dict) -> tuple:
        # bump version when output schema changes (e.g., adding meta fields)
        algo_ver = "v2_full_auto_meta"
        return (algo_ver, str(symbol).upper(), str(timeframe), int(train_split_point or -1), int(sig.get("n") or 0), sig.get("last_ts"), sig.get("last_close"))

    def _cache_path(self, key: tuple) -> str:
        # filename safe and short
        algo_ver, sym, tf, split, n, last_ts, last_close = key
        fname = f"{sym}_{tf}_split{split}_n{n}_last{last_ts}_{algo_ver}.json"
        return os.path.join(self._cache_dir, fname)

    def _cache_load(self, key: tuple) -> dict | None:
        # ttl<=0 means "disable cache"
        try:
            if int(self._cache_ttl_sec) <= 0:
                return None
        except Exception:
            pass
        # 1) memory cache
        try:
            hit = self._cache_mem.get(key)
            if isinstance(hit, dict):
                ts = float(hit.get("_cached_at", 0.0) or 0.0)
                if ts and (ts + self._cache_ttl_sec) >= float(pd.Timestamp.utcnow().timestamp()):
                    return hit.get("payload")
        except Exception:
            pass
        # 2) file cache
        try:
            path = self._cache_path(key)
            if not os.path.exists(path):
                return None
            st = os.stat(path)
            if self._cache_ttl_sec > 0:
                now = float(pd.Timestamp.utcnow().timestamp())
                if (st.st_mtime + self._cache_ttl_sec) < now:
                    return None
            import json as _json
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            if isinstance(data, dict) and "payload" in data:
                self._cache_mem[key] = data
                return data.get("payload")
        except Exception:
            return None
        return None

    def _cache_save(self, key: tuple, payload: dict) -> None:
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
        except Exception:
            return
        try:
            import json as _json
            data = {"_cached_at": float(pd.Timestamp.utcnow().timestamp()), "payload": payload}
            path = self._cache_path(key)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                _json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)
            self._cache_mem[key] = data
        except Exception:
            pass

    def _full_auto_base_profile(self, symbol: str, df: pd.DataFrame, train_split_point: int | None) -> Dict:
        """
        Full-auto базовые параметры из train-части данных (робастно, без оптимизации по PnL).
        Возвращает профиль в формате self.crypto_profiles[*].
        """
        analysis_df = df.iloc[:train_split_point] if (train_split_point is not None and train_split_point > 0) else df
        # fallbacks
        base = {
            "volatility_multiplier": 1.0,
            "volume_threshold": 0.002,
            "price_sensitivity": 1.0,
            "trend_strength": 1.0,
            "min_hold_time": 12,
            "stop_loss": -0.02,
            "take_profit": 0.02,
            # meta: откуда взят риск-прокси для SL/TP (ATR или std(returns))
            "risk_calc_source": "unknown",
            "atr_rel_med": None,
            "returns_std": None,
            "vol_proxy": None,
        }
        try:
            if analysis_df is None or len(analysis_df) < 50:
                return base

            close = analysis_df["close"].astype(float) if "close" in analysis_df.columns else None
            if close is None:
                return base
            ret = close.pct_change().dropna()
            ret_std = float(ret.std()) if len(ret) > 5 else 0.0
            base["returns_std"] = float(ret_std)

            # ATR-relative (median) if available
            atr_rel_med = None
            try:
                if {"high", "low", "close"}.issubset(set(analysis_df.columns)) and len(analysis_df) > 30:
                    high = analysis_df["high"].astype(float)
                    low = analysis_df["low"].astype(float)
                    prev_close = close.shift(1).fillna(close.iloc[0])
                    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
                    atr = tr.rolling(14, min_periods=14).mean()
                    atr_rel = (atr / close).dropna()
                    if len(atr_rel) > 0:
                        atr_rel_med = float(atr_rel.median())
            except Exception:
                atr_rel_med = None
            base["atr_rel_med"] = float(atr_rel_med) if (atr_rel_med is not None) else None

            vol_proxy = float(atr_rel_med if (atr_rel_med is not None and atr_rel_med > 0) else max(0.0, ret_std))
            vol_proxy = float(np.clip(vol_proxy, 1e-6, 0.20))
            base["vol_proxy"] = float(vol_proxy)
            base["risk_calc_source"] = "atr_rel_med" if (atr_rel_med is not None and atr_rel_med > 0) else "returns_std"

            # stop_loss / take_profit from volatility proxy
            # keep inside existing global clips later: stop_loss in [-0.05,-0.01], take_profit in [0.02,0.05]
            stop_loss = -float(np.clip(2.0 * vol_proxy, 0.01, 0.05))
            take_profit = float(np.clip(1.5 * vol_proxy, 0.02, 0.05))

            # min_hold_time from median run-length of return sign (simple regime persistence proxy)
            min_hold = 12
            try:
                sgn = np.sign(ret.fillna(0.0).values)
                # collapse zeros
                sgn = np.where(sgn == 0, 0, sgn)
                runs = []
                cur = 0
                last = 0
                for x in sgn:
                    if x == 0:
                        continue
                    if x == last:
                        cur += 1
                    else:
                        if cur > 0:
                            runs.append(cur)
                        cur = 1
                        last = x
                if cur > 0:
                    runs.append(cur)
                if runs:
                    med_run = float(np.median(runs))
                    min_hold = int(np.clip(int(med_run * 2), 12, 96))
            except Exception:
                min_hold = 12

            # volatility_multiplier relative to a baseline (2% typical)
            volatility_multiplier = float(np.clip(vol_proxy / 0.02, 0.5, 2.0))

            base.update(
                {
                    "volatility_multiplier": volatility_multiplier,
                    "min_hold_time": int(min_hold),
                    "stop_loss": float(stop_loss),
                    "take_profit": float(take_profit),
                }
            )
            return base
        except Exception:
            return base
        
    def get_crypto_profile(self, symbol: str) -> Dict:
        """Получает профиль криптовалюты"""
        # Убираем временные метки и получаем базовый символ
        base_symbol = symbol.split('_')[0] if '_' in symbol else symbol
        
        if base_symbol in self.crypto_profiles:
            return self.crypto_profiles[base_symbol].copy()
        else:
            # Дефолтный профиль (без warning-спама: динамика считается в get_trading_params()).
            return {
                'volatility_multiplier': 1.0,
                'volume_threshold': 0.002,
                'price_sensitivity': 1.0,
                'trend_strength': 1.0,
                'min_hold_time': 24,
                'stop_loss': -0.05,
                'take_profit': 0.08,
            }
    
    def analyze_market_conditions(self, df: pd.DataFrame, train_split_point: int = None) -> Dict:
        """
        Анализирует рыночные условия для адаптации параметров.
        
        Args:
            df: DataFrame с историческими данными
            train_split_point: Точка разделения train/test данных
        """
        try:
            # ИСПРАВЛЕНИЕ: Используем только обучающие данные для анализа
            if train_split_point is not None:
                analysis_df = df.iloc[:train_split_point]
            else:
                analysis_df = df
            
            # Проверяем минимальное количество данных
            if len(analysis_df) < 20:
                # Возвращаем дефолтные значения для коротких данных
                return self._get_default_market_conditions()
            
            # Волатильность (стандартное отклонение returns)
            returns = analysis_df['close'].pct_change().dropna()
            volatility = returns.std() if len(returns) > 0 else 0.02
            
            # Объем (средний объем за последние N баров)
            volume_window = min(20, len(analysis_df) - 1)
            volume_ma = analysis_df['volume'].rolling(volume_window).mean()
            current_volume = analysis_df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_ma.iloc[-1] if not pd.isna(volume_ma.iloc[-1]) and volume_ma.iloc[-1] > 0 else 1.0
            
            # Тренд (наклон линии тренда) - только по доступным данным
            if len(analysis_df) > 2:
                price_trend = np.polyfit(range(len(analysis_df)), analysis_df['close'], 1)[0]
                trend_strength = abs(price_trend) / analysis_df['close'].mean()
            else:
                trend_strength = 0.001
            
            # Рыночные условия
            market_conditions = {
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'trend_strength': trend_strength,
                'is_high_volatility': volatility > 0.02,  # >2% в день
                'is_high_volume': volume_ratio > 1.5,     # >150% от среднего
                'is_strong_trend': trend_strength > 0.001, # Сильный тренд
            }
            
            return market_conditions
            
        except Exception as e:
            logger.error(f"Ошибка анализа рыночных условий: {e}")
            return self._get_default_market_conditions()
    
    def _get_default_market_conditions(self) -> Dict:
        """Возвращает дефолтные рыночные условия"""
        return {
            'volatility': 0.02,
            'volume_ratio': 1.0,
            'trend_strength': 0.001,
            'is_high_volatility': False,
            'is_high_volume': False,
            'is_strong_trend': False,
        }
    
    def adapt_parameters(self, symbol: str, df: pd.DataFrame, train_split_point: int = None) -> Dict:
        """
        Адаптирует параметры под конкретную криптовалюту и рыночные условия.
        
        Args:
            symbol: Символ криптовалюты
            df: DataFrame с историческими данными
            train_split_point: Точка разделения train/test данных
        """
        # Full-auto базовый профиль: всегда вычисляем из train-части OHLCV.
        # Это убирает "профили от балды" и делает поведение одинаковым для известных/неизвестных символов.
        profile = self._full_auto_base_profile(symbol, df, train_split_point)
        
        # Анализ рыночных условий ТОЛЬКО по обучающим данным
        market_conditions = self.analyze_market_conditions(df, train_split_point)
        
        # Адаптация параметров
        adapted_params = profile.copy()
        
        # --- Автокалибровка volume_threshold по train-части (ATR-подобная относительная вола) ---
        try:
            analysis_df = df.iloc[:train_split_point] if train_split_point is not None else df
            if {'high','low','close'}.issubset(set(analysis_df.columns)) and len(analysis_df) > 30:
                close = analysis_df['close'].astype(float)
                high = analysis_df['high'].astype(float)
                low  = analysis_df['low'].astype(float)
                prev_close = close.shift(1).fillna(close.iloc[0])
                tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
                atr_rel = (tr.rolling(12, min_periods=12).mean() / close).dropna()
                if len(atr_rel) > 0:
                    q = float(np.clip(float(0.70), 0.50, 0.95))  # базовый квантиль (можно вынести в конфиг)
                    thr_q = float(atr_rel.quantile(q))
                    # Робастный бэкап: median + 0.5*IQR
                    med = float(atr_rel.median())
                    q1, q3 = float(atr_rel.quantile(0.25)), float(atr_rel.quantile(0.75))
                    iqr = max(1e-9, (q3 - q1))
                    thr_robust = med + 0.5 * iqr
                    thr = max(thr_q, thr_robust)
                    # Клипы под нашу шкалу фильтра в env
                    thr = float(np.clip(thr, 0.0010, 0.0060))
                    adapted_params['volume_threshold'] = thr
        except Exception:
            pass

        # --- Автоподбор трендовых окон/порогов на train-части ---
        try:
            enable_autotune = str(os.getenv('ADAPT_AUTOTUNE_TREND', '1')).strip().lower() in ('1','true','yes','on')
        except Exception:
            enable_autotune = True
        if enable_autotune:
            try:
                analysis_df = df.iloc[:train_split_point] if train_split_point is not None else df
                if 'close' in analysis_df.columns and len(analysis_df) >= 600:
                    close_s = analysis_df['close'].astype(float)
                    ret_s = close_s.pct_change().astype(float)
                    base_windows = adapted_params.get('trend_windows') if isinstance(adapted_params, dict) else None
                    base_windows = [int(x) for x in (base_windows or [60, 180, 300]) if int(x) > 1]
                    # Кандидаты окон (три масштаба)
                    def _scale(ws, k):
                        return [max(10, int(x * k)) for x in ws]
                    candidates = [
                        base_windows,
                        _scale(base_windows, 0.8),
                        _scale(base_windows, 1.25),
                        [96, 288, 576],
                        [120, 360, 720],
                        [180, 540, 1080],
                    ]
                    # Очистка кандидатов
                    uniq = []
                    seen = set()
                    for ws in candidates:
                        key = tuple(sorted(ws))
                        if key not in seen:
                            seen.add(key)
                            uniq.append(ws)
                    candidates = uniq

                    def _make_tau_map(close: pd.Series, windows: list[int]) -> dict:
                        tau_map = {}
                        for w in windows:
                            if len(close) > w + 5:
                                drift = (close - close.shift(w-1)) / close.shift(w-1)
                                # Используем 75-й перцентиль величины дрейфа как порог тренда
                                tau = float(drift.abs().quantile(0.75))
                                tau = float(np.clip(tau, 0.01, 0.12))
                                tau_map[int(w)] = tau
                        return tau_map

                    def _score_windows(close: pd.Series, ret: pd.Series, windows: list[int], tau_map: dict) -> tuple[float, dict, np.ndarray]:
                        # Формируем голосование режимов по окнам
                        mode_votes = []
                        strength_all = []
                        n = len(close)
                        for w in windows:
                            if n <= w:
                                continue
                            drift_w = (close - close.shift(w-1)) / close.shift(w-1)
                            vol_w = ret.rolling(w, min_periods=w).std()
                            strength = (drift_w.abs() / (vol_w * np.sqrt(max(w, 1))))
                            tau = float(tau_map.get(int(w), 0.03))
                            lbl = pd.Series(0, index=drift_w.index)
                            lbl[(drift_w > tau) & (strength > 0.5)] = 1
                            lbl[(drift_w < -tau) & (strength > 0.5)] = -1
                            mode_votes.append(lbl.fillna(0).astype(int).values)
                            strength_all.append(strength.fillna(0.0).values)
                        if not mode_votes:
                            return -1e9, tau_map, np.zeros(n)
                        votes = np.stack(mode_votes, axis=1)
                        sums = votes.sum(axis=1)
                        regime = np.where(sums > 0, 1, np.where(sums < 0, -1, 0))
                        # Метрики
                        strength_mat = np.stack(strength_all, axis=1)
                        strength_avg = float(np.mean(strength_mat[regime != 0])) if np.any(regime != 0) else 0.0
                        churn = float(np.mean(regime[1:] != regime[:-1])) if len(regime) > 1 else 1.0
                        up_ratio = float(np.mean(regime == 1))
                        down_ratio = float(np.mean(regime == -1))
                        flat_ratio = float(np.mean(regime == 0))
                        imbalance = 0.0
                        if flat_ratio > 0.7:
                            imbalance += (flat_ratio - 0.7) * 2.0
                        if down_ratio < 0.05:
                            imbalance += (0.05 - down_ratio) * 2.0
                        score = strength_avg - 0.5 * churn - imbalance
                        return score, tau_map, regime

                    best = (-1e9, None, None, None)  # score, windows, tau_map, regime
                    for ws in candidates:
                        tau_map = _make_tau_map(close_s, ws)
                        sc, tm, reg = _score_windows(close_s, ret_s, ws, tau_map)
                        if sc > best[0]:
                            best = (sc, ws, tm, reg)
                    if best[1] is not None and best[2] is not None:
                        ws_best = [int(x) for x in best[1]]
                        tm_best = {int(k): float(v) for k, v in best[2].items()}
                        reg_best = best[3]
                        adapted_params['trend_windows'] = ws_best
                        adapted_params['trend_tau'] = tm_best
                        # Посчитаем статистики и предвычислим последовательности
                        if reg_best is not None and hasattr(reg_best, 'shape'):
                            reg = np.asarray(reg_best, dtype=int)
                            def _runs(arr: np.ndarray, val: int) -> list:
                                runs, run = [], 0
                                for x in arr:
                                    if x == val:
                                        run += 1
                                    else:
                                        if run > 0:
                                            runs.append(run)
                                            run = 0
                                if run > 0:
                                    runs.append(run)
                                return runs
                            runs_up = _runs(reg, 1)
                            runs_down = _runs(reg, -1)
                            med_up = float(np.median(runs_up)) if runs_up else 0.0
                            med_down = float(np.median(runs_down)) if runs_down else 0.0
                            stats = {
                                'up': {
                                    'count': int((reg == 1).sum()),
                                    'ratio': float(np.mean(reg == 1)),
                                    'avg_len': float(np.mean(runs_up)) if runs_up else 0.0,
                                    'median_len': med_up,
                                    'max_len': int(np.max(runs_up)) if runs_up else 0,
                                },
                                'flat': {
                                    'count': int((reg == 0).sum()),
                                    'ratio': float(np.mean(reg == 0)),
                                    'avg_len': 0.0,
                                    'median_len': 0.0,
                                    'max_len': 0,
                                },
                                'down': {
                                    'count': int((reg == -1).sum()),
                                    'ratio': float(np.mean(reg == -1)),
                                    'avg_len': float(np.mean(runs_down)) if runs_down else 0.0,
                                    'median_len': med_down,
                                    'max_len': int(np.max(runs_down)) if runs_down else 0,
                                }
                            }
                            adapted_params['regime_stats'] = {
                                'windows': ws_best,
                                'thresholds': tm_best,
                                'stats': stats,
                            }
                            # Предвычислим последовательности runlen и прогресса
                            n = reg.shape[0]
                            run_up_seq = np.zeros(n, dtype=np.int32)
                            run_down_seq = np.zeros(n, dtype=np.int32)
                            cnt = 0
                            for i in range(n):
                                if reg[i] == 1:
                                    cnt += 1
                                else:
                                    cnt = 0
                                run_up_seq[i] = cnt
                            cnt = 0
                            for i in range(n):
                                if reg[i] == -1:
                                    cnt += 1
                                else:
                                    cnt = 0
                                run_down_seq[i] = cnt
                            eps = 1e-6
                            prog_up = np.clip(run_up_seq / max(med_up, eps), 0.0, 1.5)
                            prog_down = np.clip(run_down_seq / max(med_down, eps), 0.0, 1.5)
                            adapted_params['regime_precomputed'] = {
                                'regime_seq': reg.astype(int).tolist(),
                                'runlen_up': run_up_seq.astype(int).tolist(),
                                'runlen_down': run_down_seq.astype(int).tolist(),
                                'progress_up': prog_up.astype(float).tolist(),
                                'progress_down': prog_down.astype(float).tolist(),
                            }
            except Exception:
                pass

        # --- Статистика трендовых режимов (up/flat/down) на train-части ---
        try:
            analysis_df = df.iloc[:train_split_point] if train_split_point is not None else df
            if 'close' in analysis_df.columns and len(analysis_df) >= 400:
                close = analysis_df['close'].astype(float).values
                # Окна/пороги берём из профиля монеты (fallback — дефолты)
                prof_windows = adapted_params.get('trend_windows') if isinstance(adapted_params, dict) else None
                windows = tuple(int(w) for w in (prof_windows or [60, 180, 300]))
                prof_tau = adapted_params.get('trend_tau') if isinstance(adapted_params, dict) else None
                tau_map = dict(prof_tau or {60: 0.02, 180: 0.05, 300: 0.08})
                # Нормируем усилие тренда простым прокси: |drift| / (std(returns)*sqrt(w))
                returns = pd.Series(close).pct_change().astype(float)
                mode_votes = []  # список массивов меток по каждому окну
                for w in windows:
                    if len(close) <= w:
                        continue
                    drift_w = (pd.Series(close) - pd.Series(close).shift(w-1)) / pd.Series(close).shift(w-1)
                    vol_w = returns.rolling(w, min_periods=w).std()
                    strength = (drift_w.abs() / (vol_w * np.sqrt(max(w, 1))))
                    tau = float(tau_map.get(w, 0.03))
                    # Базовые метки по дрейфу
                    lbl = pd.Series(0, index=drift_w.index)  # 0=flat, +1=up, -1=down
                    lbl[(drift_w > tau) & (strength > 0.5)] = 1
                    lbl[(drift_w < -tau) & (strength > 0.5)] = -1
                    mode_votes.append(lbl.fillna(0).astype(int).values)
                regime = None
                if mode_votes:
                    # Мажоритарное голосование по окнам
                    votes = np.stack(mode_votes, axis=1)  # [T, K]
                    sums = votes.sum(axis=1)
                    regime = np.where(sums > 0, 1, np.where(sums < 0, -1, 0))
                # Подсчёт серий и долей
                def _run_lengths(arr: np.ndarray, val: int) -> list:
                    if arr is None or len(arr) == 0:
                        return []
                    runs = []
                    run = 0
                    for x in arr:
                        if x == val:
                            run += 1
                        else:
                            if run > 0:
                                runs.append(run)
                                run = 0
                    if run > 0:
                        runs.append(run)
                    return runs
                stats = {}
                if regime is not None:
                    T = int((regime != 0).size)
                    for name, val in (('up', 1), ('flat', 0), ('down', -1)):
                        mask = (regime == val)
                        count = int(mask.sum())
                        ratio = float(count) / float(len(regime)) if len(regime) else 0.0
                        runs = _run_lengths(regime, val)
                        stats[name] = {
                            'count': count,
                            'ratio': ratio,
                            'avg_len': (float(np.mean(runs)) if runs else 0.0),
                            'median_len': (float(np.median(runs)) if runs else 0.0),
                            'max_len': (int(np.max(runs)) if runs else 0),
                        }
                    adapted_params['regime_stats'] = {
                        'windows': list(windows),
                        'thresholds': {int(k): float(v) for k, v in tau_map.items()},
                        'stats': stats,
                    }
        except Exception:
            pass
        
        # 1. Адаптация под волатильность
        if market_conditions['is_high_volatility']:
            adapted_params['volatility_multiplier'] *= 1.2
            adapted_params['stop_loss'] *= 1.2  # Больше терпения
            adapted_params['take_profit'] *= 1.2  # Больше прибыли
            adapted_params['min_hold_time'] = max(48, adapted_params['min_hold_time'] - 4)
        
        # 2. Адаптация под объем
        if market_conditions['is_high_volume']:
            adapted_params['volume_threshold'] *= 0.8  # Снижаем порог
            adapted_params['price_sensitivity'] *= 1.1  # Повышаем чувствительность
        
        # 3. Адаптация под тренд
        if market_conditions['is_strong_trend']:
            adapted_params['trend_strength'] *= 1.3
            adapted_params['min_hold_time'] = max(12, adapted_params['min_hold_time'] - 8)
            adapted_params['take_profit'] *= 1.1  # Больше прибыли в тренде
        
        # 4. Специфичные адаптации для разных криптовалют
        if 'BTC' in symbol:
            # BTC: более консервативно в нестабильные периоды
            if market_conditions['volatility'] > 0.03:
                adapted_params['min_hold_time'] += 8
        elif 'ETH' in symbol:
            # ETH: более агрессивно в тренде
            if market_conditions['is_strong_trend']:
                adapted_params['min_hold_time'] += 12
        elif 'TON' in symbol:
            # TON: стабильная монета, меньше адаптации
            adapted_params['volatility_multiplier'] *= 0.9
        elif 'BNB' in symbol:
            # BNB: чуть мягче фильтры и быстрее реакции
            adapted_params['volume_threshold'] *= 0.9
            adapted_params['min_hold_time'] = max(16, adapted_params['min_hold_time'] - 2)
        
        # Ограничиваем значения
        adapted_params['min_hold_time'] = max(12, min(96, adapted_params['min_hold_time']))
        adapted_params['stop_loss'] = max(-0.05, min(-0.01, adapted_params['stop_loss']))
        adapted_params['take_profit'] = max(0.02, min(0.05, adapted_params['take_profit']))
        
        return adapted_params
    
    def normalize_features(self, df: pd.DataFrame, symbol: str, train_split_point: int = None) -> pd.DataFrame:
        """
        Нормализует признаки с учетом профиля криптовалюты.
        
        Args:
            df: DataFrame с данными для нормализации
            symbol: Символ криптовалюты
            train_split_point: Точка разделения train/test данных
        """
        try:
            # Получаем адаптированные параметры ТОЛЬКО по обучающим данным
            params = self.adapt_parameters(symbol, df, train_split_point)
            
            # Копируем DataFrame
            normalized_df = df.copy()
            
            # 1. Нормализация цены с учетом волатильности
            volatility_mult = params['volatility_multiplier']
            normalized_df['price_normalized'] = df['close'].pct_change() * volatility_mult
            
            # 2. Нормализация объема с учетом порога
            volume_threshold = params['volume_threshold']
            normalized_df['volume_normalized'] = np.where(
                df['volume'] > df['volume'].rolling(20).mean() * volume_threshold,
                1.0,  # Высокий объем
                0.5   # Низкий объем
            )
            
            # 3. Нормализация тренда с учетом силы
            trend_strength = params['trend_strength']
            price_ma_short = df['close'].rolling(5).mean()
            price_ma_long = df['close'].rolling(20).mean()
            normalized_df['trend_normalized'] = (
                (price_ma_short - price_ma_long) / price_ma_long * trend_strength
            )
            
            # 4. Адаптивная нормализация технических индикаторов
            if 'rsi' in df.columns:
                # RSI: более чувствительный для волатильных монет
                sensitivity = params['price_sensitivity']
                normalized_df['rsi_normalized'] = (df['rsi'] - 50) / 50 * sensitivity
            
            if 'macd' in df.columns:
                # MACD: учитываем силу тренда
                trend_mult = params['trend_strength']
                normalized_df['macd_normalized'] = df['macd'] * trend_mult
            
            # 5. Добавляем адаптивные параметры в DataFrame
            for key, value in params.items():
                normalized_df[f'param_{key}'] = value
            
            logger.info(f"Адаптивная нормализация для {symbol}: volatility_mult={params['volatility_multiplier']:.2f}, "
                       f"min_hold={params['min_hold_time']}, stop_loss={params['stop_loss']:.3f}")
            
            return normalized_df
            
        except Exception as e:
            logger.error(f"Ошибка нормализации для {symbol}: {e}")
            return df
    
    def get_trading_params(self, symbol: str, df: pd.DataFrame, train_split_point: int = None) -> Dict:
        """
        Возвращает адаптированные торговые параметры.
        
        Args:
            symbol: Символ криптовалюты
            df: DataFrame с историческими данными
            train_split_point: Точка разделения train/test данных
        """
        # Кешируем компактный результат (без больших массивов), чтобы UI/обучение не считали повторно.
        timeframe = "5m"
        try:
            sig = self._df_signature(df)
            key = self._cache_key(symbol, timeframe, train_split_point, sig)
            cached = self._cache_load(key)
            if isinstance(cached, dict):
                return cached
        except Exception:
            pass

        params = self.adapt_parameters(symbol, df, train_split_point)
        
        out = {
            'min_hold_steps': int(params['min_hold_time']),
            'stop_loss_pct': params['stop_loss'],
            'take_profit_pct': params['take_profit'],
            'volume_threshold': params['volume_threshold'],
            'volatility_multiplier': params['volatility_multiplier'],
            'price_sensitivity': params['price_sensitivity'],
            'trend_strength': params['trend_strength'],
            # meta: source for SL/TP volatility proxy + key proxy values (for UI visibility)
            'risk_calc_source': params.get('risk_calc_source'),
            'atr_rel_med': params.get('atr_rel_med'),
            'returns_std': params.get('returns_std'),
            'vol_proxy': params.get('vol_proxy'),
        }
        # Передаём трендовые настройки и статистику/предвычисления, если посчитаны
        if 'regime_stats' in params:
            out['regime_stats'] = params['regime_stats']
        if 'trend_windows' in params:
            out['trend_windows'] = params['trend_windows']
        if 'trend_tau' in params:
            out['trend_tau'] = params['trend_tau']
        if 'regime_precomputed' in params:
            out['regime_precomputed'] = params['regime_precomputed']
        # В кеш кладём только компактную часть
        try:
            compact = dict(out)
            if "regime_precomputed" in compact:
                compact.pop("regime_precomputed", None)
            self._cache_save(key, compact)  # type: ignore[name-defined]
        except Exception:
            pass
        return out

# Глобальный экземпляр
adaptive_normalizer = AdaptiveNormalizer()
