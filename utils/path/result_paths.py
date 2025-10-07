"""Вспомогательные функции для работы с папками результатов OOS."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


_AGENT_FOLDERS = {
    'dqn': 'dqn',
    'sac': 'sac',
}

_LEGACY_EXCLUDE = {'dqn', 'sac'}


def _clean_symbol(symbol: str | None) -> str:
    if not symbol:
        return ''
    return symbol.replace('\x5c', '').replace('/', '').strip()


def _dqn_symbol_dir(symbol: str | None) -> str:
    cleaned = _clean_symbol(symbol).upper()
    return cleaned or 'UNKNOWN'


def _sac_symbol_dir(symbol: str | None) -> str:
    cleaned = _clean_symbol(symbol).lower()
    if cleaned.endswith('usdt'):
        cleaned = cleaned[:-4]
    return cleaned or 'unknown'


def get_agent_root(agent_type: str | None) -> Path:
    base = Path('result')
    agent = (agent_type or '').strip().lower()
    folder = _AGENT_FOLDERS.get(agent)
    if folder:
        return base / folder
    return base


def _iter_legacy_symbol_dirs() -> Iterable[Path]:
    base = Path('result')
    if not base.exists():
        return []
    try:
        return [d for d in base.iterdir() if d.is_dir() and d.name.lower() not in _LEGACY_EXCLUDE]
    except Exception:
        return []


def list_symbol_dirs(agent_type: str | None) -> list[Path]:
    dirs: list[Path] = []
    root = get_agent_root(agent_type)
    if root.exists():
        try:
            dirs.extend([d for d in root.iterdir() if d.is_dir()])
        except Exception:
            pass
    for legacy in _iter_legacy_symbol_dirs():
        if legacy not in dirs:
            dirs.append(legacy)
    return dirs


def resolve_symbol_dir(agent_type: str | None, symbol: str | None, create: bool = False) -> Optional[Path]:
    root = get_agent_root(agent_type)
    sym_clean = _clean_symbol(symbol)
    candidates: list[str] = []
    agent = (agent_type or '').strip().lower()
    if sym_clean:
        if agent == 'sac':
            candidates.extend({
                _sac_symbol_dir(sym_clean),
                sym_clean.lower(),
                sym_clean.upper(),
            })
        else:
            candidates.extend({
                _dqn_symbol_dir(sym_clean),
                sym_clean.lower(),
            })
    if root.exists():
        for cand in candidates:
            if not cand:
                continue
            candidate_dir = root / cand
            if candidate_dir.exists():
                return candidate_dir

    for legacy in _iter_legacy_symbol_dirs():
        if sym_clean and legacy.name.lower() == sym_clean.lower():
            return legacy

    if not create:
        return None

    if agent == 'sac':
        dir_name = _sac_symbol_dir(sym_clean) or 'unknown'
    else:
        dir_name = _dqn_symbol_dir(sym_clean) or 'UNKNOWN'
    symbol_dir = root / dir_name
    symbol_dir.mkdir(parents=True, exist_ok=True)
    return symbol_dir


def resolve_run_dir(
    agent_type: str | None,
    run_id: str | None,
    symbol_hint: str | None = None,
    create: bool = False,
) -> Optional[Path]:
    if not run_id:
        return None

    agent = (agent_type or '').strip().lower()
    symbol_dir = resolve_symbol_dir(agent, symbol_hint, create=False)
    runs_subdir: Optional[Path] = None
    if symbol_dir and (symbol_dir / 'runs').exists():
        candidate = symbol_dir / 'runs' / run_id
        if candidate.exists() or create:
            runs_subdir = candidate

    if runs_subdir and runs_subdir.exists():
        return runs_subdir

    root = get_agent_root(agent)
    if root.exists():
        try:
            for sym_dir in root.iterdir():
                if not sym_dir.is_dir():
                    continue
                candidate = sym_dir / 'runs' / run_id
                if candidate.exists():
                    return candidate
        except Exception:
            pass

    for legacy in _iter_legacy_symbol_dirs():
        candidate = legacy / 'runs' / run_id
        if candidate.exists():
            return candidate

    if not create:
        return None

    # Создаём новую структуру согласно типу агента
    if agent == 'sac':
        dir_name = _sac_symbol_dir(symbol_hint) or run_id.lower()
    else:
        dir_name = _dqn_symbol_dir(symbol_hint) or run_id.upper()
    symbol_dir = root / dir_name
    runs_dir = symbol_dir / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


