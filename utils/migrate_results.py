import os
import re
import json
import shutil
import pickle
from pathlib import Path
from datetime import datetime


RESULT_DIR = Path("result")


RE_MODEL = re.compile(r"^dqn_model_([a-z]+)_([A-Za-z0-9_]+)\.pth$")
RE_REPLAY = re.compile(r"^replay_buffer_([a-z]+)_([A-Za-z0-9_]+)\.pkl$")
RE_TRAIN = re.compile(r"^train_result_([a-z]+)_([A-Za-z0-9_]+)\.pkl$")


def _strip_updates(code: str) -> str:
    # Убираем все суффиксы _update в конце строки
    while code.endswith("_update"):
        code = code[: -len("_update")]
    return code


def _parent_code(code: str) -> str | None:
    # Убрать один суффикс _update, если есть
    if code.endswith("_update"):
        return code[: -len("_update")]
    return None


def _read_train_metadata_seed(train_file: Path) -> tuple[int | None, int | None]:
    if not train_file.exists():
        return None, None
    try:
        with open(train_file, "rb") as f:
            data = pickle.load(f)
        meta = data.get("train_metadata", {}) if isinstance(data, dict) else {}
        seed = meta.get("seed") if isinstance(meta, dict) else None
        episodes = data.get("actual_episodes", data.get("episodes")) if isinstance(data, dict) else None
        try:
            if episodes is not None:
                episodes = int(episodes)
        except Exception:
            episodes = None
        return seed, episodes
    except Exception:
        return None, None


def migrate():
    if not RESULT_DIR.exists():
        print("⚠️ Папка result не найдена")
        return

    # Собираем все файлы в корне result
    files = [p for p in RESULT_DIR.iterdir() if p.is_file()]
    # Группируем по (symbol, code)
    runs: dict[tuple[str, str], dict[str, Path]] = {}

    for p in files:
        name = p.name
        m = RE_MODEL.match(name)
        if m:
            sym, code = m.group(1), m.group(2)
            runs.setdefault((sym, code), {})["model"] = p
            continue
        m = RE_REPLAY.match(name)
        if m:
            sym, code = m.group(1), m.group(2)
            runs.setdefault((sym, code), {})["replay"] = p
            continue
        m = RE_TRAIN.match(name)
        if m:
            sym, code = m.group(1), m.group(2)
            runs.setdefault((sym, code), {})["train"] = p
            continue

    if not runs:
        print("ℹ️ В корне result не найдено файлов моделей для миграции")
        return

    moved = 0
    for (sym, code), parts in sorted(runs.items()):
        symbol_dir = RESULT_DIR / sym.upper() / "runs"
        run_id = code  # используем код как run_id (включая _update*
        run_dir = symbol_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Источники
        model_src = parts.get("model")
        replay_src = parts.get("replay")
        train_src = parts.get("train")

        # Назначения
        if model_src and model_src.exists():
            shutil.move(str(model_src), str(run_dir / "model.pth"))
        if replay_src and replay_src.exists():
            shutil.move(str(replay_src), str(run_dir / "replay.pkl"))
        if train_src and train_src.exists():
            shutil.move(str(train_src), str(run_dir / "train_result.pkl"))

        # Читаем seed/эпизоды из train_result (если есть)
        seed, episodes = _read_train_metadata_seed(run_dir / "train_result.pkl")

        # Определяем parent/root
        parent = _parent_code(code)
        root = _strip_updates(code)

        manifest = {
            "run_id": run_id,
            "parent_run_id": parent,
            "root_id": root,
            "symbol": sym.upper(),
            "seed": seed,
            "episodes_end": episodes,
            "episodes_added": episodes,
            "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "artifacts": {
                "model": "model.pth" if (run_dir / "model.pth").exists() else None,
                "replay": "replay.pkl" if (run_dir / "replay.pkl").exists() else None,
                "result": "train_result.pkl" if (run_dir / "train_result.pkl").exists() else None,
            },
        }
        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        moved += 1

    print(f"✅ Миграция завершена: оформлено {moved} запусков в result/<SYMBOL>/runs/<run_id>/")


if __name__ == "__main__":
    migrate()


