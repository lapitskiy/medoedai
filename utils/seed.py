import os
import random


def set_global_seed(seed: int | None) -> int | None:
    """Устанавливает сид для Python, NumPy и Torch (если доступен).

    Возвращает применённый сид или None, если сид не задан.
    """
    if seed is None:
        return None

    try:
        os.environ["PYTHONHASHSEED"] = str(int(seed))
    except Exception:
        pass

    try:
        random.seed(int(seed))
    except Exception:
        pass

    try:
        import numpy as _np  # noqa: WPS433 (module import inside function)
        _np.random.seed(int(seed))
    except Exception:
        pass

    try:
        import torch  # noqa: WPS433
        torch.manual_seed(int(seed))
        try:
            torch.cuda.manual_seed_all(int(seed))
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        # Torch может отсутствовать в окружении некоторых воркеров
        pass

    return int(seed)


